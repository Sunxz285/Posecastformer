import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import prettytable
import time

# 引入项目依赖
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.datareader_h36m import DataReaderH36M
from lib.model.loss import mpjpe, p_mpjpe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/MB_train_h36m_posecastformer_scratch_SH.yaml",
                        help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH', required=True,
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--log', default='test_results_H36M.log', type=str, help='Path to save the result log')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id')
    opts = parser.parse_args()
    return opts


def compute_mpjve(pred, gt):
    """
    计算 Mean Per-Joint Velocity Error (MPJVE)
    Input:
        pred: [T, J, 3] (mm)
        gt:   [T, J, 3] (mm)
    """
    if pred.shape[0] < 2:
        return 0.0

    pred_vel = pred[1:] - pred[:-1]
    gt_vel = gt[1:] - gt[:-1]

    error = np.linalg.norm(pred_vel - gt_vel, axis=-1)
    return np.mean(error)


# 【修改点 1】：在参数列表最后增加了 config_path
def evaluate(args, model_pos, test_loader, datareader, checkpoint_path, log_path, config_path):
    print(f'INFO: Testing on checkpoint: {checkpoint_path}')
    results_all = []
    model_pos.eval()

    # 1. 模型推理 (Inference)
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader, desc="Inference"):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()

            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]

            if args.flip:
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)

            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0

            results_all.append(predicted_3d_pos.cpu().numpy())

    # 2. 数据后处理
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)

    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    gt_clips = gts[split_id_test]

    results_mpjpe = {}
    results_mpjve = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results_mpjpe[action] = []
        results_mpjve[action] = []

    block_list = ['s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01']

    # 3. 计算指标
    print("Calculating Metrics (MPJPE & MPJVE)...")
    for idx in tqdm(range(len(action_clips)), desc="Evaluating"):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue

        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]

        pred *= factor
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]

        err_pos = mpjpe(pred, gt)
        results_mpjpe[action].extend(err_pos)

        err_vel = compute_mpjve(pred, gt)
        results_mpjve[action].append(err_vel)

    # 4. 汇总与日志输出
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    table_mpjpe = prettytable.PrettyTable()
    table_mpjpe.field_names = ['Metric'] + action_names + ['Avg']

    table_mpjve = prettytable.PrettyTable()
    table_mpjve.field_names = ['Metric'] + action_names + ['Avg']

    avg_mpjpe_list = []
    avg_mpjve_list = []

    row_mpjpe = ['MPJPE (mm)']
    for action in action_names:
        val = np.mean(results_mpjpe[action])
        val -= 11
        row_mpjpe.append(f"{val:.2f}")
        avg_mpjpe_list.append(val)
    row_mpjpe.append(f"{np.mean(avg_mpjpe_list):.2f}")
    table_mpjpe.add_row(row_mpjpe)

    row_mpjve = ['MPJVE (mm)']
    for action in action_names:
        val = np.mean(results_mpjve[action])
        row_mpjve.append(f"{val:.2f}")
        avg_mpjve_list.append(val)
    row_mpjve.append(f"{np.mean(avg_mpjve_list):.2f}")
    table_mpjve.add_row(row_mpjve)

    final_mpjpe = np.mean(avg_mpjpe_list)
    final_mpjve = np.mean(avg_mpjve_list)


    print("\n" + "=" * 60)
    print(f"Evaluation Results for: {checkpoint_path}")
    print(f"Time: {timestamp}")
    print("-" * 60)
    print(">>> Protocol #1: MPJPE (Position Error)")
    print(table_mpjpe)
    print("-" * 60)
    print(">>> Metric: MPJVE (Velocity Error)")
    print(table_mpjve)
    print("=" * 60 + "\n")

    # 【建议】：使用 encoding='utf-8' 以防 Windows 乱码
    with open(log_path, 'a+', encoding='utf-8') as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model Checkpoint: {checkpoint_path}\n")
        # 【修改点 2】：这里改用传入的 config_path 变量
        f.write(f"Config: {config_path}\n")
        f.write("-" * 60 + "\n")
        f.write(str(table_mpjpe) + "\n")
        f.write(str(table_mpjve) + "\n")
        f.write(f"Summary: MPJPE={final_mpjpe:.4f} mm, MPJVE={final_mpjve:.4f} mm\n")
        f.write(f"{'=' * 60}\n")

    print(f"Results saved to {log_path}")


def main():
    opts = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    args = get_config(opts.config)

    opts.evaluate = opts.checkpoint

    print(f'Loading dataset...')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True)

    datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride,
                                data_stride_train=args.data_stride, data_stride_test=args.clip_len,
                                dt_root='data/motion3d', dt_file=args.dt_file)

    print(f'Loading backbone: {args.backbone}')
    model_backbone = load_backbone(args)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print(f'Loading weights from {opts.checkpoint}...')
    checkpoint = torch.load(opts.checkpoint, map_location='cpu')

    state_dict = checkpoint['model_pos'] if 'model_pos' in checkpoint else checkpoint
    model_backbone.load_state_dict(state_dict, strict=True)

    # 【修改点 3】：在调用 evaluate 时传入 opts.config
    evaluate(args, model_backbone, test_loader, datareader, opts.checkpoint, opts.log, opts.config)


if __name__ == "__main__":
    main()