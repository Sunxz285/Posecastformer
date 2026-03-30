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


def evaluate(args, model_pos, test_loader, datareader, checkpoint_path, log_path, config_path):
    print(f'INFO: Testing on checkpoint: {checkpoint_path}')
    results_all = []
    model_pos.eval()

    # 1. 模型推理 (Inference)
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader, desc="Inference"):
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

    # ========== 新增：截断数组以匹配 results_all 的长度 ==========
    n_results = len(results_all)
    if len(action_clips) != n_results:
        print(f"Warning: action_clips length ({len(action_clips)}) != results_all length ({n_results}). Truncating.")
        action_clips = action_clips[:n_results]
        factor_clips = factor_clips[:n_results]
        source_clips = source_clips[:n_results]
        gt_clips = gt_clips[:n_results]

    # 用于存储三个指标的字典
    results_mpjpe = {}
    results_mpjve = {}
    results_pampjpe = {}

    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results_mpjpe[action] = []
        results_mpjve[action] = []
        results_pampjpe[action] = []

    block_list = ['s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01']

    # 3. 计算指标
    print("Calculating Metrics (MPJPE, MPJVE, PA-MPJPE)...")
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

        err_pampjpe = p_mpjpe(pred, gt)
        results_pampjpe[action].extend(err_pampjpe)

    # 4. 汇总与日志输出
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    table_mpjpe = prettytable.PrettyTable()
    table_mpjpe.field_names = ['Metric'] + action_names + ['Avg']

    table_mpjve = prettytable.PrettyTable()
    table_mpjve.field_names = ['Metric'] + action_names + ['Avg']

    table_pampjpe = prettytable.PrettyTable()
    table_pampjpe.field_names = ['Metric'] + action_names + ['Avg']

    avg_mpjpe_list = []
    avg_mpjve_list = []
    avg_pampjpe_list = []

    # MPJPE 行
    row_mpjpe = ['MPJPE (mm)']
    for action in action_names:
        val = np.mean(results_mpjpe[action])
        row_mpjpe.append(f"{val:.2f}")
        avg_mpjpe_list.append(val)
    row_mpjpe.append(f"{np.mean(avg_mpjpe_list):.2f}")
    table_mpjpe.add_row(row_mpjpe)

    # MPJVE 行
    row_mpjve = ['MPJVE (mm)']
    for action in action_names:
        val = np.mean(results_mpjve[action])
        row_mpjve.append(f"{val:.2f}")
        avg_mpjve_list.append(val)
    row_mpjve.append(f"{np.mean(avg_mpjve_list):.2f}")
    table_mpjve.add_row(row_mpjve)

    # PA-MPJPE 行
    row_pampjpe = ['PA-MPJPE (mm)']
    for action in action_names:
        val = np.mean(results_pampjpe[action])
        row_pampjpe.append(f"{val:.2f}")
        avg_pampjpe_list.append(val)
    row_pampjpe.append(f"{np.mean(avg_pampjpe_list):.2f}")
    table_pampjpe.add_row(row_pampjpe)

    final_mpjpe = np.mean(avg_mpjpe_list)
    final_mpjve = np.mean(avg_mpjve_list)
    final_pampjpe = np.mean(avg_pampjpe_list)

    print("\n" + "=" * 60)
    print(f"Evaluation Results for: {checkpoint_path}")
    print(f"Time: {timestamp}")
    print("-" * 60)
    print(">>> Protocol #1: MPJPE (Position Error)")
    print(table_mpjpe)
    print("-" * 60)
    print(">>> Metric: MPJVE (Velocity Error)")
    print(table_mpjve)
    print("-" * 60)
    print(">>> Metric: PA-MPJPE (Procrustes-aligned MPJPE)")
    print(table_pampjpe)
    print("=" * 60 + "\n")

    # 保存日志
    with open(log_path, 'a+', encoding='utf-8') as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model Checkpoint: {checkpoint_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write("-" * 60 + "\n")
        f.write(str(table_mpjpe) + "\n")
        f.write(str(table_mpjve) + "\n")
        f.write(str(table_pampjpe) + "\n")
        f.write(f"Summary: MPJPE={final_mpjpe:.4f} mm, MPJVE={final_mpjve:.4f} mm, PA-MPJPE={final_pampjpe:.4f} mm\n")
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
    checkpoint = torch.load(opts.checkpoint, map_location='cpu', weights_only=False)  # 保留原有设置

    # 获取 checkpoint 中的状态字典
    state_dict = checkpoint['model_pos'] if 'model_pos' in checkpoint else checkpoint

    # 获取当前模型的状态字典
    model_dict = model_backbone.state_dict()

    # ========== 修改点：部分加载，跳过形状不匹配的层 ==========
    filtered_state_dict = {}
    skipped_keys = []
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                skipped_keys.append(f"{k} (shape mismatch: checkpoint {v.shape} vs model {model_dict[k].shape})")
        else:
            skipped_keys.append(f"{k} (key not found in model)")

    if skipped_keys:
        print("Warning: The following keys were skipped during loading:")
        for key in skipped_keys:
            print(f"  - {key}")

    # 使用 strict=False 加载过滤后的字典（忽略不匹配的键）
    model_backbone.load_state_dict(filtered_state_dict, strict=False)

    # 可选：打印成功加载的键数量
    print(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} keys successfully.")

    # 调用评估函数
    evaluate(args, model_backbone, test_loader, datareader, opts.checkpoint, opts.log, opts.config)


if __name__ == "__main__":
    main()