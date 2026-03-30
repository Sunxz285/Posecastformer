import os
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import prettytable

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.data.dataset_motion_3d_cpn import MotionDataset3D
from lib.data.datareader_h36m_cpn import DataReaderH36M_CPN as DataReader
from lib.utils.utils_data import flip_data
from lib.model.loss import p_mpjpe          # <-- 导入 PA-MPJPE 函数

def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output", type=str, default="", help="Path to save the results (optional).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()

def velocity_error(pred, gt):
    """
    Mean Per Joint Velocity Error (MPJVE)
    pred, gt: (T, J, 3) numpy arrays, root-relative.
    Returns scalar in mm.
    """
    vel_pred = pred[1:] - pred[:-1]          # (T-1, J, 3)
    vel_gt   = gt[1:]   - gt[:-1]
    err = np.linalg.norm(vel_pred - vel_gt, axis=-1)   # (T-1, J)
    return np.mean(err)

def main():
    opts = parse_args()
    set_random_seed(opts.seed)

    # 加载配置
    args = get_config(opts.config)

    # 初始化数据读取器（用于反归一化和获取动作标签）
    datareader = DataReader(
        n_frames=args.clip_len,
        sample_stride=args.sample_stride,
        data_stride_train=args.data_stride,
        data_stride_test=args.clip_len,          # 测试时无重叠
        dt_root='data/motion3d',
        dt_file=args.dt_file
    )
    if hasattr(datareader, 'prepare_dataset'):
        datareader.prepare_dataset()

    # 获取测试集滑动窗口索引及每个窗口对应的动作标签
    _, split_id_test = datareader.get_split_id()
    actions_list = datareader.dt_dataset['test']['action']
    print(f"Number of test clips: {len(actions_list)}")

    # 创建测试数据集和 DataLoader
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # 构建模型
    model = load_backbone(args)

    # 加载 checkpoint
    print(f"Loading checkpoint from {opts.checkpoint}")
    checkpoint = torch.load(opts.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_pos', checkpoint)
    # 去除 DataParallel 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    print("Model loaded successfully.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 收集预测结果和 ground truth
    preds_list, gts_list = [], []
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader, desc="Inference"):
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                batch_gt = batch_gt.cuda()   # 移动到 GPU 用于后续计算（如果需要）

            # 测试时翻转集成（如果配置启用）
            if args.flip:
                batch_input_flip = flip_data(batch_input)
                pred_1 = model(batch_input)
                pred_flip = model(batch_input_flip)
                pred_2 = flip_data(pred_flip)
                predicted_3d = (pred_1 + pred_2) / 2
            else:
                predicted_3d = model(batch_input)

            preds_list.append(predicted_3d.cpu().numpy())
            gts_list.append(batch_gt.cpu().numpy())

    preds_all = np.concatenate(preds_list, axis=0)   # (N, T, J, 3)
    gts_all   = np.concatenate(gts_list, axis=0)     # (N, T, J, 3)

    # 对预测和真值都进行反归一化（与训练时评估一致）
    preds_denorm = datareader.denormalize(preds_all)
    gts_denorm   = datareader.denormalize(gts_all)

    # 按动作分组存储误差
    action_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
                    'Photo', 'Posing', 'Purchases', 'Sitting', 'SittingDown',
                    'Smoking', 'Waiting', 'WalkDog', 'Walking', 'WalkTogether']
    results = {act: {'mpjpe': [], 'mpjve': [], 'pa_mpjpe': []} for act in action_names}

    N = len(preds_denorm)
    assert N == len(actions_list), f"Clip count mismatch: {N} vs {len(actions_list)}"

    for i in range(N):
        pred = preds_denorm[i]          # (T, J, 3)
        gt   = gts_denorm[i]             # (T, J, 3)
        action = actions_list[i]

        # 只处理标准动作（忽略 unknown）
        if action not in results:
            continue

        # Root-relative (以根关节为原点)
        pred = pred - pred[:, 0:1, :]
        gt   = gt   - gt[:, 0:1, :]

        # MPJPE
        err_mpjpe = np.mean(np.linalg.norm(pred - gt, axis=-1))
        results[action]['mpjpe'].append(err_mpjpe)

        # MPJVE
        err_mpjve = velocity_error(pred, gt)
        results[action]['mpjve'].append(err_mpjve)

        # PA-MPJPE (Procrustes 对齐后的 MPJPE)
        # 直接传入 (T, J, 3)，p_mpjpe 返回 (T,) 数组，取平均
        err_pa = np.mean(p_mpjpe(pred, gt))   # <-- 修复点
        results[action]['pa_mpjpe'].append(err_pa)

    # 计算每个动作的平均值
    avg_mpjpe_per_action = []
    avg_mpjve_per_action = []
    avg_pa_mpjpe_per_action = []
    for act in action_names:
        mpjpe_vals = results[act]['mpjpe']
        mpjve_vals = results[act]['mpjve']
        pa_vals    = results[act]['pa_mpjpe']
        if len(mpjpe_vals) > 0:
            avg_mpjpe_per_action.append(np.mean(mpjpe_vals))
            avg_mpjve_per_action.append(np.mean(mpjve_vals))
            avg_pa_mpjpe_per_action.append(np.mean(pa_vals))
        else:
            avg_mpjpe_per_action.append(float('nan'))
            avg_mpjve_per_action.append(float('nan'))
            avg_pa_mpjpe_per_action.append(float('nan'))

    mean_mpjpe = np.nanmean(avg_mpjpe_per_action)
    mean_mpjve = np.nanmean(avg_mpjve_per_action)
    mean_pa    = np.nanmean(avg_pa_mpjpe_per_action)

    # ----- 格式化输出 -----
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"Timestamp: {timestamp}\nModel Checkpoint: {opts.checkpoint}\nConfig: {opts.config}\n"
    print(header)

    # 构建 MPJPE 表格
    table_mpjpe = prettytable.PrettyTable()
    table_mpjpe.field_names = ['Action'] + action_names
    row_mpjpe = ['MPJPE (mm)'] + [f"{v:.2f}" for v in avg_mpjpe_per_action]
    table_mpjpe.add_row(row_mpjpe)
    print(table_mpjpe)

    # 构建 MPJVE 表格
    table_mpjve = prettytable.PrettyTable()
    table_mpjve.field_names = ['Action'] + action_names
    row_mpjve = ['MPJVE (mm)'] + [f"{v:.2f}" for v in avg_mpjve_per_action]
    table_mpjve.add_row(row_mpjve)
    print(table_mpjve)

    # 构建 PA-MPJPE 表格
    table_pa = prettytable.PrettyTable()
    table_pa.field_names = ['Action'] + action_names
    row_pa = ['PA-MPJPE (mm)'] + [f"{v:.2f}" for v in avg_pa_mpjpe_per_action]
    table_pa.add_row(row_pa)
    print(table_pa)

    summary = f"Summary: MPJPE={mean_mpjpe:.4f} mm, MPJVE={mean_mpjve:.4f} mm, PA-MPJPE={mean_pa:.4f} mm"
    print(summary)

    # 保存到文件
    if opts.output:
        output_path = opts.output
    else:
        timestamp_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"eval_result_{timestamp_file}.txt"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(str(table_mpjpe) + "\n")
        f.write(str(table_mpjve) + "\n")
        f.write(str(table_pa) + "\n")
        f.write(summary + "\n")
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()