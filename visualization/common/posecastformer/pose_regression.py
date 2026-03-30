import torch
import torch.nn as nn
import os
import numpy as np


class PoseRegressionHead(nn.Module):
    def __init__(self, in_dim=256, out_dim=3):
        """
        轻量级回归头
        LayerNorm -> Linear -> GELU -> Linear
        """
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        hidden_dim = in_dim // 2
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(self.norm(x))


def merge_sliding_windows(batch_pred, original_length, window_size=243, overlap=81):
    """
    【新增功能】将切片后的 Batch 还原为原始长序列
    逻辑：重叠部分相加，然后除以重叠次数（取平均）

    Args:
        batch_pred: [Num_Windows, Window_Size, Joints, 3] -> [8, 243, 17, 3]
        original_length: 原始视频帧数 -> 1273
        window_size: 窗口大小 -> 243
        overlap: 重叠量 -> 81
    """
    print(f"Start Merging: Batch={batch_pred.shape}, Target Length={original_length}")

    num_windows, _, num_joints, dims = batch_pred.shape
    step = window_size - overlap  # 步长 = 243 - 81 = 162

    # 1. 初始化容器
    # sum_canvas: 用于累加预测值
    # count_canvas: 用于记录每个位置被预测了多少次
    sum_canvas = torch.zeros((original_length, num_joints, dims), device=batch_pred.device)
    count_canvas = torch.zeros((original_length, num_joints, dims), device=batch_pred.device)

    for i in range(num_windows):
        # 计算当前窗口在原始序列中的起止位置
        start_idx = i * step
        end_idx = start_idx + window_size

        # 获取当前窗口的预测值
        pred_window = batch_pred[i]  # [243, 17, 3]

        # --- 边界处理 ---
        # 如果窗口超出了原始长度 (针对最后一段 Padding 的情况)
        if end_idx > original_length:
            valid_len = original_length - start_idx
            # 只取有效部分
            pred_window = pred_window[:valid_len]
            # 修正结束位置
            end_idx = original_length

        # 累加值
        sum_canvas[start_idx:end_idx] += pred_window
        # 计数 +1
        count_canvas[start_idx:end_idx] += 1.0

    # 2. 取平均 (避免除以0，虽然理论上不会有0)
    # count_canvas 中原本是0的地方（如果有的话），加上一个极小值防止报错
    merged_output = sum_canvas / (count_canvas + 1e-8)

    return merged_output


# --- 执行脚本 ---
if __name__ == "__main__":
    # 1. 加载 B-CAST 模块输出
    INPUT_FILE = "datas/final_bcast_features.pt"

    # 【重要】这里填入你最开始提取视频时的实际帧数
    # 根据你的描述，这里是 1273 (或者之前的 1257，请根据实际情况修改)
    ORIGINAL_VIDEO_LENGTH = 1273

    if os.path.exists(INPUT_FILE):
        print(f"Loading features from {INPUT_FILE}...")
        data = torch.load(INPUT_FILE)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feat_s = data['spatial'].to(device)
        feat_t = data['temporal'].to(device)

        # 2. 融合特征 (Add)
        print("Executing Element-wise Addition...")
        fused_feat = feat_s + feat_t  # [8, 243, 17, 256]

        # 3. 回归 3D 坐标 (在 Window 级别进行)
        reg_head = PoseRegressionHead(in_dim=256, out_dim=3).to(device)

        try:
            # 得到切片状态下的 3D 预测 [8, 243, 17, 3]
            windowed_pred = reg_head(fused_feat)
            print(f"Windowed Prediction Shape: {windowed_pred.shape}")

            # 4. 【核心步骤】还原为长序列
            final_pred_full = merge_sliding_windows(
                windowed_pred,
                original_length=ORIGINAL_VIDEO_LENGTH,
                window_size=243,
                overlap=81
            )

            print("-" * 30)
            print("Restoration Successful!")
            print(f"Final Full Sequence Shape: {final_pred_full.shape} (Target: [{ORIGINAL_VIDEO_LENGTH}, 17, 3])")
            print("-" * 30)

            # 保存最终的完整长序列
            torch.save(final_pred_full, "final_3d_pose_full_sequence.pt")
            print("Full sequence saved to 'final_3d_pose_full_sequence.pt'")

            # 简单验证一下最后一帧是否有数据
            print("\nCheck Last Frame Data (Should not be zero):")
            print(final_pred_full[-1, 0].detach().cpu().numpy())

        except Exception as e:
            print(f"Error during regression/merging: {e}")

    else:
        print(f"Error: {INPUT_FILE} not found.")