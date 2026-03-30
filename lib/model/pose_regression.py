import torch
import torch.nn as nn
import os
import numpy as np


class PoseRegressionHead(nn.Module):
    def __init__(self, in_dim=256, out_dim=3):
        """
        Lightweight regression head
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
    Args:
        batch_pred: [Num_Windows, Window_Size, Joints, 3] -> [8, 243, 17, 3]
        original_length: Original video frame count -> 1273
        window_size: Window size -> 243
        overlap: Overlap amount -> 81
    """
    print(f"Start Merging: Batch={batch_pred.shape}, Target Length={original_length}")

    num_windows, _, num_joints, dims = batch_pred.shape
    step = window_size - overlap  # step = 243 - 81 = 162

    # 1. Initialize containers
    # sum_canvas: accumulates predictions
    # count_canvas: records how many times each position has been predicted
    sum_canvas = torch.zeros((original_length, num_joints, dims), device=batch_pred.device)
    count_canvas = torch.zeros((original_length, num_joints, dims), device=batch_pred.device)

    for i in range(num_windows):
        # Compute the start and end positions of the current window in the original sequence
        start_idx = i * step
        end_idx = start_idx + window_size

        # Get the prediction for the current window
        pred_window = batch_pred[i]  # [243, 17, 3]

        # --- Boundary handling ---
        # If the window exceeds the original length (for the case of padding at the end)
        if end_idx > original_length:
            valid_len = original_length - start_idx
            # Take only the valid part
            pred_window = pred_window[:valid_len]
            # Correct the end index
            end_idx = original_length

        # Accumulate values
        sum_canvas[start_idx:end_idx] += pred_window
        # Increment count
        count_canvas[start_idx:end_idx] += 1.0

    # 2. Average (avoid division by zero, though theoretically there is none)
    # Add a small epsilon to positions that are still zero (if any) to prevent errors
    merged_output = sum_canvas / (count_canvas + 1e-8)

    return merged_output


