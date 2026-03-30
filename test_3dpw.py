# test_3dpw.py
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import prettytable
import pickle
import time

# Assuming the project path is correctly set, project modules can be imported
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.utils.utils_data import flip_data
from lib.model.loss import p_mpjpe  # New import: PA-MPJPE computation function

# Fixed image dimensions (consistent with 2D normalization)
IMG_W = 1920
IMG_H = 1080

# Sliding window merging function (referenced from pose_regression.py)
def merge_sliding_windows(batch_pred, original_length, window_size=243, overlap=81):
    """
    Merge sliding window predictions into a full sequence
    batch_pred: [num_windows, window_size, J, 3]
    original_length: length of the original sequence
    """
    num_windows, window_size, J, C = batch_pred.shape
    step = window_size - overlap

    sum_canvas = torch.zeros((original_length, J, C), device=batch_pred.device)
    count_canvas = torch.zeros((original_length, J, C), device=batch_pred.device)

    for i in range(num_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        if end_idx > original_length:
            valid_len = original_length - start_idx
            pred_window = batch_pred[i, :valid_len]
            end_idx = original_length
        else:
            pred_window = batch_pred[i]
        sum_canvas[start_idx:end_idx] += pred_window
        count_canvas[start_idx:end_idx] += 1.0

    merged = sum_canvas / (count_canvas + 1e-8)
    return merged

# Mapping function from COCO 18 joints to H36M 17 joints
def coco_to_h36m(coco_kps):
    """
    coco_kps: [T, 18, 3]  (x, y, conf)
    Returns: [T, 17, 3] in H36M format
    H36M joint order:
        0: Hip (pelvis center)
        1: RHip
        2: RKnee
        3: RAnkle
        4: LHip
        5: LKnee
        6: LAnkle
        7: Spine
        8: Thorax
        9: Neck
        10: Head
        11: LShoulder
        12: LElbow
        13: LWrist
        14: RShoulder
        15: RElbow
        16: RWrist
    """
    T = coco_kps.shape[0]
    h36m = np.zeros((T, 17, 3), dtype=coco_kps.dtype)

    # Define COCO joint indices
    COCO_NOSE = 0
    COCO_NECK = 1
    COCO_RSHOULDER = 2
    COCO_RELBOW = 3
    COCO_RWRIST = 4
    COCO_LSHOULDER = 5
    COCO_LELBOW = 6
    COCO_LWRIST = 7
    COCO_RHIP = 8
    COCO_RKNEE = 9
    COCO_RANKLE = 10
    COCO_LHIP = 11
    COCO_LKNEE = 12
    COCO_LANKLE = 13
    COCO_REYE = 14
    COCO_LEYE = 15
    COCO_REAR = 16
    COCO_LEAR = 17

    # 1. Directly mapped joints
    # Hip (0) is the average of left and right hips
    h36m[:, 0, :] = (coco_kps[:, COCO_RHIP, :] + coco_kps[:, COCO_LHIP, :]) / 2
    h36m[:, 1, :] = coco_kps[:, COCO_RHIP, :]      # RHip
    h36m[:, 2, :] = coco_kps[:, COCO_RKNEE, :]     # RKnee
    h36m[:, 3, :] = coco_kps[:, COCO_RANKLE, :]    # RAnkle
    h36m[:, 4, :] = coco_kps[:, COCO_LHIP, :]      # LHip
    h36m[:, 5, :] = coco_kps[:, COCO_LKNEE, :]     # LKnee
    h36m[:, 6, :] = coco_kps[:, COCO_LANKLE, :]    # LAnkle
    h36m[:, 11, :] = coco_kps[:, COCO_LSHOULDER, :] # LShoulder
    h36m[:, 12, :] = coco_kps[:, COCO_LELBOW, :]    # LElbow
    h36m[:, 13, :] = coco_kps[:, COCO_LWRIST, :]    # LWrist
    h36m[:, 14, :] = coco_kps[:, COCO_RSHOULDER, :] # RShoulder
    h36m[:, 15, :] = coco_kps[:, COCO_RELBOW, :]    # RElbow
    h36m[:, 16, :] = coco_kps[:, COCO_RWRIST, :]    # RWrist

    # 2. Joints that need interpolation
    # Neck (9) uses COCO_NECK
    h36m[:, 9, :] = coco_kps[:, COCO_NECK, :]

    # Head (10) uses the nose
    h36m[:, 10, :] = coco_kps[:, COCO_NOSE, :]

    # Spine (7) and Thorax (8) are obtained by interpolation
    # Assume Neck is (9) and Hip is (0)
    # Spine = (Neck + Hip) / 2
    h36m[:, 7, :] = (h36m[:, 9, :] + h36m[:, 0, :]) / 2
    # Thorax = (Neck + Spine) / 2
    h36m[:, 8, :] = (h36m[:, 9, :] + h36m[:, 7, :]) / 2

    # 3. Confidence handling: keep original confidence; for interpolated joints, use average confidence of adjacent joints
    h36m[:, 7, 2] = (h36m[:, 9, 2] + h36m[:, 0, 2]) / 2
    h36m[:, 8, 2] = (h36m[:, 9, 2] + h36m[:, 7, 2]) / 2
    h36m[:, 0, 2] = (coco_kps[:, COCO_RHIP, 2] + coco_kps[:, COCO_LHIP, 2]) / 2

    return h36m

# Mapping from SMPL 24 joints to H36M 17 joints
def smpl_to_h36m(smpl_joints):
    """
    smpl_joints: [T, 24, 3]
    Returns: [T, 17, 3]
    SMPL joint order (common):
        0: Pelvis
        1: L_Hip
        2: L_Knee
        3: L_Ankle
        4: R_Hip
        5: R_Knee
        6: R_Ankle
        7: Torso (spine1)
        8: Spine (spine2)
        9: Chest (spine3)
        10: Neck
        11: Head
        12: L_Clavicle
        13: L_Shoulder
        14: L_Elbow
        15: L_Wrist
        16: R_Clavicle
        17: R_Shoulder
        18: R_Elbow
        19: R_Wrist
        20: L_Hand
        21: L_Fingers
        22: R_Hand
        23: R_Fingers
    """
    T = smpl_joints.shape[0]
    h36m = np.zeros((T, 17, 3), dtype=smpl_joints.dtype)

    # H36M -> SMPL index mapping
    mapping = {
        0: 0,   # Hip -> Pelvis
        1: 4,   # RHip -> R_Hip
        2: 5,   # RKnee -> R_Knee
        3: 6,   # RAnkle -> R_Ankle
        4: 1,   # LHip -> L_Hip
        5: 2,   # LKnee -> L_Knee
        6: 3,   # LAnkle -> L_Ankle
        7: 7,   # Spine -> Torso (spine1)
        8: 8,   # Thorax -> Spine (spine2)
        9: 10,  # Neck -> Neck
        10: 11, # Head -> Head
        11: 13, # LShoulder -> L_Shoulder (skip clavicle)
        12: 14, # LElbow -> L_Elbow
        13: 15, # LWrist -> L_Wrist
        14: 17, # RShoulder -> R_Shoulder
        15: 18, # RElbow -> R_Elbow
        16: 19, # RWrist -> R_Wrist
    }

    for h_idx, smpl_idx in mapping.items():
        h36m[:, h_idx, :] = smpl_joints[:, smpl_idx, :]

    return h36m

# Convert world coordinates to camera coordinates
def world_to_camera(p_world, cam_pose):
    """
    p_world: [J, 3] or [T, J, 3]
    cam_pose: [4,4] world-to-camera transformation matrix
    Returns camera coordinates
    """
    R = cam_pose[:3, :3]
    t = cam_pose[:3, 3]
    if p_world.ndim == 2:
        # [J,3]
        p_cam = (R @ p_world.T).T + t
    else:
        # [T,J,3]
        T = p_world.shape[0]
        p_cam = np.zeros_like(p_world)
        for i in range(T):
            p_cam[i] = (R @ p_world[i].T).T + t

    return p_cam

# Denormalize: convert normalized model output to camera coordinates (millimeters)
def denormalize_to_camera(pred_norm, K, W, H):
    """
    pred_norm: [T, J, 3] normalized coordinates (x_norm, y_norm, z_norm)
    K: camera intrinsics [3,3]
    W, H: image width and height (pixels)
    Returns: [T, J, 3] 3D coordinates in camera coordinate system (millimeters)
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    T, J, _ = pred_norm.shape
    pred_cam = np.zeros_like(pred_norm)

    # Extract normalized coordinates
    x_norm = pred_norm[..., 0]
    y_norm = pred_norm[..., 1]
    z_norm = pred_norm[..., 2]

    # Convert to pixel coordinates
    u = (x_norm + 1) * W / 2
    v = (y_norm + 1) * H / 2

    # Depth recovery: assume depth normalization factor is W/2 (consistent with H36M)
    Z_c = z_norm * (W / 2)

    # Use intrinsics to get camera coordinates
    pred_cam[..., 0] = (u - cx) * Z_c / fx
    pred_cam[..., 1] = (v - cy) * Z_c / fy
    pred_cam[..., 2] = Z_c

    return pred_cam

# 3DPW dataset class (returns entire sequences)
class PW3DDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.file_list = sorted([os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.pkl')])
        print(f"Found {len(self.file_list)} sequences.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        # Take the first person (index 0)
        poses2d = data['poses2d'][0]          # [T, 3, 18] -> transpose to [T, 18, 3]
        poses2d = np.transpose(poses2d, (0, 2, 1)).astype(np.float32)

        joint_positions = data['jointPositions'][0]  # [T, 72] -> [T, 24, 3]
        joint_positions = joint_positions.reshape(-1, 24, 3).astype(np.float32)

        cam_intrinsics = data['cam_intrinsics'].astype(np.float32)          # [3,3]
        cam_poses = data['cam_poses'].astype(np.float32)                    # [T,4,4]

        # Normalize 2D keypoints to [-1,1] (using fixed image dimensions)
        poses2d_norm = poses2d.copy()
        poses2d_norm[..., 0] = poses2d[..., 0] / IMG_W * 2 - 1
        poses2d_norm[..., 1] = poses2d[..., 1] / IMG_H * 2 - 1
        # The confidence channel remains unchanged

        # Convert 3D points from world to camera coordinates
        T = joint_positions.shape[0]
        joint_cam = np.zeros_like(joint_positions)
        for i in range(T):
            joint_cam[i] = world_to_camera(joint_positions[i], cam_poses[i])

        # Map 24 joints to 17 joints
        joint_h36m = smpl_to_h36m(joint_cam)  # [T,17,3]

        # Map 2D keypoints from COCO 18 to H36M 17
        input_h36m = coco_to_h36m(poses2d_norm)  # [T,17,3]

        return {
            'input': input_h36m,           # [T,17,3] normalized 2D
            'gt_cam': joint_h36m,           # [T,17,3] 3D in camera coordinates (without root subtraction)
            'seq_name': os.path.basename(file_path),
            'length': T,
            'cam_intrinsics': cam_intrinsics,
            'cam_poses': cam_poses
        }

# Evaluation function
def evaluate_3dpw(args, model, dataset, device, window_size=243, overlap=81):
    model.eval()
    all_mpjpe = []
    all_mpjve = []
    all_pa_mpjpe = []  # New: store PA-MPJPE for each sequence
    seq_results = {}

    with torch.no_grad():
        for data in tqdm(dataset, desc="Processing sequences"):
            input_2d = torch.from_numpy(data['input']).float().to(device)   # [T,17,3]
            gt_cam = data['gt_cam']                                          # [T,17,3] numpy
            T = data['length']
            K = data['cam_intrinsics']
            # Use fixed image dimensions
            W, H = IMG_W, IMG_H

            # Sliding window prediction
            step = window_size - overlap
            windows = []
            start_idxs = list(range(0, max(1, T - window_size + 1), step))
            if start_idxs[-1] + window_size < T:
                start_idxs.append(T - window_size)

            for s in start_idxs:
                e = s + window_size
                window = input_2d[s:e]  # [window_size,17,3]
                windows.append(window)

            if len(windows) == 0:
                windows = [input_2d]

            batch_input = torch.stack(windows, dim=0)  # [num_windows, window_size, 17, 3]

            # Model inference (supports flip test)
            if args.flip:
                batch_input_flip = flip_data(batch_input)
                pred = model(batch_input)
                pred_flip = model(batch_input_flip)
                pred_flip = flip_data(pred_flip)
                pred = (pred + pred_flip) / 2
            else:
                pred = model(batch_input)

            # Merge windows to obtain full prediction (normalized coordinates)
            pred_norm = merge_sliding_windows(pred, T, window_size, overlap)  # [T,17,3]
            pred_norm = pred_norm.cpu().numpy()

            # Convert normalized prediction to camera coordinates
            pred_cam = denormalize_to_camera(pred_norm, K, W, H)  # [T,17,3]

            # Subtract root joint (index 0) from both prediction and ground truth
            pred_root = pred_cam[:, 0:1, :]
            gt_root = gt_cam[:, 0:1, :]
            pred_rel = pred_cam - pred_root
            gt_rel = gt_cam - gt_root

            # Compute MPJPE (per-frame average)
            err = np.linalg.norm(pred_rel - gt_rel, axis=-1)  # [T,17]
            mpjpe_seq = np.mean(err)
            all_mpjpe.append(mpjpe_seq)

            # Compute PA-MPJPE (Procrustes alignment)
            # p_mpjpe returns error per sample (frame), shape [T], average over frames
            pa_err = p_mpjpe(pred_rel, gt_rel)  # [T]
            pa_mpjpe_seq = np.mean(pa_err)
            pa_mpjpe_seq = pa_mpjpe_seq * 100
            all_pa_mpjpe.append(pa_mpjpe_seq)

            # Compute MPJVE (velocity error)
            if T > 1:
                pred_vel = pred_rel[1:] - pred_rel[:-1]
                gt_vel = gt_rel[1:] - gt_rel[:-1]
                err_vel = np.linalg.norm(pred_vel - gt_vel, axis=-1)  # [T-1,17]
                mpjve_seq = np.mean(err_vel)
            else:
                mpjve_seq = 0.0
            all_mpjve.append(mpjve_seq)

            seq_results[data['seq_name']] = {
                'MPJPE': mpjpe_seq,
                'PA-MPJPE': pa_mpjpe_seq,  # new
                'MPJVE': mpjve_seq
            }

    # Summary
    avg_mpjpe = np.mean(all_mpjpe)
    avg_pa_mpjpe = np.mean(all_pa_mpjpe)
    avg_mpjve = np.mean(all_mpjve)

    # Print table (new column for PA-MPJPE)
    table = prettytable.PrettyTable()
    table.field_names = ['Sequence', 'MPJPE (mm)', 'PA-MPJPE (mm)', 'MPJVE (mm)']
    for seq, vals in seq_results.items():
        table.add_row([
            seq,
            f"{vals['MPJPE']:.2f}",
            f"{vals['PA-MPJPE']:.2f}",
            f"{vals['MPJVE']:.2f}"
        ])
    table.add_row([
        'Average',
        f"{avg_mpjpe:.2f}",
        f"{avg_pa_mpjpe:.2f}",
        f"{avg_mpjve:.2f}"
    ])

    print(table)
    return avg_mpjpe, avg_pa_mpjpe, avg_mpjve, seq_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--log', type=str, default='3dpw_results.log', help='Log file path')
    parser.add_argument('--flip', action='store_true', help='Use flip test')
    parser.add_argument('--window', type=int, default=243, help='Window size')
    parser.add_argument('--overlap', type=int, default=81, help='Overlap between windows')
    args_parse = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args_parse.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    config = get_config(args_parse.config)
    config.flip = args_parse.flip  # Merge command line arguments into config

    # Load model
    print("Loading model...")
    model_backbone = load_backbone(config)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    # Load weights
    checkpoint = torch.load(args_parse.checkpoint, map_location='cpu')
    state_dict = checkpoint['model_pos'] if 'model_pos' in checkpoint else checkpoint
    model_backbone.load_state_dict(state_dict, strict=True)
    model_backbone.eval()

    # Dataset
    data_root = "F:/3DPW/sequenceFiles/sequenceFiles/test"  # Please verify the path
    dataset = PW3DDataset(data_root)

    # Evaluation
    avg_mpjpe, avg_pa_mpjpe, avg_mpjve, seq_results = evaluate_3dpw(
        config, model_backbone, dataset, device,
        window_size=args_parse.window, overlap=args_parse.overlap
    )

    # Write to log (new PA-MPJPE included)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(args_parse.log, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {args_parse.checkpoint}\n")
        f.write(f"Config: {args_parse.config}\n")
        f.write(f"Flip test: {args_parse.flip}\n")
        f.write(f"Window size: {args_parse.window}, Overlap: {args_parse.overlap}\n")
        f.write(f"Average MPJPE: {avg_mpjpe:.4f} mm\n")
        f.write(f"Average PA-MPJPE: {avg_pa_mpjpe:.4f} mm\n")
        f.write(f"Average MPJVE: {avg_mpjve:.4f} mm\n")
        for seq, vals in seq_results.items():
            f.write(f"{seq}: MPJPE={vals['MPJPE']:.2f}, PA-MPJPE={vals['PA-MPJPE']:.2f}, MPJVE={vals['MPJVE']:.2f}\n")
        f.write(f"{'='*60}\n")

    print(f"Results saved to {args_parse.log}")

if __name__ == '__main__':
    main()