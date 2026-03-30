import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import prettytable

# Import project dependencies
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.datareader_h36m import DataReaderH36M
from lib.model.loss import mpjpe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/MB_train_h36m_posecastformer_scratch_SH.yaml",
                        help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH', required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id')
    opts = parser.parse_args()
    return opts


def inject_noise(data_2d, noise_sigma_pixel, resolution=1000.0):
    """
    Inject pixel-level Gaussian noise into normalized 2D data
    data_2d: [N, T, J, 2] normalized to [-1, 1]
    noise_sigma_pixel: standard deviation of noise (in pixels)
    resolution: image resolution (H36M ~ 1000x1000)
    """
    if noise_sigma_pixel <= 0:
        return data_2d

    # Convert pixel noise to normalized coordinate noise
    # Normalized coordinate range is 2 (-1 to 1), corresponding to ~1000 pixels
    # Scale factor = 2.0 / 1000.0 = 0.002
    scale = 2.0 / resolution
    noise_sigma_norm = noise_sigma_pixel * scale

    noise = torch.randn_like(data_2d) * noise_sigma_norm
    return data_2d + noise


def evaluate_with_noise(args, model_pos, test_loader, datareader, noise_level):
    model_pos.eval()
    results_all = []

    with torch.no_grad():
        for batch_input, batch_gt in test_loader:
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()

            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]

            # --- Core step: inject noise ---
            # batch_input shape: [N, T, 17, 3] (x, y, conf)
            # We only add noise to x, y (first two channels), keeping confidence unchanged
            noisy_input = batch_input.clone()
            noisy_input[..., :2] = inject_noise(noisy_input[..., :2], noise_level)
            # ------------------------

            if args.flip:
                # When flipping augmentation is used, also flip the noisy data
                batch_input_flip = flip_data(noisy_input)
                predicted_3d_pos_1 = model_pos(noisy_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(noisy_input)

            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0

            results_all.append(predicted_3d_pos.cpu().numpy())

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

    results_mpjpe = []
    block_list = ['s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01']

    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]

        err = mpjpe(pred, gt)
        results_mpjpe.append(np.mean(err))

    return np.mean(results_mpjpe)


def main():
    opts = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    args = get_config(opts.config)

    # 1. Prepare dataset
    print('Loading dataset...')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True)
    datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride,
                                data_stride_train=args.data_stride, data_stride_test=args.clip_len,
                                dt_root='data/motion3d', dt_file=args.dt_file)

    # 2. Load model
    print(f'Loading backbone: {args.backbone}')
    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print(f'Loading checkpoint: {opts.checkpoint}')
    checkpoint = torch.load(opts.checkpoint, map_location='cpu')
    state_dict = checkpoint['model_pos'] if 'model_pos' in checkpoint else checkpoint
    model_backbone.load_state_dict(state_dict, strict=True)

    # 3. Define noise levels (in pixels) - used to generate the robustness curve
    noise_levels = [0, 5, 10, 15, 20, 30, 40, 50]

    table = prettytable.PrettyTable()
    table.field_names = ["Noise (pixels)", "MPJPE (mm)"]

    print("\nStarting Noise Robustness Test...")
    print("-" * 40)

    for sigma in noise_levels:
        print(f"Testing with Noise Sigma = {sigma} pixels...")
        error = evaluate_with_noise(args, model_backbone, test_loader, datareader, sigma)
        table.add_row([sigma, f"{error:.2f}"])
        print(f"-> Result: {error:.2f} mm")

    print("\nFinal Results (Noise Robustness Curve Data):")
    print(table)


if __name__ == "__main__":
    main()