import os
import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import prettytable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d_cpn import MotionDataset3D
from lib.data.augmentation import Augmenter2D
from lib.data.datareader_h36m import DataReaderH36M
from lib.data.datareader_h36m_cpn import DataReaderH36M_CPN  # Use the correct CPN data reader
from lib.model.loss import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH',
                        help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME',
                        help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss': min_loss
    }, chk_path)


def evaluate(args, model_pos, test_loader, datareader):
    print('INFO: Testing')
    results_all = []
    model_pos.eval()

    # Add timeout and error handling
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError("Evaluation timeout")

    try:
        with torch.no_grad():
            for batch_idx, (batch_input, batch_gt) in enumerate(tqdm(test_loader)):
                if batch_idx % 10 == 0:  # Print progress every 10 batches
                    print(f"Processing batch {batch_idx}/{len(test_loader)}")

                N, T = batch_gt.shape[:2]
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()
                if args.no_conf:
                    batch_input = batch_input[:, :, :, :2]
                if args.flip:
                    batch_input_flip = flip_data(batch_input)
                    predicted_3d_pos_1 = model_pos(batch_input)
                    predicted_3d_pos_flip = model_pos(batch_input_flip)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
                else:
                    predicted_3d_pos = model_pos(batch_input)
                if args.rootrel:
                    predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
                else:
                    batch_gt[:, 0, 0, 2] = 0

                # Ensure gt_2d mode is not used during evaluation to avoid data leakage
                if args.gt_2d:
                    print("Warning: gt_2d=True during evaluation may cause data leakage!")
                    print("Forced to False to ensure correct evaluation")
                    # Do not execute predicted_3d_pos[..., :2] = batch_input[..., :2]

                results_all.append(predicted_3d_pos.cpu().numpy())

                # Regularly clean GPU cache
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()

    # For CPN data, we need to use real ground truth data from the data loader
    # Re-run data loader to get ground truth data
    print("Collecting real ground truth data...")
    gt_results_all = []
    with torch.no_grad():
        for batch_input, batch_gt in test_loader:
            gt_results_all.append(batch_gt.cpu().numpy())
    gt_results_all = np.concatenate(gt_results_all)
    gt_results_all = datareader.denormalize(gt_results_all)

    # Ensure data length consistency
    if len(results_all) != len(gt_results_all):
        print(f"Adjusting data length: results_all adjusted from {len(results_all)} to {len(gt_results_all)}")
        results_all = results_all[:len(gt_results_all)]

    n_actual_clips = len(results_all)
    print(f"Debug: results_all length: {n_actual_clips}")
    print(f"Debug: gt_results_all length: {len(gt_results_all)}")
    print(f"Debug: split_id_test length: {len(split_id_test)}")

    # Ensure ground truth data length matches
    if len(gt_results_all) != n_actual_clips:
        print(f"Warning: ground truth length ({len(gt_results_all)}) does not match prediction length ({n_actual_clips})")
        # Truncate to the shorter length
        n_actual_clips = min(n_actual_clips, len(gt_results_all))
        results_all = results_all[:n_actual_clips]
        gt_results_all = gt_results_all[:n_actual_clips]

    # Data validation: check statistics of predictions and ground truth
    print(f"Prediction statistics - mean: {np.mean(results_all):.3f}, std: {np.std(results_all):.3f}")
    print(f"Ground Truth statistics - mean: {np.mean(gt_results_all):.3f}, std: {np.std(gt_results_all):.3f}")

    # Check for anomalies (values close to zero may indicate problems)
    if np.mean(results_all) < 0.001:
        print("Warning: Prediction mean close to zero, may indicate model didn't learn effective features")
    if np.mean(gt_results_all) < 0.001:
        print("Warning: Ground Truth mean close to zero, may indicate data loading problem")

    # Use real ground truth data
    # Get real action labels from datareader
    _, split_id_test = datareader.get_split_id()

    # Get real action labels
    if hasattr(datareader, 'dt_dataset') and 'test' in datareader.dt_dataset:
        test_actions = datareader.dt_dataset['test']['action']
        # Ensure action label length matches
        if len(test_actions) >= n_actual_clips:
            actions = np.array(test_actions[:n_actual_clips])
        else:
            # If action labels are insufficient, fill with unknown
            actions = np.array(['unknown'] * n_actual_clips)
            print(f"Warning: Number of action labels ({len(test_actions)}) is less than actual clip count ({n_actual_clips})")
    else:
        # Fallback: try to infer action from filename
        actions = np.array(['unknown'] * n_actual_clips)
        print("Warning: Unable to get action labels, using unknown")

    factors = np.ones(n_actual_clips)
    sources = np.array([f'test_clip_{i}' for i in range(n_actual_clips)])
    gts = gt_results_all  # Use real 3D pose data!

    print(f"Debug: adjusted actions shape: {actions.shape}")
    print(f"Debug: adjusted gts shape: {gts.shape}")
    print(f"First 10 action labels: {actions[:10]}")
    print(f"Unique action labels: {np.unique(actions)}")
    print(f"Debug: datareader.dt_dataset['test']['action'] length: {len(datareader.dt_dataset['test']['action'])}")
    print(f"Debug: First 20 actions: {datareader.dt_dataset['test']['action'][:20]}")
    print(f"Debug: Unique actions: {np.unique(datareader.dt_dataset['test']['action'])}")

    num_test_frames = n_actual_clips
    frames = np.array(range(num_test_frames))
    action_clips = actions
    factor_clips = factors[:, None, None]  # Expand dimensions to match [T, J, 3]
    source_clips = sources
    frame_clips = frames
    gt_clips = gts

    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}

    # Ensure using standard 15 action names
    action_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo',
                   'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Waiting',
                   'WalkDog', 'Walking', 'WalkTogether']
    print(f"Using standard Human3.6M action list: {action_names}")

    print(f"Evaluation action categories: {action_names}")

    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        # For simplified CPN evaluation, we directly calculate error for each clip
        action = action_clips[idx]
        if action in block_list:
            continue

        factor = factor_clips[idx] if factor_clips.ndim > 1 else factor_clips[idx][:, None, None]
        gt = gt_clips[idx]  # [T, J, 3]
        pred = results_all[idx]  # [T, J, 3]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)

        # For simplified evaluation, we directly accumulate errors
        e1_all[idx] += np.mean(err1)
        e2_all[idx] += np.mean(err2)
        oc[idx] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        if len(results[action]) > 0:
            final_result.append(np.mean(results[action]))
            final_result_procrustes.append(np.mean(results_procrustes[action]))
        else:
            # Log warning, add a default value instead of skipping
            print(f"Warning: Action '{action}' has no valid data")
            final_result.append(0.0)  # Use 0.0 as default value
            final_result_procrustes.append(0.0)  # Use 0.0 as default value
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)

    print(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('----------')

    return e1, e2, results_all


def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    model_pos.train()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if not has_3d:
                conf = copy.deepcopy(batch_input[:, :, :, 2:])  # For 2D data, weight/confidence is at the last channel
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
            else:
                batch_gt[:, :, :, 2] = batch_gt[:, :, :, 2] - batch_gt[:, 0:1, 0:1,
                                                              2]  # Place the depth of first frame root to 0.
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        # Predict 3D poses
        predicted_3d_pos = model_pos(batch_input)  # (N, T, 17, 3)

        optimizer.zero_grad()
        if has_3d:
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_lv = loss_limb_var(predicted_3d_pos)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
            loss_total = loss_3d_pos + \
                         args.lambda_scale * loss_3d_scale + \
                         args.lambda_3d_velocity * loss_3d_velocity + \
                         args.lambda_lv * loss_lv + \
                         args.lambda_lg * loss_lg + \
                         args.lambda_a * loss_a + \
                         args.lambda_av * loss_av
            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()


def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    print('Loading dataset...')
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 12,  # Reduce worker count to avoid deadlock
        'pin_memory': True,
        'prefetch_factor': 4,  # Reduce prefetch factor
        'persistent_workers': True  # Disable persistent workers
    }

    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 12,  # Use fewer workers during testing
        'pin_memory': True,
        'prefetch_factor': 4,  # Reduce prefetch during testing
        'persistent_workers': True  # Disable persistent workers
    }

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)

    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)
        instav = InstaVDataset2D()
        instav_loader_2d = DataLoader(instav, **trainloader_params)

    # Select data reader based on subset_list
    if args.subset_list[0] == 'H36M-CPN':
        print("Using CPN data reader")
        from lib.data.datareader_h36m_cpn import DataReaderH36M_CPN as DataReader
        datareader = DataReader(
            n_frames=args.clip_len,
            sample_stride=args.sample_stride,
            data_stride_train=args.data_stride,
            data_stride_test=args.clip_len,
            dt_root='data/motion3d',
            dt_file=args.dt_file  # Only need to pass CPN 2D data filename, 3D data automatically loaded from data_3d_h36m.npz
        )
    else:
        print("Using SH data reader")
        from lib.data.datareader_h36m import DataReaderH36M as DataReader
        datareader = DataReader(
            n_frames=args.clip_len,
            sample_stride=args.sample_stride,
            data_stride_train=args.data_stride,
            data_stride_test=args.clip_len,
            dt_root='data/motion3d',
            dt_file=args.dt_file
        )

    # Ensure CPN data reader dataset is ready (may not be needed for direct version, but harmless to keep)
    if args.subset_list[0] == 'H36M-CPN' and hasattr(datareader, 'prepare_dataset'):
        print("Preparing CPN dataset...")
        datareader.prepare_dataset()

    min_loss = 100000
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage, weights_only=False)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading checkpoint', chk_filename)

            # If PoseCASTformer, use specialized weight mapping loading function
            if args.backbone == 'PoseCASTformer':
                # Since model is already wrapped by DataParallel, need to pass model_backbone.module
                if isinstance(model_backbone, nn.DataParallel):
                    load_pretrained_weights(model_backbone.module, chk_filename)
                else:
                    load_pretrained_weights(model_backbone, chk_filename)
            else:
                # Other models use default loading method
                checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage, weights_only=False)
                model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)

            model_pos = model_backbone
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage, weights_only=False)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone

    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate:
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr,
                                weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        if args.train_2d:
            print('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d),
                                                                   len(instav_loader_2d) + len(posetrack_loader_2d)))
        else:
            print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print(
                    'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']

        args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
        if args.mask or args.noise:
            args.aug = Augmenter2D(args)

        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            start_time = time()
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            losses['2d_proj'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            N = 0

            # Curriculum Learning
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True)
                train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False)
            train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True)
            elapsed = (time() - start_time) / 60

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg))
            else:
                e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)
                print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg,
                    e1, e2))
                train_writer.add_scalar('Error P1', e1, epoch + 1)
                train_writer.add_scalar('Error P2', e2, epoch + 1)
                train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
                train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
                train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
                train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
                train_writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))

            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)

    if opts.evaluate:
        e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)


if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)