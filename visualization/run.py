import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import copy

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

# ==================== New: try importing PoseCASTformer ====================
try:
    from common.posecastformer.model_posecastformer import PoseCASTformer
    from common.posecastformer.pose_regression import merge_sliding_windows

    posecastformer_available = True
except ImportError:
    posecastformer_available = False
    print("PoseCASTformer not available, falling back to default model.")
# ======================================================================

args = parse_args()
print(args)

# ==================== New: add backbone parameter (default to 'tcn' if not defined) ====================
if not hasattr(args, 'backbone'):
    args.backbone = 'tcn'
# ========================================================================================

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset

    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset

    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset

    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
            action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]

semi_supervised = len(subjects_semi) > 0
if semi_supervised and not dataset.supports_semi_supervised():
    raise RuntimeError('Semi-supervised training is not implemented for this dataset')


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

# ==================== Model Creation ====================
if args.backbone == 'posecastformer':
    if not posecastformer_available:
        raise ImportError("PoseCASTformer model files not found. Please place them in common/posecastformer/")

    # Use the hyperparameters from training (consistent with your config)
    model_pos = PoseCASTformer(
        num_joints=dataset.skeleton().num_joints(),
        in_channels=3,  # input (x, y, conf)
        embed_dim=512,  # corresponds to dim_feat
        depth_total=5,
        depth_interaction=3,
        num_heads=8,
        maxlen=243,
        mlp_ratio=2.,
        ablation_mode='full'  # corresponds to ablation_mode
    )
    model_pos_train = copy.deepcopy(model_pos)
    # For PoseCASTformer, we do not need a trajectory model
    model_traj = None
    model_traj_train = None
else:
    # Original VideoPose3D model creation
    filter_widths = [int(x) for x in args.architecture.split(',')]
    if not args.disable_optimizations and not args.dense and args.stride == 1:
        model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                                   dataset.skeleton().num_joints(),
                                                   filter_widths=filter_widths, causal=args.causal,
                                                   dropout=args.dropout, channels=args.channels)
    else:
        model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                        dataset.skeleton().num_joints(),
                                        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                        channels=args.channels,
                                        dense=args.dense)

    model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                              filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                              channels=args.channels,
                              dense=args.dense)
    model_traj = None  # will be created for semi-supervised training if needed

# Compute receptive field (only meaningful for TemporalModel)
if args.backbone == 'posecastformer':
    # PoseCASTformer uses a fixed window length of 243
    receptive_field = 243
    pad = (receptive_field - 1) // 2  # 121
    causal_shift = 0
else:
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2
    if args.causal:
        causal_shift = pad
    else:
        causal_shift = 0

print('INFO: Receptive field: {} frames'.format(receptive_field))
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()
    if model_traj is not None:
        model_traj = model_traj.cuda()
    if model_traj_train is not None:
        model_traj_train = model_traj_train.cuda()

# ==================== Load Checkpoint ====================
if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))

    if args.backbone == 'posecastformer':
        # Handle weights saved with DataParallel (remove 'module.' prefix)
        state_dict = checkpoint['model_pos']
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model_pos_train.load_state_dict(new_state_dict, strict=False)
        model_pos.load_state_dict(new_state_dict, strict=False)
        # ignore model_traj
    else:
        model_pos_train.load_state_dict(checkpoint['model_pos'])
        model_pos.load_state_dict(checkpoint['model_pos'])
        if 'model_traj' in checkpoint and checkpoint['model_traj'] is not None:
            model_traj.load_state_dict(checkpoint['model_traj'])
            model_traj_train.load_state_dict(checkpoint['model_traj'])

# ==================== Test Generator ====================
test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                    joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

# ==================== Training part (only if not in evaluation mode) ====================
if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate
    if semi_supervised:
        # ... semi-supervised training code (same as original VideoPose3D) ...
        # Omitted here for brevity; keep as in original
        pass
    else:
        optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    train_generator = ChunkedGenerator(args.batch_size // args.stride, cameras_train, poses_train, poses_train_2d,
                                       args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                       joints_right=joints_right)
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

        lr = checkpoint['lr']

    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')

    # Training loop (same as original VideoPose3D)
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        N = 0
        model_pos_train.train()

        for _, batch_3d, batch_2d in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D pose
            predicted_3d_pos = model_pos_train(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_total = loss_3d_pos
            loss_total.backward()

            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)

        # Validation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict())
            model_pos.eval()

            epoch_loss_3d_valid = 0
            N = 0

            if not args.no_eval:
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_3d[:, :, 0] = 0

                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)

                # Evaluate on training set (eval mode)
                epoch_loss_3d_train_eval = 0
                N = 0
                for cam, batch, batch_2d in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        continue
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_3d[:, :, 0] = 0

                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_train_eval[-1] * 1000,
                losses_3d_valid[-1] * 1000))

        # Learning rate decay
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BN momentum
        momentum = initial_momentum * np.exp(-epoch / args.epochs * np.log(initial_momentum / final_momentum))
        model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                'model_traj': None,
            }, chk_path)

        # Save training curves if requested
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))
            plt.close('all')


# ==================== Evaluation function (modified to support no trajectory model) ====================
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        if use_trajectory_model and model_traj is not None:
            model_traj.eval()
            current_model = model_traj
        else:
            model_pos.eval()
            current_model = model_pos
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Predict
            predicted_3d_pos = current_model(inputs_2d)

            # Test-time augmentation
            if test_generator.augment_enabled():
                predicted_3d_pos[1, :, :, 0] *= -1
                if not use_trajectory_model:
                    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :,
                                                                         joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos,
                                                                                         inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----' + action + '----')
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev


# ==================== Sliding window prediction function (only for PoseCASTformer) ====================
def predict_with_windowing(model, input_2d, window_size=243, overlap=81):
    """
    input_2d: numpy array of shape (T, J, C), normalized 2D keypoints
    """
    T = input_2d.shape[0]
    step = window_size - overlap
    windows = []
    for start in range(0, T, step):
        end = start + window_size
        window = input_2d[start:end]
        if len(window) < window_size:
            # Pad the last frame (repeat the last frame)
            pad_len = window_size - len(window)
            pad = np.repeat(window[-1:], pad_len, axis=0)
            window = np.concatenate([window, pad], axis=0)
        windows.append(window)
    windows = np.stack(windows, axis=0)  # (num_windows, window_size, J, C)
    windows_torch = torch.from_numpy(windows).float()
    if torch.cuda.is_available():
        windows_torch = windows_torch.cuda()
    with torch.no_grad():
        preds = model(windows_torch)  # (num_windows, window_size, J, 3)
    # Merge windows
    full_pred = merge_sliding_windows(preds, original_length=T, window_size=window_size, overlap=overlap)
    return full_pred.cpu().numpy()


# ==================== Visualization ====================
if args.render:
    print('Rendering...')

    # Get input 2D keypoints
    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    if input_keypoints.shape[-1] == 2:
        conf = np.ones((input_keypoints.shape[0], input_keypoints.shape[1], 1), dtype=np.float32)
        input_keypoints = np.concatenate([input_keypoints, conf], axis=-1)

    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
            print(f"Ground truth loaded, shape: {ground_truth.shape}")
        else:
            print("No 'positions_3d' key found.")
    else:
        print("Subject or action not found in dataset.")

    # Predict
    if args.backbone == 'posecastformer':
        prediction_norm = predict_with_windowing(model_pos, input_keypoints, window_size=243, overlap=81)
    else:
        gen = UnchunkedGenerator(None, None, [input_keypoints],
                                 pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                 joints_right=joints_right)
        prediction_norm = evaluate(gen, return_predictions=True)

    print("prediction_norm min:", prediction_norm.min(), "max:", prediction_norm.max(), "mean:", prediction_norm.mean())

    # Denormalize to millimeters
    cam = dataset.cameras()[args.viz_subject][args.viz_camera]
    res_w, res_h = cam['res_w'], cam['res_h']
    prediction = np.zeros_like(prediction_norm)
    prediction[..., 0] = (prediction_norm[..., 0] + 1) * res_w / 2
    prediction[..., 1] = (prediction_norm[..., 1] + res_h / res_w) * res_w / 2
    prediction[..., 2] = prediction_norm[..., 2] * res_w / 2

    # Convert to meters (all subsequent operations are in meters)
    prediction /= 1000.0
    print("prediction (mm->m) min:", prediction.min(), "max:", prediction.max(), "mean:", prediction.mean())

    # Get camera extrinsics
    if args.viz_subject in dataset.cameras():
        cam_extrinsics = dataset.cameras()[args.viz_subject][args.viz_camera]
    else:
        for subject in dataset.cameras():
            if args.viz_camera < len(dataset.cameras()[subject]):
                cam_extrinsics = dataset.cameras()[subject][args.viz_camera]
                break

    # If ground truth is available, extract trajectory and add to predictions
    if ground_truth is not None and not args.viz_no_ground_truth:
        print("=== Ground Truth Debug ===")
        print("Raw ground truth (m) range:", ground_truth.min(), ground_truth.max())
        print("Raw ground truth Z range:", ground_truth[:, :, 2].min(), ground_truth[:, :, 2].max())

        # Extract trajectory (root joint)
        trajectory = ground_truth[:, :1, :].copy()  # (T,1,3), in meters
        print("Trajectory range:", trajectory.min(), trajectory.max())

        # Recover full ground truth: add trajectory to non-root joints
        gt_full = ground_truth.copy()
        gt_full[:, 1:, :] += trajectory
        print("gt_full (m) range:", gt_full.min(), gt_full.max())
        print("gt_full Z range:", gt_full[:, :, 2].min(), gt_full[:, :, 2].max())

        # Compute the Z range of the relative offset (for scaling)
        gt_rel_z_min = ground_truth[:, 1:, 2].min()
        gt_rel_z_max = ground_truth[:, 1:, 2].max()
        gt_rel_z_span = gt_rel_z_max - gt_rel_z_min

        # Prediction Z range
        pred_z_min = prediction[:, :, 2].min()
        pred_z_max = prediction[:, :, 2].max()
        pred_z_span = pred_z_max - pred_z_min

        # Scaling factor
        scale = gt_rel_z_span / pred_z_span if pred_z_span > 0 else 1.0
        print(f"Scaling prediction by factor: {scale} (to match true relative Z span)")

        # Isotropical scaling of prediction
        prediction_scaled = prediction * scale

        # Add trajectory to scaled prediction
        prediction_with_traj = prediction_scaled + trajectory
        print("prediction_with_traj (m) range:", prediction_with_traj.min(), prediction_with_traj.max())
        print("prediction_with_traj Z range:", prediction_with_traj[:, :, 2].min(), prediction_with_traj[:, :, 2].max())

        # Convert to world coordinates
        gt_world = camera_to_world(gt_full, R=cam_extrinsics['orientation'], t=cam_extrinsics['translation'])
        pred_world = camera_to_world(prediction_with_traj, R=cam_extrinsics['orientation'], t=cam_extrinsics['translation'])

        # Align to ground (both subtract their respective minimum Z)
        gt_world[:, :, 2] -= np.min(gt_world[:, :, 2])
        pred_world[:, :, 2] -= np.min(pred_world[:, :, 2])

        print("gt_world Z range after alignment:", gt_world[:, :, 2].min(), gt_world[:, :, 2].max())
        print("pred_world Z range after alignment:", pred_world[:, :, 2].min(), pred_world[:, :, 2].max())

        anim_output = {
            'Reconstruction': pred_world,
            'Ground truth': gt_world
        }
    else:
        # No ground truth: show only prediction, align height
        pred_world = camera_to_world(prediction, R=cam_extrinsics['orientation'], t=cam_extrinsics['translation'])
        pred_world[:, :, 2] -= np.min(pred_world[:, :, 2])
        anim_output = {'Reconstruction': pred_world}

    # Convert 2D input to pixel coordinates
    input_keypoints_pixel = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

    from common.visualization import render_animation
    render_animation(input_keypoints_pixel, keypoints_metadata, anim_output,
                     dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                     limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                     input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                     input_video_skip=args.viz_skip)
else:
    # ==================== Evaluation (non-rendering mode) ====================
    print('Evaluating...')
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))


    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)):  # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_poses_3d, out_poses_2d


    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                     joints_right=joints_right)
            e1, e2, e3, ev = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')


    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')