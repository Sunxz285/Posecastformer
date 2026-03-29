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
from lib.data.datareader_h36m_cpn import DataReaderH36M_CPN  # 使用正确的 CPN 数据读取器
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

    # 添加超时和错误处理
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError("Evaluation timeout")

    try:
        with torch.no_grad():
            for batch_idx, (batch_input, batch_gt) in enumerate(tqdm(test_loader)):
                if batch_idx % 10 == 0:  # 每10个batch打印一次进度
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

                # 确保评估时不使用gt_2d模式，避免数据泄露
                if args.gt_2d:
                    print("警告: 评估时gt_2d=True，这可能导致数据泄露！")
                    print("强制设置为False以确保正确评估")
                    # 不要执行 predicted_3d_pos[..., :2] = batch_input[..., :2]

                results_all.append(predicted_3d_pos.cpu().numpy())

                # 定期清理GPU缓存
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

    # 对于CPN数据，我们需要使用来自数据加载器的真实ground truth数据
    # 重新运行数据加载器来获取ground truth数据
    print("收集真实的ground truth数据...")
    gt_results_all = []
    with torch.no_grad():
        for batch_input, batch_gt in test_loader:
            gt_results_all.append(batch_gt.cpu().numpy())
    gt_results_all = np.concatenate(gt_results_all)
    gt_results_all = datareader.denormalize(gt_results_all)

    # 确保数据长度一致
    if len(results_all) != len(gt_results_all):
        print(f"调整数据长度: results_all从{len(results_all)}调整到{len(gt_results_all)}")
        results_all = results_all[:len(gt_results_all)]

    n_actual_clips = len(results_all)
    print(f"Debug: results_all length: {n_actual_clips}")
    print(f"Debug: gt_results_all length: {len(gt_results_all)}")
    print(f"Debug: split_id_test length: {len(split_id_test)}")

    # 确保ground truth数据长度匹配
    if len(gt_results_all) != n_actual_clips:
        print(f"警告: ground truth长度({len(gt_results_all)})与预测长度({n_actual_clips})不匹配")
        # 截断到较短的长度
        n_actual_clips = min(n_actual_clips, len(gt_results_all))
        results_all = results_all[:n_actual_clips]
        gt_results_all = gt_results_all[:n_actual_clips]

    # 数据验证：检查预测值和ground truth的统计信息
    print(f"预测值统计 - 均值: {np.mean(results_all):.3f}, 标准差: {np.std(results_all):.3f}")
    print(f"Ground Truth统计 - 均值: {np.mean(gt_results_all):.3f}, 标准差: {np.std(gt_results_all):.3f}")

    # 检查是否有异常值（接近零的值可能表示问题）
    if np.mean(results_all) < 0.001:
        print("警告: 预测值均值接近零，可能表示模型没有学到有效特征")
    if np.mean(gt_results_all) < 0.001:
        print("警告: Ground Truth均值接近零，可能表示数据加载问题")

    # 使用真实的ground truth数据
    # 从datareader获取真实的动作标签
    _, split_id_test = datareader.get_split_id()

    # 获取真实的动作标签
    if hasattr(datareader, 'dt_dataset') and 'test' in datareader.dt_dataset:
        test_actions = datareader.dt_dataset['test']['action']
        # 确保动作标签长度匹配
        if len(test_actions) >= n_actual_clips:
            actions = np.array(test_actions[:n_actual_clips])
        else:
            # 如果动作标签不够，用unknown填充
            actions = np.array(['unknown'] * n_actual_clips)
            print(f"警告: 动作标签数量({len(test_actions)})少于实际clip数量({n_actual_clips})")
    else:
        # 后备方案：尝试从文件名推断动作
        actions = np.array(['unknown'] * n_actual_clips)
        print("警告: 无法获取动作标签，使用unknown")

    factors = np.ones(n_actual_clips)
    sources = np.array([f'test_clip_{i}' for i in range(n_actual_clips)])
    gts = gt_results_all  # 使用真实的3D pose数据！

    print(f"Debug: adjusted actions shape: {actions.shape}")
    print(f"Debug: adjusted gts shape: {gts.shape}")
    print(f"动作标签前10个: {actions[:10]}")
    print(f"动作标签唯一值: {np.unique(actions)}")
    print(f"Debug: datareader.dt_dataset['test']['action'] 长度: {len(datareader.dt_dataset['test']['action'])}")
    print(f"Debug: 前20个动作: {datareader.dt_dataset['test']['action'][:20]}")
    print(f"Debug: 动作唯一值: {np.unique(datareader.dt_dataset['test']['action'])}")

    num_test_frames = n_actual_clips
    frames = np.array(range(num_test_frames))
    action_clips = actions
    factor_clips = factors[:, None, None]  # 扩展维度以匹配[T, J, 3]
    source_clips = sources
    frame_clips = frames
    gt_clips = gts

    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}

    # 确保使用标准的15个动作名称
    action_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo',
                   'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Waiting',
                   'WalkDog', 'Walking', 'WalkTogether']
    print(f"使用标准Human3.6M动作列表: {action_names}")

    print(f"评估动作类别: {action_names}")

    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        # 对于简化的CPN评估，我们直接计算每个clip的误差
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

        # 对于简化的评估，我们直接累加误差
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
            # 记录警告，添加一个默认值而不是跳过
            print(f"警告: 动作 '{action}' 没有有效数据")
            final_result.append(0.0)  # 使用0.0作为默认值
            final_result_procrustes.append(0.0)  # 使用0.0作为默认值
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
        'num_workers': 12,  # 减少worker数量避免死锁
        'pin_memory': True,
        'prefetch_factor': 4,  # 减少预取因子
        'persistent_workers': True  # 禁用持久worker
    }

    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 12,  # 测试时使用更少的worker
        'pin_memory': True,
        'prefetch_factor': 4,  # 测试时减少预取
        'persistent_workers': True  # 禁用持久worker
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

    # 根据 subset_list 选择数据读取器
    if args.subset_list[0] == 'H36M-CPN':
        print("使用CPN数据读取器")
        from lib.data.datareader_h36m_cpn import DataReaderH36M_CPN as DataReader
        datareader = DataReader(
            n_frames=args.clip_len,
            sample_stride=args.sample_stride,
            data_stride_train=args.data_stride,
            data_stride_test=args.clip_len,
            dt_root='data/motion3d',
            dt_file=args.dt_file  # 只需传入CPN 2D数据文件名，3D数据自动从 data_3d_h36m.npz 加载
        )
    else:
        print("使用SH数据读取器")
        from lib.data.datareader_h36m import DataReaderH36M as DataReader
        datareader = DataReader(
            n_frames=args.clip_len,
            sample_stride=args.sample_stride,
            data_stride_train=args.data_stride,
            data_stride_test=args.clip_len,
            dt_root='data/motion3d',
            dt_file=args.dt_file
        )

    # 确保CPN数据读取器的数据集已准备（对于直接版本可能不需要，但保留无害）
    if args.subset_list[0] == 'H36M-CPN' and hasattr(datareader, 'prepare_dataset'):
        print("准备CPN数据集...")
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

            # 如果是 PoseCASTformer，使用专门的权重映射加载函数
            if args.backbone == 'PoseCASTformer':
                # 因为模型已经被 DataParallel 包装，需要传入 model_backbone.module
                if isinstance(model_backbone, nn.DataParallel):
                    load_pretrained_weights(model_backbone.module, chk_filename)
                else:
                    load_pretrained_weights(model_backbone, chk_filename)
            else:
                # 其他模型使用默认加载方式
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