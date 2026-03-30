import os
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from lib.model.DSTformer import DSTformer
from lib.model.model_posecastformer import PoseCASTformer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_pretrained_weights(model, checkpoint_path):
    """
    Intelligently load MotionBERT (DSTformer) weights to PoseCASTformer
    """
    import torch
    print(f"Loading MotionBERT weights from {checkpoint_path}...")

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_pos' in checkpoint:
        src_state = checkpoint['model_pos']
    else:
        src_state = checkpoint

    target_state = model.state_dict()
    new_state_dict = {}

    # 2. Define layer limit (only load first 2 layers to independent_encoder)
    INDEPENDENT_DEPTH = 2

    print("Start mapping weights...")

    for k, v in src_state.items():
        if k.startswith('module.'):
            k = k[7:]

        new_k = None

        # --- A. Map Embedding layers ---
        if k.startswith('joints_embed') or k.startswith('pos_embed') or k.startswith('temp_embed'):
            new_k = f"embedding.{k}"


            if 'pos_embed' in k:
                # Original: [1, 17, 512], Target: [1, 1, 17, 512]
                if v.ndim == 3 and target_state[new_k].ndim == 4:
                    # Insert a dimension at dim=1 (Time dimension)
                    v = v.unsqueeze(1)
                    print(f"Auto-fixing shape for {k}: {v.shape}")


        # --- B. Map Independent Encoder (Layer 0 ~ N-1) ---
        elif k.startswith('blocks_st'):
            layer_idx = int(k.split('.')[1])
            if layer_idx < INDEPENDENT_DEPTH:
                suffix = ".".join(k.split('.')[2:])
                suffix = suffix.replace('_s', '')
                new_k = f"independent_encoder.spatial_blocks.{layer_idx}.{suffix}"

        elif k.startswith('blocks_ts'):
            layer_idx = int(k.split('.')[1])
            if layer_idx < INDEPENDENT_DEPTH:
                suffix = ".".join(k.split('.')[2:])
                suffix = suffix.replace('_t', '')
                new_k = f"independent_encoder.temporal_blocks.{layer_idx}.{suffix}"

        # --- C. Inject weights ---
        if new_k and new_k in target_state:
            if v.shape == target_state[new_k].shape:
                new_state_dict[new_k] = v
            else:
                print(f"Skipping {new_k}: Shape mismatch {v.shape} vs {target_state[new_k].shape}")

    # 3. Load to model
    msg = model.load_state_dict(new_state_dict, strict=False)

    print(f"Weights loaded successfully.")
    print(f"   - Source keys: {len(src_state)}")
    print(f"   - Mapped keys: {len(new_state_dict)}")
    # Expected to be 52 here

    return model

def partial_train_layers(model, partial_list):
    """Train partial layers of a given model."""
    for name, p in model.named_parameters():
        p.requires_grad = False
        for trainable in partial_list:
            if trainable in name:
                p.requires_grad = True
                break
    return model

def load_backbone(args):
    if not(hasattr(args, "backbone")):
        args.backbone = 'DSTformer' # Default
    if args.backbone=='DSTformer':
        model_backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)


    elif args.backbone == 'PoseCASTformer':
        ablation_mode = getattr(args, 'ablation_mode', 'full')

        # Print for easy confirmation of the actual model mode in logs
        print(f"Initializing model architecture: PoseCASTformer, experiment mode: {ablation_mode}")

        depth_total = getattr(args, 'depth', 5)
        depth_interaction = getattr(args, 'depth_interaction', 3)

        model_backbone = PoseCASTformer(
            num_joints=args.num_joints,
            in_channels=2 if getattr(args, 'no_conf', False) else 3,
            embed_dim=args.dim_feat,
            depth_total=depth_total,
            depth_interaction=depth_interaction,
            num_heads=args.num_heads,
            maxlen=args.maxlen,
            mlp_ratio=args.mlp_ratio,
            ablation_mode=ablation_mode  # Ensure passed to model_posecastformer.py
        )

    elif args.backbone=='TCN':
        from lib.model.model_tcn import PoseTCN
        model_backbone = PoseTCN()
    elif args.backbone=='poseformer':
        from lib.model.model_poseformer import PoseTransformer 
        model_backbone = PoseTransformer(num_frame=args.maxlen, num_joints=args.num_joints, in_chans=3, embed_dim_ratio=32, depth=4,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0, attn_mask=None) 
    elif args.backbone=='mixste':
        from lib.model.model_mixste import MixSTE2 
        model_backbone = MixSTE2(num_frame=args.maxlen, num_joints=args.num_joints, in_chans=3, embed_dim_ratio=512, depth=8,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)
    elif args.backbone=='stgcn':
        from lib.model.model_stgcn import Model as STGCN 
        model_backbone = STGCN()
    else:
        raise Exception("Undefined backbone type.")
    return model_backbone


