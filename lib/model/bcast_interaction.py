import torch
import torch.nn as nn
import os

# --- Reuse base components ---
from lib.model.dual_stream_encoder import SpatialBlock, TemporalBlock, Mlp


# --- Core Component 1: Cross Attention ---
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Note: Linear projections for Q/K/V are kept, consistent with the original code in bidir_modeling_crossattn.py
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_query, x_kv):
        B, N, C = x_query.shape
        q = self.to_q(x_query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.to_k(x_kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.to_v(x_kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# --- Core Component 2: Bottleneck Adapter ---
# [Key modification] Adjusted operator order to match the logic in bidir_modeling_crossattn.py
class BottleneckAdapter(nn.Module):
    def __init__(self, dim, bottleneck_dim, num_heads=4, drop=0.):
        super().__init__()
        # 1. Linear (Down)
        self.down_proj = nn.Linear(dim, bottleneck_dim)

        # 2. LayerNorm (Note: now applied on bottleneck dimension, after Down projection)
        self.norm = nn.LayerNorm(bottleneck_dim)

        # 3. CrossAttention
        self.cross_attn = CrossAttention(bottleneck_dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)

        # 4. Activation (GELU)
        self.act = nn.GELU()

        # 5. Linear (Up)
        self.up_proj = nn.Linear(bottleneck_dim, dim)

    def forward(self, x_query, x_kv):
        shortcut = x_query

        #  Linear (Down)
        q = self.down_proj(x_query)
        k = self.down_proj(x_kv)

        #  LayerNorm (applied to bottleneck features)
        q = self.norm(q)
        k = self.norm(k)

        #  CrossAttention
        out = self.cross_attn(q, k)

        #  Activation (GELU)
        out = self.act(out)

        #  Linear (Up)
        out = self.up_proj(out)

        # Residual Connection (note: removed the final Post-LayerNorm)
        return shortcut + out


# --- Core Component 3: Full Interaction Block ---
class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, bottleneck_ratio=0.25, mlp_ratio=4., drop=0., ablation_mode='full'):
        super().__init__()
        self.ablation_mode = ablation_mode
        bottleneck_dim = int(dim * bottleneck_ratio)

        # Self-Attention part (always kept)
        self.self_attn_s = SpatialBlock(dim, num_heads, mlp_ratio, drop)
        self.self_attn_t = TemporalBlock(dim, num_heads, mlp_ratio, drop)

        # Initialize adapters (unless explicitly disabled by 'no_', both are initialized by default)
        # New modes (time_time, joint_joint, swapped) require both adapters
        if self.ablation_mode != 'no_t2s':
            self.adapter_t2s = BottleneckAdapter(dim, bottleneck_dim, num_heads=4, drop=drop)
        else:
            self.adapter_t2s = None

        if self.ablation_mode != 'no_s2t':
            self.adapter_s2t = BottleneckAdapter(dim, bottleneck_dim, num_heads=4, drop=drop)
        else:
            self.adapter_s2t = None

    def forward(self, x_s, x_t):
        B, T, J, C = x_s.shape

        # 1. Self-Attention (always applied)
        x_s = self.self_attn_s(x_s)
        x_t = self.self_attn_t(x_t)

        # --- Define helper functions: encapsulate reshape logic ---

        # Logic A: Compute attention along the [Time] dimension
        # Input shape: (B, T, J, C) -> reshape to (B*J, T, C) for interaction -> restore
        def _apply_time_attn(adapter, query, key_value):
            q = query.permute(0, 2, 1, 3).contiguous().reshape(B * J, T, C)
            k = key_value.permute(0, 2, 1, 3).contiguous().reshape(B * J, T, C)
            out = adapter(q, k)
            return out.reshape(B, J, T, C).permute(0, 2, 1, 3).contiguous()

        # Logic B: Compute attention along the [Joint] dimension
        # Input shape: (B, T, J, C) -> reshape to (B*T, J, C) for interaction -> restore
        def _apply_joint_attn(adapter, query, key_value):
            q = query.reshape(B * T, J, C)
            k = key_value.reshape(B * T, J, C)
            out = adapter(q, k)
            return out.reshape(B, T, J, C)

        # --- 2. Cross-Attention (choose strategy based on ablation_mode) ---

        # Mode 1: Full standard mode (original paper)
        # T2S: Time (update spatial stream using temporal correlations)
        # S2T: Joint (update temporal stream using spatial correlations)
        if self.ablation_mode == 'full':
            x_s = _apply_time_attn(self.adapter_t2s, x_s, x_t)
            x_t = _apply_joint_attn(self.adapter_s2t, x_t, x_s)

        # Mode 2: Both streams use [Time] (T2S=Time, S2T=Time)
        elif self.ablation_mode == 'time_time':
            x_s = _apply_time_attn(self.adapter_t2s, x_s, x_t)
            x_t = _apply_time_attn(self.adapter_s2t, x_t, x_s)  # Modification: S2T also uses time

        # Mode 3: Both streams use [Joint] (T2S=Joint, S2T=Joint)
        elif self.ablation_mode == 'joint_joint':
            x_s = _apply_joint_attn(self.adapter_t2s, x_s, x_t)  # Modification: T2S also uses joint
            x_t = _apply_joint_attn(self.adapter_s2t, x_t, x_s)

        # Mode 4: Swapped directions (T2S=Joint, S2T=Time)
        elif self.ablation_mode == 'swapped':
            x_s = _apply_joint_attn(self.adapter_t2s, x_s, x_t)  # Modification: T2S uses joint instead
            x_t = _apply_time_attn(self.adapter_s2t, x_t, x_s)  # Modification: S2T uses time instead

        # Mode 5: Original one‑side ablation (no_t2s / no_s2t)
        elif self.ablation_mode == 'no_t2s':
            # Only do S2T (Standard Joint)
            x_t = _apply_joint_attn(self.adapter_s2t, x_t, x_s)

        elif self.ablation_mode == 'no_s2t':
            # Only do T2S (Standard Time)
            x_s = _apply_time_attn(self.adapter_t2s, x_s, x_t)

        return x_s, x_t


# --- Main Module ---
class BCASTInteractionModule(nn.Module):
    # 1. Modified __init__ to accept ablation_mode
    def __init__(self, dim=256, depth=3, num_heads=8, bottleneck_ratio=0.25, mlp_ratio=4., ablation_mode='full'):
        super().__init__()
        self.blocks = nn.ModuleList([
            InteractionBlock(dim, num_heads, bottleneck_ratio, mlp_ratio=mlp_ratio, ablation_mode=ablation_mode)
            for _ in range(depth)
        ])
        self.norm_final = nn.LayerNorm(dim)

    def forward(self, x_s, x_t):
        for blk in self.blocks:
            x_s, x_t = blk(x_s, x_t)

        x_s = self.norm_final(x_s)
        x_t = self.norm_final(x_t)
        return x_s, x_t