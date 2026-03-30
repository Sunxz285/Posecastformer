import torch
import torch.nn as nn
import os

# --- 复用基础组件 ---
from .dual_stream_encoder import SpatialBlock, TemporalBlock, Mlp


# --- 核心组件 1: 交叉注意力 (Cross Attention) ---
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 注意：这里保留了Q/K/V的线性投影，这与bidir_modeling_crossattn.py原始代码保持一致
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


# --- 核心组件 2: 瓶颈适配器 (Bottleneck Adapter) ---
# 【本次修改核心】调整了算子顺序以匹配 bidir_modeling_crossattn.py 的逻辑
class BottleneckAdapter(nn.Module):
    def __init__(self, dim, bottleneck_dim, num_heads=4, drop=0.):
        super().__init__()
        # 1. Linear (Down)
        self.down_proj = nn.Linear(dim, bottleneck_dim)

        # 2. LayerNorm (注意：现在是对瓶颈维度进行归一化，位置在Down之后)
        self.norm = nn.LayerNorm(bottleneck_dim)

        # 3. CrossAttention
        self.cross_attn = CrossAttention(bottleneck_dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)

        # 4. Activation (GELU)
        self.act = nn.GELU()

        # 5. Linear (Up)
        self.up_proj = nn.Linear(bottleneck_dim, dim)

    def forward(self, x_query, x_kv):
        shortcut = x_query

        # Step 1: Linear (Down)
        q = self.down_proj(x_query)
        k = self.down_proj(x_kv)

        # Step 2: LayerNorm (应用于瓶颈特征)
        q = self.norm(q)
        k = self.norm(k)

        # Step 3: CrossAttention
        out = self.cross_attn(q, k)

        # Step 4: Activation (GELU)
        out = self.act(out)

        # Step 5: Linear (Up)
        out = self.up_proj(out)

        # Residual Connection (注意：去掉了原来末尾的 Post-LayerNorm)
        return shortcut + out


# --- 核心组件 3: 完整的交互块 (Interaction Block) ---
class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, bottleneck_ratio=0.25, mlp_ratio=4., drop=0., ablation_mode='full'):
        super().__init__()
        self.ablation_mode = ablation_mode
        bottleneck_dim = int(dim * bottleneck_ratio)

        # Self-Attention 部分 (始终保留)
        self.self_attn_s = SpatialBlock(dim, num_heads, mlp_ratio, drop)
        self.self_attn_t = TemporalBlock(dim, num_heads, mlp_ratio, drop)

        # 初始化适配器 (只要不是显式禁止 'no_', 默认都初始化)
        # 新增的模式 (time_time, joint_joint, swapped) 都需要两个适配器
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

        # 1. Self-Attention (始终执行)
        x_s = self.self_attn_s(x_s)
        x_t = self.self_attn_t(x_t)

        # --- 定义辅助函数：封装 reshape 逻辑 ---

        # 逻辑 A: 沿 [Time] 维度计算注意力
        # 输入形状: (B, T, J, C) -> 变换为 (B*J, T, C) 进行交互 -> 还原
        def _apply_time_attn(adapter, query, key_value):
            q = query.permute(0, 2, 1, 3).contiguous().reshape(B * J, T, C)
            k = key_value.permute(0, 2, 1, 3).contiguous().reshape(B * J, T, C)
            out = adapter(q, k)
            return out.reshape(B, J, T, C).permute(0, 2, 1, 3).contiguous()

        # 逻辑 B: 沿 [Joint] 维度计算注意力
        # 输入形状: (B, T, J, C) -> 变换为 (B*T, J, C) 进行交互 -> 还原
        def _apply_joint_attn(adapter, query, key_value):
            q = query.reshape(B * T, J, C)
            k = key_value.reshape(B * T, J, C)
            out = adapter(q, k)
            return out.reshape(B, T, J, C)

        # --- 2. Cross-Attention (根据 ablation_mode 选择策略) ---

        # 模式 1: 标准全量模式 (论文原版)
        # T2S: Time (利用时间相关性更新空间流)
        # S2T: Joint (利用空间相关性更新时间流)
        if self.ablation_mode == 'full':
            x_s = _apply_time_attn(self.adapter_t2s, x_s, x_t)
            x_t = _apply_joint_attn(self.adapter_s2t, x_t, x_s)

        # 模式 2: 双流都沿 [Time] (T2S=Time, S2T=Time)
        elif self.ablation_mode == 'time_time':
            x_s = _apply_time_attn(self.adapter_t2s, x_s, x_t)
            x_t = _apply_time_attn(self.adapter_s2t, x_t, x_s)  # 修改点：S2T也用时间

        # 模式 3: 双流都沿 [Joint] (T2S=Joint, S2T=Joint)
        elif self.ablation_mode == 'joint_joint':
            x_s = _apply_joint_attn(self.adapter_t2s, x_s, x_t)  # 修改点：T2S也用关节
            x_t = _apply_joint_attn(self.adapter_s2t, x_t, x_s)

        # 模式 4: 交换方向 (T2S=Joint, S2T=Time)
        elif self.ablation_mode == 'swapped':
            x_s = _apply_joint_attn(self.adapter_t2s, x_s, x_t)  # 修改点：T2S改用关节
            x_t = _apply_time_attn(self.adapter_s2t, x_t, x_s)  # 修改点：S2T改用时间

        # 模式 5: 原始的单边消融 (no_t2s / no_s2t)
        elif self.ablation_mode == 'no_t2s':
            # 只做 S2T (Standard Joint)
            x_t = _apply_joint_attn(self.adapter_s2t, x_t, x_s)

        elif self.ablation_mode == 'no_s2t':
            # 只做 T2S (Standard Time)
            x_s = _apply_time_attn(self.adapter_t2s, x_s, x_t)

        return x_s, x_t


# --- 主模块 ---
class BCASTInteractionModule(nn.Module):
    # 1. 修改 __init__ 接收 ablation_mode
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