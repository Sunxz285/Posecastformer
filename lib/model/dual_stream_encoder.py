import torch
import torch.nn as nn
import os
import math

class Mlp(nn.Module):
    """
    Feed-Forward Network (FFN)
    Structure: Linear -> GELU -> Linear
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpatialBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        # Here int(dim * mlp_ratio) determines the intermediate layer size
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        B, T, J, C = x.shape
        x_reshaped = x.view(B * T, J, C)
        x_reshaped = x_reshaped + self.attn(self.norm1(x_reshaped))
        x_reshaped = x_reshaped + self.mlp(self.norm2(x_reshaped))
        x_out = x_reshaped.view(B, T, J, C)
        return x_out

class TemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        B, T, J, C = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * J, T, C)
        x_reshaped = x_reshaped + self.attn(self.norm1(x_reshaped))
        x_reshaped = x_reshaped + self.mlp(self.norm2(x_reshaped))
        x_out = x_reshaped.view(B, J, T, C).permute(0, 2, 1, 3)
        return x_out

class DualStreamEncoder(nn.Module):
    def __init__(self, dim=256, depth=5, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.spatial_blocks = nn.ModuleList([
            SpatialBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, spatial_x, temporal_x):
        for blk in self.spatial_blocks:
            spatial_x = blk(spatial_x)
        for blk in self.temporal_blocks:
            temporal_x = blk(temporal_x)
        return spatial_x, temporal_x