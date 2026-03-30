import torch
import torch.nn as nn

class SkeletonEmbedding(nn.Module):
    def __init__(self, num_joints=17, in_channels=3, embed_dim=256, max_len=243):
        """
        骨架序列嵌入层 (Skeleton Sequence Embedding)
        严格遵循方法论公式 (3) 和 (4)
        """
        super().__init__()

        # 1. 线性投影 (Token Embedding)
        self.joints_embed = nn.Linear(in_channels, embed_dim)

        # 2. 空间位置编码 (Spatial PosEnc)
        # Shape: [1, 1, 17, D]
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim))

        # 3. 时间位置编码 (Temporal PosEnc)
        # Shape: [1, MaxLen, 1, D]
        self.temp_embed = nn.Parameter(torch.zeros(1, max_len, 1, embed_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.joints_embed.weight)
        nn.init.constant_(self.joints_embed.bias, 0)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.temp_embed, std=0.02)

    def forward(self, x):
        # x: [Batch, T, J, C]
        B, T, J, C = x.shape

        # 1. 线性投影
        x_proj = self.joints_embed(x)

        # 2. 【核心修复】时空位置编码叠加
        # 对应方法论公式: z = xW + e_t + e_j
        # 必须同时加上时间编码和空间编码，缺一不可
        base_feat = x_proj + self.pos_embed + self.temp_embed[:, :T, :, :]

        # 3. 分发给双流
        # 对应方法论公式: FS(0) = Z, FT(0) = Z
        # 两个分支的起点完全相同，包含完整上下文
        spatial_feat = base_feat
        temporal_feat = base_feat

        return spatial_feat, temporal_feat