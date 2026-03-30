import torch
import torch.nn as nn

class SkeletonEmbedding(nn.Module):
    def __init__(self, num_joints=17, in_channels=3, embed_dim=256, max_len=243):
        """
        Skeleton sequence embedding layer (Skeleton Sequence Embedding)
        Strictly follows the methodology formulas (3) and (4)
        """
        super().__init__()

        # 1. Linear projection (Token Embedding)
        self.joints_embed = nn.Linear(in_channels, embed_dim)

        # 2. Spatial positional encoding (Spatial PosEnc)
        # Shape: [1, 1, 17, D]
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim))

        # 3. Temporal positional encoding (Temporal PosEnc)
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

        # 1. Linear projection
        x_proj = self.joints_embed(x)

        # 2. Add spatial and temporal positional encodings

        base_feat = x_proj + self.pos_embed + self.temp_embed[:, :T, :, :]

        # 3. Distribute to dual streams

        spatial_feat = base_feat
        temporal_feat = base_feat

        return spatial_feat, temporal_feat