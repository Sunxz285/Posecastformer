import torch
import torch.nn as nn

from .skeleton_embedding import SkeletonEmbedding
from .dual_stream_encoder import DualStreamEncoder
from .bcast_interaction import BCASTInteractionModule
from .pose_regression import PoseRegressionHead


class PoseCASTformer(nn.Module):
    def __init__(self,
                 num_joints=17,
                 in_channels=3,
                 embed_dim=256,  # 默认值
                 depth_total=5,
                 depth_interaction=3,
                 num_heads=8,
                 maxlen=243,
                 mlp_ratio=2.,
                 ablation_mode='full'):  # 关键参数: 默认为 2
        """
        PoseCASTformer 完整模型
        """
        super().__init__()

        # 1. 头部: Embedding
        self.embedding = SkeletonEmbedding(
            num_joints=num_joints,
            in_channels=in_channels,
            embed_dim=embed_dim,
            max_len=maxlen
        )

        # 2. 身体 (上): 独立双流编码 (N - K 层)
        depth_independent = depth_total - depth_interaction
        if depth_independent > 0:
            self.independent_encoder = DualStreamEncoder(
                dim=embed_dim,
                depth=depth_independent,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio  # 传递 mlp_ratio
            )
        else:
            self.independent_encoder = None

        # 3. 身体 (下): B-CAST 交互编码 (K 层)
        self.interaction_encoder = BCASTInteractionModule(
            dim=embed_dim,
            depth=depth_interaction,
            num_heads=num_heads,
            bottleneck_ratio=0.25,
            mlp_ratio=mlp_ratio,
            ablation_mode=ablation_mode
        )

        # 4. 尾部: 回归头
        self.regression_head = PoseRegressionHead(
            in_dim=embed_dim,
            out_dim=3
        )

    def forward(self, x):
        # 1. Embedding
        feat_s, feat_t = self.embedding(x)

        # 2. 独立编码
        if self.independent_encoder is not None:
            feat_s, feat_t = self.independent_encoder(feat_s, feat_t)

        # 3. 交互编码
        feat_s, feat_t = self.interaction_encoder(feat_s, feat_t)

        # 4. 融合
        fused_feat = feat_s + feat_t

        # 5. 回归
        pred_3d = self.regression_head(fused_feat)

        return pred_3d


if __name__ == "__main__":
    # 测试 512 维配置
    model = PoseCASTformer(embed_dim=512, mlp_ratio=2., depth_total=5, depth_interaction=3).cuda()
    dummy_input = torch.randn(2, 243, 17, 3).cuda()
    output = model(dummy_input)
    print(f"Model Integrated Successfully (Dim=512, Ratio=2). Output Shape: {output.shape}")