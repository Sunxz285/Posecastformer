import torch
import torch.nn as nn

from lib.model.skeleton_embedding import SkeletonEmbedding
from lib.model.dual_stream_encoder import DualStreamEncoder
from lib.model.bcast_interaction import BCASTInteractionModule
from lib.model.pose_regression import PoseRegressionHead


class PoseCASTformer(nn.Module):
    def __init__(self,
                 num_joints=17,
                 in_channels=3,
                 embed_dim=256,  # default value
                 depth_total=5,
                 depth_interaction=3,
                 num_heads=8,
                 maxlen=243,
                 mlp_ratio=2.,
                 ablation_mode='full'):  # key parameter: default is 2
        """
        Complete PoseCASTformer model
        """
        super().__init__()

        # 1. Head: Embedding
        self.embedding = SkeletonEmbedding(
            num_joints=num_joints,
            in_channels=in_channels,
            embed_dim=embed_dim,
            max_len=maxlen
        )

        # 2. Upper body: Independent dual-stream encoding (N - K layers)
        depth_independent = depth_total - depth_interaction
        if depth_independent > 0:
            self.independent_encoder = DualStreamEncoder(
                dim=embed_dim,
                depth=depth_independent,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
        else:
            self.independent_encoder = None

        # 3. Lower body: B-CAST interaction encoding (K layers)
        self.interaction_encoder = BCASTInteractionModule(
            dim=embed_dim,
            depth=depth_interaction,
            num_heads=num_heads,
            bottleneck_ratio=0.25,
            mlp_ratio=mlp_ratio,
            ablation_mode=ablation_mode
        )

        # 4. Tail: Regression head
        self.regression_head = PoseRegressionHead(
            in_dim=embed_dim,
            out_dim=3
        )

    def forward(self, x):
        # 1. Embedding
        feat_s, feat_t = self.embedding(x)

        # 2. Independent encoding
        if self.independent_encoder is not None:
            feat_s, feat_t = self.independent_encoder(feat_s, feat_t)

        # 3. Interaction encoding
        feat_s, feat_t = self.interaction_encoder(feat_s, feat_t)

        # 4. Fusion
        fused_feat = feat_s + feat_t

        # 5. Regression
        pred_3d = self.regression_head(fused_feat)

        return pred_3d


if __name__ == "__main__":
    # Test 512-dimensional configuration
    model = PoseCASTformer(embed_dim=512, mlp_ratio=2., depth_total=5, depth_interaction=3).cuda()
    dummy_input = torch.randn(2, 243, 17, 3).cuda()
    output = model(dummy_input)
    print(f"Model Integrated Successfully (Dim=512, Ratio=2). Output Shape: {output.shape}")