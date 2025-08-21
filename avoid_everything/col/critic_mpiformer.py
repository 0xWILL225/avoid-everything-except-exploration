import torch
from torch import nn

from avoid_everything.mpiformer import (
    MPiFormerPointNet,
    Encoder,
    TransformerLayer,
    MultiHeadAttention,
    FeedForward,
    PositionEncoding3D,
)


class CriticMPiFormer(nn.Module):
    """
    Q(s, a) critic matching the actor's state encoding:
    - PointNet++ on the point cloud
    - Linear embedder for joint state q
    - Linear embedder for action a (normalized Δq)
    - Transformer encoder over [pc tokens..., q-token, a-token]
    - MLP head -> scalar Q
    """
    def __init__(
        self,
        num_robot_points: int,
        robot_dof: int,
        *,
        feature_dim: int = 4,
        n_heads: int = 8,
        d_model: int = 512,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.point_cloud_embedder = MPiFormerPointNet(num_robot_points, feature_dim, d_model)
        self.state_embedder = nn.Linear(robot_dof, d_model)
        self.action_embedder = nn.Linear(robot_dof, d_model)
        self.token_type_embedding = nn.Embedding(3, d_model)  # 0=pc, 1=state, 2=action
        self.pe_layer = PositionEncoding3D(d_model)

        enc_layer = TransformerLayer(
            d_model=d_model,
            self_attn=MultiHeadAttention(heads=n_heads, d_model=d_model, dropout_prob=dropout),
            src_attn=None,
            feed_forward=FeedForward(d_model=d_model, d_ff=4*d_model, dropout=dropout, activation=nn.GELU, is_gated=False, bias1=True, bias2=True, bias_gate=True),
            dropout_prob=dropout,
        )
        self.encoder = Encoder(enc_layer, n_layers=n_layers)
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        point_cloud_labels: torch.Tensor,  # [B,N,1], 0 robot, 1 scene, 2 target
        point_cloud: torch.Tensor,        # [B,N,3]
        q: torch.Tensor,                  # [B,DOF] normalized
        a: torch.Tensor,                  # [B,DOF] normalized (Δq)
        bounds: torch.Tensor,             # [2,3] pc bounds
    ) -> torch.Tensor:                    # [B,1]
        pc_emb, pos = self.point_cloud_embedder(point_cloud_labels, point_cloud)
        B = q.size(0)
        s_tok = self.state_embedder(q).unsqueeze(1)
        a_tok = self.action_embedder(a).unsqueeze(1)

        seq = torch.cat((pc_emb, s_tok, a_tok), dim=1).transpose(0, 1)  # [S,B,D]

        dev = point_cloud.device
        pc_type = self.token_type_embedding(torch.tensor(0, device=dev))
        s_type  = self.token_type_embedding(torch.tensor(1, device=dev))[None, None, :]
        a_type  = self.token_type_embedding(torch.tensor(2, device=dev))[None, None, :]

        pos_emb = torch.cat(
            (self.pe_layer(pos, bounds) + pc_type, s_type.expand(B, -1, -1), a_type.expand(B, 1, -1)),
            dim=1
        ).transpose(0, 1)

        h = self.encoder(seq + pos_emb, mask=None)  # [S,B,D]
        # Use the final token (action token) as pooled representation
        h_a = h[-1]                                   # [B,D]
        return self.q_head(h_a)                       # [B,1]
