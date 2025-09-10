import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# ========================
# Attention Pooling Module
# ========================
class AttnPool(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_queries: int,
        output_dim: int,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_dim (int): Input feature dim (must be multiple of 64).
            n_queries (int): Number of learnable queries.
            output_dim (int): Output dimension after pooling.
            dropout (float): Dropout prob.
        """
        super().__init__()
        assert in_dim % 64 == 0, "in_dim must be multiple of 64"

        self.n_q = n_queries
        self.num_heads = in_dim // 64

        # Learnable queries
        self.query = nn.Parameter(torch.empty(n_queries, in_dim))

        # Projections
        self.kv   = nn.Linear(in_dim, in_dim * 2)   # keys, values
        self.proj = nn.Linear(in_dim, output_dim)   # to fixed output dim

        # Norm + dropouts
        self.norm          = nn.LayerNorm(output_dim)
        self.attn_dropout  = dropout
        self.dropout       = nn.Dropout(dropout)
        self.query_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.query)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] (patch embeddings từ DINO)
        Returns:
            [B, output_dim]
        """
        B, N, D = x.shape
        H = self.num_heads
        Dh = D // H

        # Expand queries: [n_q, D] -> [B, n_q, D] -> [B, H, n_q, Dh]
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        q = self.query_dropout(q)
        q = q.reshape(B, self.n_q, H, Dh).permute(0, 2, 1, 3)

        # Keys/values: [B, N, D] -> [2, B, H, N, Dh]
        kv = self.kv(x).reshape(B, N, 2, H, Dh).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention pooling
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout)

        # [B, H, n_q, Dh] -> [B, n_q, D]
        attn = attn.permute(0, 2, 1, 3).reshape(B, self.n_q, D)
        attn = self.dropout(attn)

        # Project and average queries: [B, n_q, D] -> [B, output_dim]
        pooled = self.proj(attn).mean(dim=1)
        return self.norm(pooled)

# ========================
# Final Classifier with DINO
# ========================
class DinoAttnClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_queries: int = 4,
        dropout: float = 0.3,
        pretrained_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        d_model = self.backbone.config.hidden_size

        # Attention pooling
        self.pool = AttnPool(d_model, n_queries, d_model, dropout)

        # Head classifier
        self.head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        # Init head
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Extract patch features
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state  # [B, N, D]

        # Bỏ CLS token (patches only)
        patch_tokens = last_hidden[:, 1:, :]  # [B, N-1, D]

        # Attn pooling
        pooled = self.pool(patch_tokens)

        # Classification
        logits = self.head(pooled)
        return logits


def build_model(num_classes: int):
    return DinoAttnClassifier(num_classes=num_classes)
