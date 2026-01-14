from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# =============================================================================
# FT-Transformer model and sklearn-style wrapper.
# =============================================================================
# Define FT-Transformer model structure.


class FeatureTokenizer(nn.Module):
    """Map numeric/categorical/geo tokens into transformer input tokens."""

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities,
        d_model: int,
        num_geo: int = 0,
        num_numeric_tokens: int = 1,
    ):
        super().__init__()

        self.num_numeric = num_numeric
        self.num_geo = num_geo
        self.has_geo = num_geo > 0

        if num_numeric > 0:
            if int(num_numeric_tokens) <= 0:
                raise ValueError("num_numeric_tokens must be >= 1 when numeric features exist.")
            self.num_numeric_tokens = int(num_numeric_tokens)
            self.has_numeric = True
            self.num_linear = nn.Linear(num_numeric, d_model * self.num_numeric_tokens)
        else:
            self.num_numeric_tokens = 0
            self.has_numeric = False

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, d_model) for card in cat_cardinalities
        ])

        if self.has_geo:
            # Map geo tokens with a linear layer to avoid one-hot on raw strings; upstream is encoded/normalized.
            self.geo_linear = nn.Linear(num_geo, d_model)

    def forward(self, X_num, X_cat, X_geo=None):
        tokens = []

        if self.has_numeric:
            batch_size = X_num.shape[0]
            num_token = self.num_linear(X_num)
            num_token = num_token.view(batch_size, self.num_numeric_tokens, -1)
            tokens.append(num_token)

        for i, emb in enumerate(self.embeddings):
            tok = emb(X_cat[:, i])
            tokens.append(tok.unsqueeze(1))

        if self.has_geo:
            if X_geo is None:
                raise RuntimeError("Geo tokens are enabled but X_geo was not provided.")
            geo_token = self.geo_linear(X_geo)
            tokens.append(geo_token.unsqueeze(1))

        x = torch.cat(tokens, dim=1)
        return x

# Encoder layer with residual scaling.


class ScaledTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, residual_scale_attn: float = 1.0,
                 residual_scale_ffn: float = 1.0, norm_first: bool = True,
                 ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization and dropout.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        # If you prefer ReLU, set: self.activation = nn.ReLU()
        self.norm_first = norm_first

        # Residual scaling coefficients.
        self.res_scale_attn = residual_scale_attn
        self.res_scale_ffn = residual_scale_ffn

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Input tensor shape: (batch, seq_len, d_model).
        x = src

        if self.norm_first:
            # Pre-norm before attention.
            x = x + self._sa_block(self.norm1(x), src_mask,
                                   src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm (usually disabled).
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        # Self-attention with residual scaling.
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        return self.res_scale_attn * self.dropout1(attn_out)

    def _ff_block(self, x):
        # Feed-forward block with residual scaling.
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.res_scale_ffn * self.dropout2(x2)

# FT-Transformer core model.


class FTTransformerCore(nn.Module):
    # Minimal FT-Transformer built from:
    #   1) FeatureTokenizer: convert numeric/categorical features to tokens;
    #   2) TransformerEncoder: model feature interactions;
    #   3) Pooling + MLP + Softplus: positive outputs for Tweedie/Gamma tasks.

    def __init__(self, num_numeric: int, cat_cardinalities, d_model: int = 64,
                 n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1,
                 task_type: str = 'regression', num_geo: int = 0,
                 num_numeric_tokens: int = 1
                 ):
        super().__init__()

        self.num_numeric = int(num_numeric)
        self.cat_cardinalities = list(cat_cardinalities or [])

        self.tokenizer = FeatureTokenizer(
            num_numeric=num_numeric,
            cat_cardinalities=cat_cardinalities,
            d_model=d_model,
            num_geo=num_geo,
            num_numeric_tokens=num_numeric_tokens
        )
        scale = 1.0 / math.sqrt(n_layers)  # Recommended default.
        encoder_layer = ScaledTransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            residual_scale_attn=scale,
            residual_scale_ffn=scale,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        self.n_layers = n_layers

        layers = [
            # If you need a deeper head, enable the sample layers below:
            # nn.LayerNorm(d_model),  # Extra normalization
            # nn.Linear(d_model, d_model),  # Extra fully connected layer
            # nn.GELU(),  # Activation
            nn.Linear(d_model, 1),
        ]

        if task_type == 'classification':
            # Classification outputs logits for BCEWithLogitsLoss.
            layers.append(nn.Identity())
        else:
            # Regression keeps positive outputs for Tweedie/Gamma.
            layers.append(nn.Softplus())

        self.head = nn.Sequential(*layers)

        # ---- Self-supervised reconstruction head (masked modeling) ----
        self.num_recon_head = nn.Linear(
            d_model, self.num_numeric) if self.num_numeric > 0 else None
        self.cat_recon_heads = nn.ModuleList([
            nn.Linear(d_model, int(card)) for card in self.cat_cardinalities
        ])

    def forward(
            self,
            X_num,
            X_cat,
            X_geo=None,
            return_embedding: bool = False,
            return_reconstruction: bool = False):

        # Inputs:
        #   X_num -> float32 tensor with shape (batch, num_numeric_features)
        #   X_cat -> long tensor with shape (batch, num_categorical_features)
        #   X_geo -> float32 tensor with shape (batch, geo_token_dim)

        if self.training and not hasattr(self, '_printed_device'):
            print(f">>> FTTransformerCore executing on device: {X_num.device}")
            self._printed_device = True

        # => tensor shape (batch, token_num, d_model)
        tokens = self.tokenizer(X_num, X_cat, X_geo)
        # => tensor shape (batch, token_num, d_model)
        x = self.encoder(tokens)

        # Mean-pool tokens, then send to the head.
        x = x.mean(dim=1)                      # => tensor shape (batch, d_model)

        if return_reconstruction:
            num_pred, cat_logits = self.reconstruct(x)
            cat_logits_out = tuple(
                cat_logits) if cat_logits is not None else tuple()
            if return_embedding:
                return x, num_pred, cat_logits_out
            return num_pred, cat_logits_out

        if return_embedding:
            return x

        # => tensor shape (batch, 1); Softplus keeps it positive.
        out = self.head(x)
        return out

    def reconstruct(self, embedding: torch.Tensor) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """Reconstruct numeric/categorical inputs from pooled embedding (batch, d_model)."""
        num_pred = self.num_recon_head(
            embedding) if self.num_recon_head is not None else None
        cat_logits = [head(embedding) for head in self.cat_recon_heads]
        return num_pred, cat_logits

# TabularDataset.


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, X_geo, y, w):

        # Input tensors:
        #   X_num: torch.float32, shape=(N, num_numeric_features)
        #   X_cat: torch.long,   shape=(N, num_categorical_features)
        #   X_geo: torch.float32, shape=(N, geo_token_dim), can be empty
        #   y:     torch.float32, shape=(N, 1)
        #   w:     torch.float32, shape=(N, 1)

        self.X_num = X_num
        self.X_cat = X_cat
        self.X_geo = X_geo
        self.y = y
        self.w = w

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (
            self.X_num[idx],
            self.X_cat[idx],
            self.X_geo[idx],
            self.y[idx],
            self.w[idx],
        )


class MaskedTabularDataset(Dataset):
    def __init__(self,
                 X_num_masked: torch.Tensor,
                 X_cat_masked: torch.Tensor,
                 X_geo: torch.Tensor,
                 X_num_true: Optional[torch.Tensor],
                 num_mask: Optional[torch.Tensor],
                 X_cat_true: Optional[torch.Tensor],
                 cat_mask: Optional[torch.Tensor]):
        self.X_num_masked = X_num_masked
        self.X_cat_masked = X_cat_masked
        self.X_geo = X_geo
        self.X_num_true = X_num_true
        self.num_mask = num_mask
        self.X_cat_true = X_cat_true
        self.cat_mask = cat_mask

    def __len__(self):
        return self.X_num_masked.shape[0]

    def __getitem__(self, idx):
        return (
            self.X_num_masked[idx],
            self.X_cat_masked[idx],
            self.X_geo[idx],
            None if self.X_num_true is None else self.X_num_true[idx],
            None if self.num_mask is None else self.num_mask[idx],
            None if self.X_cat_true is None else self.X_cat_true[idx],
            None if self.cat_mask is None else self.cat_mask[idx],
        )

