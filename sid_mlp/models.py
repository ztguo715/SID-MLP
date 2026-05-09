"""Model definitions for the public SID-MLP code path.

Only the paper decoder is kept: one cross-attention context readout and four
prefix-conditioned 256-way MLP heads.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import CODEBOOK_SIZE, DIGIT_OFFSETS


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_dim: int = 0,
    ):
        super().__init__()
        inner = attn_dim if attn_dim > 0 else d_model
        if inner % num_heads != 0:
            raise ValueError("attn_dim/d_model must be divisible by num_heads")

        self.inner = inner
        self.num_heads = num_heads
        self.head_dim = inner // num_heads
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.dropout = dropout

    def precompute_kv(self, encoder_seq: torch.Tensor):
        batch_size, seq_len, _ = encoder_seq.shape
        k = self.k_proj(encoder_seq).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2).contiguous()
        v = self.v_proj(encoder_seq).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2).contiguous()
        return k, v

    def forward(
        self,
        query_vec: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = query_vec.shape[0]
        q = self.q_proj(query_vec).view(batch_size, self.num_heads, 1, self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        any_valid = torch.isfinite(scores).any(dim=-1, keepdim=True)
        safe_scores = torch.where(any_valid, scores, torch.zeros_like(scores))
        probs = F.softmax(safe_scores, dim=-1)
        probs = torch.where(any_valid, probs, torch.zeros_like(probs))
        if self.dropout > 0 and self.training:
            probs = F.dropout(probs, p=self.dropout)

        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.inner)
        return self.out_proj(out).squeeze(1)


class MLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.fc(x))))


def make_mlp(in_dim, out_dim, hidden_dim, num_layers, dropout):
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    ]
    for _ in range(num_layers - 1):
        layers.append(MLPBlock(hidden_dim, dropout=dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class SIDMLP(nn.Module):
    """SID-MLP decoder distilled from a frozen TIGER teacher."""

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        embed_dim=128,
        d_model=128,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.1,
        head_hidden=512,
        head_layers: int = 1,
        attn_dim: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        self.cross_attn = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attn_dim=attn_dim,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)

        self.head1 = make_mlp(d_model, CODEBOOK_SIZE, head_hidden, head_layers, dropout)
        self.head2 = make_mlp(d_model + embed_dim, CODEBOOK_SIZE, head_hidden, head_layers, dropout)
        self.head3 = make_mlp(d_model + 2 * embed_dim, CODEBOOK_SIZE, head_hidden, head_layers, dropout)
        self.head4 = make_mlp(d_model + 3 * embed_dim, CODEBOOK_SIZE, head_hidden, head_layers, dropout)

    def post_attn(self, query: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        x = self.norm1(query + attn_out)
        return self.norm2(x + self.ffn(x))

    def encode(self, enc_seq, enc_mask):
        valid = (~enc_mask).to(enc_seq.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (enc_seq * valid.unsqueeze(-1)).sum(dim=1) / denom
        query = self.query_proj(pooled)
        k, v = self.cross_attn.precompute_kv(enc_seq)
        return self.post_attn(query, self.cross_attn(query, k, v, key_padding_mask=enc_mask))

    def forward(self, enc_seq, enc_mask, d1_ids=None, d2_ids=None, d3_ids=None):
        ctx = self.encode(enc_seq, enc_mask)
        logits1 = self.head1(ctx)

        if d1_ids is None:
            d1_ids = logits1.argmax(dim=-1) + DIGIT_OFFSETS[0]
        with torch.no_grad():
            e1 = self.embedding(d1_ids)
        logits2 = self.head2(torch.cat([ctx, e1], dim=1))

        if d2_ids is None:
            d2_ids = logits2.argmax(dim=-1) + DIGIT_OFFSETS[1]
        with torch.no_grad():
            e2 = self.embedding(d2_ids)
        logits3 = self.head3(torch.cat([ctx, e1, e2], dim=1))

        if d3_ids is None:
            d3_ids = logits3.argmax(dim=-1) + DIGIT_OFFSETS[2]
        with torch.no_grad():
            e3 = self.embedding(d3_ids)
        logits4 = self.head4(torch.cat([ctx, e1, e2, e3], dim=1))

        return logits1, logits2, logits3, logits4


__all__ = ["CrossAttention", "MLPBlock", "SIDMLP", "make_mlp"]
