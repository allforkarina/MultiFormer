from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, ffn_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(embed_dim * ffn_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class ReconstructionLayer(nn.Module):
    def __init__(
        self,
        token_count: int,
        embed_dim: int,
        heatmap_size: int = 36,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        self.token_count = token_count
        self.heatmap_size = heatmap_size
        target_dim = heatmap_size * heatmap_size
        self.to_heatmap = nn.Identity() if embed_dim == target_dim else nn.Linear(embed_dim, target_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(token_count, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, token_count, _ = tokens.shape
        if token_count != self.token_count:
            raise ValueError(f"Expected {self.token_count} tokens, got {token_count}")
        x = self.to_heatmap(tokens)
        x = x.reshape(bsz, token_count, self.heatmap_size, self.heatmap_size)
        return self.conv(x)


class DualAttentionExtractor(nn.Module):
    def __init__(
        self,
        freq_tokens: int,
        time_tokens: int = 64,
        embed_dim: int = 1296,
        num_heads: int = 8,
        depth: int = 8,
        dropout: float = 0.1,
        heatmap_size: int = 36,
        recon_channels: int = 64,
    ) -> None:
        super().__init__()
        self.freq_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.time_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.freq_recon = ReconstructionLayer(freq_tokens, embed_dim, heatmap_size, recon_channels)
        self.time_recon = ReconstructionLayer(time_tokens, embed_dim, heatmap_size, recon_channels)

    def forward(self, freq_tokens: torch.Tensor, time_tokens: torch.Tensor) -> torch.Tensor:
        for block in self.freq_blocks:
            freq_tokens = block(freq_tokens)
        for block in self.time_blocks:
            time_tokens = block(time_tokens)
        freq_map = self.freq_recon(freq_tokens)
        time_map = self.time_recon(time_tokens)
        return torch.cat([freq_map, time_map], dim=1)
