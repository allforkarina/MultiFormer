from __future__ import annotations

import torch
from torch import nn

from .attention_extractor import DualAttentionExtractor
from .msfn import MSFN
from .tfddt import TFDDTTokenizer


class MultiFormer(nn.Module):
    def __init__(
        self,
        time_packets: int = 64,
        rx_antennas: int = 3,
        input_subcarriers: int = 114,
        subcarrier_mode: str = "keep",
        embed_dim: int = 1296,
        num_heads: int = 8,
        depth: int = 8,
        dropout: float = 0.1,
        heatmap_size: int = 36,
        recon_channels: int = 64,
        feature_channels: int = 128,
        decoder_hidden: int = 512,
        stages: int = 3,
    ) -> None:
        super().__init__()
        self.tokenizer = TFDDTTokenizer(
            time_packets=time_packets,
            rx_antennas=rx_antennas,
            input_subcarriers=input_subcarriers,
            subcarrier_mode=subcarrier_mode,
            embed_dim=embed_dim,
        )
        freq_tokens = self.tokenizer.token_subcarriers
        self.extractor = DualAttentionExtractor(
            freq_tokens=freq_tokens,
            time_tokens=time_packets,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            dropout=dropout,
            heatmap_size=heatmap_size,
            recon_channels=recon_channels,
        )
        self.msfn = MSFN(
            feature_channels=feature_channels,
            decoder_hidden=decoder_hidden,
            stages=stages,
        )

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        freq_tokens, time_tokens = self.tokenizer(x)
        features = self.extractor(freq_tokens, time_tokens)
        return self.msfn(features)
