from __future__ import annotations

import torch
from torch import nn


class TFDDTTokenizer(nn.Module):
    """Time-Frequency Dual-Dimensional Tokenization for CSI tensors."""

    def __init__(
        self,
        time_packets: int = 64,
        rx_antennas: int = 3,
        input_subcarriers: int = 114,
        subcarrier_mode: str = "keep",
        embed_dim: int = 1296,
    ) -> None:
        super().__init__()
        self.time_packets = time_packets
        self.rx_antennas = rx_antennas
        self.input_subcarriers = input_subcarriers
        self.subcarrier_mode = subcarrier_mode
        self.token_subcarriers = 64 if subcarrier_mode == "learned64" else input_subcarriers

        if subcarrier_mode == "learned64":
            self.subcarrier_proj = nn.Linear(input_subcarriers, 64)
        elif subcarrier_mode in {"keep", "resample64"}:
            self.subcarrier_proj = None
        else:
            raise ValueError(f"Unknown subcarrier mode: {subcarrier_mode}")

        self.freq_proj = nn.Linear(time_packets * rx_antennas, embed_dim)
        self.time_proj = nn.Linear(self.token_subcarriers * rx_antennas, embed_dim)
        self.freq_pos_embed = nn.Parameter(torch.zeros(1, self.token_subcarriers, embed_dim))
        self.time_pos_embed = nn.Parameter(torch.zeros(1, time_packets, embed_dim))
        nn.init.trunc_normal_(self.freq_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B, M, NR, NS), got {tuple(x.shape)}")
        bsz, packets, rx, subcarriers = x.shape
        if packets != self.time_packets or rx != self.rx_antennas:
            raise ValueError(
                f"Expected (M, NR)=({self.time_packets}, {self.rx_antennas}), got ({packets}, {rx})"
            )
        if subcarriers != self.input_subcarriers:
            raise ValueError(f"Expected NS={self.input_subcarriers}, got {subcarriers}")

        if self.subcarrier_proj is not None:
            x = self.subcarrier_proj(x.reshape(-1, subcarriers)).reshape(bsz, packets, rx, 64)
            subcarriers = 64

        freq_tokens = x.permute(0, 3, 1, 2).reshape(bsz, subcarriers, packets * rx)
        time_tokens = x.permute(0, 1, 3, 2).reshape(bsz, packets, subcarriers * rx)
        freq_tokens = self.freq_proj(freq_tokens) + self.freq_pos_embed
        time_tokens = self.time_proj(time_tokens) + self.time_pos_embed
        return freq_tokens, time_tokens
