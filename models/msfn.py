from __future__ import annotations

import torch
from torch import nn

from .heatmap_decoder import HeatmapDecoder
from .papm import PAPM


class MSFN(nn.Module):
    def __init__(
        self,
        feature_channels: int = 128,
        decoder_hidden: int = 512,
        stages: int = 3,
        pcm_channels: int = 19,
        paf_channels: int = 38,
    ) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError("MSFN needs at least one stage")
        self.decoders = nn.ModuleList(
            [
                HeatmapDecoder(
                    in_channels=feature_channels,
                    hidden_channels=decoder_hidden,
                    pcm_channels=pcm_channels,
                    paf_channels=paf_channels,
                )
                for _ in range(stages)
            ]
        )
        self.papms = nn.ModuleList(
            [PAPM(feature_channels=feature_channels, heat_channels=pcm_channels + paf_channels) for _ in range(stages - 1)]
        )

    def forward(self, features: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        current = features
        for idx, decoder in enumerate(self.decoders):
            pcm, paf = decoder(current)
            outputs.append((pcm, paf))
            if idx < len(self.papms):
                current = self.papms[idx](current, torch.cat([pcm, paf], dim=1))
        return outputs
