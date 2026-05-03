from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import resample


NormalizeMode = Literal["zscore", "minmax", "none"]
SubcarrierMode = Literal["keep", "learned64", "resample64"]


def sanitize_csi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.all():
        return x
    fill = float(np.median(x[finite])) if finite.any() else 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def normalize_csi(x: np.ndarray, mode: NormalizeMode = "zscore", eps: float = 1e-6) -> np.ndarray:
    x = sanitize_csi(x)
    if mode == "none":
        return x
    if mode == "zscore":
        mean = float(x.mean())
        std = float(x.std())
        if not np.isfinite(std) or std < eps:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mean) / (std + eps)).astype(np.float32)
    if mode == "minmax":
        lo = float(x.min())
        hi = float(x.max())
        if not np.isfinite(hi - lo) or (hi - lo) < eps:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo + eps)).astype(np.float32)
    raise ValueError(f"Unknown CSI normalization mode: {mode}")


def resample_time(csi_amp: np.ndarray, target_packets: int = 64) -> np.ndarray:
    """Resample CSI amplitude from (NR, NS, M0) to (NR, NS, target_packets)."""
    csi_amp = sanitize_csi(csi_amp)
    if csi_amp.ndim == 2:
        csi_amp = csi_amp[:, :, None]
    if csi_amp.ndim != 3:
        raise ValueError(f"Expected CSI amplitude shape (NR, NS, M), got {csi_amp.shape}")
    if csi_amp.shape[-1] == target_packets:
        return csi_amp
    return sanitize_csi(resample(csi_amp, target_packets, axis=-1))


def resample_subcarriers(csi_amp: np.ndarray, target_subcarriers: int = 64) -> np.ndarray:
    """FFT resample along the subcarrier axis. Used only for the resample64 ablation."""
    if csi_amp.shape[1] == target_subcarriers:
        return csi_amp.astype(np.float32)
    return sanitize_csi(resample(csi_amp, target_subcarriers, axis=1))


def preprocess_csi_amp(
    csi_amp: np.ndarray,
    target_packets: int = 64,
    subcarrier_mode: SubcarrierMode = "keep",
    normalize: NormalizeMode = "zscore",
) -> np.ndarray:
    """Return CSI tensor as (M, NR, NS) float32, ready for MultiFormer."""
    x = resample_time(csi_amp, target_packets=target_packets)
    if subcarrier_mode == "resample64":
        x = resample_subcarriers(x, target_subcarriers=64)
    elif subcarrier_mode not in {"keep", "learned64"}:
        raise ValueError(f"Unknown subcarrier mode: {subcarrier_mode}")

    x = np.transpose(x, (2, 0, 1)).astype(np.float32)  # (M, NR, NS)
    return normalize_csi(x, mode=normalize)
