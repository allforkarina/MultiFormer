from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from data.heatmap_gt import LIMBS_18, heatmap_to_pose_coords


@dataclass
class Peak:
    x: float
    y: float
    score: float


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def nms_peaks(heatmap: np.ndarray, threshold: float = 0.1, max_peaks: int = 30) -> list[Peak]:
    heatmap = _to_numpy(heatmap).astype(np.float32)
    pooled = maximum_filter(heatmap, size=3, mode="constant")
    mask = (heatmap == pooled) & (heatmap >= threshold)
    ys, xs = np.where(mask)
    peaks = [Peak(float(x), float(y), float(heatmap[y, x])) for y, x in zip(ys, xs)]
    peaks.sort(key=lambda p: p.score, reverse=True)
    return peaks[:max_peaks]


def decode_single_person(
    pcm,
    paf=None,
    peak_threshold: float = 0.1,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> np.ndarray:
    """Decode a single pose by taking the strongest PCM peak for each joint."""
    pcm = _to_numpy(pcm)
    if pcm.ndim == 4:
        pcm = pcm[0]
    if pcm.shape[0] < 18:
        raise ValueError(f"PCM must have at least 18 channels, got {pcm.shape}")
    size = pcm.shape[-1]
    kpts_hm = np.zeros((18, 2), dtype=np.float32)
    for joint_idx in range(18):
        channel = pcm[joint_idx]
        y, x = np.unravel_index(int(np.argmax(channel)), channel.shape)
        if channel[y, x] >= peak_threshold:
            kpts_hm[joint_idx] = [float(x), float(y)]
    return heatmap_to_pose_coords(kpts_hm, size=size, pose_range=pose_range)


def paf_connection_score(paf: np.ndarray, limb_idx: int, p1: Peak, p2: Peak, samples: int = 10) -> float:
    paf = _to_numpy(paf)
    if paf.ndim == 4:
        paf = paf[0]
    vec = np.array([p2.x - p1.x, p2.y - p1.y], dtype=np.float32)
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return 0.0
    unit = vec / length
    xs = np.linspace(p1.x, p2.x, samples)
    ys = np.linspace(p1.y, p2.y, samples)
    h, w = paf.shape[-2:]
    score = 0.0
    valid = 0
    for x, y in zip(xs, ys):
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        px = paf[2 * limb_idx, yi, xi]
        py = paf[2 * limb_idx + 1, yi, xi]
        score += float(px * unit[0] + py * unit[1])
        valid += 1
    return score / max(valid, 1)


def match_limb_candidates(
    paf: np.ndarray,
    limb_idx: int,
    left: list[Peak],
    right: list[Peak],
    min_score: float = 0.05,
) -> list[tuple[int, int, float]]:
    if not left or not right:
        return []
    scores = np.zeros((len(left), len(right)), dtype=np.float32)
    for i, p1 in enumerate(left):
        for j, p2 in enumerate(right):
            scores[i, j] = paf_connection_score(paf, limb_idx, p1, p2)
    rows, cols = linear_sum_assignment(-scores)
    matches = []
    for row, col in zip(rows, cols):
        score = float(scores[row, col])
        if score >= min_score:
            matches.append((int(row), int(col), score))
    return matches


def decode_connections(
    pcm,
    paf,
    peak_threshold: float = 0.1,
    max_peaks: int = 30,
    min_limb_score: float = 0.05,
) -> dict:
    """Return NMS peaks and per-limb Hungarian matches for multi-person post-processing."""
    pcm = _to_numpy(pcm)
    paf = _to_numpy(paf)
    if pcm.ndim == 4:
        pcm = pcm[0]
    if paf.ndim == 4:
        paf = paf[0]
    peaks_by_joint = [nms_peaks(pcm[j], threshold=peak_threshold, max_peaks=max_peaks) for j in range(18)]
    connections = {}
    for limb_idx, (a, b) in enumerate(LIMBS_18):
        connections[(a, b)] = match_limb_candidates(
            paf,
            limb_idx,
            peaks_by_joint[a],
            peaks_by_joint[b],
            min_score=min_limb_score,
        )
    return {"peaks": peaks_by_joint, "connections": connections}
