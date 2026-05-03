from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


OPENPOSE_18_NAMES = [
    "nose",
    "neck",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
]

# COCO17 order:
# nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow,
# r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle.
COCO17_TO_OPENPOSE18 = {
    0: 0,
    2: 6,
    3: 8,
    4: 10,
    5: 5,
    6: 7,
    7: 9,
    8: 12,
    9: 14,
    10: 16,
    11: 11,
    12: 13,
    13: 15,
    14: 2,
    15: 1,
    16: 4,
    17: 3,
}

LIMBS_18 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (2, 8),
    (5, 11),
]


def valid_point(point: np.ndarray) -> bool:
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    """Convert MM-Fi COCO17 keypoints to the OpenPose-18 order used by MultiFormer."""
    kpts17 = np.asarray(kpts17, dtype=np.float32)
    if kpts17.shape[-2:] != (17, 2):
        raise ValueError(f"Expected keypoints with shape (17, 2), got {kpts17.shape}")

    kpts18 = np.zeros((18, 2), dtype=np.float32)
    valid = np.zeros(18, dtype=bool)
    for op_idx, coco_idx in COCO17_TO_OPENPOSE18.items():
        point = kpts17[coco_idx]
        if valid_point(point):
            kpts18[op_idx] = point
            valid[op_idx] = True

    left_shoulder = kpts17[5]
    right_shoulder = kpts17[6]
    if valid_point(left_shoulder) and valid_point(right_shoulder):
        kpts18[1] = (left_shoulder + right_shoulder) * 0.5
        valid[1] = True
    elif valid_point(left_shoulder):
        kpts18[1] = left_shoulder
        valid[1] = True
    elif valid_point(right_shoulder):
        kpts18[1] = right_shoulder
        valid[1] = True

    kpts18[~valid] = 0.0
    return kpts18


def pose_to_heatmap_coords(
    kpts: np.ndarray,
    size: int = 36,
    pose_range: Tuple[float, float] = (-0.8, 0.8),
    clip: bool = True,
) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float32).copy()
    lo, hi = pose_range
    scale = (size - 1) / (hi - lo)
    invalid = ~np.isfinite(kpts).all(axis=-1) | np.all(np.isclose(kpts, 0.0), axis=-1)
    kpts = (kpts - lo) * scale
    if clip:
        kpts = np.clip(kpts, 0, size - 1)
    kpts[invalid] = 0.0
    return kpts.astype(np.float32)


def heatmap_to_pose_coords(
    kpts: np.ndarray,
    size: int = 36,
    pose_range: Tuple[float, float] = (-0.8, 0.8),
) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float32).copy()
    lo, hi = pose_range
    invalid = ~np.isfinite(kpts).all(axis=-1) | np.all(np.isclose(kpts, 0.0), axis=-1)
    kpts = kpts / max(size - 1, 1) * (hi - lo) + lo
    kpts[invalid] = 0.0
    return kpts.astype(np.float32)


def gaussian_2d(center: Iterable[float], size: int = 36, sigma: float = 1.5) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    grid_y, grid_x = np.mgrid[0:size, 0:size].astype(np.float32)
    dist2 = (grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2
    heatmap = np.exp(-dist2 / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


def paf_line(
    p1: Iterable[float],
    p2: Iterable[float],
    size: int = 36,
    width: float = 1.0,
) -> np.ndarray:
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    out = np.zeros((2, size, size), dtype=np.float32)
    limb = p2 - p1
    length = float(np.linalg.norm(limb))
    if length < 1e-6:
        return out

    unit = limb / length
    grid_y, grid_x = np.mgrid[0:size, 0:size].astype(np.float32)
    rel_x = grid_x - p1[0]
    rel_y = grid_y - p1[1]
    proj = rel_x * unit[0] + rel_y * unit[1]
    proj_clamped = np.clip(proj, 0.0, length)
    closest_x = p1[0] + proj_clamped * unit[0]
    closest_y = p1[1] + proj_clamped * unit[1]
    dist = np.sqrt((grid_x - closest_x) ** 2 + (grid_y - closest_y) ** 2)
    mask = (proj >= 0.0) & (proj <= length) & (dist <= width)
    out[0, mask] = unit[0]
    out[1, mask] = unit[1]
    return out


def build_pcm_paf(
    kpts18_pose: np.ndarray,
    size: int = 36,
    sigma: float = 1.5,
    paf_width: float = 1.0,
    pose_range: Tuple[float, float] = (-0.8, 0.8),
) -> tuple[np.ndarray, np.ndarray]:
    """Build PCM(19,H,W) and PAF(38,H,W) targets from OpenPose-18 keypoints."""
    kpts18_hm = pose_to_heatmap_coords(kpts18_pose, size=size, pose_range=pose_range)
    pcm = np.zeros((19, size, size), dtype=np.float32)
    valid = np.array([valid_point(p) for p in kpts18_pose], dtype=bool)

    for idx, point in enumerate(kpts18_hm):
        if valid[idx]:
            pcm[idx] = gaussian_2d(point, size=size, sigma=sigma)
    pcm[18] = pcm[:18].mean(axis=0)

    paf = np.zeros((len(LIMBS_18) * 2, size, size), dtype=np.float32)
    for limb_idx, (a, b) in enumerate(LIMBS_18):
        if valid[a] and valid[b]:
            paf[2 * limb_idx : 2 * limb_idx + 2] = paf_line(
                kpts18_hm[a],
                kpts18_hm[b],
                size=size,
                width=paf_width,
            )
    return pcm, paf
