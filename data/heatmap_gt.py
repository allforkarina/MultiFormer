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

# H36M-17 order (MM-Fi native format):
# pelvis, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle,
# spine, thorax, neck, head, l_shoulder, l_elbow, l_wrist,
# r_shoulder, r_elbow, r_wrist.
H36M17_TO_OPENPOSE18 = {
    0: 9,   # OP nose       ← H36M neck/head_base
    2: 14,  # OP r_shoulder ← H36M r_shoulder
    3: 15,  # OP r_elbow    ← H36M r_elbow
    4: 16,  # OP r_wrist    ← H36M r_wrist
    5: 11,  # OP l_shoulder ← H36M l_shoulder
    6: 12,  # OP l_elbow    ← H36M l_elbow
    7: 13,  # OP l_wrist    ← H36M l_wrist
    8: 1,   # OP r_hip      ← H36M r_hip
    9: 2,   # OP r_knee     ← H36M r_knee
    10: 3,  # OP r_ankle    ← H36M r_ankle
    11: 4,  # OP l_hip      ← H36M l_hip
    12: 5,  # OP l_knee     ← H36M l_knee
    13: 6,  # OP l_ankle    ← H36M l_ankle
}

LIMBS_18 = [
    # Right leg
    (0, 1),
    (1, 2),
    (2, 3),
    # Left leg
    (0, 4),
    (4, 5),
    (5, 6),
    # Spine
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    # Right arm
    (8, 14),
    (14, 15),
    (15, 16),
    # Left arm
    (8, 11),
    (11, 12),
    (12, 13),
]


def valid_point(point: np.ndarray) -> bool:
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def h36m17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    """Convert MM-Fi H36M-17 keypoints to the OpenPose-18 order used by MultiFormer."""
    kpts17 = np.asarray(kpts17, dtype=np.float32)
    if kpts17.shape[-2:] != (17, 2):
        raise ValueError(f"Expected keypoints with shape (17, 2), got {kpts17.shape}")

    kpts18 = np.zeros((18, 2), dtype=np.float32)
    valid = np.zeros(18, dtype=bool)
    for op_idx, src_idx in H36M17_TO_OPENPOSE18.items():
        point = kpts17[src_idx]
        if valid_point(point):
            kpts18[op_idx] = point
            valid[op_idx] = True

    left_shoulder = kpts17[11]
    right_shoulder = kpts17[14]
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
