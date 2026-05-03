from __future__ import annotations

import numpy as np


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _valid(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & ~np.all(np.isclose(points, 0.0), axis=-1)


def _threshold_value(threshold: float) -> float:
    return threshold / 100.0 if threshold > 1.0 else threshold


def torso_scale(gt: np.ndarray) -> float:
    gt = np.asarray(gt, dtype=np.float32)
    if _valid(gt[[2, 11]]).all():
        scale = float(np.linalg.norm(gt[2] - gt[11]))
        if scale > 1e-6:
            return scale
    valid = _valid(gt)
    if valid.sum() >= 2:
        pts = gt[valid]
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        return max(diag, 1e-6)
    return 1.0


def pck(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.2) -> dict:
    pred = _to_numpy(pred).astype(np.float32)
    gt = _to_numpy(gt).astype(np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have same shape, got {pred.shape} vs {gt.shape}")
    valid = _valid(gt)
    distances = np.linalg.norm(pred - gt, axis=-1)
    limit = _threshold_value(float(threshold)) * torso_scale(gt)
    correct = (distances <= limit) & valid
    per_joint = np.full(gt.shape[0], np.nan, dtype=np.float32)
    per_joint[valid] = correct[valid].astype(np.float32)
    mean = float(np.nanmean(per_joint)) if np.isfinite(per_joint).any() else 0.0
    return {"mean": mean, "per_joint": per_joint, "valid": valid, "limit": limit}


def pck_batch(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.2) -> dict:
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    scores = [pck(p, g, threshold=threshold) for p, g in zip(pred, gt)]
    per_joint = np.stack([s["per_joint"] for s in scores], axis=0)
    mean_per_joint = np.nanmean(per_joint, axis=0)
    mean = float(np.nanmean(mean_per_joint)) if np.isfinite(mean_per_joint).any() else 0.0
    return {"mean": mean, "per_joint": mean_per_joint}
