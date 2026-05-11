"""Verify the H5 keypoint double-normalization fix.

Checks that keypoints are no longer collapsed to ~(-0.8, -0.8) and that
PCM targets have distinct Gaussian peaks spread across the heatmap.

Usage:
    python scripts/verify_kpts_fix.py --config configs/E01.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from train import build_dataset, load_config
from data.heatmap_gt import OPENPOSE_18_NAMES

POSE_MIN = -0.8
POSE_MAX = 0.8


def check_keypoints(kpts18_batch: np.ndarray) -> dict:
    """Analyze a batch of OpenPose18 keypoints."""
    all_kpts = kpts18_batch.reshape(-1, 2)  # (N*18, 2)
    valid = np.isfinite(all_kpts).all(axis=-1) & ~np.all(np.isclose(all_kpts, 0.0), axis=-1)
    valid_kpts = all_kpts[valid]

    per_joint_std = []
    for j in range(18):
        joint_kpts = kpts18_batch[:, j, :]
        v = np.isfinite(joint_kpts).all(axis=-1) & ~np.all(np.isclose(joint_kpts, 0.0), axis=-1)
        if v.sum() >= 2:
            per_joint_std.append(float(np.std(joint_kpts[v], axis=0).mean()))
        else:
            per_joint_std.append(0.0)

    avg_std = float(np.mean(per_joint_std)) if per_joint_std else 0.0
    range_min = float(valid_kpts.min()) if len(valid_kpts) > 0 else 0.0
    range_max = float(valid_kpts.max()) if len(valid_kpts) > 0 else 0.0

    print("--- Keypoints Distribution ---")
    print(f"  valid keypoints: {len(valid_kpts)} / {len(all_kpts)}")
    print(f"  range: [{range_min:.4f}, {range_max:.4f}]")
    print(f"  per-joint coordinate std (mean): {avg_std:.4f}")

    issues = []
    if range_max - range_min < 0.05:
        issues.append(f"keypoint range too narrow ({range_min:.4f} to {range_max:.4f}) — still collapsed?")
    if range_min < POSE_MIN - 0.1 or range_max > POSE_MAX + 0.1:
        issues.append(f"keypoints outside pose_range [{POSE_MIN}, {POSE_MAX}]")
    if avg_std < 0.02:
        issues.append(f"per-joint std too low ({avg_std:.4f}) — keypoints lack diversity")

    passed = len(issues) == 0
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    for issue in issues:
        print(f"     [!] {issue}")
    return {"passed": passed, "range": (range_min, range_max), "avg_std": avg_std}


def check_pcm(pcm_batch: np.ndarray, peak_threshold: float = 0.1) -> dict:
    """Analyze PCM peak positions across a batch."""
    batch_size, channels, h, w = pcm_batch.shape
    peak_positions = []
    peak_values = []
    for joint_idx in range(min(channels - 1, 18)):  # channels 0-17, skip background
        positions = []
        values = []
        for sample_idx in range(min(batch_size, 16)):  # check up to 16 samples
            channel = pcm_batch[sample_idx, joint_idx]
            y, x = np.unravel_index(int(np.argmax(channel)), channel.shape)
            val = float(channel[y, x])
            if val >= peak_threshold:
                positions.append((x, y))
                values.append(val)
        if positions:
            # Mode position across samples for this joint
            unique, counts = np.unique([f"{x},{y}" for x, y in positions], return_counts=True)
            mode_pos = tuple(map(int, unique[np.argmax(counts)].split(",")))
            avg_val = float(np.mean(values))
        else:
            mode_pos = (-1, -1)
            avg_val = 0.0
        peak_positions.append(mode_pos)
        peak_values.append(avg_val)

    unique_positions = set(p for p in peak_positions if p != (-1, -1))
    n_above_threshold = sum(1 for v in peak_values if v > peak_threshold)

    print("--- PCM Peak Positions ---")
    for j in range(18):
        name = OPENPOSE_18_NAMES[j] if j < len(OPENPOSE_18_NAMES) else f"joint_{j}"
        pos = peak_positions[j]
        val = peak_values[j]
        status = " " if val > peak_threshold else " BELOW_THRESHOLD"
        print(f"  {name:14s}  peak at ({pos[0]:2d}, {pos[1]:2d})  value={val:.4f}{status}")

    issues = []
    if n_above_threshold < 18:
        issues.append(f"only {n_above_threshold}/18 joints have peaks above threshold ({peak_threshold})")
    if len(unique_positions) < 10:
        issues.append(f"only {len(unique_positions)} unique peak positions — peaks are clustered (fix likely broken)")

    print(f"  unique peak positions: {len(unique_positions)}/18")
    passed = len(issues) == 0 and n_above_threshold == 18
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    for issue in issues:
        print(f"     [!] {issue}")
    return {"passed": passed, "unique_peaks": len(unique_positions), "above_threshold": n_above_threshold}


def main():
    parser = argparse.ArgumentParser(description="Verify H5 keypoint double-normalization fix")
    parser.add_argument("--config", default="configs/E01.yaml", help="Path to config YAML")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for sampling")
    parser.add_argument("--peak-threshold", type=float, default=0.1, help="Minimum PCM peak value")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg["dataset"].get("type", "npz") != "h5":
        print(f"WARNING: dataset.type is '{cfg['dataset'].get('type', 'npz')}', expected 'h5'")
        print("This script is designed to verify the H5 fix. Results may not be meaningful.")

    print(f"Loading dataset: {cfg['dataset']['root']}")
    ds = build_dataset(cfg, "train", max_samples=min(args.batch_size * 3, 96))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    batch = next(iter(loader))
    kpts18 = batch["kpts18"].numpy()
    pcm = batch["pcm"].numpy()

    print(f"\nBatch: {kpts18.shape[0]} samples, kpts18 shape={kpts18.shape}, pcm shape={pcm.shape}\n")

    kpts_ok = check_keypoints(kpts18)
    print()
    pcm_ok = check_pcm(pcm, peak_threshold=args.peak_threshold)

    print("\n=== Verdict ===")
    if kpts_ok["passed"] and pcm_ok["passed"]:
        print("PASS — fix is working correctly. Keypoints are diverse and PCM peaks are distinct.")
    else:
        print("FAIL — issues detected. The fix may not be sufficient or data has other problems.")
        sys.exit(1)


if __name__ == "__main__":
    main()
