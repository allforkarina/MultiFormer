"""
Build memory-mapped .npy files from MM-Fi dataset for fast training I/O.

Pre-computes 3 normalization variants and stores each as a single .npy file.
Training loader uses np.load(path, mmap_mode='r') for zero-copy OS-cached access.

Input:
    /data/WiFiPose/dataset/dataset/{ACTION}/{SUBJECT}/
        wifi-csi/frame*.mat   ← CSIamp (3, 114, 10) float64
        rgb/frame*.npy        ← COCO17 keypoints (17, 2) float32

Output:
    /data/WiFiPose/dataset/mmfi_pose_v3/
        csi_gminmax.npy  ← global_minmax normalized (N, 64, 3, 114) float32
        csi_gzscore.npy  ← global_zscore normalized (N, 64, 3, 114) float32
        csi_zscore.npy   ← per-sample zscore normalized (N, 64, 3, 114) float32
        kpts18.npy       ← OpenPose18, pose_range (N, 18, 2) float32
        meta.npz         ← environment, sample, action, frame_idx
        stats.json       ← normalization statistics

Usage:
    python scripts/build_memmap.py \
        --src /data/WiFiPose/dataset/dataset \
        --dst /data/WiFiPose/dataset/mmfi_pose_v3 \
        --train-subjects S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample


TIME_PACKETS = 64
RX_ANTENNAS = 3
SUBCARRIERS = 114

# COCO17 → OpenPose18 mapping (与 data/heatmap_gt.py 一致)
COCO17_TO_OPENPOSE18 = {
    0: 0,   # nose
    2: 6,   # r_shoulder
    3: 8,   # r_elbow
    4: 10,  # r_wrist
    5: 5,   # l_shoulder
    6: 7,   # l_elbow
    7: 9,   # l_wrist
    8: 12,  # r_hip
    9: 14,  # r_knee
    10: 16, # r_ankle
    11: 11, # l_hip
    12: 13, # l_knee
    13: 15, # l_ankle
    14: 2,  # r_eye
    15: 1,  # l_eye
    16: 4,  # r_ear
    17: 3,  # l_ear
}


def derive_env(subject: str) -> str:
    num = int(subject.lstrip("S"))
    return f"env{(num - 1) // 10 + 1}"


# ============================================================
# CSI preprocessing
# ============================================================

def sanitize_csi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.all():
        return x
    fill = float(np.median(x[finite])) if finite.any() else 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def preprocess_csi_one_frame(csi_amp: np.ndarray) -> np.ndarray:
    """sanitize → resample(10→64) → transpose → (64, 3, 114), NOT normalized."""
    csi_amp = sanitize_csi(np.asarray(csi_amp, dtype=np.float32))
    csi_amp = sanitize_csi(resample(csi_amp, TIME_PACKETS, axis=-1))
    return np.transpose(csi_amp, (2, 0, 1)).astype(np.float32, copy=False)


# ============================================================
# Keypoint conversion
# ============================================================

def _valid_point(point: np.ndarray) -> bool:
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    kpts17 = np.asarray(kpts17, dtype=np.float32)
    kpts18 = np.zeros((18, 2), dtype=np.float32)
    valid = np.zeros(18, dtype=bool)
    for op_idx, coco_idx in COCO17_TO_OPENPOSE18.items():
        p = kpts17[coco_idx]
        if _valid_point(p):
            kpts18[op_idx] = p
            valid[op_idx] = True
    l_sh, r_sh = kpts17[5], kpts17[6]
    if _valid_point(l_sh) and _valid_point(r_sh):
        kpts18[1] = (l_sh + r_sh) * 0.5
        valid[1] = True
    elif _valid_point(l_sh):
        kpts18[1] = l_sh; valid[1] = True
    elif _valid_point(r_sh):
        kpts18[1] = r_sh; valid[1] = True
    kpts18[~valid] = 0.0
    return kpts18


def normalize_kpts_to_pose_range(
    kpts: np.ndarray, pose_min: float = -0.8, pose_max: float = 0.8,
) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float32).copy()
    non_zero = kpts[kpts != 0]
    abs_max = float(np.abs(non_zero).max()) if len(non_zero) > 0 else 0.0
    if abs_max > 10.0:
        IMG_W, IMG_H = 1920.0, 1080.0
        kpts[..., 0] /= IMG_W
        kpts[..., 1] /= IMG_H
        span = pose_max - pose_min
        kpts = kpts * span + pose_min
    invalid = ~np.isfinite(kpts).all(axis=-1) | np.all(np.isclose(kpts, 0.0), axis=-1)
    kpts[invalid] = 0.0
    return kpts.astype(np.float32)


# ============================================================
# Trial enumeration
# ============================================================

def iter_trials(src_root: Path) -> list[Path]:
    trials: list[Path] = []
    for action_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("A")):
        for subj_dir in sorted(p for p in action_dir.iterdir() if p.is_dir() and p.name.startswith("S")):
            if (subj_dir / "wifi-csi").is_dir() and (subj_dir / "rgb").is_dir():
                trials.append(subj_dir)
    return trials


# ============================================================
# Single trial processing
# ============================================================

def process_trial(trial_dir: Path, pose_min: float, pose_max: float) -> dict | None:
    action = trial_dir.parent.name
    subject = trial_dir.name
    wifi_dir = trial_dir / "wifi-csi"
    rgb_dir = trial_dir / "rgb"

    mat_paths = sorted(wifi_dir.glob("frame*.mat"))
    npy_paths = sorted(rgb_dir.glob("frame*.npy"))
    if not mat_paths:
        return None

    mat_stems = {p.stem: p for p in mat_paths}
    npy_stems = {p.stem: p for p in npy_paths}
    common = sorted(set(mat_stems) & set(npy_stems))
    if not common:
        return None

    n_frames = len(common)
    csi_frames = np.empty((n_frames, TIME_PACKETS, RX_ANTENNAS, SUBCARRIERS), dtype=np.float32)
    kpts18 = np.zeros((n_frames, 18, 2), dtype=np.float32)
    frame_idx = np.zeros(n_frames, dtype=np.int64)

    for i, stem in enumerate(common):
        mat = sio.loadmat(str(mat_stems[stem]))
        csi_frames[i] = preprocess_csi_one_frame(np.asarray(mat["CSIamp"], dtype=np.float32))
        kpts_coco17 = np.load(str(npy_stems[stem]))
        kpts18[i] = normalize_kpts_to_pose_range(
            coco17_to_openpose18(kpts_coco17), pose_min, pose_max
        )
        frame_idx[i] = int(stem.replace("frame", ""))

    return {
        "csi": csi_frames,
        "kpts18": kpts18,
        "environment": derive_env(subject),
        "sample": subject,
        "action": action,
        "frame_idx": frame_idx,
    }


# ============================================================
# Worker
# ============================================================

def _worker(args):
    trial_dir, pose_min, pose_max = args
    try:
        result = process_trial(Path(trial_dir), pose_min, pose_max)
        label = f"{Path(trial_dir).parent.name}/{Path(trial_dir).name}"
        return label, result, None
    except Exception:
        label = f"{Path(trial_dir).parent.name}/{Path(trial_dir).name}"
        return label, None, traceback.format_exc()


# ============================================================
# Main
# ============================================================

def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def main():
    parser = argparse.ArgumentParser(description="Build memmap .npy files from MM-Fi dataset")
    parser.add_argument("--src", default="/data/WiFiPose/dataset/dataset")
    parser.add_argument("--dst", default="/data/WiFiPose/dataset/mmfi_pose_v3")
    parser.add_argument("--train-subjects", nargs="+",
                        default=["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10"])
    parser.add_argument("--pose-min", type=float, default=-0.8)
    parser.add_argument("--pose-max", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)
    train_set = set(args.train_subjects)

    # ---- Phase 1: Process trials ----
    trials = iter_trials(src_root)
    print(f"Found {len(trials)} trials")

    all_data: list[dict] = []
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    if args.workers <= 1:
        for i, trial in enumerate(trials):
            label = f"{trial.parent.name}/{trial.name}"
            result = process_trial(trial, args.pose_min, args.pose_max)
            if result:
                all_data.append(result)
                print(f"  [{i+1}/{len(trials)}] {label} ok N={result['csi'].shape[0]}")
            else:
                print(f"  [{i+1}/{len(trials)}] {label} SKIP")
    else:
        tasks = [(str(t), args.pose_min, args.pose_max) for t in trials]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                label, result, err = fut.result()
                if err:
                    failures.append((label, err))
                    print(f"  [{done}/{len(trials)}] {label} FAIL")
                elif result:
                    all_data.append(result)
                    print(f"  [{done}/{len(trials)}] {label} ok N={result['csi'].shape[0]}")
                else:
                    print(f"  [{done}/{len(trials)}] {label} SKIP")

    dt = time.time() - t0
    print(f"\nPhase 1: {len(all_data)} ok, {len(failures)} fail ({dt:.1f}s)")
    if not all_data:
        print("ERROR: No trials processed")
        sys.exit(1)

    # ---- Phase 2: Concatenate ----
    print("Concatenating...")
    all_csi_raw  = np.concatenate([d["csi"] for d in all_data], axis=0).astype(np.float32)
    all_kpts18   = np.concatenate([d["kpts18"] for d in all_data], axis=0).astype(np.float32)
    all_envs     = np.array([e for d in all_data for e in [d["environment"]] * d["csi"].shape[0]])
    all_subjects = np.array([s for d in all_data for s in [d["sample"]]      * d["csi"].shape[0]])
    all_actions  = np.array([a for d in all_data for a in [d["action"]]      * d["csi"].shape[0]])
    all_fidx     = np.concatenate([d["frame_idx"] for d in all_data])

    n_total = all_csi_raw.shape[0]
    n_train = int(np.isin(all_subjects.astype(str), list(train_set)).sum())
    print(f"Total: {n_total}, train: {n_train} ({n_train/n_total*100:.1f}%)")

    # ---- Phase 3: Compute stats + normalize ----
    print("Computing normalization statistics (train set only)...")
    train_mask = np.isin(all_subjects.astype(str), list(train_set))
    train_csi = all_csi_raw[train_mask]
    amp_min  = float(train_csi.min())
    amp_max  = float(train_csi.max())
    amp_mean = float(train_csi.mean())
    amp_std  = float(train_csi.std())
    print(f"  min={amp_min:.4f}  max={amp_max:.4f}  mean={amp_mean:.4f}  std={amp_std:.4f}")

    # variant 1: global_minmax
    print("  global_minmax...")
    csi_gminmax = safe_div(all_csi_raw - amp_min, amp_max - amp_min).astype(np.float32)

    # variant 2: global_zscore
    print("  global_zscore...")
    csi_gzscore = safe_div(all_csi_raw - amp_mean, amp_std).astype(np.float32)

    # variant 3: per-sample zscore
    print("  zscore...")
    csi_zscore = np.empty_like(all_csi_raw, dtype=np.float32)
    for i in range(n_total):
        x = all_csi_raw[i]
        m, s = float(x.mean()), float(x.std())
        csi_zscore[i] = safe_div(x - m, s).astype(np.float32)

    # free raw data
    del all_csi_raw

    # ---- Phase 4: Save ----
    print("Saving .npy files...")
    t_save = time.time()

    np.save(str(dst_root / "csi_gminmax.npy"), csi_gminmax)
    np.save(str(dst_root / "csi_gzscore.npy"), csi_gzscore)
    np.save(str(dst_root / "csi_zscore.npy"),  csi_zscore)
    np.save(str(dst_root / "kpts18.npy"),       all_kpts18)
    np.savez(str(dst_root / "meta.npz"),
             environment=all_envs, sample=all_subjects,
             action=all_actions, frame_idx=all_fidx)

    stats = {
        "amplitude_train_min":  amp_min,
        "amplitude_train_max":  amp_max,
        "amplitude_train_mean": amp_mean,
        "amplitude_train_std":  amp_std,
        "pose_min": args.pose_min,
        "pose_max": args.pose_max,
        "time_packets": TIME_PACKETS,
        "rx_antennas": RX_ANTENNAS,
        "subcarriers": SUBCARRIERS,
        "total_frames": n_total,
        "train_frames": n_train,
    }
    with open(dst_root / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    total_mb = sum(
        f.stat().st_size for f in dst_root.glob("*.npy") if f.is_file()
    ) / (1024 * 1024)
    print(f"Done in {time.time()-t_save:.0f}s — {total_mb:.0f} MB total .npy files")


if __name__ == "__main__":
    main()
