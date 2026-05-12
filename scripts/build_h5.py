"""
Build preprocessed H5 from MM-Fi dataset on Linux server.

Input (server data layout):
    /data/WiFiPose/dataset/dataset/
    └── {ACTION}/{SUBJECT}/
        ├── wifi-csi/frame*.mat   ← CSIamp (3, 114, 10) float64
        └── rgb/frame*.npy        ← COCO17 keypoints (17, 2) float32

Output H5:
    csi         (N, 64, 3, 114) float32  — time-resampled, NOT normalized
    kpts18      (N, 18, 2)     float32  — OpenPose18, normalized to pose_range
    environment (N,) str                 — env1-env4, derived from subject ID
    sample      (N,) str                 — subject ID
    action      (N,) str                 — action ID
    frame_idx   (N,) int64              — frame number

Environment derivation:
    S01-S10 → env1, S11-S20 → env2, S21-S30 → env3, S31-S40 → env4
    Derived from subject number: env_idx = (num - 1) // 10 + 1

Usage:
    python scripts/build_h5.py \
        --src /data/WiFiPose/dataset/dataset \
        --dst /data/WiFiPose/dataset/mmfi_pose_v2.h5 \
        --train-subjects S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 \
        --pose-min -0.8 --pose-max 0.8 \
        --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
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


# ============================================================
# Environment derivation
# ============================================================

def derive_env(subject: str) -> str:
    """Derive environment label from subject ID.

    S01-S10 → env1, S11-S20 → env2, S21-S30 → env3, S31-S40 → env4.
    """
    num = int(subject.lstrip("S"))
    env_idx = (num - 1) // 10 + 1
    return f"env{env_idx}"


# ============================================================
# CSI preprocessing
# ============================================================

def sanitize_csi(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with median of finite values."""
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.all():
        return x
    fill = float(np.median(x[finite])) if finite.any() else 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def preprocess_csi_one_frame(csi_amp: np.ndarray) -> np.ndarray:
    """Preprocess one CSI frame: sanitize → resample → transpose.

    Input:  CSIamp (3, 114, 10) from .mat file
    Output: (64, 3, 114) float32, NOT normalized
    """
    csi_amp = sanitize_csi(np.asarray(csi_amp, dtype=np.float32))
    csi_amp = sanitize_csi(resample(csi_amp, TIME_PACKETS, axis=-1))
    csi_amp = np.transpose(csi_amp, (2, 0, 1)).astype(np.float32, copy=False)
    return csi_amp


# ============================================================
# Keypoint conversion
# ============================================================

def _valid_point(point: np.ndarray) -> bool:
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    """Convert (17, 2) COCO17 → (18, 2) OpenPose18. Neck = midpoint of shoulders."""
    kpts17 = np.asarray(kpts17, dtype=np.float32)
    kpts18 = np.zeros((18, 2), dtype=np.float32)
    valid = np.zeros(18, dtype=bool)

    for op_idx, coco_idx in COCO17_TO_OPENPOSE18.items():
        p = kpts17[coco_idx]
        if _valid_point(p):
            kpts18[op_idx] = p
            valid[op_idx] = True

    l_sh = kpts17[5]
    r_sh = kpts17[6]
    if _valid_point(l_sh) and _valid_point(r_sh):
        kpts18[1] = (l_sh + r_sh) * 0.5
        valid[1] = True
    elif _valid_point(l_sh):
        kpts18[1] = l_sh
        valid[1] = True
    elif _valid_point(r_sh):
        kpts18[1] = r_sh
        valid[1] = True

    kpts18[~valid] = 0.0
    return kpts18


def normalize_kpts_to_pose_range(
    kpts: np.ndarray,
    pose_min: float = -0.8,
    pose_max: float = 0.8,
) -> np.ndarray:
    """Normalize keypoints to pose_range.

    Handles both pixel coordinates (>10 range) and already-normalized coords.
    """
    kpts = np.asarray(kpts, dtype=np.float32).copy()

    non_zero = kpts[kpts != 0]
    abs_max = float(np.abs(non_zero).max()) if len(non_zero) > 0 else 0.0

    if abs_max > 10.0:
        # Pixel coordinates → normalize by image resolution
        IMG_W, IMG_H = 1920.0, 1080.0
        kpts[..., 0] = kpts[..., 0] / IMG_W
        kpts[..., 1] = kpts[..., 1] / IMG_H
        span = pose_max - pose_min
        kpts = kpts * span + pose_min
    elif abs_max <= 2.0:
        # Already in normalized range, ensure it maps to [pose_min, pose_max]
        span = pose_max - pose_min
        kpts = kpts * span + pose_min
    # else: assume already in [pose_min, pose_max]

    invalid = ~np.isfinite(kpts).all(axis=-1) | np.all(np.isclose(kpts, 0.0), axis=-1)
    kpts[invalid] = 0.0
    return kpts.astype(np.float32)


# ============================================================
# Trial enumeration
# ============================================================

def iter_trials(src_root: Path) -> list[Path]:
    """Enumerate all {ACTION}/{SUBJECT} trial directories under src_root."""
    trials: list[Path] = []
    for action_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("A")):
        for subj_dir in sorted(p for p in action_dir.iterdir() if p.is_dir() and p.name.startswith("S")):
            wifi_dir = subj_dir / "wifi-csi"
            rgb_dir = subj_dir / "rgb"
            if wifi_dir.is_dir() and rgb_dir.is_dir():
                trials.append(subj_dir)
    return trials


# ============================================================
# Single trial processing
# ============================================================

def process_trial(
    trial_dir: Path,
    pose_min: float,
    pose_max: float,
) -> dict | None:
    """Process one trial ({ACTION}/{SUBJECT}) and return dict with arrays.

    Returns dict with:
        csi:         (N, 64, 3, 114) float32
        kpts18:      (N, 18, 2) float32
        environment: str
        sample:      str
        action:      str
        frame_idx:   (N,) int64
    """
    action = trial_dir.parent.name
    subject = trial_dir.name
    wifi_dir = trial_dir / "wifi-csi"
    rgb_dir = trial_dir / "rgb"

    mat_paths = sorted(wifi_dir.glob("frame*.mat"))
    npy_paths = sorted(rgb_dir.glob("frame*.npy"))

    if not mat_paths:
        print(f"  WARNING: {action}/{subject} has no .mat files, skipping")
        return None

    # Align frames by name (e.g. frame037.mat ↔ frame037.npy)
    mat_stems = {p.stem: p for p in mat_paths}
    npy_stems = {p.stem: p for p in npy_paths}
    common = sorted(set(mat_stems) & set(npy_stems))
    if not common:
        print(f"  WARNING: {action}/{subject} has no matching frames, skipping")
        return None

    n_frames = len(common)
    csi_frames = np.empty((n_frames, TIME_PACKETS, RX_ANTENNAS, SUBCARRIERS), dtype=np.float32)
    kpts18 = np.zeros((n_frames, 18, 2), dtype=np.float32)
    frame_idx = np.zeros(n_frames, dtype=np.int64)

    for i, stem in enumerate(common):
        # CSI
        mat = sio.loadmat(str(mat_stems[stem]))
        csi_raw = np.asarray(mat["CSIamp"], dtype=np.float32)
        csi_frames[i] = preprocess_csi_one_frame(csi_raw)

        # Keypoints
        kpts_coco17 = np.load(str(npy_stems[stem]))
        kpts_op18 = coco17_to_openpose18(kpts_coco17)
        kpts18[i] = normalize_kpts_to_pose_range(kpts_op18, pose_min, pose_max)

        # Frame index
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
# Worker for parallel execution
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
# Main pipeline
# ============================================================

def build_h5(
    src_root: Path,
    dst_path: Path,
    train_subjects: set[str],
    pose_min: float,
    pose_max: float,
    workers: int,
    chunk_size: int,
) -> None:
    """Main pipeline: enumerate → process → compute stats → write H5."""
    trials = iter_trials(src_root)
    print(f"Found {len(trials)} trials")

    all_data: list[dict] = []
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    if workers <= 1:
        for i, trial in enumerate(trials):
            label = f"{trial.parent.name}/{trial.name}"
            result = process_trial(trial, pose_min, pose_max)
            if result:
                all_data.append(result)
                print(f"  [{i+1}/{len(trials)}] {label} ok N={result['csi'].shape[0]}")
            else:
                failures.append((label, "no matching frames"))
                print(f"  [{i+1}/{len(trials)}] {label} SKIP")
    else:
        tasks = [(str(t), pose_min, pose_max) for t in trials]
        with ProcessPoolExecutor(max_workers=workers) as pool:
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
    print(f"\nProcessing: {len(all_data)} ok, {len(failures)} fail/skip ({dt:.1f}s)")

    if not all_data:
        print("ERROR: No trials processed successfully")
        sys.exit(1)

    # Concatenate all trials
    print("Concatenating arrays...")
    all_csi = np.concatenate([d["csi"] for d in all_data], axis=0)
    all_kpts18 = np.concatenate([d["kpts18"] for d in all_data], axis=0)
    all_envs = np.array(
        [e for d in all_data for e in [d["environment"]] * d["csi"].shape[0]],
        dtype=object,
    )
    all_subjects = np.array(
        [s for d in all_data for s in [d["sample"]] * d["csi"].shape[0]],
        dtype=object,
    )
    all_actions = np.array(
        [a for d in all_data for a in [d["action"]] * d["csi"].shape[0]],
        dtype=object,
    )
    all_frame_idx = np.concatenate([d["frame_idx"] for d in all_data])

    # Train-set statistics for normalization
    train_mask = np.isin(all_subjects.astype(str), list(train_subjects))
    train_csi = all_csi[train_mask]
    amp_train_min = float(train_csi.min())
    amp_train_max = float(train_csi.max())
    amp_train_mean = float(train_csi.mean())
    amp_train_std = float(train_csi.std())

    n_total = len(all_csi)
    n_train = int(train_mask.sum())
    print(f"Total frames: {n_total}")
    print(f"Train frames: {n_train} ({n_train/n_total*100:.1f}%)")
    print(f"Train CSI: min={amp_train_min:.4f} max={amp_train_max:.4f} "
          f"mean={amp_train_mean:.4f} std={amp_train_std:.4f}")

    # Write H5
    print(f"Writing H5 to {dst_path}...")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype()
    cs = min(chunk_size, all_csi.shape[0])

    with h5py.File(dst_path, "w") as f:
        f.create_dataset(
            "csi", data=all_csi,
            chunks=(cs, TIME_PACKETS, RX_ANTENNAS, SUBCARRIERS),
        )
        f.create_dataset("kpts18", data=all_kpts18, chunks=(cs, 18, 2))
        f.create_dataset("environment", data=all_envs.astype(str_dt))
        f.create_dataset("sample", data=all_subjects.astype(str_dt))
        f.create_dataset("action", data=all_actions.astype(str_dt))
        f.create_dataset("frame_idx", data=all_frame_idx)

        f.attrs["amplitude_train_min"] = amp_train_min
        f.attrs["amplitude_train_max"] = amp_train_max
        f.attrs["amplitude_train_mean"] = amp_train_mean
        f.attrs["amplitude_train_std"] = amp_train_std
        f.attrs["pose_min"] = pose_min
        f.attrs["pose_max"] = pose_max
        f.attrs["time_packets"] = TIME_PACKETS
        f.attrs["rx_antennas"] = RX_ANTENNAS
        f.attrs["subcarriers"] = SUBCARRIERS

    file_size_mb = dst_path.stat().st_size / (1024 * 1024)
    print(f"Done! {file_size_mb:.0f} MB")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build preprocessed H5 from MM-Fi dataset")
    parser.add_argument("--src", default="/data/WiFiPose/dataset/dataset",
                        help="MM-Fi dataset root (containing {ACTION}/{SUBJECT}/)")
    parser.add_argument("--dst", default="/data/WiFiPose/dataset/mmfi_pose_v2.h5",
                        help="Output H5 file path")
    parser.add_argument("--train-subjects", nargs="+",
                        default=["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"],
                        help="Subject IDs for training set statistics")
    parser.add_argument("--pose-min", type=float, default=-0.8)
    parser.add_argument("--pose-max", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    build_h5(
        src_root=Path(args.src),
        dst_path=Path(args.dst),
        train_subjects=set(args.train_subjects),
        pose_min=args.pose_min,
        pose_max=args.pose_max,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
