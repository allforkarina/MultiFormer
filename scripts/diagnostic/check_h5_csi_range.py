"""Check whether H5 CSI amplitude has been double-normalized.

Prints the raw value range of csi_amplitude and compares it with the H5
attributes to determine whether the data is:
  - RAW amplitude (range matches attrs like ~[0, 57]) → OK, proceed
  - PRE-NORMALIZED to [0,1] (range ~[0,1] but attrs say ~57) → BUG: double-normalization

Usage:
    python scripts/check_h5_csi_range.py --h5 /data/WiFiPose/dataset/mmfi_pose.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Check H5 CSI amplitude data range")
    parser.add_argument("--h5", required=True, help="Path to the H5 file")
    parser.add_argument("--n-frames", type=int, default=100,
                        help="Number of frames to sample (default: 100)")
    args = parser.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        print(f"[ERROR] H5 file not found: {h5_path}")
        return 1

    with h5py.File(h5_path, "r") as f:
        # Read sample frames
        n_total = f["csi_amplitude"].shape[0]
        n_sample = min(args.n_frames, n_total)
        raw = np.asarray(f["csi_amplitude"][:n_sample], dtype=np.float32)

        print("=" * 55)
        print(f"  H5 File: {h5_path}")
        print(f"  Total frames: {n_total}")
        print(f"  Sampled: {n_sample}")
        print("=" * 55)
        print()
        print("--- CSI Amplitude Raw Data ---")
        print(f"  shape:  {raw.shape}")
        print(f"  dtype:  {raw.dtype}")
        print(f"  min:    {raw.min():.6f}")
        print(f"  max:    {raw.max():.6f}")
        print(f"  mean:   {raw.mean():.6f}")
        print(f"  std:    {raw.std():.6f}")
        print()

        print("--- H5 Attributes ---")
        for k, v in sorted(f.attrs.items()):
            print(f"  {k}: {v}")
        print()

        # Diagnostic
        amp_min = float(f.attrs.get("amplitude_train_min", 0.0))
        amp_max = float(f.attrs.get("amplitude_train_max", 1.0))

        print("=" * 55)
        print("  DIAGNOSIS")
        print("=" * 55)
        print(f"  Raw data max:     {raw.max():.4f}")
        print(f"  Attr train_max:   {amp_max:.4f}")
        print()

        if raw.max() <= 2.0 and amp_max > 10.0:
            print("  >>> BUG: CSI appears PRE-NORMALIZED to [0,1]!")
            print("  >>> The H5 csi_amplitude was already normalized before storage,")
            print("  >>> but the loader applies global_minmax again -> double-normalization.")
            print("  >>> Fix: rebuild H5 without pre-normalizing csi_amplitude,")
            print("  >>>       or remove the loader-side normalize_global_minmax call.")
        elif abs(raw.max() - amp_max) / max(amp_max, 1.0) < 0.3:
            print("  >>> OK: Raw data range matches attributes.")
            print("  >>> CSI amplitude is stored RAW, loader normalization is correct.")
            print("  >>> Proceed to Experiment B (normalization magnitude alignment).")
        else:
            print("  >>> UNCLEAR: range mismatch but not obviously double-normalized.")
            print(f"  >>> Raw max={raw.max():.4f}, attr max={amp_max:.4f}")
            print("  >>> Investigate manually.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
