from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

from .csi_preprocess import normalize_csi, normalize_global_minmax, sanitize_csi
from .heatmap_gt import build_pcm_paf, coco17_to_openpose18


def _decode_bytes(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _resample_time_h5(csi_amp: np.ndarray, target_packets: int = 64) -> np.ndarray:
    """Resample CSI amplitude from raw (NR, NS, 10) to (NR, NS, target_packets)."""
    csi_amp = sanitize_csi(np.asarray(csi_amp, dtype=np.float32))
    if csi_amp.shape[-1] == target_packets:
        return csi_amp
    return sanitize_csi(resample(csi_amp, target_packets, axis=-1).astype(np.float32))


class H5MMFiDataset(Dataset):
    """HDF5-backed dataset reading from pre-packed MM-Fi HDF5 file.

    Reads raw CSI amplitude and COCO17 keypoints, applies time resampling
    10→64, global min-max normalization, and converts to OpenPose18 format
    matching the MultiFormer model input specification.
    """

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        envs: Iterable[str] | None = None,
        train_subjects: Iterable[str] | None = None,
        test_subjects: Iterable[str] | None = None,
        random_val_ratio: float = 0.2,
        seed: int = 42,
        time_packets: int = 64,
        subcarrier_mode: str = "keep",
        normalize: str = "global_minmax",
        heatmap_size: int = 36,
        heatmap_sigma: float = 1.5,
        paf_width: float = 1.0,
        pose_range: tuple[float, float] = (-0.8, 0.8),
        build_targets: bool = True,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be one of train/val/test/all, got {split}")
        self.h5_path = Path(h5_path)
        self.split = split
        self.time_packets = time_packets
        self.subcarrier_mode = subcarrier_mode
        self.normalize = normalize
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.paf_width = paf_width
        self.pose_range = pose_range
        self.build_targets = build_targets
        self._h5_file: h5py.File | None = None

        with h5py.File(self.h5_path, "r") as f:
            self.amp_train_min = float(f.attrs["amplitude_train_min"])
            self.amp_train_max = float(f.attrs["amplitude_train_max"])
            self.kp_x_scale = float(f.attrs.get("keypoint_x_scale", 1.0))
            self.kp_y_scale = float(f.attrs.get("keypoint_y_scale", 1.0))

            # Detect whether CSI was already normalized during H5 building.
            # If so, the loader MUST NOT re-apply global_minmax (double-norm bug).
            amp_norm = str(f.attrs.get("amplitude_normalization", ""))
            self._amp_pre_normalized = amp_norm in ("train_global_minmax", "global_minmax")

            self.indices = self._build_split(
                f, split, envs, train_subjects, test_subjects, random_val_ratio, seed
            )

    def _build_split(
        self,
        h5_file: h5py.File,
        split: str,
        envs: Iterable[str] | None,
        train_subjects: Iterable[str] | None,
        test_subjects: Iterable[str] | None,
        random_val_ratio: float,
        seed: int,
    ) -> np.ndarray:
        num_total = h5_file["action"].shape[0]
        environments = [_decode_bytes(e) for e in h5_file["environment"][:]]
        samples = [_decode_bytes(s) for s in h5_file["sample"][:]]

        env_set = set(envs) if envs else None
        subject_set = set(train_subjects) if train_subjects else None

        candidate_indices: list[int] = []
        for i in range(num_total):
            if env_set is not None and environments[i] not in env_set:
                continue
            if subject_set is not None and samples[i] not in subject_set:
                continue
            candidate_indices.append(i)

        # same_subject_random split within each sample group
        if split != "all":
            rng = random.Random(seed)
            grouped: dict[str, list[int]] = {}
            for idx in candidate_indices:
                grouped.setdefault(samples[idx], []).append(idx)

            train_indices: list[int] = []
            val_indices: list[int] = []
            for subject, indices in sorted(grouped.items()):
                shuffled = indices[:]
                rng.shuffle(shuffled)
                pivot = int(round(len(shuffled) * (1.0 - random_val_ratio)))
                train_indices.extend(shuffled[:pivot])
                val_indices.extend(shuffled[pivot:])

            if split == "train":
                return np.asarray(sorted(train_indices), dtype=np.int64)
            else:
                return np.asarray(sorted(val_indices), dtype=np.int64)

        return np.asarray(sorted(candidate_indices), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def _get_h5(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def _normalize_keypoints(self, kpts: np.ndarray) -> np.ndarray:
        """Map keypoints from [0, 1] (pre-normalized during H5 creation) to pose_range."""
        kpts = kpts.copy()
        lo, hi = self.pose_range
        kpts = kpts * (hi - lo) + lo
        return kpts.astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        h5_file = self._get_h5()
        frame_idx = int(self.indices[index])

        csi_amp = np.asarray(h5_file["csi_amplitude"][frame_idx], dtype=np.float32)
        keypoints_coco = np.asarray(h5_file["keypoints"][frame_idx], dtype=np.float32)

        # Time resample: (3, 114, 10) -> (3, 114, 64)
        csi_amp = _resample_time_h5(csi_amp, target_packets=self.time_packets)

        # Transpose to (M, NR, NS) = (64, 3, 114)
        csi_amp = np.transpose(csi_amp, (2, 0, 1)).astype(np.float32)

        # Normalize / magnitude-align CSI input.
        # H5 data is pre-normalized to [0,1] during building (amplitude_normalization
        # attr).  Re-applying global_minmax here would double-normalize — fixed.
        if self.normalize == "global_minmax":
            if not self._amp_pre_normalized:
                csi_amp = normalize_global_minmax(
                    csi_amp,
                    train_min=self.amp_train_min,
                    train_max=self.amp_train_max,
                )
            # else: already [0,1] — double-norm bug avoided
        elif self.normalize == "global_zscore":
            # Global center + scale: preserves inter-sample amplitude relationships
            # while matching the ~N(0,1) input magnitude the model expects.
            csi_amp = (csi_amp - 0.5) * 6.0  # [0,1] → [-3, +3]
        elif self.normalize == "zscore":
            csi_amp = normalize_csi(csi_amp, mode="zscore")
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize}")

        # COCO17 -> OpenPose18 keypoint conversion
        keypoints_norm = self._normalize_keypoints(keypoints_coco)
        kpts18 = coco17_to_openpose18(keypoints_norm)

        item: dict = {
            "csi": torch.from_numpy(np.ascontiguousarray(csi_amp)),
            "kpts18": torch.from_numpy(np.ascontiguousarray(kpts18)),
            "meta": {
                "env": _decode_bytes(h5_file["environment"][frame_idx]),
                "subject": _decode_bytes(h5_file["sample"][frame_idx]),
                "action": _decode_bytes(h5_file["action"][frame_idx]),
                "frame_idx": int(frame_idx),
            },
        }
        if self.build_targets:
            pcm, paf = build_pcm_paf(
                kpts18,
                size=self.heatmap_size,
                sigma=self.heatmap_sigma,
                paf_width=self.paf_width,
                pose_range=self.pose_range,
            )
            item["pcm"] = torch.from_numpy(pcm)
            item["paf"] = torch.from_numpy(paf)
        return item

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self) -> None:
        self.close()
