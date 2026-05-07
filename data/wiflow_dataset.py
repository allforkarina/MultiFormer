from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .csi_preprocess import preprocess_csi_amp
from .heatmap_gt import build_pcm_paf, coco17_to_openpose18


def _decode_string(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_env_name(name: str) -> str:
    match = re.fullmatch(r"(?:E|env)0*([1-9]\d*)", str(name), flags=re.IGNORECASE)
    if match:
        return f"env{int(match.group(1))}"
    return str(name)


def _frame_idx_from_frame_id(frame_id: str, fallback: int) -> int:
    match = re.search(r"(\d+)$", frame_id)
    if match:
        return int(match.group(1))
    return fallback


class WiFlowDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        split_scheme: str = "action_env",
        time_packets: int = 64,
        subcarrier_mode: str = "keep",
        normalize: str = "zscore",
        heatmap_size: int = 36,
        heatmap_sigma: float = 1.5,
        paf_width: float = 1.0,
        pose_range: tuple[float, float] = (-0.8, 0.8),
        build_targets: bool = True,
        max_samples: int | None = None,
        envs: Iterable[str] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.split = split
        self.split_scheme = split_scheme
        self.time_packets = time_packets
        self.subcarrier_mode = subcarrier_mode
        self.normalize = normalize
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.paf_width = paf_width
        self.pose_range = pose_range
        self.build_targets = build_targets
        self._h5_file: h5py.File | None = None

        with h5py.File(self.h5_path, "r") as h5_file:
            if split == "all":
                indices = np.arange(len(h5_file["action"]), dtype=np.int64)
            else:
                dataset_name = f"{split_scheme}_{split}_indices"
                if dataset_name not in h5_file:
                    dataset_name = f"{split}_indices"
                if dataset_name not in h5_file:
                    raise KeyError(
                        f"Split {split!r} not found in {self.h5_path} for split_scheme={split_scheme!r}"
                    )
                indices = np.asarray(h5_file[dataset_name], dtype=np.int64)

            self.keypoint_x_scale = float(
                h5_file.attrs.get(f"{split_scheme}_keypoint_x_scale", h5_file.attrs.get("keypoint_x_scale", 1.0))
            )
            self.keypoint_y_scale = float(
                h5_file.attrs.get(f"{split_scheme}_keypoint_y_scale", h5_file.attrs.get("keypoint_y_scale", 1.0))
            )

            if envs:
                normalized_filter = {_normalize_env_name(env) for env in envs}
                indices = np.asarray(
                    [
                        frame_index
                        for frame_index in indices
                        if _normalize_env_name(_decode_string(h5_file["environment"][int(frame_index)]))
                        in normalized_filter
                    ],
                    dtype=np.int64,
                )

            if max_samples is not None:
                indices = indices[:max_samples]

            self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def _get_h5_file(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getitem__(self, index: int) -> dict:
        h5_file = self._get_h5_file()
        dataset_index = int(self.indices[index])

        csi_amplitude = np.asarray(h5_file["csi_amplitude"][dataset_index], dtype=np.float32)
        keypoints17 = np.asarray(h5_file["keypoints"][dataset_index], dtype=np.float32)

        keypoints17[:, 0] /= self.keypoint_x_scale
        keypoints17[:, 1] /= self.keypoint_y_scale
        lo, hi = self.pose_range
        keypoints17 = keypoints17 * (hi - lo) + lo

        csi = preprocess_csi_amp(
            csi_amplitude,
            target_packets=self.time_packets,
            subcarrier_mode=self.subcarrier_mode,
            normalize=self.normalize,
        )
        keypoints18 = coco17_to_openpose18(keypoints17)

        action = _decode_string(h5_file["action"][dataset_index])
        subject = _decode_string(h5_file["sample"][dataset_index])
        environment = _decode_string(h5_file["environment"][dataset_index])
        frame_id = _decode_string(h5_file["frame_id"][dataset_index])
        frame_idx = _frame_idx_from_frame_id(frame_id, fallback=index + 1)

        item = {
            "csi": torch.from_numpy(csi),
            "kpts18": torch.from_numpy(keypoints18),
            "meta": {
                "env": environment,
                "subject": subject,
                "action": action,
                "frame_id": frame_id,
                "frame_idx": frame_idx,
                "csi_path": f"{self.h5_path}::csi_amplitude[{dataset_index}]",
            },
        }
        if self.build_targets:
            pcm, paf = build_pcm_paf(
                keypoints18,
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
