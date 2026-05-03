from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .csi_preprocess import preprocess_csi_amp
from .heatmap_gt import build_pcm_paf, coco17_to_openpose18


@dataclass(frozen=True)
class MMFiSample:
    env: str
    subject: str
    action: str
    frame_idx: int
    trial_dir: Path
    csi_path: Path

    @property
    def pose_path(self) -> Path:
        return self.trial_dir / "pose2d.npy"


def _sorted_dirs(path: Path, prefixes: Sequence[str] | None = None) -> list[Path]:
    if not path.exists():
        return []
    dirs = [p for p in path.iterdir() if p.is_dir()]
    if prefixes:
        dirs = [p for p in dirs if any(p.name.startswith(prefix) for prefix in prefixes)]
    return sorted(dirs, key=lambda p: p.name)


def _frame_number(path: Path) -> int | None:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def enumerate_mmfi_samples(
    root: str | Path,
    envs: Iterable[str] | None = None,
    subjects: Iterable[str] | None = None,
    max_samples: int | None = None,
) -> list[MMFiSample]:
    root = Path(root)
    env_filter = set(envs) if envs else None
    subject_filter = set(subjects) if subjects else None
    samples: list[MMFiSample] = []

    env_dirs = _sorted_dirs(root, ["E"])
    for env_dir in env_dirs:
        if env_filter and env_dir.name not in env_filter:
            continue
        for subject_dir in _sorted_dirs(env_dir, ["S"]):
            if subject_filter and subject_dir.name not in subject_filter:
                continue
            for action_dir in _sorted_dirs(subject_dir, ["A"]):
                pose_path = action_dir / "pose2d.npy"
                csi_dir = action_dir / "wifi-csi"
                if not pose_path.exists() or not csi_dir.exists():
                    continue
                try:
                    pose_len = int(np.load(pose_path, mmap_mode="r").shape[0])
                except Exception:
                    continue
                frame_paths = sorted(csi_dir.glob("frame*.mat"), key=lambda p: p.name)
                for csi_path in frame_paths:
                    frame_idx = _frame_number(csi_path)
                    if frame_idx is None or frame_idx < 1 or frame_idx > pose_len:
                        continue
                    samples.append(
                        MMFiSample(
                            env=env_dir.name,
                            subject=subject_dir.name,
                            action=action_dir.name,
                            frame_idx=frame_idx,
                            trial_dir=action_dir,
                            csi_path=csi_path,
                        )
                    )
                    if max_samples is not None and len(samples) >= max_samples:
                        return samples
    return samples


class MMFiDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        protocol: str = "subject_cross",
        envs: Iterable[str] | None = None,
        train_subjects: Iterable[str] | None = None,
        test_subjects: Iterable[str] | None = None,
        random_val_ratio: float = 0.2,
        seed: int = 42,
        max_samples: int | None = None,
        time_packets: int = 64,
        subcarrier_mode: str = "keep",
        normalize: str = "zscore",
        amp_key: str = "CSIamp",
        heatmap_size: int = 36,
        heatmap_sigma: float = 1.5,
        paf_width: float = 1.0,
        pose_range: tuple[float, float] = (-0.8, 0.8),
        build_targets: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.protocol = protocol
        self.time_packets = time_packets
        self.subcarrier_mode = subcarrier_mode
        self.normalize = normalize
        self.amp_key = amp_key
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.paf_width = paf_width
        self.pose_range = pose_range
        self.build_targets = build_targets
        self._pose_cache: dict[Path, np.ndarray] = {}

        subjects = None
        if protocol == "subject_cross" and split != "all":
            subjects = train_subjects if split == "train" else test_subjects

        all_samples = enumerate_mmfi_samples(
            self.root,
            envs=envs,
            subjects=subjects,
            max_samples=None if protocol == "random" else max_samples,
        )
        if protocol == "random" and split != "all":
            rng = random.Random(seed)
            rng.shuffle(all_samples)
            pivot = int(round(len(all_samples) * (1.0 - random_val_ratio)))
            all_samples = all_samples[:pivot] if split == "train" else all_samples[pivot:]

        if max_samples is not None:
            all_samples = all_samples[:max_samples]
        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pose_array(self, path: Path) -> np.ndarray:
        cached = self._pose_cache.get(path)
        if cached is None:
            cached = np.load(path).astype(np.float32)
            self._pose_cache[path] = cached
        return cached

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        mat = loadmat(sample.csi_path)
        if self.amp_key not in mat:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(f"{sample.csi_path} does not contain {self.amp_key}; keys={keys}")
        csi = preprocess_csi_amp(
            mat[self.amp_key],
            target_packets=self.time_packets,
            subcarrier_mode=self.subcarrier_mode,
            normalize=self.normalize,
        )

        pose17 = self._load_pose_array(sample.pose_path)[sample.frame_idx - 1]
        kpts18 = coco17_to_openpose18(pose17)
        item = {
            "csi": torch.from_numpy(csi),
            "kpts18": torch.from_numpy(kpts18),
            "meta": {
                "env": sample.env,
                "subject": sample.subject,
                "action": sample.action,
                "frame_idx": sample.frame_idx,
                "csi_path": str(sample.csi_path),
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
