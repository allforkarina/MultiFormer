from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap_gt import build_pcm_paf


@dataclass(frozen=True)
class MMFiSample:
    env: str
    subject: str
    action: str
    frame_idx: int        # 1-based index, kept for compatibility / metadata
    array_index: int      # 0-based index into the npz arrays
    npz_path: Path


def _sorted_dirs(path: Path, prefixes: Sequence[str] | None = None) -> list[Path]:
    if not path.exists():
        return []
    dirs = [p for p in path.iterdir() if p.is_dir()]
    if prefixes:
        dirs = [p for p in dirs if any(p.name.startswith(prefix) for prefix in prefixes)]
    return sorted(dirs, key=lambda p: p.name)


def enumerate_mmfi_samples(
    root: str | Path,
    envs: Iterable[str] | None = None,
    subjects: Iterable[str] | None = None,
    max_samples: int | None = None,
) -> list[MMFiSample]:
    """Enumerate every (trial, frame) pair available under the preprocessed npz tree.

    The expected layout is::

        root/E0X/S0X/A0X.npz
        with arrays: csi (N,64,3,114), kpts18 (N,18,2), frame_idx (N,)
    """
    root = Path(root)
    env_filter = set(envs) if envs else None
    subject_filter = set(subjects) if subjects else None
    samples: list[MMFiSample] = []

    for env_dir in _sorted_dirs(root, ["E"]):
        if env_filter and env_dir.name not in env_filter:
            continue
        for subject_dir in _sorted_dirs(env_dir, ["S"]):
            if subject_filter and subject_dir.name not in subject_filter:
                continue
            for npz_path in sorted(subject_dir.glob("A*.npz")):
                action = npz_path.stem
                try:
                    with np.load(npz_path, mmap_mode="r") as data:
                        n_frames = int(data["csi"].shape[0])
                        frame_ids = np.asarray(data["frame_idx"]).astype(int).tolist()
                except Exception:
                    continue
                if len(frame_ids) != n_frames:
                    frame_ids = list(range(1, n_frames + 1))
                for array_index in range(n_frames):
                    samples.append(
                        MMFiSample(
                            env=env_dir.name,
                            subject=subject_dir.name,
                            action=action,
                            frame_idx=int(frame_ids[array_index]),
                            array_index=array_index,
                            npz_path=npz_path,
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
        self._npz_cache: dict[Path, dict[str, np.ndarray]] = {}

        subjects = None
        if protocol == "subject_cross" and split != "all":
            subjects = train_subjects if split == "train" else test_subjects
        elif protocol == "same_subject_random" and split != "all":
            subjects = train_subjects if split == "train" else test_subjects

        all_samples = enumerate_mmfi_samples(
            self.root,
            envs=envs,
            subjects=subjects,
            max_samples=None if protocol in {"random", "same_subject_random"} else max_samples,
        )
        if protocol in {"random", "same_subject_random"} and split != "all":
            rng = random.Random(seed)
            rng.shuffle(all_samples)
            pivot = int(round(len(all_samples) * (1.0 - random_val_ratio)))
            all_samples = all_samples[:pivot] if split == "train" else all_samples[pivot:]

        if max_samples is not None:
            all_samples = all_samples[:max_samples]
        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_trial(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._npz_cache.get(path)
        if cached is None:
            with np.load(path) as data:
                cached = {
                    "csi": np.asarray(data["csi"], dtype=np.float32),
                    "kpts18": np.asarray(data["kpts18"], dtype=np.float32),
                }
            self._npz_cache[path] = cached
        return cached

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        trial = self._load_trial(sample.npz_path)
        csi = trial["csi"][sample.array_index]
        kpts18 = trial["kpts18"][sample.array_index]
        item = {
            "csi": torch.from_numpy(np.ascontiguousarray(csi)),
            "kpts18": torch.from_numpy(np.ascontiguousarray(kpts18)),
            "meta": {
                "env": sample.env,
                "subject": sample.subject,
                "action": sample.action,
                "frame_idx": sample.frame_idx,
                "csi_path": str(sample.npz_path),
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
