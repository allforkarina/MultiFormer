# H5 预处理优化：将训练时开销移至 H5 生成阶段 (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 Linux 服务器上的原始 MM-Fi 数据集生成一个新的 H5 文件，将时间重采样(10→64 FFT)、维度转置、关键点转换等训练时预处理操作前置到 H5 构建阶段，使训练时每个样本仅需 1 次 HDF5 读取 + 轻量算术归一化。

**Architecture:** 新脚本遍历 `/data/WiFiPose/dataset/dataset/{ACTION}/{SUBJECT}/` 目录，逐帧加载 `wifi-csi/frameXXX.mat` (CSIamp) 和 `rgb/frameXXX.npy` (COCO17 keypoints)，执行时间重采样 + 转置得到 `(64,3,114)` 未归一化 CSI，同时将 COCO17 关键点转换为 OpenPose18 并归一化到 pose_range。所有预处理后的张量写入单个 H5 文件，附带训练集全局统计量作为 attributes。

**Tech Stack:** Python, h5py, numpy, scipy.signal.resample, scipy.io.loadmat

---

## 服务器数据布局

```
/data/WiFiPose/dataset/dataset/    ← 数据根目录 (--src)
├── A01/                           ← 动作 (action)
│   ├── S01/                       ← 受试者 (subject)
│   │   ├── wifi-csi/
│   │   │   ├── frame001.mat       ← CSIamp (3, 114, 10) float64
│   │   │   ├── frame002.mat
│   │   │   └── ...
│   │   └── rgb/
│   │       ├── frame001.npy       ← COCO17 keypoints (17, 2) float32
│   │       ├── frame002.npy
│   │       └── ...
│   ├── S02/
│   └── ...
├── A02/
├── ...
└── Axx/
```

**与旧 `preprocess_mmfi.py` 的差异：**
- 无显式 environment (E0X) 目录层级 → environment 由 subject ID 派生: S01-S10 → env1, S11-S20 → env2, S21-S30 → env3, S31-S40 → env4
- 关键点每帧一个 `.npy` 文件，而非按 trial 聚合的 `pose2d.npy`
- CSI 是 `.mat` 文件，与旧脚本一致

**Environment 派生规则:**
```python
def derive_env(subject: str) -> str:
    num = int(subject.lstrip("S"))
    env_idx = (num - 1) // 10 + 1
    return f"env{env_idx}"
# S01-S10 → env1, S11-S20 → env2, S21-S30 → env3, S31-S40 → env4
```

---

## 数据需求分析

### 当前 H5 结构 (`mmfi_pose.h5`) — 训练时预处理

| 数据集 | Shape | 训练时操作 |
|--------|-------|-----------|
| `csi_amplitude` | `(N, 3, 114, 10)` | FFT resample(10→64) + transpose + normalize |
| `keypoints` | `(N, 17, 2)` | COCO17→OpenPose18 + normalize to pose_range |
| `environment` | `(N,)` str | 用于划分 |
| `sample` | `(N,)` str | 用于划分 |
| `action` | `(N,)` str | 元信息 |
| `frame_idx` | `(N,)` int | 帧序号 |

### 目标 H5 结构 (新 v2) — 预处理前置

| 数据集 | Shape | 训练时操作 |
|--------|-------|-----------|
| `csi` | `(N, 64, 3, 114)` | **仅归一化** (纯算术, 4 模式可选) |
| `kpts18` | `(N, 18, 2)` | **无需转换** (已是 OpenPose18, 已在 pose_range) |
| `environment` | `(N,)` str | env1-env4, 由 subject ID 派生 (S01-S10→env1, ...) |
| `sample` | `(N,)` str | 受试者ID (S01-Sxx) |
| `action` | `(N,)` str | 动作ID (A01-Axx) |
| `frame_idx` | `(N,)` int64 | 帧序号 |

**Attributes:**

| Attribute | 用途 |
|-----------|------|
| `amplitude_train_min` | global_minmax 归一化 |
| `amplitude_train_max` | global_minmax 归一化 |
| `amplitude_train_mean` | global_zscore 归一化 |
| `amplitude_train_std` | global_zscore 归一化 |
| `pose_min` | 关键点坐标范围下界 |
| `pose_max` | 关键点坐标范围上界 |
| `time_packets` | 64 (记录) |
| `rx_antennas` | 3 (记录) |
| `subcarriers` | 114 (记录) |

### 归一化消融实验接口

新 H5 存储**未归一化**的 CSI，训练时根据 `csi.normalize` config 分发：

| normalize 模式 | 训练时操作 | 所需 H5 attr |
|---------------|-----------|-------------|
| `global_minmax` | `(x - amp_train_min) / (amp_train_max - amp_train_min)` | `amplitude_train_min`, `amplitude_train_max` |
| `global_zscore` | `(x - amp_train_mean) / amp_train_std` | `amplitude_train_mean`, `amplitude_train_std` |
| `zscore` | per-sample `(x - mean) / std` | 无需 attr |
| `none` | pass | 无需 attr |

---

### Task 1: 更新 `data/h5_dataset.py` 适配新 H5 结构

**Files:**
- Modify: `data/h5_dataset.py`

- [ ] **Step 1: 修改 `__init__` — 替换 attr 和数据集读取**

```python
# 替换 lines 72-96 的 attr 读取和数组预加载:

with h5py.File(self.h5_path, "r") as f:
    # 归一化统计量 (训练时用，纯算术)
    self.amp_train_min = float(f.attrs["amplitude_train_min"])
    self.amp_train_max = float(f.attrs["amplitude_train_max"])
    self.amp_train_mean = float(f.attrs.get("amplitude_train_mean", 0.0))
    self.amp_train_std = float(f.attrs.get("amplitude_train_std", 1.0))
    self.pose_min = float(f.attrs.get("pose_min", -0.8))
    self.pose_max = float(f.attrs.get("pose_max", 0.8))

    # 预加载轻量数组（keypoints ~43 MB, meta ~6 MB）
    self._keypoints_h5 = np.asarray(f["kpts18"][:], dtype=np.float32)
    self._envs_h5 = np.array(
        [_decode_bytes(e) for e in f["environment"][:]], dtype=object
    )
    self._samples_h5 = np.array(
        [_decode_bytes(s) for s in f["sample"][:]], dtype=object
    )
    self._actions_h5 = np.array(
        [_decode_bytes(a) for a in f["action"][:]], dtype=object
    )

    self.indices = self._build_split(
        split, envs, train_subjects, test_subjects, random_val_ratio, seed
    )
```

- [ ] **Step 2: 修改 `__getitem__` — 移除 resample/transpose/keypoint 转换**

```python
def __getitem__(self, index: int) -> dict:
    h5_file = self._get_h5()
    frame_idx = int(self.indices[index])

    # CSI: 已是 (64, 3, 114)，仅需归一化
    csi_amp = np.asarray(h5_file["csi"][frame_idx], dtype=np.float32)

    if self.normalize == "global_minmax":
        csi_amp = normalize_global_minmax(
            csi_amp,
            train_min=self.amp_train_min,
            train_max=self.amp_train_max,
        )
    elif self.normalize == "global_zscore":
        csi_amp = (csi_amp - self.amp_train_mean) / (self.amp_train_std + 1e-6)
    elif self.normalize == "zscore":
        csi_amp = normalize_csi(csi_amp, mode="zscore")
    elif self.normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    # 关键点: 已是 (18, 2) 在 pose_range，直接使用
    kpts18 = self._keypoints_h5[frame_idx].copy()

    item: dict = {
        "csi": torch.from_numpy(csi_amp),
        "kpts18": torch.from_numpy(np.ascontiguousarray(kpts18)),
        "meta": {
            "env": str(self._envs_h5[frame_idx]),
            "subject": str(self._samples_h5[frame_idx]),
            "action": str(self._actions_h5[frame_idx]),
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
```

- [ ] **Step 3: 删除不再需要的方法和 import**

```python
# 删除 _resample_time_h5 函数 (lines 23-29)
# 删除 _normalize_keypoints 方法 (lines 157-162)
# 删除 import 中的: from scipy.signal import resample (line 10)
# 删除 import 中的: from .heatmap_gt import build_pcm_paf, coco17_to_openpose18 中不需要的 coco17_to_openpose18
#    → 仅保留 build_pcm_paf

# 清理后的 import:
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .csi_preprocess import normalize_csi, normalize_global_minmax
from .heatmap_gt import build_pcm_paf
```

- [ ] **Step 4: 修改 `_build_split` — 保留 env + subject 双重过滤**

```python
def _build_split(
    self,
    split: str,
    envs: Iterable[str] | None,
    train_subjects: Iterable[str] | None,
    test_subjects: Iterable[str] | None,
    random_val_ratio: float,
    seed: int,
) -> np.ndarray:
    num_total = len(self._samples_h5)
    env_list = [str(e) for e in self._envs_h5]
    sample_list = [str(s) for s in self._samples_h5]

    env_set = set(envs) if envs else None
    subject_set = set(train_subjects) if train_subjects else None

    candidate_indices: list[int] = []
    for i in range(num_total):
        if env_set is not None and env_list[i] not in env_set:
            continue
        if subject_set is not None and sample_list[i] not in subject_set:
            continue
        candidate_indices.append(i)

    if split != "all":
        rng = random.Random(seed)
        grouped: dict[str, list[int]] = {}
        for idx in candidate_indices:
            grouped.setdefault(sample_list[idx], []).append(idx)

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
```

- [ ] **Step 5: 更新 `__init__` 签名 — 保留 envs 参数**

```python
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
    normalize: str = "global_minmax",
    heatmap_size: int = 36,
    heatmap_sigma: float = 1.5,
    paf_width: float = 1.0,
    pose_range: tuple[float, float] = (-0.8, 0.8),
    build_targets: bool = True,
) -> None:
```

- [ ] **Step 6: 验证 meta dict**

meta dict 中保留 `"env"` 键（已在 Step 2 中体现）。

- [ ] **Step 7: 验证修改**

```bash
cd /data/WiFiPose/multiformer
python -c "
from data.h5_dataset import H5MMFiDataset
ds = H5MMFiDataset('/data/WiFiPose/dataset/mmfi_pose_v2.h5', split='train', normalize='global_zscore')
sample = ds[0]
print(f'csi: {sample[\"csi\"].shape}')       # torch.Size([64, 3, 114])
print(f'kpts18: {sample[\"kpts18\"].shape}') # torch.Size([18, 2])
print(f'csi range: [{sample[\"csi\"].min():.2f}, {sample[\"csi\"].max():.2f}]')
print(f'kpts range: [{sample[\"kpts18\"].min():.2f}, {sample[\"kpts18\"].max():.2f}]')
print(f'meta keys: {list(sample[\"meta\"].keys())}')
"
```

Expected: csi `torch.Size([64, 3, 114])`, kpts18 `torch.Size([18, 2])`, meta keys `['env', 'subject', 'action', 'frame_idx']`.

---

### Task 2: 编写 H5 生成脚本 `scripts/build_h5.py`

**Files:**
- Create: `scripts/build_h5.py`

- [ ] **Step 1: 脚本骨架 + CLI**

```python
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
```

- [ ] **Step 2: CSI 预处理函数 (no normalization)**

```python
def sanitize_csi(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with median of finite values."""
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.all():
        return x
    fill = float(np.median(x[finite])) if finite.any() else 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def preprocess_csi_one_frame(csi_amp: np.ndarray) -> np.ndarray:
    """
    Preprocess one CSI frame: sanitize → resample → transpose.

    Input:  CSIamp (3, 114, 10) from .mat file
    Output: (64, 3, 114) float32, NOT normalized
    """
    csi_amp = sanitize_csi(np.asarray(csi_amp, dtype=np.float32))
    csi_amp = sanitize_csi(resample(csi_amp, TIME_PACKETS, axis=-1))
    csi_amp = np.transpose(csi_amp, (2, 0, 1)).astype(np.float32, copy=False)
    return csi_amp
```

- [ ] **Step 3: 关键点转换函数**

```python
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
    """
    Normalize keypoints to pose_range.

    MM-Fi pose2d coordinates are already in normalized range (approx [-0.8, 0.8]).
    If values appear to be pixel coordinates (range > 10), scale by image dimensions.
    """
    kpts = np.asarray(kpts, dtype=np.float32).copy()

    abs_max = np.abs(kpts[kpts != 0]).max() if (kpts != 0).any() else 0.0

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
```

- [ ] **Step 4: 枚举所有 trial + env 派生函数**

```python
def derive_env(subject: str) -> str:
    """Derive environment label from subject ID.
    S01-S10 → env1, S11-S20 → env2, S21-S30 → env3, S31-S40 → env4.
    """
    num = int(subject.lstrip("S"))
    env_idx = (num - 1) // 10 + 1
    return f"env{env_idx}"


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
```

- [ ] **Step 5: 单 trial 处理函数**

```python
def process_trial(
    trial_dir: Path,
    pose_min: float,
    pose_max: float,
) -> dict | None:
    """
    Process one trial ({ACTION}/{SUBJECT}) and return dict with arrays.

    Returns dict with:
        csi:      (N, 64, 3, 114) float32
        kpts18:   (N, 18, 2) float32
        sample:   str
        action:   str
        frame_idx: (N,) int64
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
```

- [ ] **Step 6: 主流程 — 并行处理 + 写 H5**

```python
def _worker(args):
    trial_dir, pose_min, pose_max = args
    try:
        result = process_trial(Path(trial_dir), pose_min, pose_max)
        label = f"{Path(trial_dir).parent.name}/{Path(trial_dir).name}"
        return label, result, None
    except Exception:
        label = f"{Path(trial_dir).parent.name}/{Path(trial_dir).name}"
        return label, None, traceback.format_exc()


def build_h5(
    src_root: Path,
    dst_path: Path,
    train_subjects: set[str],
    pose_min: float,
    pose_max: float,
    workers: int,
    chunk_size: int,
) -> None:
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
        print("ERROR: No trials processed")
        sys.exit(1)

    # Concatenate
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

    # Train-set statistics
    train_mask = np.isin(all_subjects.astype(str), list(train_subjects))
    train_csi = all_csi[train_mask]
    amp_train_min = float(train_csi.min())
    amp_train_max = float(train_csi.max())
    amp_train_mean = float(train_csi.mean())
    amp_train_std = float(train_csi.std())

    print(f"Total frames: {len(all_csi)}")
    print(f"Train frames: {train_mask.sum()} ({train_mask.sum()/len(all_csi)*100:.1f}%)")
    print(f"Train CSI: min={amp_train_min:.4f} max={amp_train_max:.4f} mean={amp_train_mean:.4f} std={amp_train_std:.4f}")

    # Write H5
    print(f"Writing H5 to {dst_path}...")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype()
    cs = min(chunk_size, all_csi.shape[0])

    with h5py.File(dst_path, "w") as f:
        f.create_dataset(
            "csi", data=all_csi,
            chunks=(cs, TIME_PACKETS, RX_ANTENNAS, SUBCARRIERS),
            compression="gzip", compression_opts=4,
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
```

- [ ] **Step 7: 在服务器上测试（小数据子集）**

```bash
# 先测试单个 trial
python scripts/build_h5.py \
    --src /data/WiFiPose/dataset/dataset \
    --dst /tmp/test_mmfi_v2.h5 \
    --workers 1
```

```bash
# 验证输出
python -c "
import h5py, numpy as np
with h5py.File('/tmp/test_mmfi_v2.h5', 'r') as f:
    print('csi:', f['csi'].shape, f['csi'].dtype)
    print('kpts18:', f['kpts18'].shape, f['kpts18'].dtype)
    print('sample:', f['sample'][:5])
    print('action:', f['action'][:5])
    print('attrs:', dict(f.attrs))
"
```

Expected: `csi (N, 64, 3, 114) float32`, `kpts18 (N, 18, 2) float32`, `environment` dataset with values like "env1", attrs contain all 9 keys.

- [ ] **Step 8: 全量处理**

```bash
nohup python scripts/build_h5.py \
    --src /data/WiFiPose/dataset/dataset \
    --dst /data/WiFiPose/dataset/mmfi_pose_v2.h5 \
    --workers 8 \
    > logs/build_h5.log 2>&1 &
```

---

### Task 3: 更新配置文件

**Files:**
- Modify: `configs/E01.yaml`, `configs/E01_B1.yaml`, `configs/E01_B2.yaml`

- [ ] **Step 1: 更新 dataset section**

保留 `envs` 字段，更新 `root` 路径：

```yaml
# configs/E01.yaml (B0: global_minmax)
dataset:
  type: "h5"
  root: "/data/WiFiPose/dataset/mmfi_pose_v2.h5"
  split: "same_subject_random"
  envs: ["env1"]
  train_subjects: ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"]
  test_subjects: ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"]
  random_val_ratio: 0.2
  seed: 42

csi:
  time_packets: 64
  rx_antennas: 3
  subcarriers: 114
  subcarrier_mode: "keep"
  normalize: "global_minmax"
```

```yaml
# configs/E01_B1.yaml (global_zscore)
# Same as E01.yaml, except:
csi:
  normalize: "global_zscore"
```

```yaml
# configs/E01_B2.yaml (zscore)
# Same as E01.yaml, except:
csi:
  normalize: "zscore"
```

---

### Task 4: 训练速度验证

- [ ] **Step 1: 旧 H5 基准测试**

```bash
python -c "
import time
from data.h5_dataset import H5MMFiDataset
ds = H5MMFiDataset('/data/WiFiPose/dataset/mmfi_pose.h5', split='train', normalize='global_minmax')
t0 = time.time()
for i in range(200):
    _ = ds[i]
dt = time.time() - t0
print(f'Old H5: {dt:.3f}s for 200 samples = {dt/200*1000:.1f}ms/sample')
"
```

- [ ] **Step 2: 新 H5 速度测试**

```bash
python -c "
import time
from data.h5_dataset import H5MMFiDataset
ds = H5MMFiDataset('/data/WiFiPose/dataset/mmfi_pose_v2.h5', split='train', normalize='global_minmax')
t0 = time.time()
for i in range(200):
    _ = ds[i]
dt = time.time() - t0
print(f'New H5: {dt:.3f}s for 200 samples = {dt/200*1000:.1f}ms/sample')
"
```

Expected: 新 H5 每样本耗时显著降低（移除了 FFT resample 开销）。

---

## 执行顺序

```
Task 2 (生成新 H5) → Task 3 (更新 config)
                     ↓
                Task 1 (修改 loader 适配新 H5)
                     ↓
                Task 4 (速度验证)
                     ↓
                Linux 服务器: 训练 B0/B1/B2 各 5 epoch
```

---

## 磁盘空间预估

- CSI `(N, 64, 3, 114)` float32: 每帧 87.5 KB
- 预估总帧数 ~30,000 (约 100 trial × 297 帧)
- 未压缩: ~2.6 GB, gzip 压缩后: ~1.5-2 GB
