# 归一化消融实验 — 实现计划

## 目标

验证"归一化输出量级不匹配"是否为 multiformer (H5) 性能远低于 referVersion (NPZ) 的根本原因，并在保留"全局对齐"设计意图的前提下寻找最优归一化策略。

## 背景

- referVersion: NPZ + 逐帧 z-score → 输入量级 ≈ N(0,1)，epoch 1 PCK@20 = 0.827
- multiformer: H5 + 全局 min-max → 输入量级 [0,1]，epoch 21-52 PCK@20 ≈ 0.55-0.63
- 两个管线的 CSI 时序处理完全相同（10→64 FFT 上采样）
- `TFDDTTokenizer` 第一层是 Linear 投影（无 LayerNorm），对输入量级敏感

---

## 阶段 1: 实验 A — 验证 H5 内 CSI 原始数据范围

**状态**: pending

**目标**: 确认 H5 中的 `csi_amplitude` 是原始幅度还是已被归一化到 [0,1]。

**操作**:
在 Linux 服务器上运行以下诊断脚本：

```python
# scripts/check_h5_csi_range.py
import h5py
import numpy as np

h5_path = "/data/WiFiPose/dataset/mmfi_pose.h5"
with h5py.File(h5_path, "r") as f:
    raw = f["csi_amplitude"][:100]  # 前 100 帧
    print("=== CSI Amplitude Raw Data ===")
    print(f"shape: {raw.shape}")
    print(f"min: {raw.min():.6f}")
    print(f"max: {raw.max():.6f}")
    print(f"mean: {raw.mean():.6f}")
    print(f"std: {raw.std():.6f}")
    print()
    print("=== H5 Attributes ===")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")

# 判断逻辑:
# 若 raw.max() ≈ 1 而 attrs.amplitude_train_max ≈ 57 → 二次归一化 bug，需修复 H5 构建脚本
# 若 raw.max() 与 attrs.amplitude_train_max 量级一致 → 数据正常，继续阶段 2
```

**判断与决策**:
- 若二次归一化 → **修复 H5 构建脚本**，重新生成 H5 文件（移除构建阶段的归一化，保留原始 amplitude）
- 若数据正常 → 继续阶段 2

---

## 阶段 2: 实验 B — 归一化量级对齐

**状态**: pending (依赖阶段 1 通过)

**目标**: 在保留"全局对齐"的前提下，将输入量级从 [0,1] 对齐到 ≈ N(0,1)。

### 涉及文件

| 文件 | 修改内容 | 变更量 |
|------|---------|--------|
| `data/csi_preprocess.py` | 新增 `normalize_global_zscore()` 函数 | ~15 行 |
| `data/h5_dataset.py` | `__getitem__` 中归一化分支逻辑 | ~10 行 |
| `data/h5_dataset.py` | `__init__` 读取新的 H5 attrs（可选） | ~5 行 |
| `configs/E01.yaml` | `csi.normalize` 字段扩展 | ~1 行 |

### 2.1 新增归一化函数 (`data/csi_preprocess.py`)

在 `normalize_csi()` 之后新增：

```python
def normalize_global_zscore(
    x: np.ndarray,
    train_mean: float,
    train_std: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply global z-score normalization using pre-computed train split statistics.
    
    Unlike per-sample zscore, all samples share the same mean/std,
    preserving inter-sample amplitude relationships (action discriminability).
    Unlike global minmax, output is centered at 0 with unit variance,
    matching the input distribution the model was designed for.
    """
    x = sanitize_csi(x)
    denom = train_std
    if not np.isfinite(denom) or denom < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - train_mean) / (denom + eps)).astype(np.float32)
```

**设计理由**: 全局 z-score 保留动作间相对幅度差异（与 minmax 一致），同时将输出对齐到 N(0,1) 量级（与 referVersion zscore 一致）。

### 2.2 修改 H5MMFiDataset 的归一化逻辑 (`data/h5_dataset.py`)

#### 2.2.1 `__init__` 读取新的 H5 attrs（方案 A 所需）

在 `__init__` 的 H5 attrs 读取部分增加：

```python
# 仅当 normalize == "global_zscore" 时需要
if self.normalize == "global_zscore":
    self.amp_train_mean = float(f.attrs["amplitude_train_mean"])
    self.amp_train_std = float(f.attrs["amplitude_train_std"])
```

**前提**: H5 文件需要预先计算并存储 `amplitude_train_mean` 和 `amplitude_train_std` 属性。如果 H5 中尚无这两个 attr，需先用诊断脚本计算并写入。

#### 2.2.2 `__getitem__` 中的归一化分支

将第 162-167 行的归一化调用改为根据 `self.normalize` 分发：

```python
# Normalize (strategy selected by config csi.normalize)
if self.normalize == "global_minmax":
    csi_amp = normalize_global_minmax(
        csi_amp,
        train_min=self.amp_train_min,
        train_max=self.amp_train_max,
    )
elif self.normalize == "global_zscore":
    csi_amp = normalize_global_zscore(
        csi_amp,
        train_mean=self.amp_train_mean,
        train_std=self.amp_train_std,
    )
elif self.normalize == "zscore":
    csi_amp = normalize_csi(csi_amp, mode="zscore")
elif self.normalize == "minmax":
    csi_amp = normalize_csi(csi_amp, mode="minmax")
elif self.normalize == "none":
    pass  # already sanitized in _resample_time_h5
else:
    raise ValueError(f"Unknown normalize mode: {self.normalize}")
```

### 2.3 配置变体

在 `configs/E01.yaml` 和新增的临时实验配置中切换 `csi.normalize`：

| 配置 | `csi.normalize` | 说明 |
|------|----------------|------|
| E01.yaml (当前) | `"global_minmax"` | 基线 B0 |
| E01_B1.yaml | `"global_zscore"` | 全局 z-score，保留全局对齐 |
| E01_B2.yaml | `"zscore"` | 逐样本 z-score，等同 referVersion（对照） |
| E01_B3.yaml | `"minmax"` | 逐样本 min-max，额外对照 |

**每个变体的 YAML 仅需修改 `csi.normalize` 和 `train.output_dir`**：

```yaml
# configs/E01_B1.yaml (示例)
dataset:
  type: "h5"
  root: "/data/WiFiPose/dataset/mmfi_pose.h5"
  # ... 其余与 E01.yaml 完全相同 ...

csi:
  normalize: "global_zscore"    # ← 唯一关键差异
  # ... 其余相同 ...

train:
  output_dir: "outputs/E01_B1"  # ← 避免覆盖原输出
  # ... 其余相同 ...
```

### 2.4 训练与评估流程

1. 创建 4 个配置文件（或直接在服务器上临时修改 normalize 字段）
2. 每个变体训练 **5 epoch**（batch_size=32, seed=42）
3. 记录每个 epoch 的 `train_log.csv`
4. 对比指标：PCK@20 曲线、val_loss 曲线、train_loss 下降速度

### 2.5 预期结果

| 变体 | 预期 epoch 5 PCK@20 | 理由 |
|------|---------------------|------|
| B0 (global_minmax) | ~0.55-0.60 | 当前基线，输入量级不匹配 |
| B1 (global_zscore) | **~0.75-0.85** | 量级对齐 + 保留全局关系 |
| B2 (zscore) | ~0.80-0.88 | 等同 referVersion，最接近参考 |
| B3 (minmax) | ~0.55-0.65 | 逐样本 min-max，量级仍不匹配 |

---

## 阶段 3: 实验 C — 划分作用域对齐（可选）

**状态**: pending (若阶段 2 确认问题后仍有残余差距)

**目标**: 排除训练/验证划分差异对 PCK 评估的影响。

**修改位置**: `data/h5_dataset.py:_build_split()`

将当前的"按 subject 分组 shuffle"改为"全局 shuffle"：

```python
# 修改前 (L102-125 附近):
for subject, indices in sorted(grouped.items()):
    shuffled = indices[:]
    rng.shuffle(shuffled)
    pivot = int(round(len(shuffled) * (1.0 - random_val_ratio)))
    train_indices.extend(shuffled[:pivot])
    val_indices.extend(shuffled[pivot:])

# 修改后:
all_candidates = candidate_indices[:]
rng.shuffle(all_candidates)
pivot = int(round(len(all_candidates) * (1.0 - random_val_ratio)))
train_indices = all_candidates[:pivot]
val_indices = all_candidates[pivot:]
```

**仅当阶段 2 的 B1 变体 PCK 与 referVersion 仍有 ≥ 0.05 差距时才需要执行**。

---

## 阶段 4: H5 属性补充（若选择全局 z-score）

**状态**: pending (依赖阶段 2 结论 — B1 变体方案被采纳)

若实验验证 B1 (global_zscore) 最优，需要在 H5 文件中写入 `amplitude_train_mean` 和 `amplitude_train_std`。

**操作**: 运行一次性脚本计算训练集的全局统计量：

```python
# scripts/compute_h5_stats.py
import h5py
import numpy as np

h5_path = "/data/WiFiPose/dataset/mmfi_pose.h5"
with h5py.File(h5_path, "r+") as f:
    # 获取训练索引（与 h5_dataset 的 _build_split 逻辑一致）
    # ... 计算训练集的 csi_amplitude mean 和 std ...
    
    f.attrs["amplitude_train_mean"] = float(train_mean)
    f.attrs["amplitude_train_std"] = float(train_std)
    print(f"Written: amplitude_train_mean={train_mean:.6f}, amplitude_train_std={train_std:.6f}")
```

**注意**: 需要在 `_resample_time_h5` 和转置之后计算统计量，使统计量与模型实际看到的输入分布一致。

---

## 执行顺序

```
阶段 1 (实验 A) ──→ 阶段 2 (实验 B) ──→ 决策
                     │                      │
                     │               B1/B2 效果最好？
                     │                 ├── B1 → 阶段 4 (补充 H5 attrs)
                     │                 ├── B2 → 考虑是否接受逐样本 zscore
                     │                 └── B0/B3 → 排查其他原因
                     │
                     └── 若仍有残差 → 阶段 3 (实验 C)
```

## 文件变更汇总

| 阶段 | 文件 | 操作 | 状态 |
|------|------|------|------|
| 1 | `scripts/check_h5_csi_range.py` | 新建 (诊断脚本) | pending |
| 2.1 | `data/csi_preprocess.py` | 新增 `normalize_global_zscore()` | pending |
| 2.2 | `data/h5_dataset.py` | 修改 `__init__` 读取新 attrs + `__getitem__` 归一化分支 | pending |
| 2.3 | `configs/E01_B1.yaml` (等) | 新建实验配置 | pending |
| 3 | `data/h5_dataset.py` | 修改 `_build_split` 划分逻辑 | pending |
| 4 | `scripts/compute_h5_stats.py` | 新建 (统计量计算脚本) | pending |

## 不修改的文件

| 文件 | 原因 |
|------|------|
| `models/*` | 与归一化无关 |
| `decode/pose_decoder.py` | 与归一化无关 |
| `utils/metrics.py` | 与归一化无关 |
| `data/heatmap_gt.py` | 与归一化无关 |
| `train.py` | 无需修改（归一化逻辑在 dataset 内部） |
| `eval.py` | 无需修改 |
