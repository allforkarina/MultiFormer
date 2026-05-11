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

**状态**: complete

**结果**: CONFIRMED — H5 CSI 已在构建阶段归一化到 [0,1]。
- raw max = 0.9354, attr amplitude_train_max = 57.1957
- `amplitude_normalization: train_global_minmax` — 确认数据是预归一化的
- **结论**: 加载侧调用 `normalize_global_minmax` 是二次归一化，与关键点 bug 同性质
- **修复方案**: 不重建 H5，而是在 loader 中检测 `amplitude_normalization` attr，跳过重复归一化

---

## 阶段 2: 实验 B — 归一化量级对齐

**状态**: implemented (代码已修改，等待在服务器上训练验证)

**背景**: 实验 A 确认 H5 CSI 已在 [0,1] 区间（预归一化）。实验 B 在此基础上调整输入量级，使其匹配模型期望的 ≈ N(0,1) 分布。

### 已完成的修改

#### `data/h5_dataset.py`

1. **修复二次归一化**: `__init__` 中读取 `amplitude_normalization` attr（第 77-80 行），若值为 `train_global_minmax` 则标记 `_amp_pre_normalized = True`

2. **归一化分发逻辑**: `__getitem__` 第 167-187 行替换为：
```python
if self.normalize == "global_minmax":
    if not self._amp_pre_normalized:
        csi_amp = normalize_global_minmax(...)  # only for non-pre-normalized data
    # else: already [0,1] → double-norm bug avoided
elif self.normalize == "global_zscore":
    csi_amp = (csi_amp - 0.5) * 6.0  # [0,1] → [-3,+3], preserves global alignment
elif self.normalize == "zscore":
    csi_amp = normalize_csi(csi_amp, mode="zscore")  # per-sample, referVersion equiv
elif self.normalize == "none":
    pass
```

#### 新增配置文件

| 文件 | `csi.normalize` | 说明 |
|------|----------------|------|
| `configs/E01.yaml` | `"global_minmax"` | B0 基线: [0,1] 不变（二次归一化已修复） |
| `configs/E01_B1.yaml` | `"global_zscore"` | B1: [0,1]→[-3,+3] 中心化扩展 |
| `configs/E01_B2.yaml` | `"zscore"` | B2: 逐样本 z-score (referVersion 等价) |

### 在服务器上执行的命令

```bash
git pull
# B0（基线，二次归一化修复）
python train.py --config configs/E01.yaml

# B1（全局 z-score）
python train.py --config configs/E01_B1.yaml

# B2（逐样本 z-score，等同 referVersion）
python train.py --config configs/E01_B2.yaml
```

每个变体训练 5 epoch 即可观察 PCK@20 趋势。预期 B1 和 B2 的 PCK@20 显著高于 B0。

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
| 1 | `scripts/check_h5_csi_range.py` | 新建 (诊断脚本) | complete |
| 2.1 | `data/h5_dataset.py` | 修复二次归一化 + 归一化分发逻辑 | complete |
| 2.2 | `configs/E01_B1.yaml` | 新建 (global_zscore 变体) | complete |
| 2.2 | `configs/E01_B2.yaml` | 新建 (zscore 变体) | complete |
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
