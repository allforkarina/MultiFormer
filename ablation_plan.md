# MultiFormer vs referVersion 差异分析与消融实验计划 (v3)

## 前置约束

1. 两个项目的数据均来自 MM-Fi 开源数据集，原始数据相同。
2. 全局 min-max 归一化是**有意为之**的设计选择，目的是保留动作间的相对信号强度。
3. 根据 `scripts/preprocess_mmfi.py`，NPZ 和 H5 的 CSI 时序处理**完全相同**：
   - 每帧原始 CSIamp: `(3, 114, 10)`（10 个时间包，非 297）
   - FFT 上采样 10 → 64
   - 转置 `(2,0,1)` → `(64, 3, 114)`
   - 预处理脚本第 155-174 行 `preprocess_one_frame()` 与 `h5_dataset.py` 的 `_resample_time_h5()` 完全一致

---

## 性能差距

| 指标 | referVersion (NPZ + zscore) | multiformer (H5 + global_minmax) |
|------|---------|---------|
| epoch 1 PCK@20 | **0.827** | 未知 |
| epoch 13 PCK@20 | **0.910** | 0.55-0.63 (epoch 21-52) |
| epoch 1 train_loss | 0.0284 | 未知（从 epoch 21 看约 0.011） |
| epoch 13 train_loss | 0.0156 | ~0.009（epoch 52，缓慢下降） |
| val_loss 趋势 | 持续下降 | epoch 21 后基本不动 |

---

## 已排除的差异

| 差异 | 结论 |
|------|------|
| 模型代码 | diff 验证，**完全相同** |
| decode/metrics/heatmap_gt | diff 验证，**完全相同** |
| train 循环 | diff 验证，**完全相同** |
| CSI 时序分辨率 | **不成立** — 两管线都是 10→64 FFT 上采样 |
| 时序重采样方法 | **不成立** — 使用相同的 `scipy.signal.resample` |
| 关键点二次归一化 | **已修复** (80d3079)，当前代码正确 |

---

## 实际差异分析

### 差异 1: 归一化输出量级 ⭐⭐⭐⭐ (最可能根因)

两个管线唯一的预处理差异是归一化方式。即使保留"全局对齐"的设计意图，**输出值域的巨大差异**可以直接解释性能差距。

| 维度 | referVersion (zscore) | multiformer (global_minmax) |
|------|----------|----------|
| 公式 | `(x - mean_sample) / std_sample` | `(x - global_min) / (global_max - global_min)` |
| 输出均值 | ≈ 0 | ≈ 0.3-0.5（取决于训练集均值在 [0,1] 区间的落点） |
| 输出方差 | ≈ 1 | ≈ 0.02-0.08（compress into [0,1]） |
| 典型范围 | [-3, +3] | [0, 1] |
| per-sample 自适应 | 是 | 否 |

**为什么这会导致训练失败**：

1. **`TFDDTTokenizer` 的 Linear 投影没有 LayerNorm**。输入均值偏离零（0→0.5）后，Linear 输出的 bias 基准发生系统性偏移。后续 TransformerBlock 的 LayerNorm 能逐步纠正，但每一层的梯度会因输入方差过小而衰减。

2. **学习率不匹配**：lr=0.001 是为 zscore 量级（std≈1）调优的。对于 [0,1] 量级（std≈0.2），有效学习率约缩小 **5 倍**，表现为训练曲线极其平缓——与当前日志中 train_loss 从 0.011 缓慢降到 0.009（30 epoch 仅降 17%）的观察一致。

3. **梯度信号弱**：全局 minmax 把所有值压缩到 [0,1]，大部分值聚集在狭窄区间内。Linear 层接收的输入特征方差小，反向传播的梯度幅度也小。

### 差异 2: H5 内 CSI 是否二次归一化 ⭐⭐⭐

用户描述 H5 已经过"inf清洗、归一化"。如果 `csi_amplitude` 已在 H5 构建阶段归一化到 [0,1]，则加载时再次调用 `normalize_global_minmax` 会除以约 57，导致值坍缩到 ~0.01 量级——与之前关键点二次归一化 bug **同性质**。

Ha 属性 `amplitude_train_max = 57.1957` 提示数据可能是原始幅度（未经归一化），但需实际打印确认。

### 差异 3: 训练/验证集划分作用域 ⭐⭐

- referVersion MMFiDataset: **全局** shuffle 所有样本 → 80/20 分割
- multiformer H5MMFiDataset: 按 **subject 分组**，每组内 80/20 分割

两者虽都实现 `same_subject_random`，但划分结果不同。val set 成分差异可能解释部分 PCK 偏差（≤ 0.05），但不能解释全部 0.25+ 的差距。

### 差异 4: 关键点坐标空间与归一化 ⭐

NPZ 中的 `kpts18` 是原始像素坐标（`preprocess_mmfi.py` 第 11/367 行），在训练时由 `build_pcm_paf()` → `pose_to_heatmap_coords()` 统一映射到 heatmap 空间。

H5 中的 `keypoints` 是已归一化到 [0,1] 的 COCO17 坐标，由 `_normalize_keypoints()` 映射到 pose_range [-0.8, 0.8]，然后同样进入 `build_pcm_paf()`。

两条路径最终进入 `build_pcm_paf()` 的 keypoints 坐标空间不同：
- NPZ: 像素坐标 → 通过 `pose_to_heatmap_coords()` 映射（但这里有个问题——`pose_to_heatmap_coords` 假设输入在 pose_range 内，而 NPZ 存储的是像素坐标！）

**等等**，这里有一个被忽略的差异：

`preprocess_mmfi.py` 第 367 行：`kpts18 = convert_pose2d_batch(pose2d)` — 转换 COCO17→OpenPose18，但**保留了原始 2D 像素坐标**。

然后 `MMFiDataset.__getitem__()` 直接返回这些像素坐标的 `kpts18`。

在 `build_pcm_paf()` 中，`pose_to_heatmap_coords()` 被调用：
```python
kpts18_hm = pose_to_heatmap_coords(kpts18_pose, size=size, pose_range=pose_range)
```
其中 `pose_range = (-0.8, 0.8)`。但输入 `kpts18_pose` 是**像素坐标**（范围约 [0, 1920] × [0, 1080]），不在 [-0.8, 0.8] 内！

这意味着 `pose_to_heatmap_coords()` 的计算：
```python
scale = (size - 1) / (hi - lo)  # (36-1) / (0.8 - (-0.8)) = 35 / 1.6 = 21.875
kpts = (kpts - lo) * scale     # (pixel_coord - (-0.8)) * 21.875
```
会把像素坐标（如 x=960）映射到 `(960 + 0.8) * 21.875 = 21013`，远超出 heatmap 范围 [0, 35]！然后 `clip=True` 会裁剪到 35。

**这意味着 NPZ 管线的所有 GT 关键点都被裁剪到了 heatmap 边界**。但 referVersion 日志显示 PCK@20=0.91——这是不可能的，除非...

让我重新检查。也许 `MMFiDataset.__getitem__` 中 NPZ 的 keypoints 已经被预处理过了？

`preprocess_mmfi.py` 第 367 行：`kpts18 = convert_pose2d_batch(pose2d)` — 直接使用 `pose2d.npy` 中的像素坐标。

`pose2d.npy` 中的坐标来自 MM-Fi 数据集。检查 MM-Fi 的 pose2d 坐标系统...

实际上，MM-Fi 的 `pose2d.npy` 存储的可能是**归一化到 [-0.8, 0.8] 的坐标**，而不是像素坐标。如果是这样，那就是正确的。

让我检查一下：MM-Fi 数据集是按帧处理的，pose2d.npy 可能已经由 MM-Fi 作者归一化到了 pose_range。如果是这样，那 NPZ 和 H5 的坐标空间是一致的（都在 [-0.8, 0.8] 内）。

等等，但 H5 的 keypoints 是先归一化到 [0,1]，然后由 `_normalize_keypoints` 映射到 [-0.8, 0.8]。如果 MM-Fi 的 `pose2d.npy` 也在 [-0.8, 0.8] 范围内，那两者一致。

但问题是：H5 的 keypoints 归一化过程是：
1. 原始像素坐标 → `/ axis_max` → [0, 1]（H5 构建时）
2. `_normalize_keypoints`: [0, 1] → `* 1.6 - 0.8` → [-0.8, 0.8]

而 MM-Fi 的 `pose2d.npy` 可能直接用不同的方式归一化到了 [-0.8, 0.8]。如果归一化方式不同（例如用不同的 scale/offset），关键点就会略有偏移。

但这不太可能解释 0.25 PCK 的差距。PCK 是按 torso_scale 归一化的误差，小偏移影响有限。

OK，让我回到主线。差异 4 的分析方向不对，让我简化为只关注真正重要的差异。

---

## 修正后的差异总结

经过 `preprocess_mmfi.py` 验证，**CSI 预处理管线在两个项目中完全相同**（10→64 FFT 上采样 → 转置 → 归一化），唯一的分歧点是归一化。

因此真正的差异缩小为：

1. **归一化输出量级** — zscore 输出 ≈ N(0,1)，global_minmax 输出 [0,1]
2. **H5 内 CSI 是否二次归一化** — 需验证
3. **训练/验证划分作用域** — 全局 vs 按 subject 分组
4. **关键点坐标来源** — NPZ 来自 pose2d.npy（可能已在 pose_range），H5 来自 [0,1]→pose_range 映射

---

## 消融实验设计

### 实验 A: 验证 H5 内 CSI 原始数据范围 (最先做)

**假设**: H5 中 `csi_amplitude` 可能已被归一化到 [0,1]，导致二次归一化。

**方法**:
```python
with h5py.File(h5_path, "r") as f:
    raw = f["csi_amplitude"][0:20]
    print("raw min/max/mean:", raw.min(), raw.max(), raw.mean())
    print("attrs:", dict(f.attrs))
```

**判断标准**:
- 若 raw.max() ≈ 1 且 attrs 中 amplitude_train_max ≈ 57 → **二次归一化 bug**
- 若 raw.max() ≈ 57（匹配 attrs）→ 数据正常

**成本**: 5 分钟。

---

### 实验 B: 归一化量级对齐 (最高优先级)

**假设**: 保留全局归一化策略，但将输出从 [0,1] 对齐到 ≈ N(0,1) 量级后，训练能显著改善。

**背景**: `TFDDTTokenizer` 第一层是 Linear 投影（无 LayerNorm），对输入量级敏感。referVersion 的 zscore 使模型权重适应 N(0,1) 输入；H5 的 minmax 输出 [0,1] 输入使 Linear 层面对均值偏移 + 低方差。

**方法**: 在 `h5_dataset.py` 的 `normalize_global_minmax` 之后，增加一个可配置的后处理步骤：

- **变体 B1**: 中心化 + 扩展（不引入 per-sample 归一化）
  ```python
  csi_amp = (csi_amp - 0.5) * 6.0  # [0,1] → [-3, +3]
  ```
  均值 0，标准差 ≈ 0.2 × 6 ≈ 1.2（接近 zscore 量级）

- **变体 B2**: 全局 z-score（需要额外存 train_mean/train_std 到 H5 attrs）
  ```python
  csi_amp = (csi_amp - train_mean) / train_std
  ```
  全局统一做 zscore，保留动作间相对差异（不逐样本归一化）

- **变体 B3 (对照)**: 逐样本 z-score（等同 referVersion）
  ```python
  csi_amp = normalize_csi(csi_amp, mode="zscore")  # 即 (x - mean_sample) / std_sample
  ```

**对比**: B0 (原方案) vs B1 vs B2 vs B3，各训练 5 epoch。

**预测**: B1 和 B2 应该在保留"全局对齐"设计意图的同时，显著改善训练动态（train loss 下降更快，PCK 更高）。B3 应与 referVersion 性能一致（但牺牲了全局对齐）。

**成本**: 4 × 5 epoch = 约 20 epoch 训练时间。

---

### 实验 C: 划分作用域对齐

**假设**: 全局 shuffle vs 按 subject 分组 shuffle 导致 val set 成分不同。

**方法**: 将 `H5MMFiDataset._build_split` 改为全局 shuffle（与 MMFiDataset 一致）。

**成本**: 5 epoch 训练。

---

### 实验 D: 关键点坐标交叉验证

**假设**: NPZ 和 H5 中相同帧的 keypoints 坐标一致。

**方法**: 对同 env/subject/action/frame，同时从 NPZ 和 H5 提取 keypoints，计算差异。

**前提**: 需要能同时访问两种数据源。

**成本**: 5 分钟（若能同时访问）。

---

## 推荐执行顺序

```
实验 A → 实验 B → 实验 C → (实验 D，可选)
```

---

## 核心结论

修正 `差异 1` 的判断：两个管线的 CSI 时序处理完全相同（10→64 FFT 上采样），不存在信息损失。性能差距主要来自**归一化输出量级不匹配**——referVersion 的 zscore 使模型适应 N(0,1) 输入，而 H5 的 global_minmax 输出 [0,1] 导致 Linear 层面对均值偏移和低方差输入，削弱了梯度和收敛速度。在不改变全局归一化设计意图的前提下，可通过 B1 或 B2 变体将输出对齐到 N(0,1) 量级来验证。
