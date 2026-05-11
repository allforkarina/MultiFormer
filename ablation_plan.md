# MultiFormer vs referVersion 差异分析与消融实验计划

## 性能对比

| 指标 | referVersion (NPZ+zscore) | multiformer (H5+global_minmax) |
|------|--------------------------|-------------------------------|
| epoch 1 PCK@20 | **0.827** | ~0.60 (推测) |
| epoch 13 PCK@20 | **0.910** | ~0.55-0.63 (epoch 21-52) |
| epoch 1 train_loss | 0.0284 | ~0.011 (epoch 21) |
| epoch 13 train_loss | 0.0156 | ~0.009 (epoch 52, 缓慢下降) |

**结论**: multiformer 的 PCK@20 比 referVersion 低约 0.25-0.30，差距巨大。

---

## 两个项目的代码一致性确认

以下模块 **完全相同** (diff 验证):

- `models/tfddt.py`
- `models/attention_extractor.py`
- `models/heatmap_decoder.py`
- `models/papm.py`
- `models/msfn.py`
- `models/multiformer.py`
- `decode/pose_decoder.py`
- `utils/metrics.py`
- `data/heatmap_gt.py`
- `eval.py`
- `visualize.py`
- `train.py` (仅 `build_dataset` 分发逻辑不同)

---

## 差异分析

### 差异 1: CSI 归一化方式 ⭐⭐⭐ (最高优先级)

| 维度 | referVersion | multiformer |
|------|-------------|-------------|
| 方式 | 逐样本 z-score | 全局 min-max |
| 公式 | `(x - mean_sample) / std_sample` | `(x - global_min) / (global_max - global_min)` |
| 每样本均值 | ~0 | 取决于样本 |
| 每样本方差 | ~1 | 取决于样本 |
| 配置 | `normalize: "zscore"` | `normalize: "global_minmax"` |

**分析**: 逐样本 z-score 确保每个输入样本具有相同的统计分布（均值≈0, 方差≈1），这对于 transformer 模型的训练稳定性至关重要。全局 min-max 将所有值压缩到 [0, 1]，但不同样本的有效范围可能差异很大——WiFi CSI 信号强度受距离、遮挡等因素影响，全局 min-max 无法消除这种样本间差异。

**预期影响**: **极大**。这是最可能导致性能差距的因素。模型设计时使用的训练数据经过 z-score 归一化，如果输入分布发生显著偏移，模型在第一层就会得到完全不同的激活模式。

### 差异 2: CSI 时序重采样方式 ⭐⭐

| 维度 | referVersion | multiformer |
|------|-------------|-------------|
| 时机 | 离线预处理，存储在 NPZ 中 | 在线，每次 `__getitem__` 时执行 |
| 输入形状 | `(NR, NS, M0)` 原始 → FFT resample | `(3, 114, 10)` → FFT resample |
| sanitize 调用 | 1次 (resample 前) | 2次 (resample 前后各一次) |
| 数据源 | NPZ 中的 csi 数组 | HDF5 中的 csi_amplitude 数组 |

**分析**: 两者都使用 `scipy.signal.resample` 做 FFT 重采样。但 `_resample_time_h5` 在 resample 之后额外调用了一次 `sanitize_csi()`。`resample` 输出本身应为有限值，额外 sanitize 不太可能引入显著差异。真正的差异可能在于源数据——NPZ 中的 csi 可能经过了额外处理（如截断、滤波）。

**预期影响**: **中等**。resample 本身的差异应很小，但源数据预处理链路的差异可能重要。

### 差异 3: CSI 源数据格式 ⭐⭐⭐

| 维度 | referVersion | multiformer |
|------|-------------|-------------|
| 数组键 | `data["csi"]` | `h5_file["csi_amplitude"]` |
| 已预处理 | 是 (已重采样、转置、z-score归一化) | 否 (原始数据) |
| 形状 | `(N, 64, 3, 114)` | `(3, 114, 10)` |
| 预归一化 | 逐样本 z-score 已固化在文件中 | 无 |

**分析**: referVersion 使用的 NPZ 文件中的 CSI 数据已经经过完整的 `preprocess_csi_amp()` 管线：
```
原始 CSI → resample_time(10→64) → transpose(2,0,1) → normalize_csi(zscore)
```
而 H5 的原始 `csi_amplitude` 没有经过这些处理。如果 NPZ 的预处理管线中还包括了其他未记录的步骤（如按天线归一化、去趋势等），这些差异会累积。

**预期影响**: **极大**。如果 H5 中的 `csi_amplitude` 和 NPZ 中的 `csi` 是同一个原始数据源的不同版本，那么预处理管线的差异直接导致模型看到不同的输入。

### 差异 4: 关键点坐标处理 ⭐⭐

| 维度 | referVersion | multiformer |
|------|-------------|-------------|
| 原始格式 | OpenPose18 `(18, 2)` | COCO17 `(17, 2)` |
| 坐标空间 | 已在 pose_range [-0.8, 0.8] | [0, 1] 归一化 → 映射到 pose_range |
| 转换 | 无 | `coco17_to_openpose18()` + `_normalize_keypoints()` |

**分析**: 此差异已经修复过一次（二次归一化 bug: 80d3079）。当前代码中 `_normalize_keypoints` 正确地将 [0, 1] 映射到 [-0.8, 0.8]：
```python
kpts = kpts * (hi - lo) + lo  # [0, 1] * 1.6 - 0.8 = [-0.8, 0.8]
```
但需要验证 COCO17→OpenPose18 的映射是否正确。重点检查：
- `coco17_to_openpose18()` 映射表是否与 H5 中 COCO17 的关节顺序一致
- neck 关键点 (索引 1) 由左右肩中点计算是否正确

**预期影响**: **中等**。如果验证脚本通过（keypoints 范围合理、PCM 峰值分散），则关键点处理正确。

### 差异 5: 环境命名 ⭐⭐

| 维度 | referVersion | multiformer |
|------|-------------|-------------|
| 配置 envs | `["E01"]` | `["env1"]` |
| 来源 | NPZ 目录名 | H5 中 environment 字段 |

**分析**: NPZ 目录结构使用 `E01`, `E02` 等命名。H5 文件中的 `environment` 字段可能使用不同命名（如 `"env1"`, `"E01"`, `"E1"` 等）。如果命名不匹配，环境过滤可能返回错误数量的样本。

**预期影响**: **中等**。如果过滤不匹配导致使用了错误的环境数据或混合了多个环境。

### 差异 6: 训练/验证集划分逻辑 ⭐

| 维度 | referVersion | multiformer |
|------|-------------|-------------|
| 划分方式 | 全局样本列表 shuffle + 80/20 分割 | 按 subject 分组，每组内 shuffle + 80/20 分割 |
| 种子 | 42 | 42 |

**分析**: 两种方式在 `same_subject_random` 协议下理论上应该产生相同的随机划分（相同种子）。但由于采样列表构建方式不同（NPZ 文件枚举 vs H5 索引过滤），样本顺序可能不同，导致划分结果略有不同。

**预期影响**: **低**。不应该解释 0.25 PCK 的差距。

### 差异 7: CSI 归一化函数 `sanitize_csi` 中的 NaN 填充 ⭐

| 维度 | referVersion | multiformer |
|------|-------------|-------------|
| 填充方式 | `np.median(x[finite])` | 相同 |

**分析**: 两个项目的 `sanitize_csi` 完全相同。

**预期影响**: **无**。

### 差异 8: `preprocess_csi_amp` 逐样本 z-score 的特殊值处理 ⭐⭐

**分析**: `normalize_csi` 在 std 过小或 hi-lo 过小时，会返回全零数组：
```python
if not np.isfinite(std) or std < eps:
    return np.zeros_like(x, dtype=np.float32)
```
如果 H5 中某些样本的 CSI 幅度变化极小（如环境噪声底），全局 min-max 会将这些样本映射到接近零的值，而逐样本 z-score 可能会返回全零数组。这种行为差异可能导致某些样本在两个管线中的表示完全不同。

**预期影响**: **低-中等**，取决于数据中异常样本的比例。

---

## 消融实验设计

### 实验 A: 归一化方式切换 (最高优先级)

**假设**: 全局 min-max 归一化导致输入分布不一致，是性能差异的主要原因。

**方法**:
1. 在 `h5_dataset.py` 的 `__getitem__` 中，将 `normalize_global_minmax()` 替换为 `normalize_csi(x, mode="zscore")`
2. 保持其他所有代码不变
3. 使用相同配置 `configs/E01.yaml`（仅修改 `normalize` 字段为 `zscore`）
4. 训练 10 个 epoch，对比 PCK@20 曲线

**预测**: 如果切换到逐样本 z-score 后 PCK@20 显著提升（接近 0.8+），则确认归一化是主要原因。

**实现注意事项**:
- 修改 `__getitem__` 中的第 163-167 行
- 无需修改 HDF5 文件
- 可先跑 2-3 epoch 快速验证方向

### 实验 B: CSI 源数据交叉验证 (高优先级)

**假设**: H5 中的 `csi_amplitude` 和 NPZ 中的 `csi` 是不同的预处理版本。

**方法**:
1. 在本地有 NPZ 数据的环境下，同时加载 H5 和 NPZ 数据
2. 对同一帧 (相同 env, subject, action, frame_idx):
   - 从 NPZ 读取 csi → shape (64, 3, 114)
   - 从 H5 读取 csi_amplitude → 经过完整的 H5 管线处理 → shape (64, 3, 114)
3. 计算两组数据的统计差异 (均值差、标准差差、相关系数)
4. 用 referVersion 训练好的模型分别推理两组数据，对比 PCM 输出差异

**预测**: 如果两组数据在统计上高度相关（相关系数 > 0.95），则源数据不是问题。如果差异很大，说明 `build_h5_dataset` 的预处理与 NPZ 的 `preprocess_csi_amp` 不一致。

**实现注意事项**:
- 需要同时能访问 H5 文件和 NPZ 文件的环境
- 可写一个独立脚本来执行此对比

### 实验 C: 时序重采样方法对比 (中等优先级)

**假设**: 在线重采样 (H5) 和离线重采样 (NPZ) 产生不同结果。

**方法**:
1. 构建一个测试脚本，对相同原始数据分别执行:
   a. NPZ 路径: `preprocess_csi_amp(csi_raw)` — 使用 `csi_preprocess.py` 的 `resample_time`
   b. H5 路径: `_resample_time_h5(csi_raw)` — 使用 `h5_dataset.py` 的方法
2. 比较两种方法输出的数值差异 (MAE, max abs diff)
3. 在相同原始数据上，分别将两种重采样结果送入模型，比较最终 PCM 输出

**预测**: 两者的数值差异应该很小 (MAE < 1e-5)，对模型输出的影响微乎其微。

### 实验 D: 环境过滤命名验证 (中等优先级)

**假设**: H5 中 environment 字段的值与配置中的 `env1` 不匹配。

**方法**:
1. 编写脚本打印 H5 文件中所有唯一的 `environment` 值和 `sample` 值:
   ```python
   with h5py.File(h5_path, "r") as f:
       print("Unique environments:", set(f["environment"][:]))
       print("Unique samples:", set(f["sample"][:]))
   ```
2. 验证 `envs: ["env1"]` 过滤后实际包含了多少样本
3. 确认 `sample` 字段命名与 NPZ 的 subject 目录命名一致 (如 `S01`, `S02`)

**预测**: 环境命名可能是 `E01` 而非 `env1`。但这不会导致更差的性能（只会导致过滤掉所有数据），所以如果有数据被加载，命名就是匹配的。

### 实验 E: 关键点转换完整性验证 (中优先级)

**假设**: COCO17→OpenPose18 转换可能在某些边缘情况下产生错误的关键点。

**方法**:
1. 扩展 `scripts/verify_kpts_fix.py`，增加以下检查:
   - 所有 18 个关节的有效率（当前 NPZ 管线 vs H5 管线）
   - Neck 关键点（索引 1）是否在左右肩中点
   - 对称关节（左右肩、左右髋等）的分布是否对称
   - 与 NPZ 中同帧的关键点进行直接对比
2. 可视化对比 20 个样本的 PCM 热力图（NPZ vs H5）

**预测**: 如果 PCM 峰值分散且关键点范围在 [-0.8, 0.8]，则转换正确。

### 实验 F: 全链路端到端对比 (确认性实验)

**假设**: 上述实验确认的差异能够完整解释性能差距。

**方法**:
1. 在能同时运行两个管线的环境中:
   a. 用 referVersion 的 NPZ 管线训练 10 epoch
   b. 用 multiformer 的 H5 管线（应用修复）训练 10 epoch
   c. 用 NPZ 数据但使用 multiformer 的 train.py 训练 10 epoch
2. 三条训练曲线绘制在同一张图上对比
3. 确认 (a) 和 (c) 性能一致，(b) 在应用归一化修复后达到相近性能

**预测**: 切换归一化方式后的 H5 管线性能应接近 NPZ 管线。

---

## 实验执行优先级

```
1. 实验 A (归一化)       ← 可能性最高，最容易验证
2. 实验 D (环境命名)      ← 快速排查，5分钟即可完成
3. 实验 B (源数据对比)    ← 需要能访问两种数据源
4. 实验 E (关键点验证)    ← 已有脚本基础
5. 实验 C (重采样方法)    ← 预期影响小，可最后做
6. 实验 F (全链路)        ← 确认性实验，前提是前面找到问题
```

---

## 推荐执行顺序

**第一步**: 实验 D（先确认数据加载正确）
**第二步**: 实验 A（最可能的根因）
**第三步**: 根据实验结果决定是否继续 B/C/E/F
