# MultiFormer 项目差异分析报告

**当前项目**: `multiformer` (D:\Files\Projects\PythonProjects\PaperResuming\multiformer)  
**参考项目**: `referVersion` (C:\Users\LuvRene\Downloads\referVersion)  
**对比日期**: 2026-05-13

---

## 核心结论

**模型架构完全一致。** `models/` 目录下的全部 6 个文件经过逐行比对，字节级别完全相同，无任何差异。两项目使用相同的 MultiFormer 模型结构。

所有差异集中在：数据管道（新增后端）、配置系统、训练入口、辅助脚本和文档。

---

## 一、模型文件对比（全部一致）

以下 6 个模型核心文件在两个项目中 **完全一致**：

| 文件 | 说明 | 状态 |
|------|------|------|
| `models/__init__.py` | 导出 MultiFormer | 一致 |
| `models/multiformer.py` | 顶层模型组装 (tokenizer → extractor → MSFN) | 一致 |
| `models/tfddt.py` | TFDDTTokenizer (时频双维token化) | 一致 |
| `models/attention_extractor.py` | DualAttentionExtractor + TransformerBlock + ReconstructionLayer | 一致 |
| `models/heatmap_decoder.py` | HeatmapDecoder (CNN 解码器: PCM/PAF 头) | 一致 |
| `models/papm.py` | PAPM (姿态注意力感知模块: 通道+空间注意力) | 一致 |
| `models/msfn.py` | MSFN (多阶段特征网络, 3阶段堆叠) | 一致 |

### 模型架构回顾

```
CSI Input (B, 64, 3, 114)
  └─ TFDDTTokenizer
       ├─ freq_tokens (B, 114, 1296)  — Linear(M·NR → embed_dim) + pos_embed
       └─ time_tokens (B, 64, 1296)   — Linear(NS·NR → embed_dim) + pos_embed
  └─ DualAttentionExtractor (depth=8)
       ├─ 8× TransformerBlock (freq)  → ReconstructionLayer → (B, 64, 36, 36)
       ├─ 8× TransformerBlock (time)  → ReconstructionLayer → (B, 64, 36, 36)
       └─ cat → features (B, 128, 36, 36)
  └─ MSFN (stages=3)
       ├─ Stage1: HeatmapDecoder → (pcm_1, paf_1)
       │   └─ PAPM_1: features × attention(pcm_1, paf_1)
       ├─ Stage2: HeatmapDecoder → (pcm_2, paf_2)
       │   └─ PAPM_2: features × attention(pcm_2, paf_2)
       └─ Stage3: HeatmapDecoder → (pcm_3, paf_3)
```

---

## 二、非模型文件差异

### 2.1 train.py

| 差异项 | 参考版本 | 当前版本 |
|--------|----------|----------|
| 导入 | 仅 `from data import MMFiDataset` | 增加 `from data.h5_dataset import H5MMFiDataset` |
| `build_dataset()` 返回类型 | `-> MMFiDataset` | `-> MMFiDataset \| H5MMFiDataset \| MemmapDataset` |
| `build_dataset()` 分发逻辑 | 直接构造 `MMFiDataset` | 根据 `cfg["dataset"]["type"]` 分发到三种后端 (`"h5"`, `"memmap"`, `"npz"`) |

### 2.2 data/__init__.py

| 差异项 | 参考版本 | 当前版本 |
|--------|----------|----------|
| 导入 | 仅导出 `MMFiDataset` | 增加 `from .h5_dataset import H5MMFiDataset` |
| `__all__` | `["MMFiDataset", ...]` | 增加 `"H5MMFiDataset"` |

### 2.3 data/csi_preprocess.py

当前版本新增函数 `normalize_global_minmax()` （约12行）：

```python
def normalize_global_minmax(x, train_min, train_max, eps=1e-6):
    """使用预计算的训练集统计量进行全局 min-max 归一化"""
    x = sanitize_csi(x)
    denom = train_max - train_min
    if not np.isfinite(denom) or denom < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - train_min) / (denom + eps)).astype(np.float32)
```

参考版本仅有 `normalize_csi()`（逐样本 z-score 归一化），没有全局归一化函数。

### 2.4 配置文件差异 (configs/*.yaml)

#### 配置参数对比

| 配置项 | 参考版本 | 当前版本 |
|--------|----------|----------|
| `dataset.type` | 不存在（始终 NPZ） | `"h5"` / `"memmap"` / `"npz"` |
| `dataset.root` | `/root/autodl-fs/...` (服务器路径) | `/data/WiFiPose/dataset/...` (不同服务器路径) |
| `dataset.envs` | `["E01"]` 等 | `["env1"]` 等 |
| `dataset.max_samples` | 显式设为 `null` | 不存在该字段 |
| `csi.normalize` | 全部为 `"zscore"` | `"global_minmax"` (E01/E02/E03/E04), `"global_zscore"` (E01_B1), `"zscore"` (E01_B2) |
| `train.early_stop_*` | E01: delta=0.001, patience=1; E02/E03: 未启用 | E01: delta=0.00001, patience=3, warmup=10; E02/E03: 未启用; E04: delta=0.001 |

#### 当前版本新增配置文件

| 文件 | 用途 | 说明 |
|------|------|------|
| `E01_B1.yaml` | memmap 数据集 + `global_zscore` 归一化 | 归一化消融实验变体 B1 |
| `E01_B2.yaml` | memmap 数据集 + `zscore` 归一化 | 归一化消融实验变体 B2 |

---

## 三、当前版本新增文件

### 3.1 新增数据管道模块

| 文件 | 说明 |
|------|------|
| `data/h5_dataset.py` | HDF5 后端数据集 `H5MMFiDataset`。从单个 H5 文件读取原始 CSI `(3, 114, 10)`，支持 4 种归一化模式（global_minmax / global_zscore / zscore / none），内部做时间重采样 (10→64)、转置、归一化、COCO17→OpenPose18 转换，并为关键点和元数据做内存预加载 |
| `data/memmap_dataset.py` | 内存映射 `.npy` 后端数据集 `MemmapDataset`。读取预归一化的 CSI `(N, 64, 3, 114)` 和相关键点 `(N, 18, 2)`，利用 OS 页缓存实现高效随机读取 |

### 3.2 新增脚本

| 文件 | 说明 |
|------|------|
| `scripts/preprocess_mmfi.py` | MM-Fi 原始数据预处理流水线 |
| `scripts/build_h5.py` | 从 NPZ 文件构建 HDF5 数据集 |
| `scripts/build_memmap.py` | 构建内存映射 .npy 数据集 |
| `scripts/diagnostic/check_h5_csi_range.py` | H5 CSI 范围诊断工具 |
| `scripts/diagnostic/verify_kpts_fix.py` | 关键点双重归一化修复验证 |

### 3.3 新增文档

| 文件 | 说明 |
|------|------|
| `CLAUDE.md` | 项目指南（供 Claude Code 使用） |
| `ablation_plan.md` | 消融实验计划 |
| `task_plan.md` | 任务规划 |
| `progress.md` | 进度追踪 |
| `docs/superpowers/plans/2026-05-12-h5-preprocessing-optimization.md` | H5 预处理优化方案 |

---

## 四、参考版本独有文件（输出产物类）

以下文件仅存在于参考版本，属于训练输出产物或参考材料，非源代码层面的差异：

| 文件 | 类型 |
|------|------|
| `README.md` | 英文 README |
| `复现计划.md` | 中文复现计划 |
| `参考论文/鸿鹄实验参考论文.pdf` | 参考论文 PDF |
| `outputs/E01~E04/train_log.csv` | 训练日志（4个文件） |
| `outputs/viz/E01~E04_quickcheck/*.png` | 可视化对比图（32张） |

---

## 五、完全一致的文件

以下文件在两个项目中 **逐字节相同**，无需关注：

- `eval.py` — 评估脚本
- `visualize.py` — 可视化脚本
- `data/heatmap_gt.py` — 热力图真值生成（COCO17 ↔ OpenPose18 转换、PCM/PAF 构建）
- `data/mmfi_dataset.py` — NPZ 格式数据集类
- `decode/__init__.py` — 解码模块导出
- `decode/pose_decoder.py` — 姿态解码（NMS + Hungarian 匹配）
- `utils/__init__.py` — 工具模块导出
- `utils/metrics.py` — PCK 指标计算
- `utils/viz.py` — 骨架渲染
- `requirements.txt` — 依赖列表
- `scripts/train_all_envs.sh` — Linux 批量训练脚本
- `scripts/train_all_envs.bat` — Windows 批量训练脚本

---

## 六、差异总结

| 类别 | 差异数量 | 影响范围 |
|------|----------|----------|
| 模型架构 | **0** | 无差异，完全一致 |
| 训练入口 | 3 处修改 | `train.py` 支持多数据后端分发 |
| 数据管道 | 2 个新模块 + 1 个新函数 | 新增 H5 和 memmap 两种数据集后端 |
| 配置文件 | 7 个文件有差异 | 数据路径、环境命名、归一化策略、早停参数均不同 |
| 脚本 | 5 个新脚本 | 数据预处理、构建和诊断工具 |
| 文档 | 5 个新文档 | 项目指南、消融计划、进度追踪等 |
| 输出产物 | ~40 个文件 | 仅参考版本有（训练日志、可视化图像） |

**核心结论**：当前项目与参考项目使用**完全相同的模型架构**（Transformer + Dual Attention + MSFN），所有差异都属于工程优化层面——扩展数据管道支持多种后端（NPZ/HDF5/Memmap）、增加全局归一化策略、以及配套的构建脚本和诊断工具。模型结构本身无需关注，差异聚焦于数据加载和配置管理。
