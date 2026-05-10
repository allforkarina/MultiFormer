# MultiFormer 数据加载适配方案

## 目标
将当前项目的 NPZ 数据加载方式对齐到 Linux 服务器上的 HDF5 数据集，保持所有核心模块（models/, decode/, utils/）不变。

## 差异总览

| 维度 | 当前 (NPZ) | 目标 (HDF5) |
|------|-----------|------------|
| 路径 | `root/E0X/S0X/A0X.npz` | `/data/WiFiPose/dataset/mmfi_pose.h5` |
| CSI 形状 | `(64, 3, 114)` 已重采样 | `(3, 114, 10)` 原始 |
| 关键点 | `(18, 2)` OpenPose18 | `(17, 2)` COCO17 |
| 归一化 | 逐样本 z-score | 全局 min-max (H5属性) |
| 划分 | subject枚举 + shuffle | 从sample字段构建same_subject_random |
| 元数据 | env/subject/action | action/sample/environment |

## 决策记录

1. **输入**: 仅振幅 (不改模型)
2. **归一化**: 全局 min-max，使用 HDF5 预计算统计量
3. **划分**: same_subject_random，sample→subject 映射
4. **环境过滤**: 通过 config 中的 envs 列表过滤

## 修改清单

### 新建文件
- `data/h5_dataset.py` — HDF5 数据集类，输出格式对齐 MMFiDataset

### 修改文件
- `data/csi_preprocess.py` — 新增 10→64 时间重采样 + 全局 min-max 归一化
- `data/__init__.py` — 导出 H5MMFiDataset
- `train.py` — build_dataset 支持 H5 数据源分发
- `configs/default.yaml` — Linux 服务器路径和 dataset_type 字段

### 不动文件
- `models/` — 全部
- `decode/pose_decoder.py`
- `utils/metrics.py`, `utils/viz.py`
- `data/heatmap_gt.py`

## 数据流

```
HDF5 mmfi_pose.h5
  ├─ csi_amplitude: (N, 3, 114, 10)  cleaned_raw
  ├─ keypoints:      (N, 17, 2)       COCO17
  ├─ action/sample/environment/frame_id 元数据
  └─ train/val/test_indices (action_env, frame_random)
       ↓
  H5MMFiDataset.__getitem__
  1. 读取单帧 csi_amp(3,114,10), kpts(17,2)
  2. 时间重采样: (3,114,10) → (3,114,64)
  3. 转置: (3,114,64) → (64,3,114)
  4. 全局min-max归一化
  5. COCO17→OpenPose18 转换
  6. 构建 PCM(19,36,36), PAF(38,36,36)
  7. 返回 {"csi": (64,3,114), "kpts18": (18,2), "pcm":..., "paf":..., "meta":...}
       ↓
  MultiFormer.forward()
```
