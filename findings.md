# 诊断发现 — H5 数据管线 Bug

## 根因：关键点二次归一化

### 数据流追溯

```
1. HDF5 构建时 (build_h5_dataset):
   原始 COCO17 像素关键点 → 除以 axis_max (626.7, 475.9) → [0, 1] 区间
   写入 HDF5: keypoints 已经是 [0, 1] 归一化坐标

2. H5MMFiDataset 加载时 (_normalize_keypoints):
   读取 [0, 1] 关键点 → 再次除以 axis_max → ≈ 0.0016 → * 1.6 - 0.8 → ≈ -0.8
   ⚡ 所有关键点坍缩到 heatmap 原点 (0, 0)
```

### 症状验证

| 观察 | 解释 |
|------|------|
| loss = 0.000032 (epoch 3+) | 所有高斯峰重叠在 (0, 0)，模型轻易学会退化模式 |
| val loss = 0.00003 | 验证集同样退化 |
| PCK@20/30/40 = 0.0000 | 预测全部在 (-0.8, -0.8)，GT 之间 torso_scale ≈ 0.0005，distance ≈ 0.002 > limit ≈ 0.0001 |

### HDF5 属性确认

```yaml
keypoint_normalization: train_axis_max
keypoint_x_scale: 626.7066    # 构建时已用于归一化
keypoint_y_scale: 475.9098    # 构建时已用于归一化
amplitude_train_min: 0.0
amplitude_train_max: 57.1957
```

### 诊断结果

```
原始关键点范围: [0.3817, 0.9188] → [0, 1] ✓（已归一化）
PCM 峰值: 0.9968 ✓（高斯峰正确）
初始 loss: 0.031 ✓（未训练模型合理）
训练后 loss: 0.000032 ✗（退化到平凡解）
PCK: 0.0000 ✗
```

## 修复方案

移除 `_normalize_keypoints` 中的 scale 除法。关键点已在 [0, 1]，只需做 pose_range 映射：

```python
# Before (Bug):
kpts[:, 0] = kpts[:, 0] / self.kp_x_scale  # double-normalizes!
kpts[:, 1] = kpts[:, 1] / self.kp_y_scale
kpts = kpts * (hi - lo) + lo

# After (Fix):
kpts = kpts * (hi - lo) + lo  # [0,1] → [pose_min, pose_max]
```
