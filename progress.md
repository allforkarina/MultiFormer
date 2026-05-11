# 修复进度

## 会话 2026-05-11

### 诊断阶段
- [x] 审查 csi_preprocess.py — CSI 归一化正确
- [x] 审查 h5_dataset.py — 发现 `_normalize_keypoints` 二次归一化
- [x] 审查 heatmap_gt.py — COCO17→OpenPose18 映射正确
- [x] 审查 mmfi_dataset.py — NPZ 管线作为对照
- [x] 审查 train.py / eval.py — 训练和评估管线正确
- [x] 审查模型架构 — shape 匹配
- [x] 审查 metrics.py — PCK 计算正确
- [x] 审查 pose_decoder.py — 解码正确
- [x] 确认 HDF5 attrs — amplitude_train_min/max, kp_x/y_scale 存在且合理
- [x] 确认无遮挡关键点 — 0%
- [x] 用户诊断: 原始关键点 [0, 1] 范围 — 已归一化
- [x] **根因确认**: `_normalize_keypoints` 对已归一化关键点再次除法

### 修复阶段
- [x] 修改 h5_dataset.py `_normalize_keypoints` 移除 scale 除法
- [x] 创建验证脚本 scripts/verify_kpts_fix.py
- [ ] 在 Linux 服务器运行验证脚本确认修复生效
- [ ] 重新训练验证 loss 正常下降 + PCK > 0
