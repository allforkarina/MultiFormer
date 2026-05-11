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

---

## 会话 2026-05-12 — 性能差距诊断

### 差异分析 (v1 → v2 → v3)
- [x] 对比 referVersion 和 multiformer 的文件级差异 (diff 验证)
- [x] 确认模型/decode/metrics/heatmap_gt 代码完全相同
- [x] 分析所有差异并创建 ablation_plan.md v1
- [x] 读取 scripts/preprocess_mmfi.py，确认 CSI 时序处理完全相同
- [x] 排除"时序分辨率不一致"假说 (v3 修正)
- [x] 确认归一化输出量级不匹配为最可能根因

### 规划输出
- [x] 创建 ablation_plan.md (v1 → v2 → v3，每次修正)
- [x] 创建 task_plan.md — 归一化消融实验实现计划
- [x] 更新 progress.md

### 实验 A 实现
- [x] 创建 `scripts/check_h5_csi_range.py` — H5 CSI 二次归一化诊断脚本
- [x] 在 Linux 服务器执行脚本，确认 **CSI 已被预归一化到 [0,1]**
  - raw max=0.9354, attr amplitude_train_max=57.1957 → 二次归一化确认
  - `amplitude_normalization: train_global_minmax` → 与关键点 bug 同性质

### 实验 B 实现
- [x] 修复 `data/h5_dataset.py` 二次归一化:
  - `__init__` 检测 `amplitude_normalization` attr
  - `__getitem__` 跳过已归一化数据的 `global_minmax` 调用
- [x] 添加归一化分发逻辑: `global_minmax` / `global_zscore` / `zscore` / `none`
- [x] 创建 `configs/E01_B1.yaml` (global_zscore 变体)
- [x] 创建 `configs/E01_B2.yaml` (zscore 变体)
- [ ] 在 Linux 服务器训练三个变体各 5 epoch，对比 PCK@20 曲线
