# MultiFormer MM-Fi 复现工程

这个目录按 `复现计划.md` 实现了 MultiFormer 学生网络：MM-Fi CSI 读取、TFDDT、双路 Transformer、MSFN 三阶段 PCM/PAF 输出、PCK 评估和 GT/Pred 骨架可视化。

## 环境

```powershell
pip install -r requirements.txt
```

推荐在已有 PyTorch 环境里运行；数据集默认路径写在 `configs/default.yaml`：

```yaml
dataset:
  root: "D:/MM-Fi数据集/MMFi_Dataset"
```

## 快速检查

```powershell
python -c "from models.multiformer import MultiFormer; import torch; m=MultiFormer(embed_dim=128,num_heads=4,depth=1); x=torch.randn(2,64,3,114); print([(p.shape,a.shape) for p,a in m(x)])"
```

期望输出 3 组：

```text
(torch.Size([2, 19, 36, 36]), torch.Size([2, 38, 36, 36]))
```

## 训练

```powershell
python train.py --config configs/default.yaml
```

先跑小样本：

```powershell
python train.py --config configs/default.yaml --max-train-samples 256 --max-val-samples 64
```

单 batch overfit 检查：

```powershell
python train.py --config configs/default.yaml --max-train-samples 32 --max-val-samples 32 --overfit-batch
```

## 评估

```powershell
python eval.py --config configs/default.yaml --ckpt outputs/best.pth
```

## 可视化

```powershell
python visualize.py --config configs/default.yaml --ckpt outputs/best.pth --n-per-env 8
```

结果输出到 `outputs/viz/E01` 到 `outputs/viz/E04`。
