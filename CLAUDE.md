# MultiFormer — WiFi-based Human Pose Estimation

## Project overview

MultiFormer is a Transformer-based architecture for 2D human pose estimation from WiFi CSI (Channel State Information) signals. The model takes CSI amplitude data as input and outputs Part Confidence Maps (PCM) and Part Affinity Fields (PAF), which are then decoded into 18-keypoint human poses (OpenPose format).

## Dataset

The project uses the **MM-Fi** dataset with memory-mapped `.npy` files for efficient training I/O.

### Memmap dataset (`data/memmap_dataset.py`)

Memory-mapped `.npy` files built by `scripts/build_memmap.py`. CSI is pre-normalized at build time (three variants stored: `global_minmax`, `global_zscore`, `zscore`). At training time, the dataset selects the variant specified by `csi.normalize` in the config.

- **Format**: Directory of `.npy` files: `csi_global_minmax.npy`, `csi_global_zscore.npy`, `csi_zscore.npy`, `ground_truth.npy`, `meta.npz`, `stats.json`
- **CSI shape**: `(N, 64, 3, 114)` — pre-resampled to 64 time packets
- **Normalization**: Pre-computed at build time; selected at dataset init via `normalize` parameter
- **Keypoints**: Pre-converted COCO17→OpenPose18, stored as pixel coordinates
- **I/O**: Uses `np.memmap` for zero-copy reads; benefits from OS page cache
- **Use case**: Primary training backend for all environments

### Data split logic

The dataset uses **per-subject grouped random split**:
1. Group all frame indices by subject
2. For each subject, shuffle its frames and split 80/20 (train/val) using `random_val_ratio`
3. `split="train"` returns the 80% portion; `split="test"` returns the 20% portion
4. `split="all"` returns all frames (for visualization)

This ensures frames from the same subject don't leak between train and val.

## Model architecture

```
CSI input (B, 64, 3, 114)
  └─ TFDDTTokenizer                    # Time-Frequency Dual-Dimensional Tokenization
       ├─ Optional subcarrier projection: Linear(114→64) when subcarrier_mode="learned64"
       ├─ freq tokens (B, NS, embed_dim) via Linear(M·NR → embed_dim) + freq_pos_embed
       └─ time tokens (B, M, embed_dim) via Linear(NS·NR → embed_dim) + time_pos_embed
  └─ DualAttentionExtractor            # 8 stacked TransformerBlocks per branch + ReconstructionLayer
       ├─ freq_blocks (8× TransformerBlock) → freq_recon → freq_map (B, 64, 36, 36)
       ├─ time_blocks (8× TransformerBlock) → time_recon → time_map (B, 64, 36, 36)
       └─ cat → features (B, 128, 36, 36)
  └─ MSFN                              # Multi-Stage Feature Network (3 stages)
       ├─ Stage 1: HeatmapDecoder → pcm (B, 19, 36, 36), paf (B, 38, 36, 36)
       ├─ PAPM: channel + spatial attention refines features using stage-1 heatmaps
       ├─ Stage 2: HeatmapDecoder → pcm, paf
       ├─ PAPM: refine using stage-2 heatmaps
       └─ Stage 3: HeatmapDecoder → pcm, paf
```

### Key components

| Component | File | Description |
|-----------|------|-------------|
| `TFDDTTokenizer` | `models/multiformer.py` | Splits CSI into time and frequency token sequences with learnable position embeddings |
| `DualAttentionExtractor` | `models/multiformer.py` | Parallel Transformer encoders for time and frequency domains, each with 8 blocks |
| `ReconstructionLayer` | `models/multiformer.py` | Projects token sequences back to 2D feature maps (36×36) |
| `MSFN` | `models/multiformer.py` | 3-stage refinement with PAPM attention between stages |
| `PAPM` | `models/multiformer.py` | Pose Attention Perception Module: channel attention + spatial attention guided by previous stage heatmaps |
| `HeatmapDecoder` | `models/multiformer.py` | CNN decoder: ConvTranspose2d → BatchNorm → ReLU → Conv2d → output (19 PCM + 38 PAF channels) |

### Output format

- **PCM** (Part Confidence Maps): 19 channels (18 keypoints + background), 36×36 spatial
- **PAF** (Part Affinity Fields): 38 channels (19 limbs × 2 for x,y vectors), 36×36 spatial

## Configuration

All configs are YAML files in `configs/`. Key sections:

| Section | Key fields |
|---------|-----------|
| `dataset` | `root`, `envs`, `train_subjects`, `test_subjects`, `random_val_ratio`, `seed` |
| `csi` | `time_packets` (64), `rx_antennas` (3), `subcarriers` (114), `subcarrier_mode` ("keep"/"resample64"/"learned64"), `normalize` ("global_minmax"/"global_zscore"/"zscore") |
| `heatmap` | `size` (36), `sigma` (1.5), `paf_width` (1.0), `pose_min`/`pose_max` (-0.8/0.8) |
| `model` | `embed_dim` (1296), `num_heads` (8), `depth` (8), `dropout` (0.1), `recon_channels` (64), `feature_channels` (128), `decoder_hidden` (512), `stages` (3) |
| `train` | `epochs`, `batch_size`, `lr`, `lr_step`, `lr_gamma`, `num_workers`, `device`, `output_dir`, `early_stop_*` |
| `eval` | `batch_size`, `pck_thresholds`, `peak_threshold` |
| `visualize` | `n_per_env`, `seed` |

### Config files

| File | Environment | Subjects | Normalization | Notes |
|------|------------|----------|---------------|-------|
| `default.yaml` | env1 | S01-S10 | global_minmax | Default config |
| `E01.yaml` | env1 | S01-S10 | global_minmax | Baseline |
| `E01_B1.yaml` | env1 | S01-S10 | global_zscore | Ablation: zscore normalization |
| `E01_B2.yaml` | env1 | S01-S10 | zscore | Ablation: per-sample zscore |
| `E02.yaml` | env2 | S11-S20 | global_minmax | Cross-env |
| `E03.yaml` | env3 | S21-S30 | global_minmax | Cross-env |
| `E04.yaml` | env4 | S31-S40 | global_minmax | Cross-env |

### Normalization variants

The `build_memmap.py` script pre-computes three normalization variants:

| Config value | Description | Statistics source |
|-------------|-------------|-------------------|
| `global_minmax` | Min-max scaling to [0, 1] | Global min/max from train subjects |
| `global_zscore` | Z-score standardization | Global mean/std from train subjects |
| `zscore` | Per-sample z-score | Per-frame mean/std |

## Training

### Entry point

```bash
python train.py --config configs/E01.yaml
```

Additional flags:
- `--max-train-samples N` / `--max-val-samples N`: Limit dataset size for debugging
- `--overfit-batch`: Overfit on a single batch (200 steps per epoch)

### Training loop

1. `build_dataset(cfg, "train")` creates `MemmapDataset` with `split="train"`
2. `build_dataset(cfg, "test")` creates `MemmapDataset` with `split="test"` for validation
3. Multi-stage MSE loss: sum of MSE(pcm_pred, pcm_gt) + MSE(paf_pred, paf_gt) across all stages
4. Adam optimizer with StepLR scheduler
5. Early stopping: monitors train_loss improvement; stops if improvement < `early_stop_min_delta` for `early_stop_patience` consecutive epochs (after `early_stop_warmup`)

### Evaluation metrics

- **PCK** (Percentage of Correct Keypoints): computed at thresholds [0.2, 0.3, 0.4] (normalized by person bounding box)
- Pose decoding: `decode_single_person()` finds peaks in PCM heatmaps above `peak_threshold`

## Scripts

### `scripts/build_memmap.py`

Three-phase pipeline to build memory-mapped `.npy` files from raw MM-Fi data:

1. **Phase 1 — Scan**: Walk the raw dataset directory (`{ACTION}/{SUBJECT}/`), collect all `.mat` (CSI) and `.npy` (pose2d) file paths
2. **Phase 2 — Process**: For each frame, resample CSI from 10→64 time packets via FFT, convert COCO17→OpenPose18 keypoints
3. **Phase 3 — Normalize**: Compute statistics from train subjects only, then produce three variants:
   - `csi_global_minmax.npy`: `(x - min) / (max - min)`
   - `csi_global_zscore.npy`: `(x - mean) / std`
   - `csi_zscore.npy`: per-sample `(x - mean_i) / std_i`

Key arguments:
- `--src`: Raw MM-Fi dataset path (default: `/data/WiFiPose/dataset/dataset`)
- `--dst`: Output directory (default: `/data/WiFiPose/dataset/mmfi_pose_v3`)
- `--train-subjects`: Subjects used for normalization statistics (default: S01-S10)
- `--pose-min`/`--pose-max`: Keypoint coordinate range for heatmap mapping (default: -0.8/0.8)
- `--workers`: Parallel processing workers (default: 4)

### `scripts/train_all_envs.sh` / `train_all_envs.bat`

Orchestration scripts that run training sequentially for all 4 environments (E01-E04).

## Project structure

```
multiformer/
├── train.py                  # Training entry point
├── eval.py                   # Evaluation script (PCK metrics)
├── visualize.py              # Visualization (GT vs predicted poses)
├── configs/                  # YAML configuration files
│   ├── default.yaml
│   ├── E01.yaml
│   ├── E01_B1.yaml
│   ├── E01_B2.yaml
│   ├── E02.yaml
│   ├── E03.yaml
│   └── E04.yaml
├── data/
│   ├── __init__.py           # Exports MemmapDataset
│   ├── memmap_dataset.py     # Memory-mapped dataset
│   ├── csi_preprocess.py     # CSI sanitization, resampling, normalization
│   └── heatmap_gt.py         # Keypoint conversion, heatmap/PAF generation
├── models/
│   ├── __init__.py
│   └── multiformer.py        # MultiFormer model architecture
├── decode/
│   ├── __init__.py
│   └── pose_decoder.py       # Heatmap → keypoint decoding
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # PCK computation
│   └── viz.py                # Visualization utilities
├── scripts/
│   ├── build_memmap.py       # Build memmap .npy files from raw MM-Fi
│   ├── train_all_envs.sh     # Linux training orchestration
│   └── train_all_envs.bat    # Windows training orchestration
├── CLAUDE.md                 # This file
└── .gitignore
```

## Key design decisions

1. **Memory-mapped I/O**: CSI data is stored as `.npy` files and accessed via `np.memmap`. This enables zero-copy reads and leverages OS page cache for fast repeated access.
2. **Pre-computed normalization**: Three normalization variants are pre-computed at build time, avoiding per-sample computation during training.
3. **Per-subject split**: Train/val split is done per-subject (not global shuffle) to prevent data leakage between splits.
4. **Multi-stage refinement**: The MSFN uses 3 stages with PAPM attention, where each stage's heatmap output guides the next stage's feature refinement.
5. **Dual-domain attention**: Separate Transformer encoders for time and frequency domains, with reconstruction layers projecting back to 2D spatial feature maps.