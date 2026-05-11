# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

MultiFormer — WiFi CSI-based human pose estimation (17/18 keypoints). Uses a transformer architecture with dual time-frequency attention for reconstructing human pose heatmaps from WiFi channel state information.

## Commands

```bash
# Training (env-specific configs: E01-E04)
python train.py --config configs/E01.yaml

# Evaluation
python eval.py --config configs/E01.yaml --ckpt outputs/E01/best.pth

# Visualization (save GT vs prediction comparison images)
python visualize.py --config configs/E01.yaml --ckpt outputs/E01/best.pth --out outputs/viz
```

## Model architecture

```
CSI input (B, 64, 3, 114)
  └─ TFDDTTokenizer         # Time-Frequency Dual-Dimensional Tokenizer
       ├─ freq tokens (B, NS, embed_dim) via Linear(M·NR → embed_dim)
       └─ time tokens (B, M, embed_dim)  via Linear(NS·NR → embed_dim)
  └─ DualAttentionExtractor # 8 stacked TransformerBlocks per branch + ReconstructionLayer
       ├─ freq_blocks → freq_map (B, 64, 36, 36)
       └─ time_blocks → time_map (B, 64, 36, 36)
       └─ cat → features (B, 128, 36, 36)
  └─ MSFN                   # Multi-Stage Feature Network (3 stages)
       ├─ HeatmapDecoder → pcm (B, 19, 36, 36), paf (B, 38, 36, 36)
       └─ PAPM (Pose Attention Perception Module) between stages
```

Key modules:
- `models/tfddt.py` — Tokenizes CSI into separate freq/time token streams with learned positional embeddings
- `models/attention_extractor.py` — Parallel transformer branches (freq + time) with reconstruction heads
- `models/heatmap_decoder.py` — 4-layer CNN trunk → bottleneck → PCM/PAF prediction heads
- `models/papm.py` — Channel + spatial attention module that refines features using previous stage heatmaps
- `models/msfn.py` — Stacks N HeatmapDecoders with PAPM inter-stage refinement
- `models/multiformer.py` — Top-level module composing tokenizer → extractor → msfn

## Data pipeline

Two dataset backends, selected via `dataset.type` in config:

| Config key | Backend | Format | Shape (raw) | Normalization |
|------------|---------|--------|-------------|---------------|
| `"npz"` | `data/mmfi_dataset.py:MMFiDataset` | NPZ per action | `(64, 3, 114)` | per-sample z-score |
| `"h5"` | `data/h5_dataset.py:H5MMFiDataset` | Single HDF5 | `(3, 114, 10)` | configurable (4 modes) |

Both return the same dict: `{"csi": (64,3,114), "kpts18": (18,2), "pcm": (19,36,36), "paf": (38,36,36), "meta": {...}}`

**HDF5 data flow** (server deployment):
1. H5 attrs read at init: `amplitude_train_min/max`, `keypoint_x/y_scale`, `amplitude_normalization`
2. Keypoints (~43 MB) and meta arrays (~5 MB) preloaded into RAM at init — no per-sample HDF5 read for these
3. Per sample: read raw CSI `(3, 114, 10)` from HDF5 (only 1 HDF5 random read per sample)
4. Time resample `10 → 64` via `scipy.signal.resample`
5. Transpose to `(64, 3, 114)`
6. Normalize according to `csi.normalize` config (see Config system below)
7. COCO17 `(17,2)` → OpenPose18 `(18,2)` keypoint conversion via `data/heatmap_gt.py:coco17_to_openpose18()`
8. Build PCM (19ch) and PAF (38ch) heatmap targets

**NPZ data flow** (local):
1. Load pre-resampled CSI `(64, 3, 114)` from compressed `.npz`
2. Per-sample z-score or min-max normalization via `data/csi_preprocess.py:normalize_csi()`
3. Keypoints already in OpenPose18 format

## Preprocessing (`data/csi_preprocess.py`)

- `sanitize_csi()` — Replace NaN/Inf with median-fill
- `resample_time()` — Resample time packets via FFT (scipy)
- `resample_subcarriers()` — Resample subcarrier dimension
- `normalize_csi()` — Per-sample z-score or min-max
- `normalize_global_minmax()` — Global min-max using pre-computed stats
- `preprocess_csi_amp()` — End-to-end pipeline for NPZ format

## Keypoint coordinates

All keypoints are in `pose_range` (default `[-0.8, 0.8]`). Heatmap targets use grid coordinates `[0, size-1]` (default size=36). Conversion functions in `data/heatmap_gt.py`:
- `pose_to_heatmap_coords()` — Pose range → grid coords
- `heatmap_to_pose_coords()` — Grid coords → pose range
- `build_pcm_paf()` — OpenPose18 → PCM (19, H, W) + PAF (38, H, W)

## Config system

Self-contained YAML files in `configs/`. No inheritance — each file is complete. Key sections:
- `dataset` — `type` (`"h5"`|`"npz"`), `root`, `envs`, `train_subjects`, `test_subjects`, `random_val_ratio`, `seed`
- `csi` — `time_packets` (64), `rx_antennas` (3), `subcarriers` (114), `subcarrier_mode`, `normalize` (one of: `global_minmax` / `global_zscore` / `zscore` / `none`)
- `heatmap` — `size` (36), `sigma` (1.5), `paf_width` (1.0), `pose_min`/`pose_max`
- `model` — `embed_dim` (1296), `num_heads` (8), `depth` (8), `dropout`, `stages` (3)
- `train` — `epochs`, `batch_size`, `lr`, `lr_step`, `lr_gamma`, `num_workers`, `output_dir`, early-stop params
- `eval` — `batch_size`, `pck_thresholds`, `peak_threshold`

## Evaluation metrics

- **PCK** (Percentage of Correct Keypoints) at configurable thresholds (default @20, @30, @40)
- Based on torso-scale normalized distance between predicted and GT keypoints
- `decode_single_person()` — Argmax peak extraction from PCM heatmaps
- `decode_connections()` — NMS + Hungarian matching for multi-person (with PAF)
