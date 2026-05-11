"""
MM-Fi → MultiFormer 离线预处理脚本
==================================

功能：
    把原始 MM-Fi 数据集（D:\\MM-Fi数据集\\MMFi_Dataset）里每一个
    trial 目录 (E0X/S0X/A0X/) 转成训练用的单个 .npz 文件，
    输出到 C:\\Users\\杜迪安\\Desktop\\MM-Fi_pre\\E0X/S0X/A0X.npz。

    每个 npz 内三个数组：
        csi        float32  (N, 64, 3, 114)   — 已时间上采样 + z-score
        kpts18     float32  (N, 18, 2)        — OpenPose-18 顺序，2D 像素坐标
        frame_idx  int32    (N,)              — 1-based 原始帧号

    其中 N 通常 = 297（每个 trial 的有效帧数）。

输入约定：
    源数据布局：
        SRC/E0X/S0X/A0X/
            ├── wifi-csi/frame001.mat … frame297.mat   每帧含 CSIamp (3,114,10)
            ├── pose2d.npy                              (N, 17, 2) float32, COCO17
            └── ground_truth.npy                        (N, 17, 3) float32, 3D（不用）

预处理流程（与论文及 data/csi_preprocess.py 等价）：
    1. 加载每帧的 CSIamp (NR=3, NS=114, M0=10)
    2. 时间维 FFT 重采样 10 → 64：得到 (3, 114, 64)
    3. 转置成 (M=64, NR=3, NS=114)
    4. NaN / Inf 用 finite 中位数填充
    5. z-score 归一化（per-frame，整张 64×3×114 一起算 mean/std）
    6. 堆叠 N 帧 → (N, 64, 3, 114)
    7. pose2d.npy (N, 17, 2) → OpenPose-18 顺序 (N, 18, 2)
       （neck = 左右肩中点；缺失点保持 0）
    8. 写出 npz + 在输出根目录写 index.json 汇总元信息

用法：
    python preprocess_mmfi.py
        # 默认源 = D:\\MM-Fi数据集\\MMFi_Dataset
        # 默认目标 = C:\\Users\\杜迪安\\Desktop\\MM-Fi_pre

    python preprocess_mmfi.py --src "D:/xxx" --dst "E:/xxx" --workers 8
        # 多进程并行（每个 trial 一个任务）

    python preprocess_mmfi.py --envs E01 E02 --subjects S01 S02
        # 只处理指定 env / subject

依赖：
    numpy, scipy（仅用 scipy.signal.resample 和 scipy.io.loadmat）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
from scipy.signal import resample


# =============================================================================
# 1. 默认参数（可被 CLI 覆盖）
# =============================================================================

DEFAULT_SRC = r"D:\MM-Fi数据集\MMFi_Dataset"
DEFAULT_DST = r"C:\Users\杜迪安\Desktop\MM-Fi_pre"

# CSI 预处理目标尺寸
TIME_PACKETS = 64        # M：时间维上采样后的 packet 数（论文值）
RX_ANTENNAS = 3          # NR：接收天线数（MM-Fi 固定）
INPUT_SUBCARRIERS = 114  # NS：原始子载波数（MM-Fi 固定）

# 子载波模式：keep（默认，保留 114）/ resample64（FFT 降到 64，对照实验用）
# learned64 是网络内部的 nn.Linear，不在离线阶段做，等价于 keep。
SUBCARRIER_MODE = "keep"

# 归一化模式：zscore（默认）/ minmax / none
NORMALIZE = "zscore"

# 数值稳定项
EPS = 1e-6


# =============================================================================
# 2. CSI 数值清洗 + 归一化（与 data/csi_preprocess.py 对齐）
# =============================================================================

def sanitize_csi(x: np.ndarray) -> np.ndarray:
    """
    把 NaN / +Inf / -Inf 替换成 finite 元素的中位数。
    全部都是非 finite 时返回 0，避免归一化炸掉。
    """
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.all():
        return x
    fill = float(np.median(x[finite])) if finite.any() else 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def normalize_csi(x: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """
    Per-sample 归一化：把整张 (M, NR, NS) 当作一个张量统一计算 mean/std 或 min/max。
    """
    x = sanitize_csi(x)
    if mode == "none":
        return x
    if mode == "zscore":
        mean = float(x.mean())
        std = float(x.std())
        if not np.isfinite(std) or std < EPS:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mean) / (std + EPS)).astype(np.float32)
    if mode == "minmax":
        lo = float(x.min())
        hi = float(x.max())
        if not np.isfinite(hi - lo) or (hi - lo) < EPS:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo + EPS)).astype(np.float32)
    raise ValueError(f"未知归一化模式: {mode}")


def resample_time(csi_amp: np.ndarray, target_packets: int = TIME_PACKETS) -> np.ndarray:
    """
    时间维 FFT 重采样：(NR, NS, M0) → (NR, NS, target_packets)。
    论文用 "插零 + 低通滤波"，scipy.signal.resample 是 FFT-based，等价。
    """
    csi_amp = sanitize_csi(csi_amp)
    if csi_amp.ndim == 2:
        # 兜底：少数 mat 可能只有 (NR, NS)，扩一维当 1 packet
        csi_amp = csi_amp[:, :, None]
    if csi_amp.ndim != 3:
        raise ValueError(f"CSIamp 形状期望 (NR, NS, M)，实际 {csi_amp.shape}")
    if csi_amp.shape[-1] == target_packets:
        return csi_amp
    return sanitize_csi(resample(csi_amp, target_packets, axis=-1))


def resample_subcarriers(csi_amp: np.ndarray, target: int = 64) -> np.ndarray:
    """
    子载波维 FFT 降采样：仅在 subcarrier_mode='resample64' 时使用。
    输入形状 (NR, NS, M)，axis=1 是子载波。
    """
    if csi_amp.shape[1] == target:
        return csi_amp.astype(np.float32)
    return sanitize_csi(resample(csi_amp, target, axis=1))


def preprocess_one_frame(
    csi_amp: np.ndarray,
    target_packets: int = TIME_PACKETS,
    subcarrier_mode: str = SUBCARRIER_MODE,
    normalize: str = NORMALIZE,
) -> np.ndarray:
    """
    单帧 CSI 预处理主入口。
    输入：CSIamp (NR=3, NS=114, M0=10)
    输出：(M=64, NR=3, NS=114 或 64) float32，已归一化
    """
    x = resample_time(csi_amp, target_packets=target_packets)            # (3, 114, 64)
    if subcarrier_mode == "resample64":
        x = resample_subcarriers(x, target=64)                            # (3, 64, 64)
    elif subcarrier_mode not in {"keep", "learned64"}:
        raise ValueError(f"未知 subcarrier_mode: {subcarrier_mode}")

    # 维度顺序对齐 MultiFormer：(M, NR, NS)
    x = np.transpose(x, (2, 0, 1)).astype(np.float32)
    return normalize_csi(x, mode=normalize)


# =============================================================================
# 3. 关键点格式转换：COCO17 → OpenPose-18
#    （与 data/heatmap_gt.py 中 coco17_to_openpose18 一致）
# =============================================================================

# OpenPose-18 关键点顺序：
#   0 nose, 1 neck, 2 R-Sh, 3 R-El, 4 R-Wr, 5 L-Sh, 6 L-El, 7 L-Wr,
#   8 R-Hip, 9 R-Kn, 10 R-An, 11 L-Hip, 12 L-Kn, 13 L-An,
#   14 R-Eye, 15 L-Eye, 16 R-Ear, 17 L-Ear

# COCO17 → OpenPose18 的逐点映射（neck=1 单独由左右肩中点合成，不在此表）
COCO17_TO_OPENPOSE18 = {
    0: 0,    # nose
    14: 2,   # r_eye
    15: 1,   # 占位：L-eye 索引这里只是为了避免与 neck 冲突，下面会被覆盖
    # 注意：以上注释只是占位提示，下表才是真实映射
}

# 真实映射（与 data/heatmap_gt.py 完全一致）：
# OpenPose 索引 → COCO17 索引
OP18_FROM_COCO17 = {
    0: 0,    # nose      ← coco17[0]
    2: 6,    # r_shoulder← coco17[6]
    3: 8,    # r_elbow   ← coco17[8]
    4: 10,   # r_wrist   ← coco17[10]
    5: 5,    # l_shoulder← coco17[5]
    6: 7,    # l_elbow   ← coco17[7]
    7: 9,    # l_wrist   ← coco17[9]
    8: 12,   # r_hip     ← coco17[12]
    9: 14,   # r_knee    ← coco17[14]
    10: 16,  # r_ankle   ← coco17[16]
    11: 11,  # l_hip     ← coco17[11]
    12: 13,  # l_knee    ← coco17[13]
    13: 15,  # l_ankle   ← coco17[15]
    14: 2,   # r_eye     ← coco17[2]
    15: 1,   # l_eye     ← coco17[1]
    16: 4,   # r_ear     ← coco17[4]
    17: 3,   # l_ear     ← coco17[3]
}


def _valid_point(point: np.ndarray) -> bool:
    """非 finite 或全 0 视为无效（MM-Fi 用 (0,0) 表示缺失）。"""
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    """
    输入：(17, 2) COCO17 关键点
    输出：(18, 2) OpenPose-18 关键点（neck=1 用左右肩中点合成；缺失点为 0）
    """
    kpts17 = np.asarray(kpts17, dtype=np.float32)
    if kpts17.shape[-2:] != (17, 2):
        raise ValueError(f"期望 (17, 2)，实际 {kpts17.shape}")

    kpts18 = np.zeros((18, 2), dtype=np.float32)
    valid = np.zeros(18, dtype=bool)

    # 1. 直接映射的 17 个点
    for op_idx, coco_idx in OP18_FROM_COCO17.items():
        p = kpts17[coco_idx]
        if _valid_point(p):
            kpts18[op_idx] = p
            valid[op_idx] = True

    # 2. neck（OpenPose-18 索引 1）= 左右肩中点
    l_sh = kpts17[5]
    r_sh = kpts17[6]
    if _valid_point(l_sh) and _valid_point(r_sh):
        kpts18[1] = (l_sh + r_sh) * 0.5
        valid[1] = True
    elif _valid_point(l_sh):
        kpts18[1] = l_sh
        valid[1] = True
    elif _valid_point(r_sh):
        kpts18[1] = r_sh
        valid[1] = True

    # 3. 无效点强制 0
    kpts18[~valid] = 0.0
    return kpts18


def convert_pose2d_batch(pose2d: np.ndarray) -> np.ndarray:
    """批量转换 (N, 17, 2) → (N, 18, 2)。"""
    n = pose2d.shape[0]
    out = np.zeros((n, 18, 2), dtype=np.float32)
    for i in range(n):
        out[i] = coco17_to_openpose18(pose2d[i])
    return out


# =============================================================================
# 4. Trial 级处理：把一个 E0X/S0X/A0X 目录打包成一个 npz
# =============================================================================

@dataclass
class TrialInfo:
    env: str
    subject: str
    action: str
    file: str
    num_frames: int
    csi_shape: tuple
    kpts_shape: tuple


def list_frame_mats(wifi_dir: Path) -> list[Path]:
    """按文件名排序列出 frame*.mat。"""
    return sorted(wifi_dir.glob("frame*.mat"))


def load_csi_amp(mat_path: Path) -> np.ndarray:
    """
    读取一帧 .mat 的 CSIamp，返回 (NR, NS, M0)，dtype float32。
    MM-Fi 原始 CSIamp 是 float64，这里立刻 cast 到 float32 节省内存。
    """
    mat = sio.loadmat(str(mat_path))
    if "CSIamp" not in mat:
        raise KeyError(f"{mat_path} 中没有 CSIamp 键")
    return np.asarray(mat["CSIamp"], dtype=np.float32)


def process_trial(
    src_trial: Path,
    dst_npz: Path,
    subcarrier_mode: str = SUBCARRIER_MODE,
    normalize: str = NORMALIZE,
    overwrite: bool = False,
) -> TrialInfo:
    """
    把一个 trial 目录 (E0X/S0X/A0X) 打包成一个 npz 文件。
    """
    if dst_npz.exists() and not overwrite:
        # 已存在，读元信息直接返回（不报错，方便断点续跑）
        with np.load(dst_npz) as d:
            return TrialInfo(
                env=src_trial.parent.parent.name,
                subject=src_trial.parent.name,
                action=src_trial.name,
                file=str(dst_npz.relative_to(dst_npz.parents[2])).replace("\\", "/"),
                num_frames=int(d["csi"].shape[0]),
                csi_shape=tuple(d["csi"].shape),
                kpts_shape=tuple(d["kpts18"].shape),
            )

    wifi_dir = src_trial / "wifi-csi"
    pose_npy = src_trial / "pose2d.npy"
    if not wifi_dir.is_dir():
        raise FileNotFoundError(f"缺少 wifi-csi 目录: {wifi_dir}")
    if not pose_npy.is_file():
        raise FileNotFoundError(f"缺少 pose2d.npy: {pose_npy}")

    # ---- 1. 收集所有 frame*.mat
    mats = list_frame_mats(wifi_dir)
    if not mats:
        raise RuntimeError(f"{wifi_dir} 没有 frame*.mat")
    n_frames = len(mats)

    # ---- 2. 加载 pose2d 并对齐帧数（取 min，少数 trial 帧数会差几帧）
    pose2d = np.load(pose_npy)  # (N, 17, 2) float32
    if pose2d.ndim != 3 or pose2d.shape[1:] != (17, 2):
        raise ValueError(f"pose2d.npy 形状异常: {pose2d.shape}")
    if pose2d.shape[0] != n_frames:
        n_aligned = min(pose2d.shape[0], n_frames)
        mats = mats[:n_aligned]
        pose2d = pose2d[:n_aligned]
        n_frames = n_aligned

    # ---- 3. 逐帧 CSI 预处理
    # 提前用第一帧探出输出形状，避免 list+stack 的内存峰值
    first = preprocess_one_frame(
        load_csi_amp(mats[0]),
        target_packets=TIME_PACKETS,
        subcarrier_mode=subcarrier_mode,
        normalize=normalize,
    )
    csi_out = np.empty((n_frames, *first.shape), dtype=np.float32)
    csi_out[0] = first
    for i in range(1, n_frames):
        amp = load_csi_amp(mats[i])
        csi_out[i] = preprocess_one_frame(
            amp,
            target_packets=TIME_PACKETS,
            subcarrier_mode=subcarrier_mode,
            normalize=normalize,
        )

    # ---- 4. 关键点格式转换
    kpts18 = convert_pose2d_batch(pose2d)  # (N, 18, 2)

    # ---- 5. frame_idx：1-based 原始帧号，从文件名提取（如 frame037.mat → 37）
    frame_idx = np.array(
        [int(p.stem.replace("frame", "")) for p in mats],
        dtype=np.int32,
    )

    # ---- 6. 写出 npz（compressed=False 写入更快，读取也快；MM-Fi 单 trial ~50MB）
    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst_npz, csi=csi_out, kpts18=kpts18, frame_idx=frame_idx)

    return TrialInfo(
        env=src_trial.parent.parent.name,
        subject=src_trial.parent.name,
        action=src_trial.name,
        file=str(dst_npz.relative_to(dst_npz.parents[2])).replace("\\", "/"),
        num_frames=n_frames,
        csi_shape=tuple(csi_out.shape),
        kpts_shape=tuple(kpts18.shape),
    )


# =============================================================================
# 5. 数据集枚举
# =============================================================================

def iter_trials(
    src_root: Path,
    envs: Iterable[str] | None,
    subjects: Iterable[str] | None,
    actions: Iterable[str] | None,
) -> list[Path]:
    """
    枚举所有匹配的 trial 目录。返回 src_root/E0X/S0X/A0X 的 Path 列表。
    """
    env_filter = set(envs) if envs else None
    subj_filter = set(subjects) if subjects else None
    act_filter = set(actions) if actions else None

    trials: list[Path] = []
    for env_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("E")):
        if env_filter and env_dir.name not in env_filter:
            continue
        for subj_dir in sorted(p for p in env_dir.iterdir() if p.is_dir() and p.name.startswith("S")):
            if subj_filter and subj_dir.name not in subj_filter:
                continue
            for act_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir() and p.name.startswith("A")):
                if act_filter and act_dir.name not in act_filter:
                    continue
                trials.append(act_dir)
    return trials


# =============================================================================
# 6. 多进程 worker（用于 ProcessPoolExecutor）
# =============================================================================

def _worker(args: tuple) -> tuple[str, dict | None, str | None]:
    """子进程入口：成功返回 (key, info_dict, None)，失败返回 (key, None, traceback_str)。"""
    src_trial, dst_npz, subcarrier_mode, normalize, overwrite = args
    key = f"{src_trial.parent.parent.name}/{src_trial.parent.name}/{src_trial.name}"
    try:
        info = process_trial(
            Path(src_trial),
            Path(dst_npz),
            subcarrier_mode=subcarrier_mode,
            normalize=normalize,
            overwrite=overwrite,
        )
        return key, asdict(info), None
    except Exception:
        return key, None, traceback.format_exc()


# =============================================================================
# 7. 主入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MM-Fi → MultiFormer 离线预处理")
    p.add_argument("--src", default=DEFAULT_SRC, help="原始数据集根目录（含 E0X/S0X/A0X/...）")
    p.add_argument("--dst", default=DEFAULT_DST, help="输出根目录（会写 E0X/S0X/A0X.npz）")
    p.add_argument("--envs", nargs="*", default=None, help="只处理这些 env，例如 --envs E01 E02")
    p.add_argument("--subjects", nargs="*", default=None, help="只处理这些 subject")
    p.add_argument("--actions", nargs="*", default=None, help="只处理这些 action")
    p.add_argument("--subcarrier-mode", default=SUBCARRIER_MODE,
                   choices=["keep", "resample64", "learned64"],
                   help="子载波处理方式（learned64 等价 keep，因实际投影在网络内做）")
    p.add_argument("--normalize", default=NORMALIZE, choices=["zscore", "minmax", "none"])
    p.add_argument("--workers", type=int, default=4, help="并行进程数（每个 trial 一个任务）")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的 npz")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.is_dir():
        print(f"[错误] 源目录不存在: {src_root}", file=sys.stderr)
        return 1
    dst_root.mkdir(parents=True, exist_ok=True)

    trials = iter_trials(src_root, args.envs, args.subjects, args.actions)
    if not trials:
        print("[警告] 未找到匹配的 trial，检查 --envs / --subjects / --actions")
        return 0

    print(f"[信息] 共 {len(trials)} 个 trial 待处理")
    print(f"[信息] 源:    {src_root}")
    print(f"[信息] 目标:  {dst_root}")
    print(f"[信息] 配置:  subcarrier_mode={args.subcarrier_mode}, normalize={args.normalize}, "
          f"workers={args.workers}, overwrite={args.overwrite}")

    # 构建任务列表（要传给子进程，所有参数必须可 pickle，所以转成基础类型）
    tasks = []
    for t in trials:
        rel = t.relative_to(src_root)  # E0X/S0X/A0X
        dst_npz = dst_root / rel.parent / f"{rel.name}.npz"
        tasks.append((str(t), str(dst_npz), args.subcarrier_mode, args.normalize, args.overwrite))

    results: list[dict] = []
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    if args.workers <= 1:
        # 单进程：方便看 traceback、调试
        for i, task in enumerate(tasks, 1):
            key, info, err = _worker(task)
            if err:
                failures.append((key, err))
                print(f"  [{i}/{len(tasks)}] {key}  失败")
            else:
                results.append(info)
                print(f"  [{i}/{len(tasks)}] {key}  ok  N={info['num_frames']}")
    else:
        # 多进程
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, task): task for task in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                key, info, err = fut.result()
                if err:
                    failures.append((key, err))
                    print(f"  [{done}/{len(tasks)}] {key}  失败")
                else:
                    results.append(info)
                    print(f"  [{done}/{len(tasks)}] {key}  ok  N={info['num_frames']}")

    dt = time.time() - t0
    print(f"\n[信息] 处理完成：成功 {len(results)} / 失败 {len(failures)}，"
          f"耗时 {dt:.1f}s（{dt/max(len(tasks),1):.2f}s/trial）")

    # 写 index.json（与 MM-Fi_pre/index.json 字段一致，便于核对）
    index = {
        "source": str(src_root),
        "csi": {
            "time_packets": TIME_PACKETS,
            "rx_antennas": RX_ANTENNAS,
            "subcarriers": INPUT_SUBCARRIERS,
            "subcarrier_mode": args.subcarrier_mode,
            "normalize": args.normalize,
            "amp_key": "CSIamp",
        },
        "trials": sorted(results, key=lambda d: (d["env"], d["subject"], d["action"])),
    }
    if failures:
        index["failures"] = [{"trial": k, "traceback": tb} for k, tb in failures]

    index_path = dst_root / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[信息] 写入索引: {index_path}")

    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
