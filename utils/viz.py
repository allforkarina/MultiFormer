from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.heatmap_gt import LIMBS_18
from utils.metrics import pck


def _valid(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def draw_skeleton(ax, kpts18, color: str, label: str, marker: str = "o", linestyle: str = "-") -> None:
    kpts18 = np.asarray(kpts18, dtype=np.float32)
    first = True
    for a, b in LIMBS_18:
        if _valid(kpts18[a]) and _valid(kpts18[b]):
            ax.plot(
                [kpts18[a, 0], kpts18[b, 0]],
                [kpts18[a, 1], kpts18[b, 1]],
                color=color,
                linewidth=2.0,
                linestyle=linestyle,
                label=label if first else None,
            )
            first = False
    valid = np.array([_valid(p) for p in kpts18])
    if valid.any():
        ax.scatter(kpts18[valid, 0], kpts18[valid, 1], c=color, s=28, marker=marker)


def render_compare(
    gt_kpts,
    pred_kpts,
    title: str = "",
    save_path: str | Path | None = None,
    pck_threshold: float = 0.2,
) -> None:
    gt = np.asarray(gt_kpts, dtype=np.float32)
    pred = np.asarray(pred_kpts, dtype=np.float32)
    score = pck(pred, gt, threshold=pck_threshold)["mean"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=140)
    for ax, kpts, color, label, marker, linestyle in [
        (axes[0], gt, "tab:green", "GT", "o", "-"),
        (axes[1], pred, "tab:red", "Pred", "^", "--"),
    ]:
        draw_skeleton(ax, kpts, color=color, label=label, marker=marker, linestyle=linestyle)
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(0.9, -0.9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#e6e6e6", linewidth=0.7)
        ax.legend(loc="upper right")
    axes[0].set_title("Ground Truth")
    axes[1].set_title(f"Prediction PCK@{int(pck_threshold * 100)}={score:.3f}")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
