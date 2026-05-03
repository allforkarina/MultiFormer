from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from decode.pose_decoder import decode_single_person
from train import build_dataset, build_model, load_config, multistage_loss
from utils.metrics import pck_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    dataset = build_dataset(cfg, "test", max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["eval"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    losses = []
    preds = []
    gts = []
    pose_range = (float(cfg["heatmap"].get("pose_min", -0.8)), float(cfg["heatmap"].get("pose_max", 0.8)))
    peak_threshold = float(cfg["eval"].get("peak_threshold", 0.1))
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            csi = batch["csi"].to(device).float()
            pcm_gt = batch["pcm"].to(device).float()
            paf_gt = batch["paf"].to(device).float()
            outputs = model(csi)
            losses.append(float(multistage_loss(outputs, pcm_gt, paf_gt).item()))
            pcm_last, _ = outputs[-1]
            for pcm in pcm_last:
                preds.append(decode_single_person(pcm, peak_threshold=peak_threshold, pose_range=pose_range))
            gts.extend(batch["kpts18"].cpu().numpy())

    print(f"samples: {len(dataset)}")
    print(f"loss: {np.mean(losses):.6f}")
    for th in cfg["eval"].get("pck_thresholds", [0.2, 0.3, 0.4]):
        score = pck_batch(np.asarray(preds), np.asarray(gts), threshold=float(th))
        print(f"PCK@{int(float(th) * 100)}: {score['mean']:.4f}")


if __name__ == "__main__":
    main()
