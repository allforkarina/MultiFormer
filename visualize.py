from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from decode.pose_decoder import decode_single_person
from train import build_dataset, build_model, load_config
from utils.viz import render_compare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-per-env", type=int)
    parser.add_argument("--out", default="outputs/viz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
    device_name = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    rng = random.Random(int(cfg["visualize"].get("seed", 123)))
    n_per_env = args.n_per_env or int(cfg["visualize"].get("n_per_env", 8))
    out_root = Path(args.out)
    pose_range = (float(cfg["heatmap"].get("pose_min", -0.8)), float(cfg["heatmap"].get("pose_max", 0.8)))

    for env in cfg["dataset"].get("envs", ["env1"]):
        dataset = build_dataset(cfg, "all")
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        for index in indices[:n_per_env]:
            item = dataset[index]
            csi = item["csi"][None].to(device).float()
            with torch.no_grad():
                pcm, paf = model(csi)[-1]
            pred = decode_single_person(pcm[0], peak_threshold=float(cfg["eval"].get("peak_threshold", 0.1)), pose_range=pose_range)
            meta = item["meta"]
            title = f"{meta['env']}/{meta['subject']}/{meta['action']} frame={meta['frame_idx']:03d}"
            save_path = out_root / meta["env"] / f"{meta['subject']}_{meta['action']}_f{meta['frame_idx']:03d}.png"
            render_compare(item["kpts18"].numpy(), pred, title=title, save_path=save_path)
            print(save_path)


if __name__ == "__main__":
    main()