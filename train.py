from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.mmfi_dataset import MMFiDataset
from data.h5_dataset import H5MMFiDataset
from decode.pose_decoder import decode_single_person
from models.multiformer import MultiFormer
from utils.metrics import pck_batch

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        return yaml.safe_load(text)
    return _load_simple_yaml(text)


def _strip_comment(line: str) -> str:
    in_quote = False
    quote_char = ""
    for idx, ch in enumerate(line):
        if ch in {"'", '"'} and (idx == 0 or line[idx - 1] != "\\"):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
        elif ch == "#" and not in_quote:
            return line[:idx]
    return line


def _logical_yaml_lines(text: str) -> list[str]:
    lines = []
    pending = ""
    balance = 0
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        pending = f"{pending} {line.strip()}" if pending else line
        balance += line.count("[") - line.count("]")
        if balance <= 0:
            lines.append(pending)
            pending = ""
            balance = 0
    if pending:
        lines.append(pending)
    return lines


def _parse_yaml_scalar(value: str):
    import ast

    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for line in _logical_yaml_lines(text):
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_yaml_scalar(value)
    return root


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: dict, split: str, max_samples: int | None = None) -> MMFiDataset | H5MMFiDataset | MemmapDataset:
    ds_cfg = cfg["dataset"]
    csi_cfg = cfg["csi"]
    hm_cfg = cfg["heatmap"]
    dataset_type = ds_cfg.get("type", "npz")

    common = dict(
        split=split,
        time_packets=int(csi_cfg.get("time_packets", 64)),
        subcarrier_mode=csi_cfg.get("subcarrier_mode", "keep"),
        normalize=csi_cfg.get("normalize", "zscore"),
        heatmap_size=int(hm_cfg.get("size", 36)),
        heatmap_sigma=float(hm_cfg.get("sigma", 1.5)),
        paf_width=float(hm_cfg.get("paf_width", 1.0)),
        pose_range=(float(hm_cfg.get("pose_min", -0.8)), float(hm_cfg.get("pose_max", 0.8))),
    )

    if dataset_type == "h5":
        return H5MMFiDataset(
            h5_path=ds_cfg["root"],
            envs=ds_cfg.get("envs"),
            train_subjects=ds_cfg.get("train_subjects"),
            test_subjects=ds_cfg.get("test_subjects"),
            random_val_ratio=float(ds_cfg.get("random_val_ratio", 0.2)),
            seed=int(ds_cfg.get("seed", 42)),
            **common,
        )

    return MMFiDataset(
        root=ds_cfg["root"],
        protocol=ds_cfg.get("split", "subject_cross"),
        envs=ds_cfg.get("envs"),
        train_subjects=ds_cfg.get("train_subjects"),
        test_subjects=ds_cfg.get("test_subjects"),
        random_val_ratio=float(ds_cfg.get("random_val_ratio", 0.2)),
        seed=int(ds_cfg.get("seed", 42)),
        max_samples=max_samples if max_samples is not None else ds_cfg.get("max_samples"),
        amp_key=csi_cfg.get("amp_key", "CSIamp"),
        **common,
    )


def build_model(cfg: dict) -> MultiFormer:
    csi_cfg = cfg["csi"]
    model_cfg = cfg["model"]
    subcarrier_mode = csi_cfg.get("subcarrier_mode", "keep")
    input_subcarriers = 64 if subcarrier_mode == "resample64" else int(csi_cfg.get("subcarriers", 114))
    return MultiFormer(
        time_packets=int(csi_cfg.get("time_packets", 64)),
        rx_antennas=int(csi_cfg.get("rx_antennas", 3)),
        input_subcarriers=input_subcarriers,
        subcarrier_mode=subcarrier_mode,
        embed_dim=int(model_cfg.get("embed_dim", 1296)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        depth=int(model_cfg.get("depth", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        heatmap_size=int(cfg["heatmap"].get("size", 36)),
        recon_channels=int(model_cfg.get("recon_channels", 64)),
        feature_channels=int(model_cfg.get("feature_channels", 128)),
        decoder_hidden=int(model_cfg.get("decoder_hidden", 512)),
        stages=int(model_cfg.get("stages", 3)),
    )


def multistage_loss(outputs, pcm_gt, paf_gt) -> torch.Tensor:
    loss = torch.zeros((), device=pcm_gt.device)
    for pcm, paf in outputs:
        loss = loss + F.mse_loss(pcm, pcm_gt) + F.mse_loss(paf, paf_gt)
    return loss


@torch.no_grad()
def evaluate(model: MultiFormer, loader: DataLoader, device: torch.device, cfg: dict) -> dict:
    model.eval()
    losses = []
    preds = []
    gts = []
    peak_threshold = float(cfg["eval"].get("peak_threshold", 0.1))
    pose_range = (float(cfg["heatmap"].get("pose_min", -0.8)), float(cfg["heatmap"].get("pose_max", 0.8)))
    for batch in tqdm(loader, desc="val", leave=False):
        csi = batch["csi"].to(device, non_blocking=True).float()
        pcm_gt = batch["pcm"].to(device, non_blocking=True).float()
        paf_gt = batch["paf"].to(device, non_blocking=True).float()
        outputs = model(csi)
        losses.append(float(multistage_loss(outputs, pcm_gt, paf_gt).item()))
        pcm_last, paf_last = outputs[-1]
        for pcm in pcm_last:
            preds.append(decode_single_person(pcm, peak_threshold=peak_threshold, pose_range=pose_range))
        gts.extend(batch["kpts18"].cpu().numpy())
    result = {"loss": float(np.mean(losses)) if losses else 0.0}
    if preds:
        for th in cfg["eval"].get("pck_thresholds", [0.2]):
            result[f"pck@{int(float(th) * 100)}"] = pck_batch(np.asarray(preds), np.asarray(gts), threshold=float(th))["mean"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--overfit-batch", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["dataset"].get("seed", 42)))
    train_cfg = cfg["train"]
    device_name = train_cfg.get("device", "cuda")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    train_ds = build_dataset(cfg, "train", max_samples=args.max_train_samples)
    val_ds = build_dataset(cfg, "test", max_samples=args.max_val_samples)
    if len(train_ds) == 0:
        raise RuntimeError("No training samples found. Check dataset.root and split subjects.")
    if len(val_ds) == 0:
        print("Warning: no validation samples found; training without validation.")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["eval"].get("batch_size", train_cfg.get("batch_size", 32))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    ) if len(val_ds) else None

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg.get("lr_step", 15)),
        gamma=float(train_cfg.get("lr_gamma", 0.7)),
    )

    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.csv"
    best_loss = float("inf")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "pck@20", "lr"])
        writer.writeheader()

    overfit_batch = None
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 1))
    early_stop_warmup = int(train_cfg.get("early_stop_warmup", 3))
    prev_train_loss = float("inf")
    plateau_count = 0
    for epoch in range(1, int(train_cfg.get("epochs", 100)) + 1):
        model.train()
        train_losses = []
        iterator = tqdm(train_loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(iterator, 1):
            if args.overfit_batch:
                if overfit_batch is None:
                    overfit_batch = batch
                batch = overfit_batch
            csi = batch["csi"].to(device, non_blocking=True).float()
            pcm_gt = batch["pcm"].to(device, non_blocking=True).float()
            paf_gt = batch["paf"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(csi)
            loss = multistage_loss(outputs, pcm_gt, paf_gt)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            iterator.set_postfix(loss=f"{np.mean(train_losses):.5f}")
            if args.overfit_batch and step >= 200:
                break

        scheduler.step()
        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate(model, val_loader, device, cfg) if val_loader is not None else {}
        val_loss = float(val_metrics.get("loss", train_loss))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "pck@20": val_metrics.get("pck@20", ""),
            "lr": optimizer.param_groups[0]["lr"],
        }
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_loss": val_loss}
            torch.save(ckpt, output_dir / "best.pth")
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if early_stop_min_delta > 0 and epoch >= early_stop_warmup:
            improvement = prev_train_loss - train_loss
            if improvement < early_stop_min_delta:
                plateau_count += 1
                print(
                    f"early-stop check: train_loss improvement {improvement:.6f}"
                    f" < {early_stop_min_delta} ({plateau_count}/{early_stop_patience})"
                )
                if plateau_count >= early_stop_patience:
                    print(f"early stopping at epoch {epoch}: train_loss plateaued")
                    break
            else:
                plateau_count = 0
        prev_train_loss = train_loss


if __name__ == "__main__":
    main()
