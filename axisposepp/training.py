import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import ShapeNetPairs
from .losses import AxisPosePPLoss
from .model import AxisPosePP


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in (".yml", ".yaml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML config requires PyYAML. Install it with: pip install pyyaml") from exc
            return yaml.safe_load(f)
        return json.load(f)


def move_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def build_dataset(cfg, split):
    data_cfg = cfg["data"]
    if data_cfg.get("dataset", "shapenet").lower() != "shapenet":
        raise ValueError("This release supports only data.dataset=shapenet.")
    return ShapeNetPairs(
        root_dir=data_cfg["root"],
        split="train" if split == "train" else "test",
        image_size=data_cfg.get("image_size", 336),
        train_pair_file=data_cfg.get("train_pair_file"),
        test_pair_file=data_cfg.get("test_pair_file"),
    )


def _sample_object_id(sample):
    prefix = sample.get("query", sample.get("ref", ""))
    return Path(prefix).parent.name


def subset_first_n_objects(dataset, max_objects):
    if max_objects in (None, "", 0, False) or not hasattr(dataset, "images"):
        return dataset
    max_objects = int(max_objects)
    object_ids = []
    selected = []
    seen = set()
    for index, sample in enumerate(dataset.images):
        object_id = _sample_object_id(sample)
        if object_id not in seen:
            if len(object_ids) >= max_objects:
                continue
            seen.add(object_id)
            object_ids.append(object_id)
        if object_id in seen:
            selected.append(index)
    print(f"[AxisPose++] validation subset: objects={len(object_ids)} samples={len(selected)}")
    subset = Subset(dataset, selected)
    subset.object_count = len(object_ids)
    return subset


def build_loader(cfg, split, shuffle):
    dataset = build_dataset(cfg, split)
    if dataset is None:
        return None
    train_cfg = cfg["train"]
    if split != "train":
        dataset = subset_first_n_objects(dataset, train_cfg.get("val_max_objects"))
    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=split == "train",
    )


def save_checkpoint(path, model, optimizer, epoch, step, best_val, scaler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
            "best_val": best_val,
        },
        path,
    )


def resolve_resume_path(resume, output_dir):
    if resume in (None, "", False):
        return None
    if isinstance(resume, str) and resume.lower() == "auto":
        latest_path = output_dir / "latest.pt"
        best_path = output_dir / "best.pt"
        if latest_path.exists():
            return latest_path
        return best_path if best_path.exists() else None
    return Path(resume)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    epoch = int(checkpoint.get("epoch", 0))
    step = int(checkpoint.get("step", 0))
    best_val = float(checkpoint.get("best_val", float("inf")))
    return epoch, step, best_val


def append_metrics(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fieldnames = list(reader.fieldnames or [])
        old_rows = list(reader)

    fieldnames = old_fieldnames.copy()
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    if fieldnames != old_fieldnames:
        missing_keys = [key for key in fieldnames if key not in old_fieldnames]
        for old_row in old_rows:
            extras = old_row.pop(None, None)
            if extras:
                for key, value in zip(missing_keys, extras):
                    old_row[key] = value
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerow(row)
    else:
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)


def compute_inference_metrics(outputs, batch):
    metrics = {}
    pred_axis = outputs["heatmap"][:, 0:3].float()
    target_axis = batch["target_axis_heatmap"].float()
    metrics["axis_mae"] = (pred_axis - target_axis).abs().mean()

    if "target_center_norm" in batch:
        target_center = batch["target_center_norm"].float()
    else:
        target_center = None
    if "target_directions" in batch:
        target_dirs = batch["target_directions"].float()
    else:
        target_dirs = None

    if target_center is not None:
        h, w = target_axis.shape[-2:]
        scale = torch.tensor([w - 1, h - 1], device=target_axis.device, dtype=target_axis.dtype)
        center_err = ((outputs["center"].float() - target_center) * scale).norm(dim=-1)
        metrics["center_px_err"] = center_err.mean()

    if target_dirs is not None:
        pred_dirs = F.normalize(outputs["directions"].float(), dim=-1, eps=1e-6)
        target_dirs = F.normalize(target_dirs.float(), dim=-1, eps=1e-6)
        cos = (pred_dirs * target_dirs).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.acos(cos))
        metrics["dir_angle_deg"] = angle.mean()
        metrics["dir_acc_10deg"] = (angle < 10.0).float().mean()
        metrics["dir_acc_20deg"] = (angle < 20.0).float().mean()
    return metrics


def optional_positive_int(value, default=None):
    if value in (None, "", 0, False):
        return default
    return int(value)


def summarize_batch_meta(batch, limit=3):
    meta = batch.get("meta")
    if not isinstance(meta, dict):
        return ""
    refs = meta.get("ref", [])
    queries = meta.get("query", [])
    parts = []
    for idx in range(min(limit, len(refs), len(queries))):
        parts.append(f"{refs[idx]}->{queries[idx]}")
    return "; ".join(parts)


def train_one_epoch(model, criterion, loader, optimizer, scaler, device, epoch, log_interval, grad_clip=None):
    model.train()
    running = 0.0
    finite_steps = 0
    skipped_steps = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(batch)
            losses = criterion(outputs, batch)
            loss = losses["loss"]
        if not torch.isfinite(loss):
            skipped_steps += 1
            print(
                f"[AxisPose++] skipped non-finite loss at epoch={epoch} step={step}/{len(loader)} "
                f"loss={float(loss.detach().cpu())} samples={summarize_batch_meta(batch)}"
            )
            continue
        if scaler is None:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(grad_clip) if grad_clip else float("inf"),
            )
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[AxisPose++] skipped non-finite grad at epoch={epoch} step={step}/{len(loader)} "
                    f"grad_norm={float(grad_norm.detach().cpu())} samples={summarize_batch_meta(batch)}"
                )
                continue
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(grad_clip) if grad_clip else float("inf"),
            )
            if not torch.isfinite(grad_norm):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                print(
                    f"[AxisPose++] skipped non-finite grad at epoch={epoch} step={step}/{len(loader)} "
                    f"grad_norm={float(grad_norm.detach().cpu())} samples={summarize_batch_meta(batch)}"
                )
                continue
            scaler.step(optimizer)
            scaler.update()

        running += float(loss.detach())
        finite_steps += 1
        if step % log_interval == 0:
            parts = [f"{k}={float(v):.4f}" for k, v in losses.items()]
            print(f"epoch={epoch} step={step}/{len(loader)} " + " ".join(parts))
    if skipped_steps:
        print(f"[AxisPose++] epoch={epoch} skipped_nonfinite_steps={skipped_steps}")
    return running / max(finite_steps, 1)


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    if loader is None:
        return None
    model.eval()
    sums = {}
    count = 0
    sample_count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        batch_size = batch["target_image"].shape[0]
        outputs = model(batch)
        losses = criterion(outputs, batch)
        inference_metrics = compute_inference_metrics(outputs, batch)
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        for key, value in inference_metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        sample_count += batch_size
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics["num_batches"] = count
    metrics["num_samples"] = sample_count
    if hasattr(loader.dataset, "object_count"):
        metrics["num_objects"] = loader.dataset.object_count
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train AxisPose++")
    parser.add_argument("--config", default="configs/train_shapenet.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dino-weights", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None, help="'auto', empty/false, or a checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    if args.dino_weights:
        cfg["model"]["dino"]["weights_path"] = args.dino_weights
    if args.resume is not None:
        cfg["train"]["resume"] = args.resume

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AxisPose++] device={device}")

    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)
    if train_loader is None:
        raise ValueError("Could not construct the ShapeNet training dataset.")

    model = AxisPosePP(**cfg["model"]).to(device)
    criterion = AxisPosePPLoss(**cfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    scaler = torch.cuda.amp.GradScaler() if cfg["train"].get("amp", True) and device.type == "cuda" else None

    output_dir = Path(cfg["train"].get("output_dir", "outputs/train_shapenet"))
    output_dir.mkdir(parents=True, exist_ok=True)
    val_interval = cfg["train"].get("val_interval", 1)
    checkpoint_interval = cfg["train"].get("checkpoint_interval", 1)
    save_latest = cfg["train"].get("save_latest", True)
    metrics_path = output_dir / "metrics.csv"
    best_val = float("inf")
    global_step = 0
    start_epoch = 1
    resume_path = resolve_resume_path(cfg["train"].get("resume"), output_dir)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        loaded_epoch, global_step, best_val = load_checkpoint(resume_path, model, optimizer, scaler, device=device)
        start_epoch = loaded_epoch + 1
        print(
            f"[AxisPose++] resumed from {resume_path} "
            f"(last_epoch={loaded_epoch}, next_epoch={start_epoch}, step={global_step}, best_val={best_val:.6f})"
        )

    total_epochs = cfg["train"].get("epochs", 100)
    if start_epoch > total_epochs:
        print(f"[AxisPose++] checkpoint already reached epochs={total_epochs}; nothing to train.")
        return

    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            cfg["train"].get("log_interval", 20),
            grad_clip=cfg["train"].get("grad_clip", 1.0),
        )
        global_step += len(train_loader)
        print(f"epoch={epoch} train_loss={train_loss:.4f}")

        should_validate = val_loader is not None and val_interval > 0 and epoch % val_interval == 0
        val_metrics = None
        val_loss = train_loss
        if should_validate:
            val_metrics = evaluate(
                model,
                criterion,
                val_loader,
                device,
            )
            val_prefix = (
                f"epoch={epoch} val "
                f"objects={int(val_metrics.get('num_objects', -1))} "
                f"samples={int(val_metrics.get('num_samples', 0))} "
                f"batches={int(val_metrics.get('num_batches', 0))} "
            )
            printable_val_metrics = {
                key: value
                for key, value in val_metrics.items()
                if key not in ("num_objects", "num_samples", "num_batches")
            }
            print(val_prefix + " ".join(f"{k}={v:.4f}" for k, v in printable_val_metrics.items()))
            val_loss = val_metrics["loss"]

        metric_row = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
        }
        if val_metrics is not None:
            metric_row.update({f"val_{key}": value for key, value in val_metrics.items()})
        append_metrics(metrics_path, metric_row)
        if save_latest:
            save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, global_step, best_val, scaler=scaler)
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, global_step, best_val, scaler=scaler)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, global_step, best_val, scaler=scaler)
            print(f"[AxisPose++] saved best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

