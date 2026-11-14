"""Evaluate sensitivity to test-time changes of the object coordinate frame."""

import argparse
import copy
import csv
import hashlib
import json
import platform
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

import test_shapenet as evaluator
from axisposepp.dataset import estimate_axis_geometry
from axisposepp.heatmaps import make_center_heatmap


ROOT = Path(__file__).resolve().parents[1]


def frame_rotations(count, degrees, seed):
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angle = np.deg2rad(degrees)
    matrices = []
    for x, y, z in axes:
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        matrices.append(np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross))
    return np.asarray(matrices)


def change_frame(pose, q):
    changed = pose.copy()
    changed[:3, :3] = pose[:3, :3] @ q.T
    return changed


def render_axes(pose, size, cfg):
    width, height = size
    intrinsics = np.array(cfg["K"], dtype=np.float64)
    if cfg.get("scale_to_image"):
        src_width, src_height = cfg["src_size"]
        intrinsics[0] *= width / src_width
        intrinsics[1] *= height / src_height

    # Match the public ShapeNet axis rasterization at the source image size.
    points = np.vstack((np.zeros(3), np.eye(3)))
    camera_points = points @ pose[:3, :3].T + pose[:3, 3]
    if not np.all(camera_points[:, 2] > 1e-8):
        raise ValueError("Cannot project axes whose origin or unit endpoints are behind the camera")
    pixels = camera_points @ intrinsics.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    start = tuple(round(float(v)) for v in pixels[0])
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for end, color in zip(pixels[1:], ((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        direction = end - pixels[0]
        length = np.linalg.norm(direction)
        if length >= 1e-6:
            end = pixels[0] + direction * (max(width, height) / length)
        finish = tuple(round(float(v)) for v in end)
        cv2.arrowedLine(canvas, start, finish, color, thickness=4, line_type=cv2.LINE_8, tipLength=0.04)
    return Image.fromarray(canvas)


class ReframedPairs(Dataset):
    def __init__(self, dataset, rotations, intrinsics):
        self.dataset = dataset
        self.rotations = rotations
        self.intrinsics = intrinsics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        root = self.dataset.root_dir
        q = self.rotations[index]
        ref, query = sample["meta"]["ref"], sample["meta"]["query"]
        ref_pose = change_frame(evaluator.load_target_pose(root, ref), q)
        target_pose = change_frame(evaluator.load_target_pose(root, query), q)
        with Image.open(root / (ref + "_axisRot.png")) as original:
            ref_axes = render_axes(ref_pose, original.size, self.intrinsics)
        sample["ref_axis"] = self.dataset.axis_transform(ref_axes)
        sample["ref_pose"] = self.dataset._flatten_pose(ref_pose)
        sample["ref_pose_matrix"] = torch.from_numpy(ref_pose)
        sample["target_pose_matrix"] = torch.from_numpy(target_pose)

        # These target annotations are used by evaluator diagnostics only.
        # The network consumes the two RGB images, reference pose and reference axes.
        with Image.open(root / (query + "_axisRot.png")) as original:
            target_axes = render_axes(target_pose, original.size, self.intrinsics)
        center, directions = estimate_axis_geometry(self.dataset.axis_transform(target_axes))
        if center is None:
            center, directions = torch.zeros(2), torch.zeros(3, 2)
        size = self.dataset.image_size
        sample["target_center_norm"] = center / (size - 1)
        sample["target_directions"] = directions
        sample["target_center_heatmap"] = make_center_heatmap(center, size, size)
        return sample


def angular_error(a, b):
    cosine = (np.sum(a * b, axis=(-2, -1)) - 1) / 2
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(out, runs):
    _, baseline, _ = runs[0]
    gt = baseline["gt_RT"][:, :3, :3]
    base_valid = baseline["valid"].astype(bool)
    summary, per_pair = [], []
    for angle, result, q in runs:
        if not all(np.array_equal(result[key], baseline[key]) for key in ("ref", "query")):
            raise ValueError("Evaluated pair order changed between conditions")
        if not np.allclose(result["relative_rot_deg"], baseline["relative_rot_deg"], atol=1e-6):
            raise ValueError("Ground-truth relative rotation changed between conditions")
        for key in ("ref_RT", "gt_RT"):
            if not np.allclose(result[key][:, :3, :3] @ q, baseline[key][:, :3, :3], atol=1e-7):
                raise ValueError(f"{key} does not map back to the original frame")
        restored = result["pred_RT_c2"][:, :3, :3] @ q
        valid = result["valid"].astype(bool)
        error = np.where(valid, angular_error(restored, gt), np.nan)
        if not np.isfinite(error[valid]).all():
            raise ValueError("A prediction marked valid has a nonfinite rotation error")
        drift = angular_error(restored, baseline["pred_RT_c2"][:, :3, :3]) if angle else np.zeros(len(error))
        drift = np.where(valid & base_valid, drift, np.nan)
        acc = 100 * np.mean(error < 30)
        median = float(np.median(error[valid])) if valid.any() else None
        base_acc = summary[0]["acc30_pct"] if summary else acc
        base_median = summary[0]["median_error_deg"] if summary else median
        summary.append(dict(frame_change_deg=angle, pairs=len(error), valid=int(valid.sum()),
                            acc30_pct=acc, delta_acc30_pp=acc - base_acc, median_error_deg=median,
                            delta_median_deg=median - base_median if None not in (median, base_median) else None,
                            median_prediction_drift_deg=float(np.nanmedian(drift)) if np.isfinite(drift).any() else None))
        for i, (ref, query) in enumerate(zip(result["ref"], result["query"])):
            per_pair.append(dict(frame_change_deg=angle, pair_number=i + 1, object_id=result["object_id"][i],
                                 ref=ref, query=query, valid=int(valid[i]), error_deg=error[i], prediction_drift_deg=drift[i]))
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_errors.csv", per_pair)
    print(json.dumps(summary, indent=2))


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def write_figures(out, runs, cfg, selected):
    source = Path(cfg["data"]["root"])
    bbox = evaluator.BBoxInfoCache(cfg["visualization"])
    _, baseline, _ = runs[0]
    size, step, left = 202, 212, 112
    title_font = ImageFont.load_default(size=20)
    label_font = ImageFont.load_default(size=15)
    out.mkdir(exist_ok=True)
    panels = []
    for pair in selected:
        i = pair - 1
        ref, query = str(baseline["ref"][i]), str(baseline["query"][i])
        object_id = str(baseline["object_id"][i])
        ref_image = read_rgb(source / (ref + "_rot.png")).resize((size, size), Image.Resampling.LANCZOS)
        axis_image = read_rgb(source / (ref + "_axisRot.png"))
        target_image = read_rgb(source / (query + "_rot.png")).resize((size, size), Image.Resampling.LANCZOS)
        corners, edges = bbox.get(object_id)
        gt = baseline["gt_RT"][i]
        intrinsics = baseline["K"][i].copy()
        intrinsics[:2] *= size / cfg["data"]["image_size"]
        panel = Image.new("RGB", (left + len(runs) * step, 548), "#f4f7fb")
        draw = ImageDraw.Draw(panel)
        draw.text((12, 10), f"ShapeNet object {object_id}  |  Pair {pair:03d}", fill="#18263e", font=title_font)
        draw.text((12, 42), "Green: ground truth   Red: prediction   Both use ground-truth translation",
                  fill="#18263e", font=label_font)
        draw.text((12, 187), "Reference", fill="#18263e", font=label_font)
        draw.text((12, 397), "Target", fill="#18263e", font=label_font)
        for col, (angle, result, q) in enumerate(runs):
            x = left + step * col
            axes = axis_image if angle == 0 else render_axes(result["ref_RT"][i], axis_image.size, cfg["intrinsics"])
            axes = axes.resize((size, size), Image.Resampling.LANCZOS)
            red, green, blue = axes.split()
            mask = ImageChops.lighter(red, ImageChops.lighter(green, blue))
            panel.paste(Image.composite(axes, ref_image, mask), (x, 91))
            target = target_image.copy()
            evaluator.draw_bbox(target, corners, gt, intrinsics, color=(0, 236, 179), edges=edges)
            label = "No valid prediction"
            if result["valid"][i]:
                pred = result["pred_RT_c2"][i].copy()
                pred[:3, :3] = pred[:3, :3] @ q[i]
                pred[:3, 3] = gt[:3, 3]
                evaluator.draw_bbox(target, corners, pred, intrinsics, color=(255, 84, 99), edges=edges)
                label = f"Error {angular_error(pred[:3, :3], gt[:3, :3]):.2f}°"
            panel.paste(target, (x, 303))
            draw.text((x + size // 2, 76), f"{angle:g}°", fill="#18263e", font=label_font, anchor="mm")
            draw.text((x + size // 2, 524), label, fill="#18263e", font=label_font, anchor="mm")
        panel.save(out / f"pair_{pair:03d}_object_{object_id}.png")
        if len(selected) <= 4:
            panels.append(panel)
    if len(panels) > 1:
        width, height = panels[0].size
        sheet = Image.new("RGB", (2 * width + 12, ((len(panels) + 1) // 2) * (height + 12) - 12), "white")
        for i, panel in enumerate(panels):
            sheet.paste(panel, ((i % 2) * (width + 12), (i // 2) * (height + 12)))
        sheet.save(out / "overview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/reference_frame")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, help="Limit evaluated pairs for a quick check")
    parser.add_argument("--angles", nargs="+", type=float, default=[0, 15, 30, 60])
    parser.add_argument("--seed", type=int, default=2027)
    figures = parser.add_mutually_exclusive_group()
    figures.add_argument("--pairs", type=int, nargs="+", help="One-based pair numbers to visualize")
    figures.add_argument("--all-figures", action="store_true")
    figures.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.angles[0] != 0 or len(set(args.angles)) != len(args.angles) or any(not 0 <= a <= 180 for a in args.angles):
        parser.error("--angles must start at 0 and contain unique values in [0, 180]")
    if args.batch_size < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("batch size and sample limit must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    out = args.out.resolve()
    config_path = ROOT / "configs/test_shapenet.yaml"
    cfg = evaluator.load_merged_config(config_path)
    source = ROOT / cfg["data"]["root"]
    cfg["data"]["root"] = str(source)
    cfg["test"]["checkpoint"] = str(ROOT / cfg["test"]["checkpoint"])
    cfg["model"]["dino"]["weights_path"] = str(ROOT / cfg["model"]["dino"]["weights_path"])
    cfg["test"].update(device=args.device, batch_size=args.batch_size, num_workers=0,
                       max_samples=args.max_samples, save_pred_poses=True, pred_pose_filename="pred_poses.npz")
    cfg["visualization"].update(enabled=False, bbox_info_file=str(ROOT / cfg["visualization"]["bbox_info_file"]))
    cfg["c2"].setdefault("axis_reproj_ref_rot_filter_deg", 45.0)
    base_loader = evaluator.build_test_loader(cfg)
    dataset = base_loader.dataset
    pairs = [(sample["ref"], sample["query"]) for sample in dataset.samples]
    if len(set(pairs)) != len(pairs):
        raise ValueError("The pair list contains duplicates")
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    for angle in args.angles:
        condition = copy.deepcopy(cfg)
        run_dir = out / f"q{angle:g}"
        condition["test"]["output_dir"] = str(run_dir)
        q = frame_rotations(len(dataset), angle, args.seed)
        if angle == 0:
            loader = base_loader
        else:
            data = ReframedPairs(dataset, q, cfg["intrinsics"])
            loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"Evaluating {angle:g}° frame change", flush=True)
        with patch.object(evaluator, "build_test_loader", return_value=loader):
            evaluator.run_test(condition)
        with np.load(run_dir / "pred_poses.npz") as contents:
            result = dict(contents)
        retained = list(zip(result["ref"], result["query"]))
        if not retained:
            raise ValueError("No pairs passed the evaluation filters")
        pair_to_q = dict(zip(pairs, q))
        q = np.stack([pair_to_q[key] for key in retained])
        np.savez_compressed(run_dir / "frame_changes.npz", ref=result["ref"], query=result["query"],
                            Q=q, rotation_seed=args.seed)
        runs.append((angle, result, q))

    count = len(runs[0][1]["ref"])
    selected = args.pairs or [pair for pair in (71, 18, 86, 27) if pair <= count]
    if not selected:
        selected = list(range(1, min(count, 4) + 1))
    if args.all_figures:
        selected = list(range(1, count + 1))
    if any(not 1 <= pair <= count for pair in selected) or len(set(selected)) != len(selected):
        parser.error(f"--pairs must contain unique pair numbers in [1, {count}]")
    write_metrics(out, runs)
    if not args.no_figures:
        write_figures(out / "figures", runs, cfg, selected)
    metadata = dict(config=str(config_path.relative_to(ROOT)), angles=args.angles, seed=args.seed,
                    device=args.device, batch_size=args.batch_size, max_samples=args.max_samples,
                    visualized_pairs=[] if args.no_figures else selected, python=platform.python_version(),
                    torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__, opencv=cv2.__version__,
                    pair_list_sha256=hashlib.sha256(dataset.pair_file.read_bytes()).hexdigest())
    (out / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
