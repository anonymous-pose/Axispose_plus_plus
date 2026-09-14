import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axisposepp.c2_adapter import solve_c2_candidates_from_axes, scale_intrinsics
from axisposepp.model import AxisPosePP
from axisposepp.training import build_dataset, load_config, move_to_device, subset_first_n_objects
from axisposepp.visualization import (
    _center_from_heatmap,
    _geometry_guided_heatmap_geometry,
    _geometry_guided_ray_search_geometry,
    _line_image_from_px,
    _local_peak_candidates,
    _ray_response,
)


PARSER_NAMES = ("peak_multi",)
AXIS_SOURCES = ("pred_heatmap",)


def _deep_update(base, updates):
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_merged_config(path):
    cfg = load_config(path)
    base_config = cfg.get("base_config")
    if not base_config:
        return cfg
    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = Path(path).resolve().parent / base_path
    base = load_merged_config(base_path)
    return _deep_update(base, {k: v for k, v in cfg.items() if k != "base_config"})


def load_model_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    epoch = checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None
    step = checkpoint.get("step", None) if isinstance(checkpoint, dict) else None
    return epoch, step


def build_test_loader(cfg):
    test_cfg = cfg.get("test", {})
    split = test_cfg.get("split", "val")
    dataset = build_dataset(cfg, split)
    if dataset is None:
        raise ValueError("Could not build test dataset. Check data.root and pair file paths.")

    max_objects = test_cfg.get("max_objects", cfg.get("train", {}).get("val_max_objects"))
    if split != "train" and max_objects not in (None, "", 0, False):
        dataset = subset_first_n_objects(dataset, max_objects)

    return DataLoader(
        dataset,
        batch_size=test_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=test_cfg.get("num_workers", cfg.get("train", {}).get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )


def parse_heatmap(method, axis_heatmap, center_heatmap, geometry_dirs, test_cfg):
    common_kwargs = {
        "min_center_distance": test_cfg.get("min_center_distance", 8),
        "geo_weight": test_cfg.get("geo_weight", 0.5),
    }
    if method == "peak":
        return _geometry_guided_heatmap_geometry(
            axis_heatmap,
            center_heatmap=center_heatmap,
            geometry_dirs=geometry_dirs,
            top_k=test_cfg.get("peak_top_k", 32),
            **common_kwargs,
        )
    if method == "ray":
        return _geometry_guided_ray_search_geometry(
            axis_heatmap,
            center_heatmap=center_heatmap,
            geometry_dirs=geometry_dirs,
            num_angles=test_cfg.get("num_angles", 360),
            num_samples=test_cfg.get("num_ray_samples", 128),
            **common_kwargs,
        )
    raise ValueError(f"Unknown parser '{method}'. Valid choices: {PARSER_NAMES}")


def _dedupe_scored_directions(scored, max_count, min_angle_deg=5.0):
    if not scored:
        return []
    min_cos = float(np.cos(np.deg2rad(float(min_angle_deg))))
    kept = []
    for item in sorted(scored, key=lambda x: float(x["score"]), reverse=True):
        direction = item["direction"]
        duplicate = False
        for prev in kept:
            if float((direction * prev["direction"]).sum()) > min_cos:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(item)
        if len(kept) >= int(max_count):
            break
    return kept


def parse_heatmap_multi_candidates(axis_heatmap, center_heatmap, geometry_dirs, test_cfg):
    """Return multiple tri-axis candidates from heatmap top responses.

    Each axis channel keeps top-N directed ray candidates. The Cartesian product
    of x/y/z candidates is evaluated by C2 later.
    """
    axis_heatmap = axis_heatmap.detach().float().cpu()
    _, height, width = axis_heatmap.shape
    center = _center_from_heatmap(axis_heatmap, center_heatmap)

    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    coords = torch.stack([xs.float(), ys.float()], dim=-1)
    dist = (coords - center.view(1, 1, 2)).norm(dim=-1)
    valid = dist >= float(test_cfg.get("min_center_distance", 8))

    per_axis_top_k = int(test_cfg.get("multi_axis_top_k", 3))
    peak_top_k = int(test_cfg.get("multi_peak_top_k", test_cfg.get("peak_top_k", 64)))
    min_angle_deg = float(test_cfg.get("multi_min_angle_deg", 5.0))
    geo_weight = float(test_cfg.get("geo_weight", 0.5))
    min_center_distance = float(test_cfg.get("min_center_distance", 8))

    axis_candidates = []
    for axis_idx in range(3):
        channel = axis_heatmap[axis_idx]
        peak_values, peak_indices = _local_peak_candidates(channel, valid, top_k=peak_top_k)
        geo_dir = None
        if geometry_dirs is not None:
            geo_dir = geometry_dirs[axis_idx].detach().float().cpu()
            geo_dir = geo_dir / geo_dir.norm().clamp_min(1e-6)

        scored = []
        for rank, (peak_value, flat_idx) in enumerate(zip(peak_values, peak_indices)):
            point = torch.tensor([int(flat_idx) % width, int(flat_idx) // width], dtype=torch.float32)
            direction = point - center
            if direction.norm() < 1e-6:
                continue
            direction = direction / direction.norm().clamp_min(1e-6)
            ray_score = _ray_response(channel, center, direction, min_distance=min_center_distance)
            heatmap_score = 0.6 * ray_score + 0.4 * peak_value
            if geo_dir is not None:
                geo_score = ((direction * geo_dir).sum().clamp(-1.0, 1.0) + 1.0) * 0.5
            else:
                geo_score = torch.tensor(0.0)
            score = heatmap_score + geo_weight * geo_score
            scored.append(
                {
                    "direction": direction,
                    "score": float(score),
                    "peak_value": float(peak_value),
                    "ray_score": float(ray_score),
                    "geo_score": float(geo_score),
                    "peak_rank": int(rank),
                    "point": [float(point[0]), float(point[1])],
                }
            )

        kept = _dedupe_scored_directions(scored, per_axis_top_k, min_angle_deg=min_angle_deg)
        if not kept:
            kept = [
                {
                    "direction": torch.tensor([1.0, 0.0], dtype=torch.float32),
                    "score": 0.0,
                    "peak_value": 0.0,
                    "ray_score": 0.0,
                    "geo_score": 0.0,
                    "peak_rank": -1,
                    "point": [float(center[0] + 1.0), float(center[1])],
                }
            ]
        axis_candidates.append(kept)

    candidate_mode = str(test_cfg.get("multi_axis_candidate_mode", "full")).lower()
    if candidate_mode in ("full", "cartesian", "all"):
        combos = list(itertools.product(*[range(len(items)) for items in axis_candidates]))
        mode_meta = {"candidate_mode": "full"}
    elif candidate_mode in ("z_guided", "z_top1", "z_guided_top1"):
        search_top_k_value = test_cfg.get("z_guided_search_top_k")
        if search_top_k_value in (None, "", 0, False):
            search_top_k = per_axis_top_k
        else:
            search_top_k = int(search_top_k_value)
        search_top_k = max(search_top_k, 1)
        x_score = float(axis_candidates[0][0]["score"])
        y_score = float(axis_candidates[1][0]["score"])
        if x_score >= y_score:
            combos = [(0, cand_idx, 0) for cand_idx in range(min(search_top_k, len(axis_candidates[1])))]
            mode_meta = {
                "candidate_mode": "z_guided",
                "z_guided_fixed_axis": "x",
                "z_guided_search_axis": "y",
            }
        else:
            combos = [(cand_idx, 0, 0) for cand_idx in range(min(search_top_k, len(axis_candidates[0])))]
            mode_meta = {
                "candidate_mode": "z_guided",
                "z_guided_fixed_axis": "y",
                "z_guided_search_axis": "x",
            }
    else:
        raise ValueError(
            f"Unknown multi_axis_candidate_mode={candidate_mode}. "
            "Use 'full' or 'z_guided'."
        )

    tri_axis_candidates = []
    for combo_idx, combo in enumerate(combos):
        dirs = torch.stack([axis_candidates[axis_idx][cand_idx]["direction"] for axis_idx, cand_idx in enumerate(combo)])
        score = sum(float(axis_candidates[axis_idx][cand_idx]["score"]) for axis_idx, cand_idx in enumerate(combo))
        tri_axis_candidates.append(
            {
                "center": center,
                "dirs": dirs,
                "axis_indices": tuple(int(v) for v in combo),
                "heatmap_candidate_index": int(combo_idx),
                "heatmap_candidate_score": float(score),
                **mode_meta,
            }
        )
    return tri_axis_candidates


def gt_axes_from_batch(batch, idx, image_size):
    center_norm = batch["target_center_norm"][idx].detach().cpu().numpy().astype(np.float64)
    directions = batch["target_directions"][idx].detach().cpu().numpy().astype(np.float64)
    scale = np.asarray([image_size - 1, image_size - 1], dtype=np.float64)
    center = center_norm * scale
    return center, directions


def parse_sign_candidates(value):
    if value in (None, "", "all"):
        return None
    if value in ("none", "identity", "fixed"):
        return [(1, 1, 1)]
    candidates = []
    for item in value:
        if len(item) != 3:
            raise ValueError(f"Each sign candidate must have 3 entries, got {item}")
        candidates.append(tuple(int(v) for v in item))
    return candidates


def parse_float_list(value, fallback=None):
    if value in (None, "", False):
        return fallback
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


def get_meta_value(meta, key, idx):
    value = meta[key]
    if torch.is_tensor(value):
        return value[idx].item()
    return value[idx]


def object_id_from_query(query_prefix):
    return Path(query_prefix).parent.name


def canonical_category(name):
    name = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "display": "display",
        "monitor": "display",
        "screen": "display",
        "loudspeaker": "loudspeaker",
        "speaker": "loudspeaker",
        "sofa": "sofa",
        "couch": "sofa",
        "vessel": "vessel",
        "watercraft": "vessel",
        "boat": "vessel",
    }
    return aliases.get(name, name)


def metadata_category(item):
    for key in ("category_name", "synset_name", "class_name", "class"):
        value = item.get(key)
        if value is not None and not str(value).isdigit():
            return canonical_category(value)
    value = item.get("category_name", item.get("class_name", item.get("category", "")))
    return canonical_category(value)


def metadata_id_candidates(item):
    obj_id = str(item.get("obj_id", item.get("model_id", item.get("instance_id", "")))).strip()
    seq_id = str(item.get("id", item.get("seq_id", item.get("seqID", "")))).strip()
    ids = {value for value in (obj_id, seq_id) if value}
    if seq_id:
        try:
            ids.add(f"{int(seq_id):06d}")
        except ValueError:
            pass
    return ids


def load_category_lookup(path):
    if path in (None, "", False):
        return {}
    path = Path(path)
    if not path.exists():
        fallback = ROOT / "scripts" / "metaData_shapeNet_with_seqID.json"
        if fallback.exists():
            print(f"[AxisPose++] category metadata not found: {path}; using fallback={fallback}")
            path = fallback
        else:
            print(f"[AxisPose++] category metadata not found: {path}")
            return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise ValueError(f"Category metadata must be a list or a dict containing a list: {path}")

    lookup = {}
    for item in data:
        category = metadata_category(item)
        for key in metadata_id_candidates(item):
            lookup[key] = category
            if key.isdigit():
                lookup[f"{int(key):06d}"] = category
                lookup[str(int(key))] = category
    return lookup


def resolve_category_metadata_path(cfg):
    sym_cfg = cfg.get("symmetry", {})
    data_cfg = cfg.get("data", {})
    path = (
        sym_cfg.get("metadata_file")
        or data_cfg.get("metadata_file")
        or data_cfg.get("metadata")
        or data_cfg.get("metadata_path")
    )
    if path in (None, "", False):
        default_path = ROOT / "scripts" / "metaData_shapeNet_with_seqID.json"
        if default_path.exists():
            return default_path
        return None
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def category_for_object(object_id, category_lookup):
    return category_lookup.get(object_id, category_lookup.get(str(object_id).zfill(6), ""))


def category_from_query_prefix(query_prefix):
    """Infer a category from category/sequence/view dataset layouts.

    ShapeNet categories remain metadata-driven. This fallback is for datasets
    such as a flat export where a query looks like
    ``book_test/book_batch-10_12/00050``.
    """
    parts = Path(str(query_prefix)).parts
    if not parts:
        return ""
    category = canonical_category(parts[0])
    if category.endswith("_test"):
        category = category[:-5]
    return category


def load_target_pose(data_root, query_prefix):
    pose_path = Path(data_root) / f"{query_prefix}_pose.txt"
    pose = np.loadtxt(pose_path).astype(np.float64)
    if pose.shape == (3, 4):
        rt = np.eye(4, dtype=np.float64)
        rt[:3, :] = pose
        return rt
    if pose.shape == (4, 4):
        return pose
    raise ValueError(f"Unsupported pose shape {pose.shape}: {pose_path}")


def matrix_from_batch(batch, key, idx):
    if key not in batch:
        return None
    value = batch[key]
    if torch.is_tensor(value):
        return value[idx].detach().cpu().numpy().astype(np.float64)
    return np.asarray(value[idx], dtype=np.float64)


def pose_from_batch_or_file(batch, key, idx, data_root, prefix):
    pose = matrix_from_batch(batch, key, idx)
    if pose is not None:
        if pose.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :] = pose
            return out
        if pose.shape == (4, 4):
            return pose
        flat = pose.reshape(-1)
        if flat.size == 16:
            return flat.reshape(4, 4)
        if flat.size == 12:
            out = np.eye(4, dtype=np.float64)
            out[:3, :] = flat.reshape(3, 4)
            return out
        raise ValueError(f"Unsupported batch pose shape {pose.shape} for key={key}")
    return load_target_pose(data_root, prefix)


def intrinsics_from_batch_or_default(batch, idx, default_K):
    K = matrix_from_batch(batch, "K", idx)
    if K is None:
        return default_K
    if K.shape == (3, 3):
        return K
    if K.shape == (4, 4):
        return K[:3, :3]
    flat = K.reshape(-1)
    if flat.size >= 9:
        return flat[:9].reshape(3, 3)
    raise ValueError(f"Unsupported batch K shape {K.shape}")


def load_points_file(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        for key in ("points", "vertices", "pts", "xyz"):
            if key in data:
                points = data[key]
                break
        else:
            first_key = list(data.keys())[0]
            points = data[first_key]
    elif suffix in (".txt", ".xyz", ".pts"):
        points = np.loadtxt(path)
    elif suffix == ".ply":
        points = load_ply_points(path)
    else:
        raise ValueError(f"Unsupported model point file suffix: {path}")

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Model points must have shape N x 3, got {points.shape}: {path}")
    return points[:, :3]


def load_ply_points(path):
    try:
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
        if len(points) > 0:
            return points
    except Exception:
        pass

    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        vertex_count = None
        header_done = False
        rows = []
        for line in f:
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line == "end_header":
                header_done = True
                break
        if not header_done or vertex_count is None:
            raise ValueError(f"Could not parse ascii PLY header: {path}")
        for _ in range(vertex_count):
            parts = f.readline().strip().split()
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(rows, dtype=np.float64)


class ModelPointCache:
    def __init__(self, cfg):
        self.enabled = bool(cfg.get("enabled", True)) and cfg.get("root") not in (None, "")
        # YAML `root:` is parsed as None. Keep disabled model-point evaluation
        # usable without requiring a dummy path in every config.
        self.root = Path(cfg.get("root") or "")
        self.patterns = cfg.get(
            "patterns",
            [
                "{object_id}.npy",
                "{object_id}.npz",
                "{object_id}.ply",
                "{object_id}/points.npy",
                "{object_id}/points.npz",
                "{object_id}/model.ply",
                "{object_id}/pointcloud.ply",
            ],
        )
        self.max_points = cfg.get("max_points")
        self.seed = int(cfg.get("sample_seed", 123))
        self.cache = {}

    def resolve(self, object_id):
        for pattern in self.patterns:
            path = self.root / pattern.format(object_id=object_id)
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No model point file found for object_id={object_id}. "
            f"root={self.root}, patterns={self.patterns}"
        )

    def get(self, object_id):
        if not self.enabled:
            return None
        if object_id in self.cache:
            return self.cache[object_id]
        path = self.resolve(object_id)
        points = load_points_file(path)
        if self.max_points not in (None, "", 0, False) and len(points) > int(self.max_points):
            rng = np.random.default_rng(self.seed)
            keep = rng.choice(len(points), size=int(self.max_points), replace=False)
            points = points[keep]
        diameter = compute_diameter(points)
        self.cache[object_id] = {"points": points, "diameter": diameter, "path": path}
        return self.cache[object_id]


def transform_points(points, RT):
    return points @ RT[:3, :3].T + RT[:3, 3].reshape(1, 3)


def project_points(points_cam, K):
    z = points_cam[:, 2:3]
    valid = np.abs(z[:, 0]) > 1e-9
    uvw = points_cam @ K.T
    uv = np.full((len(points_cam), 2), np.nan, dtype=np.float64)
    uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    return uv, valid


BBOX_EDGES = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

# ``sorted_key_info.json`` stores ShapeNetData's original eight bbox vertices.
# Keep its vertex order and edge topology instead of rebuilding a cube.
SHAPENET_DATA_BBOX_EDGES = (
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7),
)


def _normalized_object_keys(object_id):
    value = str(object_id).strip()
    keys = {value}
    try:
        keys.update((str(int(value)), f"{int(value):06d}"))
    except ValueError:
        pass
    return keys


def _as_bbox_points(value):
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    # ``sorted_key_3d_info.json`` stores an object-frame center followed by
    # the eight GT bbox corners.  The center is metadata, not a drawable
    # corner, so discard it before constructing bbox edges.
    if points.ndim == 2 and points.shape[0] == 9 and points.shape[1] >= 3:
        return points[1:, :3]
    if points.ndim == 2 and points.shape[0] == 8 and points.shape[1] >= 3:
        return points[:, :3]
    if points.size == 24:
        return points.reshape(8, 3)
    return None


def _find_bbox_points(value, visited=None):
    """Find explicit 8x3 bbox points in a JSON record."""
    if visited is None:
        visited = set()
    value_id = id(value)
    if value_id in visited:
        return None
    visited.add(value_id)

    direct = _as_bbox_points(value)
    if direct is not None:
        return direct
    if isinstance(value, dict):
        for key in (
            "bbox_3d_points", "bbox3d_points", "bbox_3d", "bbox3d",
            "bbox_points", "box_3d", "box3d", "corners", "bbox",
        ):
            if key in value:
                found = _find_bbox_points(value[key], visited)
                if found is not None:
                    return found
        for item in value.values():
            found = _find_bbox_points(item, visited)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_bbox_points(item, visited)
            if found is not None:
                return found
    return None


def infer_bbox_edges(corners):
    """Infer cuboid edges from the stored object-frame corner coordinates."""
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        return None
    extent = np.ptp(corners, axis=0)
    tolerance = max(float(extent.max()), 1.0) * 1e-8
    edges = []
    for first in range(len(corners)):
        for second in range(first + 1, len(corners)):
            changed_axes = np.count_nonzero(np.abs(corners[first] - corners[second]) > tolerance)
            if changed_axes == 1:
                edges.append((first, second))
    return tuple(edges) if len(edges) == 12 else None


class BBoxInfoCache:
    """Read per-instance 3D bbox vertices from JSON for visualization only."""

    def __init__(self, cfg):
        self.path = cfg.get("bbox_info_file")
        self.required = bool(cfg.get("bbox_required", False))
        self.enabled = self.path not in (None, "", False)
        self.cache = {}
        self.data = None
        if self.enabled:
            self.path = Path(self.path)
            if not self.path.exists():
                raise FileNotFoundError(f"3D bbox metadata file not found: {self.path}")
            with self.path.open("r", encoding="utf-8") as f:
                self.data = json.load(f)

    def _record_for_object(self, object_id):
        keys = _normalized_object_keys(object_id)
        if isinstance(self.data, dict):
            for key in keys:
                if key in self.data:
                    return self.data[key]
            records = self.data
            for container_key in ("data", "items", "objects", "models", "key_info", "sorted_key_info"):
                if isinstance(self.data.get(container_key), (dict, list)):
                    records = self.data[container_key]
                    break
        else:
            records = self.data
        if isinstance(records, dict):
            for key in keys:
                if key in records:
                    return records[key]
            records = list(records.values())
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                record_keys = set()
                for key in ("id", "seq_id", "seqID", "instance_id", "model_id", "object_id"):
                    if record.get(key) is not None:
                        record_keys.update(_normalized_object_keys(record[key]))
                if keys & record_keys:
                    return record
        return None

    def get(self, object_id):
        if not self.enabled:
            return None, None
        object_id = str(object_id)
        if object_id in self.cache:
            return self.cache[object_id]
        record = self._record_for_object(object_id)
        points = _find_bbox_points(record) if record is not None else None
        if points is None:
            message = f"Could not find eight 3D bbox points for object_id={object_id} in {self.path}"
            if self.required:
                raise KeyError(message)
            print(f"[AxisPose++] warning: {message}; using configured visualization bbox")
            result = (None, None)
        else:
            # Prefer geometry-derived edges: different ShapeNet metadata files
            # preserve different corner orders.  Retain the older topology as
            # a fallback for malformed or non-axis-aligned records.
            result = (points, infer_bbox_edges(points) or SHAPENET_DATA_BBOX_EDGES)
        self.cache[object_id] = result
        return result


def make_bbox_corners(vis_cfg, points=None):
    if points is not None and len(points) > 0:
        points = np.asarray(points, dtype=np.float64)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
    elif vis_cfg.get("bbox_min") is not None and vis_cfg.get("bbox_max") is not None:
        mins = np.asarray(vis_cfg["bbox_min"], dtype=np.float64).reshape(3)
        maxs = np.asarray(vis_cfg["bbox_max"], dtype=np.float64).reshape(3)
    else:
        size = np.asarray(vis_cfg.get("bbox_size", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3)
        center = np.asarray(vis_cfg.get("bbox_center", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        mins = center - size * 0.5
        maxs = center + size * 0.5

    xs = [mins[0], maxs[0]]
    ys = [mins[1], maxs[1]]
    zs = [mins[2], maxs[2]]
    return np.asarray([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float64)


def draw_bbox(image, corners, RT, K, color=(255, 60, 40), width=2, edges=BBOX_EDGES):
    draw = ImageDraw.Draw(image)
    if RT is None or corners is None:
        return image
    corners_cam = transform_points(corners, RT)
    uv, valid_z = project_points(corners_cam, K)
    valid = valid_z & np.isfinite(uv).all(axis=1)
    for i, j in edges:
        if valid[i] and valid[j]:
            draw.line(
                (float(uv[i, 0]), float(uv[i, 1]), float(uv[j, 0]), float(uv[j, 1])),
                fill=color,
                width=int(width),
            )
    return image


def add_label(image, text):
    draw = ImageDraw.Draw(image)
    pad = 4
    box = draw.textbbox((pad, pad), text)
    draw.rectangle((0, 0, box[2] + pad, box[3] + pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255))
    return image


def load_rgb_image(data_root, prefix, image_size):
    root = Path(data_root)
    candidates = [
        root / f"{prefix}_rot.png",
        root / f"{prefix}.png",
        root / f"{prefix}.jpg",
        root / f"{prefix}.jpeg",
    ]
    parts = Path(prefix).parts
    if len(parts) >= 2:
        object_name, view = parts[0], parts[-1]
        sequence = Path(*parts[:-1])
        candidates.extend(
            [
                root / object_name / "color" / f"{view}.png",
                root / object_name / "color" / f"{view}.jpg",
                root / object_name / "color" / f"{view}.jpeg",
                root / "testShapeNet" / object_name / f"{view}.png",
                root / "testShapeNet" / object_name / f"{view}.jpg",
                root / "testShapeNet" / object_name / f"{view}.jpeg",
                root / "testShapenet" / object_name / f"{view}.png",
                root / "testShapenet" / object_name / f"{view}.jpg",
                root / "testShapenet" / object_name / f"{view}.jpeg",
                root / sequence / f"{view}_color.png",
                root / sequence / f"{view}_color.jpg",
                root / sequence / f"{view}_color.jpeg",
            ]
        )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Could not find RGB image for prefix={prefix}; tried={candidates}")
    image = Image.open(path).convert("RGB")
    return image.resize((image_size, image_size), Image.BILINEAR)


def make_axis_heatmap_image(axis_heatmap, center_heatmap=None):
    if axis_heatmap is None:
        return None
    heatmap = axis_heatmap.detach().float().cpu()
    if heatmap.min() < 0.0 or heatmap.max() > 1.0:
        heatmap = torch.sigmoid(heatmap)
    heatmap = heatmap.clamp(0.0, 1.0)
    for channel_idx in range(min(3, heatmap.shape[0])):
        channel = heatmap[channel_idx]
        max_val = channel.max().clamp_min(1e-6)
        heatmap[channel_idx] = (channel / max_val).clamp(0.0, 1.0).pow(0.65)
    array = (heatmap[:3].permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    image = Image.fromarray(array)

    if center_heatmap is not None:
        center = center_heatmap.detach().float().cpu().squeeze(0)
        if center.min() < 0.0 or center.max() > 1.0:
            center = torch.sigmoid(center)
        flat_idx = int(center.argmax())
        cx = flat_idx % center.shape[1]
        cy = flat_idx // center.shape[1]
        draw = ImageDraw.Draw(image)
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 0))
    return image


def save_pose_bbox_visual(
    output_dir,
    sample_idx,
    data_root,
    ref_prefix,
    query_prefix,
    ref_RT,
    gt_RT,
    pred_RT,
    K,
    image_size,
    corners,
    bbox_edges,
    pred_axis_center,
    pred_axis_dirs,
    gt_axis_center,
    gt_axis_dirs,
    pred_axis_heatmap=None,
    pred_center_heatmap=None,
    metrics=None,
    line_width=2,
    target_image_root=None,
):
    ref_image = load_rgb_image(data_root, ref_prefix, image_size)
    target_root = data_root if target_image_root in (None, "", False) else target_image_root
    try:
        target_bbox = load_rgb_image(target_root, query_prefix, image_size)
    except FileNotFoundError:
        # Preserve the normal clean-image fallback used by ShapeNetDataset.
        target_bbox = load_rgb_image(data_root, query_prefix, image_size)

    draw_bbox(ref_image, corners, ref_RT, K, color=(80, 220, 80), width=line_width, edges=bbox_edges)
    draw_bbox(target_bbox, corners, gt_RT, K, color=(80, 220, 80), width=line_width, edges=bbox_edges)
    draw_bbox(target_bbox, corners, pred_RT, K, color=(255, 70, 50), width=line_width, edges=bbox_edges)

    pred_axes = _line_image_from_px(
        image_size,
        image_size,
        torch.as_tensor(pred_axis_center, dtype=torch.float32),
        torch.as_tensor(pred_axis_dirs, dtype=torch.float32),
    )
    gt_axes = _line_image_from_px(
        image_size,
        image_size,
        torch.as_tensor(gt_axis_center, dtype=torch.float32),
        torch.as_tensor(gt_axis_dirs, dtype=torch.float32),
    )
    pred_heatmap = make_axis_heatmap_image(pred_axis_heatmap, pred_center_heatmap)
    if pred_heatmap is None:
        pred_heatmap = Image.new("RGB", (image_size, image_size), (0, 0, 0))

    add_label(ref_image, "ref GT bbox")
    target_name = "target degraded" if target_image_root not in (None, "", False) else "target"
    add_label(target_bbox, f"{target_name} bbox GT green / pred red")
    add_label(pred_heatmap, "pred heatmap RGB")
    if metrics is not None and np.isfinite(metrics.get("rot_deg", np.nan)):
        pred_label = f"pred axes rot={metrics['rot_deg']:.2f}"
    else:
        pred_label = "pred axes invalid"
    add_label(pred_axes, pred_label)
    add_label(gt_axes, "GT axes")

    panel = Image.new("RGB", (image_size * 5, image_size), (20, 20, 20))
    panel.paste(ref_image, (0, 0))
    panel.paste(target_bbox, (image_size, 0))
    panel.paste(pred_heatmap, (image_size * 2, 0))
    panel.paste(pred_axes, (image_size * 3, 0))
    panel.paste(gt_axes, (image_size * 4, 0))

    vis_dir = Path(output_dir) / "bbox_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    path = vis_dir / f"sample_{sample_idx:06d}.png"
    panel.save(path)
    return path


def reprojection_error(points, pred_RT, gt_RT, K):
    pred_cam = transform_points(points, pred_RT)
    gt_cam = transform_points(points, gt_RT)
    pred_uv, pred_valid = project_points(pred_cam, K)
    gt_uv, gt_valid = project_points(gt_cam, K)
    valid = pred_valid & gt_valid & np.isfinite(pred_uv).all(axis=1) & np.isfinite(gt_uv).all(axis=1)
    if not valid.any():
        return float("inf")
    return float(np.linalg.norm(pred_uv[valid] - gt_uv[valid], axis=1).mean())


def add_error(points, pred_RT, gt_RT):
    pred = transform_points(points, pred_RT)
    gt = transform_points(points, gt_RT)
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def adds_error(points, pred_RT, gt_RT):
    pred = transform_points(points, pred_RT)
    gt = transform_points(points, gt_RT)
    try:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(gt).query(pred, k=1)
        return float(dist.mean())
    except Exception:
        return float(chunked_nearest_mean(pred, gt))


def chunked_nearest_mean(pred, gt, chunk=2048):
    total = 0.0
    count = 0
    for start in range(0, len(pred), chunk):
        part = pred[start : start + chunk]
        dist2 = ((part[:, None, :] - gt[None, :, :]) ** 2).sum(axis=-1)
        total += np.sqrt(dist2.min(axis=1)).sum()
        count += len(part)
    return total / max(count, 1)


def compute_diameter(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) > 5000:
        rng = np.random.default_rng(123)
        points = points[rng.choice(len(points), size=5000, replace=False)]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def rotation_error_deg(R_pred, R_gt):
    R_delta = R_pred @ R_gt.T
    cos = (np.trace(R_delta) - 1.0) * 0.5
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def rotation_axis_error_deg(R_pred, R_gt, axis="z"):
    """Angular error for one object axis after applying the two rotations."""
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    try:
        axis_index = axis_to_index[str(axis).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown symmetry axis: {axis}") from exc
    pred_axis = np.asarray(R_pred, dtype=np.float64)[:, axis_index]
    gt_axis = np.asarray(R_gt, dtype=np.float64)[:, axis_index]
    denom = max(np.linalg.norm(pred_axis) * np.linalg.norm(gt_axis), 1e-9)
    cos = float(np.clip(np.dot(pred_axis, gt_axis) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def rotation_z(deg):
    angle = np.deg2rad(float(deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def parse_symmetry_categories(sym_cfg):
    if not bool(sym_cfg.get("enabled", True)):
        return {}
    categories = sym_cfg.get("categories")
    if categories in (None, "", False):
        categories = {"mug": "z_180"}
    if not isinstance(categories, dict):
        raise ValueError("symmetry.categories must be a mapping, e.g. {mug: z_180}")
    return {canonical_category(key): value for key, value in categories.items()}


def symmetry_rotations_for_category(category, symmetry_categories):
    category = canonical_category(category or "")
    spec = symmetry_categories.get(category)
    if spec in (None, "", False):
        return []
    if isinstance(spec, str):
        specs = [spec]
    else:
        specs = list(spec)

    rotations = []
    for item in specs:
        if isinstance(item, str):
            name = item.lower().strip()
            if name in ("z_180", "rz_180", "object_z_180"):
                rotations.append(rotation_z(180.0))
            elif name in ("z_axis", "axis_z", "continuous_z", "z_continuous"):
                # Handled by axis_only_rotation_for_category below.
                continue
            elif name in ("none", "identity"):
                continue
            else:
                raise ValueError(f"Unknown symmetry spec for category={category}: {item}")
        elif isinstance(item, (int, float)):
            rotations.append(rotation_z(float(item)))
        else:
            raise ValueError(f"Unsupported symmetry spec for category={category}: {item}")
    return rotations


def axis_only_rotation_for_category(category, symmetry_categories):
    """Return the evaluated object axis for a continuous axial symmetry."""
    spec = symmetry_categories.get(canonical_category(category or ""))
    if spec in (None, "", False):
        return None
    specs = [spec] if isinstance(spec, str) else list(spec)
    for item in specs:
        if isinstance(item, str) and item.lower().strip() in (
            "z_axis", "axis_z", "continuous_z", "z_continuous"
        ):
            return "z"
    return None


def symmetry_aware_rotation_error_deg(R_pred, R_gt, symmetry_rotations=None, axis_only=None):
    if axis_only is not None:
        error = rotation_axis_error_deg(R_pred, R_gt, axis=axis_only)
        return error, error, False
    raw = rotation_error_deg(R_pred, R_gt)
    if not symmetry_rotations:
        return raw, raw, False
    best = raw
    used_symmetry = False
    for R_sym in symmetry_rotations:
        err = rotation_error_deg(R_pred, R_gt @ R_sym)
        if err < best:
            best = err
            used_symmetry = True
    return best, raw, used_symmetry


def translation_error_cm(t_pred, t_gt, unit_to_meter=1.0):
    return float(np.linalg.norm(t_pred.reshape(3) - t_gt.reshape(3)) * float(unit_to_meter) * 100.0)


def align_translation_scale(pred_RT, gt_RT, mode="none", fixed_scale=1.0, eps=1e-9):
    """Apply a scalar translation scale to a predicted pose.

    C2 recovers translation up to the scale implied by lamda0. In diagnostic
    evaluation, GT pose can estimate this scalar so rotation/sign errors are not
    mixed with a poorly calibrated depth scale.
    """
    pred_RT = np.asarray(pred_RT, dtype=np.float64).copy()
    mode = (mode or "none").lower()
    pred_t = pred_RT[:3, 3].reshape(3)
    gt_t = np.asarray(gt_RT, dtype=np.float64)[:3, 3].reshape(3)

    if mode in ("none", "off", "false"):
        scale = 1.0
    elif mode == "fixed":
        scale = float(fixed_scale)
    elif mode in ("gt_norm", "norm"):
        scale = np.linalg.norm(gt_t) / max(np.linalg.norm(pred_t), eps)
    elif mode in ("gt_lstsq", "lstsq", "least_squares"):
        scale = float(np.dot(pred_t, gt_t) / max(np.dot(pred_t, pred_t), eps))
    elif mode in ("gt_lstsq_pos", "lstsq_pos", "least_squares_pos"):
        scale = float(np.dot(pred_t, gt_t) / max(np.dot(pred_t, pred_t), eps))
        scale = max(scale, eps)
    elif mode in ("gt_z", "z"):
        scale = float(gt_t[2] / pred_t[2]) if abs(pred_t[2]) > eps else 1.0
    else:
        raise ValueError(f"Unknown translation scale mode: {mode}")

    if not np.isfinite(scale):
        scale = 1.0
    pred_RT[:3, 3] = pred_t * scale
    return pred_RT, float(scale)


def make_intrinsics(cfg, image_size):
    intr_cfg = cfg["intrinsics"]
    K = np.asarray(intr_cfg["K"], dtype=np.float64).reshape(3, 3)
    if intr_cfg.get("scale_to_image", False):
        src_size = intr_cfg.get("src_size")
        if src_size is None:
            raise ValueError("intrinsics.src_size is required when scale_to_image=true")
        K = scale_intrinsics(K, src_size=tuple(src_size), dst_size=(image_size, image_size))
    return K


def candidate_pose_metrics(
    candidate,
    points,
    gt_RT,
    K,
    unit_to_meter=1.0,
    translation_scale_mode="none",
    fixed_translation_scale=1.0,
    symmetry_rotations=None,
    axis_only=None,
):
    if not candidate["valid"] or candidate["RT"] is None:
        return {
            "reproj": None,
            "add": None,
            "adds": None,
            "rot_deg": float("inf"),
            "trans_cm": float("inf"),
            "translation_scale": None,
        }
    pred_RT, scale = align_translation_scale(
        candidate["RT"],
        gt_RT,
        mode=translation_scale_mode,
        fixed_scale=fixed_translation_scale,
    )
    rot_deg, rot_deg_no_sym, symmetry_used = symmetry_aware_rotation_error_deg(
        pred_RT[:3, :3],
        gt_RT[:3, :3],
        symmetry_rotations=symmetry_rotations,
        axis_only=axis_only,
    )
    metrics = {
        "reproj": None,
        "add": None,
        "adds": None,
        "rot_deg": rot_deg,
        "rot_deg_no_sym": rot_deg_no_sym,
        "symmetry_used": int(symmetry_used),
        "rotation_metric": f"{axis_only}_axis" if axis_only is not None else "so3",
        "trans_cm": translation_error_cm(pred_RT[:3, 3], gt_RT[:3, 3], unit_to_meter=unit_to_meter),
        "translation_scale": scale,
    }
    if points is not None:
        metrics["reproj"] = reprojection_error(points, pred_RT, gt_RT, K)
        metrics["add"] = add_error(points, pred_RT, gt_RT)
        metrics["adds"] = adds_error(points, pred_RT, gt_RT)
    return metrics


def project_pose_axes(RT, K, axis_length=1.0):
    axis_length = float(axis_length)
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )
    points_cam = transform_points(points, RT)
    return project_points(points_cam, K)


def refine_candidate_pose_from_axes(candidate, K, score_cfg=None, ref_RT=None):
    """Refine one C2 pose by matching projected canonical axes to predicted 2D axes."""
    score_cfg = score_cfg or {}
    if not bool(score_cfg.get("pose_refine_enabled", False)):
        return candidate
    if not candidate.get("valid") or candidate.get("RT") is None or candidate.get("points_h") is None:
        candidate["pose_refine_success"] = 0
        candidate["pose_refine_cost"] = None
        candidate["pose_refine_nfev"] = 0
        return candidate

    try:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation as SciRotation
    except Exception as exc:
        candidate["pose_refine_success"] = 0
        candidate["pose_refine_error"] = f"scipy unavailable: {exc}"
        candidate["pose_refine_cost"] = None
        candidate["pose_refine_nfev"] = 0
        return candidate

    points_h = np.asarray(candidate["points_h"], dtype=np.float64)
    if points_h.shape != (3, 4):
        candidate["pose_refine_success"] = 0
        candidate["pose_refine_error"] = "points_h shape is not 3x4"
        candidate["pose_refine_cost"] = None
        candidate["pose_refine_nfev"] = 0
        return candidate

    target_uv = (points_h[:2, :] / np.maximum(np.abs(points_h[2:3, :]), 1e-9)).T
    target_dirs = target_uv[1:] - target_uv[:1]
    target_norm = np.linalg.norm(target_dirs, axis=1, keepdims=True)
    if np.any(target_norm[:, 0] < 1e-6):
        candidate["pose_refine_success"] = 0
        candidate["pose_refine_error"] = "degenerate target axes"
        candidate["pose_refine_cost"] = None
        candidate["pose_refine_nfev"] = 0
        return candidate
    target_dirs = target_dirs / np.maximum(target_norm, 1e-9)

    RT0 = np.asarray(candidate["RT"], dtype=np.float64)
    try:
        rvec0 = SciRotation.from_matrix(RT0[:3, :3]).as_rotvec()
    except Exception as exc:
        candidate["pose_refine_success"] = 0
        candidate["pose_refine_error"] = f"bad initial rotation: {exc}"
        candidate["pose_refine_cost"] = None
        candidate["pose_refine_nfev"] = 0
        return candidate
    t0 = RT0[:3, 3].reshape(3)
    params0 = np.concatenate([rvec0, t0])

    axis_length = float(score_cfg.get("pose_refine_axis_length", 1.0))
    image_scale = float(score_cfg.get("pose_refine_image_scale", max(abs(K[0, 2]) * 2.0, abs(K[1, 2]) * 2.0, 1.0)))
    center_weight = float(score_cfg.get("pose_refine_center_weight", 1.0))
    dir_weight = float(score_cfg.get("pose_refine_dir_weight", 1.0))
    endpoint_weight = float(score_cfg.get("pose_refine_endpoint_weight", 0.0))
    init_rot_weight = float(score_cfg.get("pose_refine_init_rot_weight", 0.0))
    init_trans_weight = float(score_cfg.get("pose_refine_init_trans_weight", 0.0))
    ref_rot_weight = float(score_cfg.get("pose_refine_ref_rot_weight", 0.0))
    ref_rot_max_deg = score_cfg.get("pose_refine_ref_rot_max_deg", None)
    residual_size = 0
    if center_weight > 0:
        residual_size += 2
    if dir_weight > 0:
        residual_size += 6
    if endpoint_weight > 0:
        residual_size += 6
    if init_rot_weight > 0:
        residual_size += 3
    if init_trans_weight > 0:
        residual_size += 3
    if ref_RT is not None and ref_rot_weight > 0:
        residual_size += 1
    residual_size = max(residual_size, 1)

    def residual(params):
        RT = np.eye(4, dtype=np.float64)
        try:
            RT[:3, :3] = SciRotation.from_rotvec(params[:3]).as_matrix()
        except Exception:
            return np.ones(residual_size, dtype=np.float64) * 1e3
        RT[:3, 3] = params[3:6]

        pred_uv, valid_z = project_pose_axes(RT, K, axis_length=axis_length)
        if not valid_z.all() or not np.isfinite(pred_uv).all():
            return np.ones(residual_size, dtype=np.float64) * 1e3

        pred_dirs = pred_uv[1:] - pred_uv[:1]
        pred_norm = np.linalg.norm(pred_dirs, axis=1, keepdims=True)
        if np.any(pred_norm[:, 0] < 1e-6):
            return np.ones(residual_size, dtype=np.float64) * 1e3
        pred_dirs = pred_dirs / np.maximum(pred_norm, 1e-9)

        parts = []
        if center_weight > 0:
            parts.append(center_weight * (pred_uv[0] - target_uv[0]) / image_scale)
        if dir_weight > 0:
            parts.append(dir_weight * (pred_dirs - target_dirs).reshape(-1))
        if endpoint_weight > 0:
            parts.append(endpoint_weight * ((pred_uv[1:] - target_uv[1:]) / image_scale).reshape(-1))
        if init_rot_weight > 0:
            rel = SciRotation.from_matrix(RT[:3, :3] @ RT0[:3, :3].T).as_rotvec()
            parts.append(init_rot_weight * rel)
        if init_trans_weight > 0:
            denom = max(np.linalg.norm(t0), 1e-6)
            parts.append(init_trans_weight * (RT[:3, 3] - t0) / denom)
        if ref_RT is not None and ref_rot_weight > 0:
            ref_rot_deg = rotation_error_deg(RT[:3, :3], np.asarray(ref_RT, dtype=np.float64)[:3, :3])
            if ref_rot_max_deg not in (None, "", 0, False):
                ref_rot_deg = max(0.0, ref_rot_deg - float(ref_rot_max_deg))
            parts.append(np.asarray([ref_rot_weight * ref_rot_deg / 180.0], dtype=np.float64))
        if not parts:
            return np.zeros(1, dtype=np.float64)
        return np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in parts])

    try:
        result = least_squares(
            residual,
            params0,
            max_nfev=int(score_cfg.get("pose_refine_max_nfev", 50)),
            loss=str(score_cfg.get("pose_refine_loss", "linear")),
            ftol=float(score_cfg.get("pose_refine_ftol", 1e-6)),
            xtol=float(score_cfg.get("pose_refine_xtol", 1e-6)),
            gtol=float(score_cfg.get("pose_refine_gtol", 1e-6)),
        )
    except Exception as exc:
        candidate["pose_refine_success"] = 0
        candidate["pose_refine_error"] = str(exc)
        candidate["pose_refine_cost"] = None
        candidate["pose_refine_nfev"] = 0
        return candidate

    refined = np.eye(4, dtype=np.float64)
    refined[:3, :3] = SciRotation.from_rotvec(result.x[:3]).as_matrix()
    refined[:3, 3] = result.x[3:6]

    candidate["RT_before_refine"] = RT0.copy()
    candidate["R_before_refine"] = candidate.get("R")
    candidate["t_before_refine"] = candidate.get("t")
    candidate["RT"] = refined
    candidate["R"] = refined[:3, :3]
    candidate["t"] = refined[:3, 3]
    candidate["pose_refine_success"] = int(bool(result.success) and np.isfinite(refined).all())
    candidate["pose_refine_cost"] = float(result.cost)
    candidate["pose_refine_nfev"] = int(result.nfev)
    candidate["pose_refine_axis_length"] = axis_length
    return candidate


def candidate_axis_reprojection_score(candidate, K, score_cfg=None, ref_RT=None):
    """Score a C2 candidate by reprojecting its final pose back to the input 2D axes.

    This does not use GT pose. It checks whether the orthonormalized C2 pose still
    projects to the same center/rays that were given to C2.
    """
    score_cfg = score_cfg or {}
    if not candidate.get("valid") or candidate.get("RT") is None:
        return {
            "axis_reproj_score": float("inf"),
            "axis_reproj_dir_deg": float("inf"),
            "axis_reproj_endpoint_px": float("inf"),
            "axis_reproj_center_px": float("inf"),
        }

    points_h = candidate.get("points_h")
    X = candidate.get("X")
    if points_h is None or X is None:
        return {
            "axis_reproj_score": float("inf"),
            "axis_reproj_dir_deg": float("inf"),
            "axis_reproj_endpoint_px": float("inf"),
            "axis_reproj_center_px": float("inf"),
        }

    points_h = np.asarray(points_h, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if points_h.shape != (3, 4) or X.shape != (3, 4) or not np.isfinite(X).all():
        return {
            "axis_reproj_score": float("inf"),
            "axis_reproj_dir_deg": float("inf"),
            "axis_reproj_endpoint_px": float("inf"),
            "axis_reproj_center_px": float("inf"),
        }

    target_uv = (points_h[:2, :] / np.maximum(np.abs(points_h[2:3, :]), 1e-9)).T
    target_dirs = target_uv[1:] - target_uv[:1]
    target_norm = np.linalg.norm(target_dirs, axis=1, keepdims=True)
    if np.any(target_norm[:, 0] < 1e-6):
        return {
            "axis_reproj_score": float("inf"),
            "axis_reproj_dir_deg": float("inf"),
            "axis_reproj_endpoint_px": float("inf"),
            "axis_reproj_center_px": float("inf"),
        }
    target_dirs = target_dirs / np.maximum(target_norm, 1e-9)

    RT = np.asarray(candidate["RT"], dtype=np.float64)
    center_cam = RT[:3, 3]
    axis_length_cfg = score_cfg.get("axis_reproj_axis_length", None)
    if axis_length_cfg in (None, "", 0, False):
        axis_lengths = np.linalg.norm(X[:, 1:] - X[:, :1], axis=0)
        axis_lengths = np.maximum(axis_lengths, 1e-6)
    else:
        axis_lengths = np.full(3, float(axis_length_cfg), dtype=np.float64)
    axis_points_cam = np.vstack(
        [
            center_cam.reshape(1, 3),
            center_cam + RT[:3, 0] * axis_lengths[0],
            center_cam + RT[:3, 1] * axis_lengths[1],
            center_cam + RT[:3, 2] * axis_lengths[2],
        ]
    )
    pred_uv, valid_z = project_points(axis_points_cam, K)
    if not valid_z.all() or not np.isfinite(pred_uv).all():
        return {
            "axis_reproj_score": float("inf"),
            "axis_reproj_dir_deg": float("inf"),
            "axis_reproj_endpoint_px": float("inf"),
            "axis_reproj_center_px": float("inf"),
        }

    pred_dirs = pred_uv[1:] - pred_uv[:1]
    pred_norm = np.linalg.norm(pred_dirs, axis=1, keepdims=True)
    if np.any(pred_norm[:, 0] < 1e-6):
        return {
            "axis_reproj_score": float("inf"),
            "axis_reproj_dir_deg": float("inf"),
            "axis_reproj_endpoint_px": float("inf"),
            "axis_reproj_center_px": float("inf"),
        }
    pred_dirs = pred_dirs / np.maximum(pred_norm, 1e-9)

    dots = np.sum(pred_dirs * target_dirs, axis=1).clip(-1.0, 1.0)
    dir_deg = float(np.degrees(np.arccos(dots)).mean())
    endpoint_px = float(np.linalg.norm(pred_uv[1:] - target_uv[1:], axis=1).mean())
    center_px = float(np.linalg.norm(pred_uv[0] - target_uv[0]))

    dir_weight = float(score_cfg.get("axis_reproj_dir_weight", 1.0))
    endpoint_weight = float(score_cfg.get("axis_reproj_endpoint_weight", 0.05))
    center_weight = float(score_cfg.get("axis_reproj_center_weight", 0.1))
    imag_weight = float(score_cfg.get("axis_reproj_imag_weight", 0.05))
    distance_weight = float(score_cfg.get("axis_reproj_distance_weight", 0.0))
    heatmap_weight = float(score_cfg.get("axis_reproj_heatmap_weight", 0.0))
    preferred_distance = score_cfg.get("axis_reproj_preferred_distance", None)
    ref_rot_weight = float(score_cfg.get("axis_reproj_ref_rot_weight", 0.0))
    ref_rot_max_deg = score_cfg.get("axis_reproj_ref_rot_max_deg", None)
    ref_rot_filter_deg = score_cfg.get("axis_reproj_ref_rot_filter_deg", None)

    imag = 0.0
    for key in ("max_imag_X", "max_imag_R"):
        value = candidate.get(key)
        if value is not None and np.isfinite(value):
            imag += float(value)

    distance_term = 0.0
    point_distance = candidate.get("point_distance")
    if preferred_distance not in (None, "", 0, False) and point_distance is not None:
        preferred_distance = float(preferred_distance)
        distance_term = abs(float(point_distance) - preferred_distance) / max(abs(preferred_distance), 1e-6)

    ref_rot_deg = None
    ref_rot_term = 0.0
    needs_ref_rot = ref_rot_weight > 0 or ref_rot_filter_deg not in (None, "", 0, False)
    if ref_RT is not None and needs_ref_rot:
        ref_rot_deg = rotation_error_deg(RT[:3, :3], np.asarray(ref_RT, dtype=np.float64)[:3, :3])
        if ref_rot_max_deg not in (None, "", 0, False):
            ref_rot_term = max(0.0, ref_rot_deg - float(ref_rot_max_deg))
        else:
            ref_rot_term = ref_rot_deg

    score = (
        dir_weight * dir_deg
        + endpoint_weight * endpoint_px
        + center_weight * center_px
        + imag_weight * imag
        + distance_weight * distance_term
        + ref_rot_weight * ref_rot_term
        - heatmap_weight * float(candidate.get("heatmap_candidate_score") or 0.0)
    )
    return {
        "axis_reproj_score": float(score),
        "axis_reproj_dir_deg": dir_deg,
        "axis_reproj_endpoint_px": endpoint_px,
        "axis_reproj_center_px": center_px,
        "axis_reproj_ref_rot_deg": ref_rot_deg,
    }


def select_candidate(
    candidates,
    points,
    gt_RT,
    K,
    selection,
    unit_to_meter=1.0,
    translation_scale_mode="none",
    fixed_translation_scale=1.0,
    symmetry_rotations=None,
    axis_only=None,
    ref_RT=None,
    score_cfg=None,
):
    score_cfg = score_cfg or {}
    scored = []
    for candidate in candidates:
        metrics = candidate_pose_metrics(
            candidate,
            points,
            gt_RT,
            K,
            unit_to_meter=unit_to_meter,
            translation_scale_mode=translation_scale_mode,
            fixed_translation_scale=fixed_translation_scale,
            symmetry_rotations=symmetry_rotations,
            axis_only=axis_only,
        )
        scored.append((candidate, metrics))

    valid = [(c, m) for c, m in scored if c["valid"] and c["RT"] is not None]
    if not valid:
        return None, None

    if selection == "first_valid":
        return valid[0]
    if selection in ("axis_reprojection", "c2_axis_reprojection", "axis_reproj"):
        for candidate, _ in valid:
            candidate.update(candidate_axis_reprojection_score(candidate, K, score_cfg=score_cfg, ref_RT=ref_RT))
        filter_deg = score_cfg.get("axis_reproj_ref_rot_filter_deg", None)
        filtered = valid
        if filter_deg not in (None, "", 0, False):
            filter_deg = float(filter_deg)
            within_ref = [
                item
                for item in valid
                if item[0].get("axis_reproj_ref_rot_deg") is not None
                and np.isfinite(item[0].get("axis_reproj_ref_rot_deg"))
                and float(item[0].get("axis_reproj_ref_rot_deg")) <= filter_deg
            ]
            if within_ref:
                filtered = within_ref
        selected = min(filtered, key=lambda item: item[0].get("axis_reproj_score", float("inf")))
        selected[0]["axis_reproj_ref_rot_filter_deg"] = filter_deg if filter_deg not in (None, "", 0, False) else None
        selected[0]["axis_reproj_ref_rot_filter_used"] = int(len(filtered) < len(valid))
        selected[0]["axis_reproj_ref_rot_filter_candidates"] = len(filtered)
        return selected
    if selection in ("confidence", "heatmap_score", "non_oracle_score", "c2_stability"):
        heatmap_weight = float(score_cfg.get("confidence_heatmap_weight", 1.0))
        imag_weight = float(score_cfg.get("confidence_imag_weight", 0.05))
        ref_rot_weight = float(score_cfg.get("confidence_ref_rot_weight", 0.0))
        ref_rot_max_deg = score_cfg.get("confidence_ref_rot_max_deg", None)
        dist_weight = float(score_cfg.get("confidence_point_distance_weight", 0.0))
        dist_mode = str(score_cfg.get("confidence_point_distance_mode", "preferred")).lower()
        preferred_distance = score_cfg.get("confidence_preferred_point_distance", None)
        if preferred_distance in (None, "", 0, False):
            preferred_distance = None
        else:
            preferred_distance = float(preferred_distance)
        max_candidate_distance = max(
            [
                abs(float(c.get("point_distance")))
                for c, _ in valid
                if c.get("point_distance") is not None and np.isfinite(float(c.get("point_distance")))
            ]
            or [1.0]
        )

        def non_oracle_score(item):
            candidate, _ = item
            score = heatmap_weight * float(candidate.get("heatmap_candidate_score") or 0.0)
            imag_values = []
            for key in ("max_imag_X", "max_imag_R"):
                value = candidate.get(key)
                if value is not None and np.isfinite(value):
                    imag_values.append(float(value))
            score -= imag_weight * sum(imag_values)

            if ref_RT is not None and ref_rot_weight > 0 and candidate.get("RT") is not None:
                rel_rot = rotation_error_deg(candidate["RT"][:3, :3], ref_RT[:3, :3])
                if ref_rot_max_deg not in (None, "", 0, False):
                    rel_rot = max(0.0, rel_rot - float(ref_rot_max_deg))
                score -= ref_rot_weight * rel_rot

            if dist_weight > 0:
                point_distance = candidate.get("point_distance")
                if point_distance is not None:
                    point_distance = float(point_distance)
                    if dist_mode in ("max", "large", "larger", "largest"):
                        score += dist_weight * point_distance / max(max_candidate_distance, 1e-6)
                    elif dist_mode in ("min", "small", "smaller", "smallest"):
                        score -= dist_weight * point_distance / max(max_candidate_distance, 1e-6)
                    elif dist_mode not in ("none", "off", "false"):
                        if preferred_distance is not None:
                            denom = max(abs(preferred_distance), 1e-6)
                            score -= dist_weight * abs(point_distance - preferred_distance) / denom
            return score

        return max(valid, key=non_oracle_score)
    if selection == "oracle_pose":
        return min(valid, key=lambda item: item[1]["rot_deg"] + item[1]["trans_cm"])
    if points is None:
        raise ValueError(
            f"selection={selection} needs model points. Use selection=oracle_pose or first_valid "
            "when the dataset has only GT poses."
        )
    if selection == "oracle_adds":
        return min(valid, key=lambda item: item[1]["adds"])
    if selection == "oracle_add":
        return min(valid, key=lambda item: item[1]["add"])
    if selection == "oracle_reproj":
        return min(valid, key=lambda item: item[1]["reproj"])
    raise ValueError(f"Unknown candidate selection: {selection}")


def rank_candidates_for_selection(
    candidates,
    points,
    gt_RT,
    K,
    selection,
    unit_to_meter=1.0,
    translation_scale_mode="none",
    fixed_translation_scale=1.0,
    symmetry_rotations=None,
    ref_RT=None,
    score_cfg=None,
):
    """Rank candidates by the same non-GT selector used for single-output inference."""
    score_cfg = score_cfg or {}
    scored = []
    for candidate in candidates:
        metrics = candidate_pose_metrics(
            candidate,
            points,
            gt_RT,
            K,
            unit_to_meter=unit_to_meter,
            translation_scale_mode=translation_scale_mode,
            fixed_translation_scale=fixed_translation_scale,
            symmetry_rotations=symmetry_rotations,
        )
        if not candidate.get("valid") or candidate.get("RT") is None:
            continue
        scored.append((candidate, metrics))

    if not scored:
        return []

    if selection == "first_valid":
        for rank, (candidate, _) in enumerate(scored):
            candidate["selection_rank"] = rank
        return scored

    if selection in ("axis_reprojection", "c2_axis_reprojection", "axis_reproj"):
        for candidate, _ in scored:
            candidate.update(candidate_axis_reprojection_score(candidate, K, score_cfg=score_cfg, ref_RT=ref_RT))
        filter_deg = score_cfg.get("axis_reproj_ref_rot_filter_deg", None)
        filtered = scored
        if filter_deg not in (None, "", 0, False):
            filter_deg = float(filter_deg)
            within_ref = [
                item
                for item in scored
                if item[0].get("axis_reproj_ref_rot_deg") is not None
                and np.isfinite(item[0].get("axis_reproj_ref_rot_deg"))
                and float(item[0].get("axis_reproj_ref_rot_deg")) <= filter_deg
            ]
            if within_ref:
                filtered = within_ref
        ranked = sorted(filtered, key=lambda item: item[0].get("axis_reproj_score", float("inf")))
        for rank, (candidate, _) in enumerate(ranked):
            candidate["selection_rank"] = rank
            candidate["axis_reproj_ref_rot_filter_deg"] = filter_deg if filter_deg not in (None, "", 0, False) else None
            candidate["axis_reproj_ref_rot_filter_used"] = int(len(filtered) < len(scored))
            candidate["axis_reproj_ref_rot_filter_candidates"] = len(filtered)
        return ranked

    if selection in ("confidence", "heatmap_score", "non_oracle_score", "c2_stability"):
        heatmap_weight = float(score_cfg.get("confidence_heatmap_weight", 1.0))
        imag_weight = float(score_cfg.get("confidence_imag_weight", 0.05))
        ref_rot_weight = float(score_cfg.get("confidence_ref_rot_weight", 0.0))
        ref_rot_max_deg = score_cfg.get("confidence_ref_rot_max_deg", None)
        dist_weight = float(score_cfg.get("confidence_point_distance_weight", 0.0))
        dist_mode = str(score_cfg.get("confidence_point_distance_mode", "preferred")).lower()
        preferred_distance = score_cfg.get("confidence_preferred_point_distance", None)
        if preferred_distance in (None, "", 0, False):
            preferred_distance = None
        else:
            preferred_distance = float(preferred_distance)
        max_candidate_distance = max(
            [
                abs(float(c.get("point_distance")))
                for c, _ in scored
                if c.get("point_distance") is not None and np.isfinite(float(c.get("point_distance")))
            ]
            or [1.0]
        )

        def non_oracle_score(item):
            candidate, _ = item
            score = heatmap_weight * float(candidate.get("heatmap_candidate_score") or 0.0)
            for key in ("max_imag_X", "max_imag_R"):
                value = candidate.get(key)
                if value is not None and np.isfinite(value):
                    score -= imag_weight * float(value)
            if ref_RT is not None and ref_rot_weight > 0 and candidate.get("RT") is not None:
                rel_rot = rotation_error_deg(candidate["RT"][:3, :3], ref_RT[:3, :3])
                if ref_rot_max_deg not in (None, "", 0, False):
                    rel_rot = max(0.0, rel_rot - float(ref_rot_max_deg))
                score -= ref_rot_weight * rel_rot
            if dist_weight > 0:
                point_distance = candidate.get("point_distance")
                if point_distance is not None:
                    point_distance = float(point_distance)
                    if dist_mode in ("max", "large", "larger", "largest"):
                        score += dist_weight * point_distance / max(max_candidate_distance, 1e-6)
                    elif dist_mode in ("min", "small", "smaller", "smallest"):
                        score -= dist_weight * point_distance / max(max_candidate_distance, 1e-6)
                    elif dist_mode not in ("none", "off", "false") and preferred_distance is not None:
                        score -= dist_weight * abs(point_distance - preferred_distance) / max(abs(preferred_distance), 1e-6)
            return -score

        ranked = sorted(scored, key=non_oracle_score)
        for rank, (candidate, _) in enumerate(ranked):
            candidate["selection_rank"] = rank
        return ranked

    raise ValueError(f"topk_eval currently supports non-oracle selectors, got selection={selection}")


def topk_metrics_from_ranked(ranked, top_ks):
    out = {}
    for k in top_ks:
        subset = ranked[: int(k)]
        valid_metrics = [metrics for _, metrics in subset if metrics is not None and np.isfinite(metrics["rot_deg"])]
        if not valid_metrics:
            out[f"top{k}_valid"] = 0
            out[f"top{k}_rot_deg"] = float("inf")
            out[f"top{k}_trans_cm"] = float("inf")
            out[f"top{k}_rot_acc_5deg"] = 0
            out[f"top{k}_rot_acc_30deg"] = 0
            out[f"top{k}_pose_acc_30cm_30deg"] = 0
            continue
        best_rot = min(valid_metrics, key=lambda item: item["rot_deg"])
        best_pose = min(valid_metrics, key=lambda item: item["rot_deg"] + item["trans_cm"])
        out[f"top{k}_valid"] = 1
        out[f"top{k}_rot_deg"] = best_rot["rot_deg"]
        out[f"top{k}_trans_cm"] = best_pose["trans_cm"]
        out[f"top{k}_rot_acc_5deg"] = int(best_rot["rot_deg"] < 5.0)
        out[f"top{k}_rot_acc_30deg"] = int(best_rot["rot_deg"] < 30.0)
        out[f"top{k}_pose_acc_30cm_30deg"] = int(best_pose["trans_cm"] < 30.0 and best_pose["rot_deg"] < 30.0)
    return out


def rank_candidates_for_inference(candidates, K, selection, ref_RT=None, score_cfg=None):
    """Rank candidates without using target GT. Used for fair top-K inference."""
    score_cfg = score_cfg or {}
    valid = [candidate for candidate in candidates if candidate.get("valid") and candidate.get("RT") is not None]
    if not valid:
        return []

    if selection == "first_valid":
        ranked = list(valid)
    elif selection in ("axis_reprojection", "c2_axis_reprojection", "axis_reproj"):
        for candidate in valid:
            candidate.update(candidate_axis_reprojection_score(candidate, K, score_cfg=score_cfg, ref_RT=ref_RT))
        filter_deg = score_cfg.get("axis_reproj_ref_rot_filter_deg", None)
        filtered = valid
        if filter_deg not in (None, "", 0, False):
            filter_deg = float(filter_deg)
            within_ref = [
                candidate
                for candidate in valid
                if candidate.get("axis_reproj_ref_rot_deg") is not None
                and np.isfinite(candidate.get("axis_reproj_ref_rot_deg"))
                and float(candidate.get("axis_reproj_ref_rot_deg")) <= filter_deg
            ]
            if within_ref:
                filtered = within_ref
        ranked = sorted(filtered, key=lambda candidate: candidate.get("axis_reproj_score", float("inf")))
        for candidate in ranked:
            candidate["axis_reproj_ref_rot_filter_deg"] = filter_deg if filter_deg not in (None, "", 0, False) else None
            candidate["axis_reproj_ref_rot_filter_used"] = int(len(filtered) < len(valid))
            candidate["axis_reproj_ref_rot_filter_candidates"] = len(filtered)
    elif selection in ("confidence", "heatmap_score", "non_oracle_score", "c2_stability"):
        heatmap_weight = float(score_cfg.get("confidence_heatmap_weight", 1.0))
        imag_weight = float(score_cfg.get("confidence_imag_weight", 0.05))
        ref_rot_weight = float(score_cfg.get("confidence_ref_rot_weight", 0.0))
        ref_rot_max_deg = score_cfg.get("confidence_ref_rot_max_deg", None)
        dist_weight = float(score_cfg.get("confidence_point_distance_weight", 0.0))
        dist_mode = str(score_cfg.get("confidence_point_distance_mode", "preferred")).lower()
        preferred_distance = score_cfg.get("confidence_preferred_point_distance", None)
        if preferred_distance in (None, "", 0, False):
            preferred_distance = None
        else:
            preferred_distance = float(preferred_distance)
        max_candidate_distance = max(
            [
                abs(float(candidate.get("point_distance")))
                for candidate in valid
                if candidate.get("point_distance") is not None and np.isfinite(float(candidate.get("point_distance")))
            ]
            or [1.0]
        )

        def rank_score(candidate):
            score = heatmap_weight * float(candidate.get("heatmap_candidate_score") or 0.0)
            for key in ("max_imag_X", "max_imag_R"):
                value = candidate.get(key)
                if value is not None and np.isfinite(value):
                    score -= imag_weight * float(value)
            if ref_RT is not None and ref_rot_weight > 0:
                rel_rot = rotation_error_deg(candidate["RT"][:3, :3], ref_RT[:3, :3])
                if ref_rot_max_deg not in (None, "", 0, False):
                    rel_rot = max(0.0, rel_rot - float(ref_rot_max_deg))
                score -= ref_rot_weight * rel_rot
            if dist_weight > 0:
                point_distance = candidate.get("point_distance")
                if point_distance is not None:
                    point_distance = float(point_distance)
                    if dist_mode in ("max", "large", "larger", "largest"):
                        score += dist_weight * point_distance / max(max_candidate_distance, 1e-6)
                    elif dist_mode in ("min", "small", "smaller", "smallest"):
                        score -= dist_weight * point_distance / max(max_candidate_distance, 1e-6)
                    elif dist_mode not in ("none", "off", "false") and preferred_distance is not None:
                        score -= dist_weight * abs(point_distance - preferred_distance) / max(abs(preferred_distance), 1e-6)
            return -score

        ranked = sorted(valid, key=rank_score)
    else:
        raise ValueError(f"topk_eval currently supports non-oracle selectors, got selection={selection}")

    for rank, candidate in enumerate(ranked):
        candidate["selection_rank"] = rank
    return ranked


def pack_ranked_candidate_poses(ranked_candidates, top_k):
    """Pack the highest-ranked inference candidates into fixed-size NPZ arrays.

    Candidate pools can be smaller than ``top_k`` after C2 validity and the
    reference-rotation filter.  NaN/-1 padding keeps every sample aligned while
    ``candidate_count`` records how many rows are meaningful.
    """
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError(f"candidate pose top_k must be positive, got {top_k}")

    invalid_RT = np.full((top_k, 4, 4), np.nan, dtype=np.float64)
    packed = {
        "candidate_count": min(len(ranked_candidates), top_k),
        "candidate_RT_c2": invalid_RT,
        "candidate_selection_rank": np.full(top_k, -1, dtype=np.int32),
        "candidate_axis_reproj_score": np.full(top_k, np.nan, dtype=np.float64),
        "candidate_axis_reproj_dir_deg": np.full(top_k, np.nan, dtype=np.float64),
        "candidate_axis_reproj_endpoint_px": np.full(top_k, np.nan, dtype=np.float64),
        "candidate_axis_reproj_center_px": np.full(top_k, np.nan, dtype=np.float64),
        "candidate_axis_reproj_ref_rot_deg": np.full(top_k, np.nan, dtype=np.float64),
        "candidate_heatmap_score": np.full(top_k, np.nan, dtype=np.float64),
        "candidate_max_imag_X": np.full(top_k, np.nan, dtype=np.float64),
        "candidate_max_imag_R": np.full(top_k, np.nan, dtype=np.float64),
        "candidate_flag": np.full(top_k, -1, dtype=np.int16),
        "candidate_point_distance": np.full(top_k, np.nan, dtype=np.float64),
        "candidate_signs": np.zeros((top_k, 3), dtype=np.int8),
        "candidate_heatmap_candidate_index": np.full(top_k, -1, dtype=np.int32),
        "candidate_axis_indices": np.full((top_k, 3), -1, dtype=np.int16),
    }

    for output_idx, candidate in enumerate(ranked_candidates[:top_k]):
        RT = candidate.get("RT")
        if RT is not None:
            RT = np.asarray(RT, dtype=np.float64)
            if RT.shape == (4, 4):
                packed["candidate_RT_c2"][output_idx] = RT
        packed["candidate_selection_rank"][output_idx] = int(candidate.get("selection_rank", output_idx))
        for archive_key, candidate_key in (
            ("candidate_axis_reproj_score", "axis_reproj_score"),
            ("candidate_axis_reproj_dir_deg", "axis_reproj_dir_deg"),
            ("candidate_axis_reproj_endpoint_px", "axis_reproj_endpoint_px"),
            ("candidate_axis_reproj_center_px", "axis_reproj_center_px"),
            ("candidate_axis_reproj_ref_rot_deg", "axis_reproj_ref_rot_deg"),
            ("candidate_heatmap_score", "heatmap_candidate_score"),
            ("candidate_max_imag_X", "max_imag_X"),
            ("candidate_max_imag_R", "max_imag_R"),
            ("candidate_point_distance", "point_distance"),
        ):
            value = candidate.get(candidate_key)
            if value is not None:
                try:
                    packed[archive_key][output_idx] = float(value)
                except (TypeError, ValueError):
                    pass
        if candidate.get("flag") is not None:
            packed["candidate_flag"][output_idx] = int(candidate["flag"])
        signs = candidate.get("signs")
        if signs is not None:
            packed["candidate_signs"][output_idx] = np.asarray(signs, dtype=np.int8).reshape(3)
        if candidate.get("heatmap_candidate_index") is not None:
            packed["candidate_heatmap_candidate_index"][output_idx] = int(candidate["heatmap_candidate_index"])
        axis_indices = candidate.get("axis_indices")
        if axis_indices is not None:
            packed["candidate_axis_indices"][output_idx] = np.asarray(axis_indices, dtype=np.int16).reshape(3)
    return packed


def topk_metrics_from_ranked_candidates(
    ranked_candidates,
    top_ks,
    points,
    gt_RT,
    K,
    unit_to_meter=1.0,
    translation_scale_mode="none",
    fixed_translation_scale=1.0,
    symmetry_rotations=None,
    axis_only=None,
):
    scored = []
    max_k = max(top_ks) if top_ks else 0
    for candidate in ranked_candidates[:max_k]:
        metrics = candidate_pose_metrics(
            candidate,
            points,
            gt_RT,
            K,
            unit_to_meter=unit_to_meter,
            translation_scale_mode=translation_scale_mode,
            fixed_translation_scale=fixed_translation_scale,
            symmetry_rotations=symmetry_rotations,
            axis_only=axis_only,
        )
        scored.append((candidate, metrics))
    return topk_metrics_from_ranked(scored, top_ks)


def init_sums():
    return {
        "samples": 0,
        "valid_pose": 0,
        "reproj": 0.0,
        "add": 0.0,
        "adds": 0.0,
        "rot_deg": 0.0,
        "trans_cm": 0.0,
        "translation_scale": 0.0,
        "rot_deg_values": [],
        "trans_cm_values": [],
        "translation_scale_values": [],
        "adds_0.1d": 0,
        "adds_0.2d": 0,
        "rot_acc_1deg": 0,
        "rot_acc_3deg": 0,
        "rot_acc_5deg": 0,
        "rot_acc_30deg": 0,
        "trans_acc_1cm": 0,
        "trans_acc_3cm": 0,
        "trans_acc_5cm": 0,
        "trans_acc_30cm": 0,
        "pose_acc_1cm_1deg": 0,
        "pose_acc_3cm_3deg": 0,
        "pose_acc_5cm_5deg": 0,
        "pose_acc_30cm_30deg": 0,
    }


def update_sums(sums, metrics, diameter):
    sums["samples"] += 1
    if metrics is None:
        return
    finite = np.isfinite([metrics["rot_deg"], metrics["trans_cm"]]).all()
    if not finite:
        return
    sums["valid_pose"] += 1
    for key in ("rot_deg", "trans_cm"):
        sums[key] += float(metrics[key])
        sums[f"{key}_values"].append(float(metrics[key]))
    if metrics.get("translation_scale") is not None and np.isfinite(metrics["translation_scale"]):
        sums["translation_scale"] += float(metrics["translation_scale"])
        sums["translation_scale_values"].append(float(metrics["translation_scale"]))
        sums["translation_scale_count"] = sums.get("translation_scale_count", 0) + 1
    for key in ("reproj", "add", "adds"):
        if metrics.get(key) is not None and np.isfinite(metrics[key]):
            sums[key] += float(metrics[key])
            sums[f"{key}_count"] = sums.get(f"{key}_count", 0) + 1
    if diameter is not None and metrics.get("adds") is not None and np.isfinite(metrics["adds"]):
        sums["adds_0.1d"] += int(metrics["adds"] < 0.1 * diameter)
        sums["adds_0.2d"] += int(metrics["adds"] < 0.2 * diameter)
    sums["rot_acc_1deg"] += int(metrics["rot_deg"] < 1.0)
    sums["rot_acc_3deg"] += int(metrics["rot_deg"] < 3.0)
    sums["rot_acc_5deg"] += int(metrics["rot_deg"] < 5.0)
    sums["rot_acc_30deg"] += int(metrics["rot_deg"] < 30.0)
    sums["trans_acc_1cm"] += int(metrics["trans_cm"] < 1.0)
    sums["trans_acc_3cm"] += int(metrics["trans_cm"] < 3.0)
    sums["trans_acc_5cm"] += int(metrics["trans_cm"] < 5.0)
    sums["trans_acc_30cm"] += int(metrics["trans_cm"] < 30.0)
    sums["pose_acc_1cm_1deg"] += int(metrics["trans_cm"] < 1.0 and metrics["rot_deg"] < 1.0)
    sums["pose_acc_3cm_3deg"] += int(metrics["trans_cm"] < 3.0 and metrics["rot_deg"] < 3.0)
    sums["pose_acc_5cm_5deg"] += int(metrics["trans_cm"] < 5.0 and metrics["rot_deg"] < 5.0)
    sums["pose_acc_30cm_30deg"] += int(metrics["trans_cm"] < 30.0 and metrics["rot_deg"] < 30.0)


def finalize_sums(sums, num_batches):
    valid = max(sums["valid_pose"], 1)
    total = max(sums["samples"], 1)
    row = {
        "samples": sums["samples"],
        "valid_pose": sums["valid_pose"],
        "batches": num_batches,
        "valid_rate": sums["valid_pose"] / total,
    }
    for key in ("reproj", "add", "adds", "rot_deg", "trans_cm", "translation_scale"):
        count = sums.get(f"{key}_count", sums["valid_pose"] if key in ("rot_deg", "trans_cm") else 0)
        row[key] = sums[key] / max(count, 1) if count > 0 else None
    for key in ("rot_deg", "trans_cm", "translation_scale"):
        values = np.asarray(sums.get(f"{key}_values", []), dtype=np.float64)
        if values.size > 0:
            row[f"{key}_median"] = float(np.median(values))
            row[f"{key}_p90"] = float(np.percentile(values, 90))
            row[f"{key}_p95"] = float(np.percentile(values, 95))
            row[f"{key}_min"] = float(values.min())
            row[f"{key}_max"] = float(values.max())
        else:
            row[f"{key}_median"] = None
            row[f"{key}_p90"] = None
            row[f"{key}_p95"] = None
            row[f"{key}_min"] = None
            row[f"{key}_max"] = None
    for key in (
        "adds_0.1d",
        "adds_0.2d",
        "rot_acc_1deg",
        "rot_acc_3deg",
        "rot_acc_5deg",
        "rot_acc_30deg",
        "trans_acc_1cm",
        "trans_acc_3cm",
        "trans_acc_5cm",
        "trans_acc_30cm",
        "pose_acc_1cm_1deg",
        "pose_acc_3cm_3deg",
        "pose_acc_5cm_5deg",
        "pose_acc_30cm_30deg",
    ):
        row[key] = sums[key] / total
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt_optional(value, precision=4):
    if value is None:
        return "NA"
    return f"{float(value):.{precision}f}"


@torch.no_grad()
def run_test(cfg):
    test_cfg = cfg["test"]
    c2_cfg = cfg["c2"]
    data_root = Path(cfg["data"]["root"])
    target_image_root = cfg["data"].get("target_image_root")
    image_size = int(cfg["data"].get("image_size", 256))
    K = make_intrinsics(cfg, image_size)
    unit_to_meter = float(cfg.get("pose", {}).get("unit_to_meter", 1.0))
    output_dir = Path(test_cfg.get("output_dir", "outputs/c2_pose_test"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pred_poses = bool(test_cfg.get("save_pred_poses", False))
    save_candidate_poses = bool(test_cfg.get("save_candidate_poses", False))
    candidate_pose_top_k = int(test_cfg.get("candidate_pose_top_k", 5))
    if save_candidate_poses and not save_pred_poses:
        raise ValueError("test.save_candidate_poses requires test.save_pred_poses=true")
    if save_candidate_poses and candidate_pose_top_k <= 0:
        raise ValueError("test.candidate_pose_top_k must be positive when candidate pose export is enabled")
    pred_pose_filename = str(test_cfg.get("pred_pose_filename", "pred_poses.npz"))
    pred_pose_path = Path(pred_pose_filename)
    if not pred_pose_path.is_absolute():
        pred_pose_path = output_dir / pred_pose_path

    device = torch.device(test_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    loader = build_test_loader(cfg)
    axis_source = test_cfg.get("axis_source", "pred_heatmap")
    if axis_source not in AXIS_SOURCES:
        raise ValueError(f"Unknown axis_source '{axis_source}'. Valid choices: {AXIS_SOURCES}")
    needs_model = axis_source in ("pred_heatmap", "pred_geometry")
    model = None
    checkpoint_path = test_cfg.get("checkpoint")
    epoch = None
    step = None
    if needs_model:
        model = AxisPosePP(**cfg["model"]).to(device)
        checkpoint_path = Path(checkpoint_path)
        epoch, step = load_model_checkpoint(checkpoint_path, model, device)
        model.eval()

    parser_name = test_cfg.get("parser", "peak")
    point_cache = ModelPointCache(cfg.get("model_points", {"enabled": False}))
    point_distances = parse_float_list(
        c2_cfg.get("point_distances"),
        fallback=[float(c2_cfg.get("point_distance", 0.42 * image_size))],
    )
    flags = c2_cfg.get("flags", [1])
    sign_candidates = parse_sign_candidates(c2_cfg.get("sign_candidates"))
    selection = c2_cfg.get("selection", "oracle_reproj")
    include_xy_swap = bool(c2_cfg.get("include_xy_swap", False))
    c2_imag_tol = float(c2_cfg.get("imag_tol", 1e-6))
    c2_force_real = bool(c2_cfg.get("force_real", False))
    translation_scale_mode = c2_cfg.get("translation_scale", "none")
    fixed_translation_scale = float(c2_cfg.get("fixed_translation_scale", 1.0))
    topk_eval_cfg = cfg.get("topk_eval", {})
    topk_eval_enabled = bool(topk_eval_cfg.get("enabled", False))
    topk_values = sorted({int(v) for v in topk_eval_cfg.get("ks", [1, 3, 5]) if int(v) > 0})
    max_batches = test_cfg.get("max_batches")
    max_samples = test_cfg.get("max_samples")
    max_relative_rot_deg = test_cfg.get("max_relative_rot_deg")
    if max_relative_rot_deg not in (None, "", 0, False):
        max_relative_rot_deg = float(max_relative_rot_deg)
    else:
        max_relative_rot_deg = None
    requested_categories = test_cfg.get("categories", test_cfg.get("include_categories"))
    if requested_categories in (None, "", False):
        requested_categories = None
    else:
        requested_categories = {canonical_category(value) for value in requested_categories}
    vis_cfg = cfg.get("visualization", test_cfg.get("visualization", {}))
    save_visualizations = bool(vis_cfg.get("enabled", test_cfg.get("save_visualizations", False)))
    vis_stride = int(vis_cfg.get("stride", 1))
    vis_max_samples = vis_cfg.get("max_samples")
    if vis_max_samples in (None, "", 0, False):
        vis_max_samples = None
    else:
        vis_max_samples = int(vis_max_samples)
    vis_line_width = int(vis_cfg.get("line_width", 2))
    use_model_points_bbox = bool(vis_cfg.get("use_model_points_bbox", True))
    bbox_info_cache = BBoxInfoCache(vis_cfg)
    symmetry_cfg = cfg.get("symmetry", {})
    symmetry_categories = parse_symmetry_categories(symmetry_cfg)
    metadata_path = resolve_category_metadata_path(cfg)
    category_lookup = load_category_lookup(metadata_path)

    sums = init_sums()
    rows = []
    pose_archive = {
        "sample": [], "object_id": [], "category": [], "ref": [], "query": [], "valid": [],
        "relative_rot_deg": [], "ref_RT": [], "gt_RT": [], "K": [], "pred_RT_c2": [],
        "pred_RT_scaled": [], "pred_RT_visual": [], "translation_scale": [], "selected_flag": [],
        "selected_point_distance": [], "selected_signs": [], "heatmap_candidate_index": [], "axis_indices": [],
        "geometry_center": [], "geometry_dirs": [],
    } if save_pred_poses else None
    if pose_archive is not None and save_candidate_poses:
        pose_archive.update(
            {
                "candidate_count": [], "candidate_RT_c2": [], "candidate_selection_rank": [],
                "candidate_axis_reproj_score": [], "candidate_axis_reproj_dir_deg": [],
                "candidate_axis_reproj_endpoint_px": [], "candidate_axis_reproj_center_px": [],
                "candidate_axis_reproj_ref_rot_deg": [], "candidate_heatmap_score": [],
                "candidate_max_imag_X": [], "candidate_max_imag_R": [], "candidate_flag": [],
                "candidate_point_distance": [], "candidate_signs": [],
                "candidate_heatmap_candidate_index": [], "candidate_axis_indices": [],
            }
        )
    topk_sums = {
        k: {
            "valid": 0,
            "rot_deg_sum": 0.0,
            "rot_deg_values": [],
            "trans_cm_sum": 0.0,
            "rot_acc_5deg": 0,
            "rot_acc_30deg": 0,
            "pose_acc_30cm_30deg": 0,
        }
        for k in topk_values
    }
    total_batches = 0
    total_samples = 0
    total_seen = 0
    total_filtered = 0
    total_category_excluded = 0
    saved_visuals = 0

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches not in (None, "", 0, False) and batch_idx > int(max_batches):
            break
        batch = move_to_device(batch, device)
        if needs_model:
            outputs = model(batch)
            pred_axis = outputs["heatmap"][:, 0:3].detach().cpu()
            pred_center_hm = outputs["heatmap"][:, 3:4].detach().cpu()
            geometry_centers = outputs["center"].detach().cpu()
            geometry_dirs = outputs["directions"].detach().cpu()
            batch_size = pred_axis.shape[0]
        else:
            batch_size = batch["target_image"].shape[0]
        meta = batch["meta"]

        for idx in range(batch_size):
            if max_samples not in (None, "", 0, False) and total_samples >= int(max_samples):
                break

            ref_prefix = get_meta_value(meta, "ref", idx)
            query_prefix = get_meta_value(meta, "query", idx)
            object_id = object_id_from_query(query_prefix)
            category = category_for_object(object_id, category_lookup)
            if not category:
                category = category_from_query_prefix(query_prefix)
            if requested_categories is not None and category not in requested_categories:
                total_category_excluded += 1
                continue
            symmetry_rotations = symmetry_rotations_for_category(category, symmetry_categories)
            axis_only_rotation = axis_only_rotation_for_category(category, symmetry_categories)
            sample_K = intrinsics_from_batch_or_default(batch, idx, K)
            ref_RT = pose_from_batch_or_file(batch, "ref_pose_matrix", idx, data_root, ref_prefix)
            gt_RT = pose_from_batch_or_file(batch, "target_pose_matrix", idx, data_root, query_prefix)
            relative_rot_deg = rotation_error_deg(gt_RT[:3, :3], ref_RT[:3, :3])
            total_seen += 1
            if max_relative_rot_deg is not None and relative_rot_deg > max_relative_rot_deg:
                total_filtered += 1
                continue
            model_info = point_cache.get(object_id)
            points = model_info["points"] if model_info is not None else None
            diameter = model_info["diameter"] if model_info is not None else None
            gt_center, gt_dirs = gt_axes_from_batch(batch, idx, image_size)
            pred_axis_heatmap = None
            pred_center_heatmap = None

            if axis_source in ("pred_heatmap", "pred_geometry"):
                pred_axis_heatmap = pred_axis[idx]
                pred_center_heatmap = pred_center_hm[idx]
                candidates = []
                if axis_source == "pred_geometry":
                    center_t = geometry_centers[idx] * float(image_size - 1)
                    dirs_t = geometry_dirs[idx]
                    heatmap_axis_candidates = [
                        {
                            "center": center_t,
                            "dirs": dirs_t,
                            "axis_indices": (0, 0, 0),
                            "heatmap_candidate_index": 0,
                            "heatmap_candidate_score": 0.0,
                            "candidate_mode": "geometry_only",
                        }
                    ]
                elif parser_name == "peak_multi":
                    heatmap_axis_candidates = parse_heatmap_multi_candidates(
                        pred_axis[idx],
                        pred_center_hm[idx],
                        geometry_dirs[idx],
                        test_cfg,
                    )
                else:
                    center_t, dirs_t = parse_heatmap(
                        parser_name,
                        pred_axis[idx],
                        pred_center_hm[idx],
                        geometry_dirs[idx],
                        test_cfg,
                    )
                    heatmap_axis_candidates = [
                        {
                            "center": center_t,
                            "dirs": dirs_t,
                            "axis_indices": (0, 0, 0),
                            "heatmap_candidate_index": 0,
                            "heatmap_candidate_score": 0.0,
                        }
                    ]
                pred_axis_center = heatmap_axis_candidates[0]["center"].numpy()
                pred_axis_dirs = heatmap_axis_candidates[0]["dirs"].numpy()
                for axis_candidate in heatmap_axis_candidates:
                    center = axis_candidate["center"].numpy()
                    dirs = axis_candidate["dirs"].numpy()
                    for flag in flags:
                        for point_distance in point_distances:
                            solved_candidates = solve_c2_candidates_from_axes(
                                    sample_K,
                                    center,
                                    dirs,
                                    lamda0=c2_cfg["lamda0"],
                                    flag=int(flag),
                                    point_distance=point_distance,
                                    sign_candidates=sign_candidates,
                                    include_xy_swap=include_xy_swap,
                                    orthonormalize=bool(c2_cfg.get("orthonormalize", True)),
                                    imag_tol=c2_imag_tol,
                                    force_real=c2_force_real,
                            )
                            for solved in solved_candidates:
                                solved.update(
                                    {
                                        "axis_center": center,
                                        "axis_dirs": dirs,
                                        "axis_indices": axis_candidate["axis_indices"],
                                        "heatmap_candidate_index": axis_candidate["heatmap_candidate_index"],
                                        "heatmap_candidate_score": axis_candidate["heatmap_candidate_score"],
                                        "candidate_mode": axis_candidate.get("candidate_mode", ""),
                                        "z_guided_fixed_axis": axis_candidate.get("z_guided_fixed_axis", ""),
                                        "z_guided_search_axis": axis_candidate.get("z_guided_search_axis", ""),
                                    }
                                )
                            candidates.extend(solved_candidates)
            elif axis_source == "gt_axes":
                center, dirs = gt_axes_from_batch(batch, idx, image_size)
                pred_axis_center = center
                pred_axis_dirs = dirs
                candidates = []
                for flag in flags:
                    for point_distance in point_distances:
                        candidates.extend(
                            solve_c2_candidates_from_axes(
                                sample_K,
                                center,
                                dirs,
                                lamda0=c2_cfg["lamda0"],
                                flag=int(flag),
                                point_distance=point_distance,
                                sign_candidates=sign_candidates,
                                include_xy_swap=include_xy_swap,
                                orthonormalize=bool(c2_cfg.get("orthonormalize", True)),
                                imag_tol=c2_imag_tol,
                                force_real=c2_force_real,
                            )
                        )
            else:
                raise ValueError(f"Unknown axis_source '{axis_source}'")

            if bool(c2_cfg.get("pose_refine_enabled", False)):
                candidates = [
                    refine_candidate_pose_from_axes(candidate, sample_K, score_cfg=c2_cfg, ref_RT=ref_RT)
                    for candidate in candidates
                ]
            topk_metrics = {}
            ranked_candidates = None
            if (topk_eval_enabled and topk_values) or save_candidate_poses:
                ranked_candidates = rank_candidates_for_inference(
                    candidates,
                    sample_K,
                    selection,
                    ref_RT=ref_RT,
                    score_cfg=c2_cfg,
                )
                selected = ranked_candidates[0] if ranked_candidates else None
                metrics = (
                    candidate_pose_metrics(
                        selected,
                        points,
                        gt_RT,
                        sample_K,
                        unit_to_meter=unit_to_meter,
                        translation_scale_mode=translation_scale_mode,
                        fixed_translation_scale=fixed_translation_scale,
                        symmetry_rotations=symmetry_rotations,
                        axis_only=axis_only_rotation,
                    )
                    if selected is not None
                    else None
                )
                if topk_eval_enabled and topk_values:
                    topk_metrics = topk_metrics_from_ranked_candidates(
                        ranked_candidates,
                        topk_values,
                        points,
                        gt_RT,
                        sample_K,
                        unit_to_meter=unit_to_meter,
                        translation_scale_mode=translation_scale_mode,
                        fixed_translation_scale=fixed_translation_scale,
                        symmetry_rotations=symmetry_rotations,
                        axis_only=axis_only_rotation,
                    )
                    for k in topk_values:
                        if topk_metrics.get(f"top{k}_valid", 0):
                            stats = topk_sums[k]
                            stats["valid"] += 1
                            stats["rot_deg_sum"] += float(topk_metrics[f"top{k}_rot_deg"])
                            stats["rot_deg_values"].append(float(topk_metrics[f"top{k}_rot_deg"]))
                            stats["trans_cm_sum"] += float(topk_metrics[f"top{k}_trans_cm"])
                            stats["rot_acc_5deg"] += int(topk_metrics[f"top{k}_rot_acc_5deg"])
                            stats["rot_acc_30deg"] += int(topk_metrics[f"top{k}_rot_acc_30deg"])
                            stats["pose_acc_30cm_30deg"] += int(topk_metrics[f"top{k}_pose_acc_30cm_30deg"])
            else:
                selected, metrics = select_candidate(
                    candidates,
                    points,
                    gt_RT,
                    sample_K,
                    selection,
                    unit_to_meter=unit_to_meter,
                    translation_scale_mode=translation_scale_mode,
                    fixed_translation_scale=fixed_translation_scale,
                    symmetry_rotations=symmetry_rotations,
                    axis_only=axis_only_rotation,
                    ref_RT=ref_RT,
                    score_cfg=c2_cfg,
                )
            if selected is not None and selected.get("axis_center") is not None:
                pred_axis_center = selected["axis_center"]
                pred_axis_dirs = selected["axis_dirs"]
            update_sums(sums, metrics, diameter)

            visual_path = ""
            pred_RT_c2 = None
            pred_RT_scaled = None
            pred_RT_vis = None
            if selected is not None and selected.get("valid") and selected.get("RT") is not None:
                pred_RT_c2 = np.asarray(selected["RT"], dtype=np.float64).copy()
                pred_RT_scaled, _ = align_translation_scale(
                    pred_RT_c2,
                    gt_RT,
                    mode=translation_scale_mode,
                    fixed_scale=fixed_translation_scale,
                )
                pred_RT_vis = pred_RT_scaled.copy()
                if bool(vis_cfg.get("pred_bbox_use_gt_translation", False)):
                    pred_RT_vis = pred_RT_vis.copy()
                    pred_RT_vis[:3, 3] = gt_RT[:3, 3]
            should_save_vis = (
                save_visualizations
                and vis_stride > 0
                and total_samples % vis_stride == 0
                and (vis_max_samples is None or saved_visuals < vis_max_samples)
            )
            if should_save_vis:
                metadata_bbox_points, metadata_bbox_edges = bbox_info_cache.get(object_id)
                if metadata_bbox_points is not None:
                    corners = metadata_bbox_points
                    bbox_edges = metadata_bbox_edges
                else:
                    bbox_points = points if use_model_points_bbox else None
                    corners = make_bbox_corners(vis_cfg, points=bbox_points)
                    bbox_edges = BBOX_EDGES
                visual_path = str(
                    save_pose_bbox_visual(
                        output_dir,
                        total_samples,
                        data_root,
                        ref_prefix,
                        query_prefix,
                        ref_RT,
                        gt_RT,
                        pred_RT_vis,
                        sample_K,
                        image_size,
                        corners,
                        bbox_edges,
                        pred_axis_center,
                        pred_axis_dirs,
                        gt_center,
                        gt_dirs,
                        pred_axis_heatmap=pred_axis_heatmap,
                        pred_center_heatmap=pred_center_heatmap,
                        metrics=metrics,
                        line_width=vis_line_width,
                        target_image_root=target_image_root,
                    )
                )
                saved_visuals += 1

            row = {
                "sample": total_samples,
                "axis_source": axis_source,
                "object_id": object_id,
                "category": category,
                "ref": ref_prefix,
                "query": query_prefix,
                "relative_rot_deg": relative_rot_deg,
                "visual_path": visual_path,
                "valid": int(
                    metrics is not None
                    and np.isfinite(metrics["rot_deg"])
                    and np.isfinite(metrics["trans_cm"])
                ),
                "num_candidates": len(candidates),
                "diameter": diameter,
            }
            if selected is not None:
                row.update(
                    {
                        "signs": str(selected["signs"]),
                        "swap_xy": int(selected["swap_xy"]),
                        "flag": selected["flag"],
                        "selected_point_distance": selected.get("point_distance"),
                        "heatmap_candidate_index": selected.get("heatmap_candidate_index"),
                        "axis_indices": str(selected.get("axis_indices", "")),
                        "heatmap_candidate_score": selected.get("heatmap_candidate_score"),
                        "candidate_mode": selected.get("candidate_mode", ""),
                        "z_guided_fixed_axis": selected.get("z_guided_fixed_axis", ""),
                        "z_guided_search_axis": selected.get("z_guided_search_axis", ""),
                        "max_imag_X": selected.get("max_imag_X"),
                        "max_imag_R": selected.get("max_imag_R"),
                        "axis_reproj_score": selected.get("axis_reproj_score"),
                        "axis_reproj_dir_deg": selected.get("axis_reproj_dir_deg"),
                        "axis_reproj_endpoint_px": selected.get("axis_reproj_endpoint_px"),
                        "axis_reproj_center_px": selected.get("axis_reproj_center_px"),
                        "axis_reproj_ref_rot_deg": selected.get("axis_reproj_ref_rot_deg"),
                        "axis_reproj_ref_rot_filter_deg": selected.get("axis_reproj_ref_rot_filter_deg"),
                        "axis_reproj_ref_rot_filter_used": selected.get("axis_reproj_ref_rot_filter_used"),
                        "axis_reproj_ref_rot_filter_candidates": selected.get("axis_reproj_ref_rot_filter_candidates"),
                        "pose_refine_success": selected.get("pose_refine_success"),
                        "pose_refine_cost": selected.get("pose_refine_cost"),
                        "pose_refine_nfev": selected.get("pose_refine_nfev"),
                        "pose_refine_axis_length": selected.get("pose_refine_axis_length"),
                        "force_real": int(bool(selected.get("force_real", False))),
                    }
                )
            if metrics is not None:
                row.update(metrics)
                row["valid"] = int(np.isfinite(metrics["rot_deg"]) and np.isfinite(metrics["trans_cm"]))
                row["adds_0.1d"] = int(metrics["adds"] < 0.1 * diameter) if diameter is not None and metrics.get("adds") is not None else ""
                row["adds_0.2d"] = int(metrics["adds"] < 0.2 * diameter) if diameter is not None and metrics.get("adds") is not None else ""
                row["rot_acc_1deg"] = int(metrics["rot_deg"] < 1.0)
                row["rot_acc_3deg"] = int(metrics["rot_deg"] < 3.0)
                row["rot_acc_5deg"] = int(metrics["rot_deg"] < 5.0)
                row["rot_acc_30deg"] = int(metrics["rot_deg"] < 30.0)
                row["trans_acc_1cm"] = int(metrics["trans_cm"] < 1.0)
                row["trans_acc_3cm"] = int(metrics["trans_cm"] < 3.0)
                row["trans_acc_5cm"] = int(metrics["trans_cm"] < 5.0)
                row["trans_acc_30cm"] = int(metrics["trans_cm"] < 30.0)
                row["pose_acc_1cm_1deg"] = int(metrics["trans_cm"] < 1.0 and metrics["rot_deg"] < 1.0)
                row["pose_acc_3cm_3deg"] = int(metrics["trans_cm"] < 3.0 and metrics["rot_deg"] < 3.0)
                row["pose_acc_5cm_5deg"] = int(metrics["trans_cm"] < 5.0 and metrics["rot_deg"] < 5.0)
                row["pose_acc_30cm_30deg"] = int(metrics["trans_cm"] < 30.0 and metrics["rot_deg"] < 30.0)
            if topk_metrics:
                row.update(topk_metrics)
            rows.append(row)
            if pose_archive is not None:
                invalid_RT = np.full((4, 4), np.nan, dtype=np.float64)
                signs = selected.get("signs") if selected is not None else None
                axis_indices = selected.get("axis_indices") if selected is not None else None
                pose_archive["sample"].append(total_samples)
                pose_archive["object_id"].append(object_id)
                pose_archive["category"].append(category)
                pose_archive["ref"].append(ref_prefix)
                pose_archive["query"].append(query_prefix)
                pose_archive["valid"].append(int(pred_RT_c2 is not None))
                pose_archive["relative_rot_deg"].append(float(relative_rot_deg))
                pose_archive["ref_RT"].append(np.asarray(ref_RT, dtype=np.float64))
                pose_archive["gt_RT"].append(np.asarray(gt_RT, dtype=np.float64))
                pose_archive["K"].append(np.asarray(sample_K, dtype=np.float64))
                pose_archive["pred_RT_c2"].append(pred_RT_c2 if pred_RT_c2 is not None else invalid_RT)
                pose_archive["pred_RT_scaled"].append(pred_RT_scaled if pred_RT_scaled is not None else invalid_RT)
                pose_archive["pred_RT_visual"].append(pred_RT_vis if pred_RT_vis is not None else invalid_RT)
                pose_archive["translation_scale"].append(float(metrics.get("translation_scale")) if metrics is not None and metrics.get("translation_scale") is not None else np.nan)
                pose_archive["selected_flag"].append(int(selected.get("flag")) if selected is not None and selected.get("flag") is not None else -1)
                pose_archive["selected_point_distance"].append(float(selected.get("point_distance")) if selected is not None and selected.get("point_distance") is not None else np.nan)
                pose_archive["selected_signs"].append(np.asarray(signs, dtype=np.int8).reshape(3) if signs is not None else np.zeros(3, dtype=np.int8))
                pose_archive["heatmap_candidate_index"].append(int(selected.get("heatmap_candidate_index")) if selected is not None and selected.get("heatmap_candidate_index") is not None else -1)
                pose_archive["axis_indices"].append(np.asarray(axis_indices, dtype=np.int16).reshape(3) if axis_indices is not None else np.full(3, -1, dtype=np.int16))
                if needs_model:
                    pose_archive["geometry_center"].append(geometry_centers[idx].numpy())
                    pose_archive["geometry_dirs"].append(geometry_dirs[idx].numpy())
                else:
                    pose_archive["geometry_center"].append(np.full(2, np.nan, dtype=np.float32))
                    pose_archive["geometry_dirs"].append(np.full((3, 2), np.nan, dtype=np.float32))
                if save_candidate_poses:
                    packed_candidates = pack_ranked_candidate_poses(
                        ranked_candidates or [],
                        candidate_pose_top_k,
                    )
                    for key, value in packed_candidates.items():
                        pose_archive[key].append(value)
            total_samples += 1

        total_batches += 1
        if max_samples not in (None, "", 0, False) and total_samples >= int(max_samples):
            break

    summary = finalize_sums(sums, total_batches)
    if topk_eval_enabled:
        total_for_topk = max(summary["samples"], 1)
        for k in topk_values:
            stats = topk_sums[k]
            valid_topk = max(stats["valid"], 1)
            values = np.asarray(stats["rot_deg_values"], dtype=np.float64)
            summary[f"top{k}_valid"] = stats["valid"]
            summary[f"top{k}_valid_rate"] = stats["valid"] / total_for_topk
            summary[f"top{k}_rot_deg"] = stats["rot_deg_sum"] / valid_topk if stats["valid"] else None
            summary[f"top{k}_rot_deg_median"] = float(np.median(values)) if values.size else None
            summary[f"top{k}_rot_deg_p90"] = float(np.percentile(values, 90)) if values.size else None
            summary[f"top{k}_trans_cm"] = stats["trans_cm_sum"] / valid_topk if stats["valid"] else None
            summary[f"top{k}_rot_acc_5deg"] = stats["rot_acc_5deg"] / total_for_topk
            summary[f"top{k}_rot_acc_30deg"] = stats["rot_acc_30deg"] / total_for_topk
            summary[f"top{k}_pose_acc_30cm_30deg"] = stats["pose_acc_30cm_30deg"] / total_for_topk
    summary.update(
        {
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else "",
            "axis_source": axis_source,
            "parser": parser_name,
            "multi_axis_candidate_mode": test_cfg.get("multi_axis_candidate_mode", "full") if parser_name == "peak_multi" else "",
            "multi_axis_top_k": test_cfg.get("multi_axis_top_k", "") if parser_name == "peak_multi" else "",
            "z_guided_search_top_k": test_cfg.get("z_guided_search_top_k", "") if parser_name == "peak_multi" else "",
            "selection": selection,
            "lamda0": c2_cfg["lamda0"],
            "point_distance": point_distances[0] if len(point_distances) == 1 else "",
            "point_distances": str(point_distances),
            "flags": str(flags),
            "sign_candidates": str(sign_candidates if sign_candidates is not None else "all"),
            "include_xy_swap": int(include_xy_swap),
            "c2_force_real": int(c2_force_real),
            "c2_imag_tol": c2_imag_tol,
            "translation_scale_mode": translation_scale_mode,
            "fixed_translation_scale": fixed_translation_scale,
            "symmetry_categories": str(symmetry_categories),
            "symmetry_metadata": str(metadata_path) if metadata_path is not None else "",
            "max_relative_rot_deg": max_relative_rot_deg if max_relative_rot_deg is not None else "",
            "scanned_samples": total_seen,
            "filtered_samples": total_filtered,
            "category_excluded_samples": total_category_excluded,
            "included_categories": sorted(requested_categories) if requested_categories is not None else None,
            "saved_visualizations": saved_visuals,
            "topk_eval_enabled": int(topk_eval_enabled),
            "topk_eval_ks": str(topk_values),
            "candidate_pose_export_enabled": int(save_candidate_poses),
            "candidate_pose_export_top_k": candidate_pose_top_k if save_candidate_poses else "",
        }
    )
    if pose_archive is not None:
        np.savez_compressed(
            pred_pose_path,
            archive_version=np.asarray("2" if save_candidate_poses else "1"),
            sample=np.asarray(pose_archive["sample"], dtype=np.int64),
            object_id=np.asarray(pose_archive["object_id"], dtype=np.str_),
            category=np.asarray(pose_archive["category"], dtype=np.str_),
            ref=np.asarray(pose_archive["ref"], dtype=np.str_),
            query=np.asarray(pose_archive["query"], dtype=np.str_),
            valid=np.asarray(pose_archive["valid"], dtype=np.uint8),
            relative_rot_deg=np.asarray(pose_archive["relative_rot_deg"], dtype=np.float64),
            ref_RT=np.asarray(pose_archive["ref_RT"], dtype=np.float64).reshape(-1, 4, 4),
            gt_RT=np.asarray(pose_archive["gt_RT"], dtype=np.float64).reshape(-1, 4, 4),
            K=np.asarray(pose_archive["K"], dtype=np.float64).reshape(-1, 3, 3),
            pred_RT_c2=np.asarray(pose_archive["pred_RT_c2"], dtype=np.float64).reshape(-1, 4, 4),
            pred_RT_scaled=np.asarray(pose_archive["pred_RT_scaled"], dtype=np.float64).reshape(-1, 4, 4),
            pred_RT_visual=np.asarray(pose_archive["pred_RT_visual"], dtype=np.float64).reshape(-1, 4, 4),
            translation_scale=np.asarray(pose_archive["translation_scale"], dtype=np.float64),
            selected_flag=np.asarray(pose_archive["selected_flag"], dtype=np.int16),
            selected_point_distance=np.asarray(pose_archive["selected_point_distance"], dtype=np.float64),
            selected_signs=np.asarray(pose_archive["selected_signs"], dtype=np.int8).reshape(-1, 3),
            heatmap_candidate_index=np.asarray(pose_archive["heatmap_candidate_index"], dtype=np.int32),
            axis_indices=np.asarray(pose_archive["axis_indices"], dtype=np.int16).reshape(-1, 3),
            **(
                {
                    "candidate_selection_method": np.asarray(selection),
                    "candidate_pose_top_k": np.asarray(candidate_pose_top_k, dtype=np.int32),
                    "candidate_count": np.asarray(pose_archive["candidate_count"], dtype=np.int32),
                    "candidate_RT_c2": np.asarray(pose_archive["candidate_RT_c2"], dtype=np.float64).reshape(-1, candidate_pose_top_k, 4, 4),
                    "candidate_selection_rank": np.asarray(pose_archive["candidate_selection_rank"], dtype=np.int32).reshape(-1, candidate_pose_top_k),
                    "candidate_axis_reproj_score": np.asarray(pose_archive["candidate_axis_reproj_score"], dtype=np.float64).reshape(-1, candidate_pose_top_k),
                    "candidate_axis_reproj_dir_deg": np.asarray(pose_archive["candidate_axis_reproj_dir_deg"], dtype=np.float64).reshape(-1, candidate_pose_top_k),
                    "candidate_axis_reproj_endpoint_px": np.asarray(pose_archive["candidate_axis_reproj_endpoint_px"], dtype=np.float64).reshape(-1, candidate_pose_top_k),
                    "candidate_axis_reproj_center_px": np.asarray(pose_archive["candidate_axis_reproj_center_px"], dtype=np.float64).reshape(-1, candidate_pose_top_k),
                    "candidate_axis_reproj_ref_rot_deg": np.asarray(pose_archive["candidate_axis_reproj_ref_rot_deg"], dtype=np.float64).reshape(-1, candidate_pose_top_k),
                    "candidate_heatmap_score": np.asarray(pose_archive["candidate_heatmap_score"], dtype=np.float64).reshape(-1, candidate_pose_top_k),
                    "candidate_max_imag_X": np.asarray(pose_archive["candidate_max_imag_X"], dtype=np.float64).reshape(-1, candidate_pose_top_k),
                    "candidate_max_imag_R": np.asarray(pose_archive["candidate_max_imag_R"], dtype=np.float64).reshape(-1, candidate_pose_top_k),
                    "candidate_flag": np.asarray(pose_archive["candidate_flag"], dtype=np.int16).reshape(-1, candidate_pose_top_k),
                    "candidate_point_distance": np.asarray(pose_archive["candidate_point_distance"], dtype=np.float64).reshape(-1, candidate_pose_top_k),
                    "candidate_signs": np.asarray(pose_archive["candidate_signs"], dtype=np.int8).reshape(-1, candidate_pose_top_k, 3),
                    "candidate_heatmap_candidate_index": np.asarray(pose_archive["candidate_heatmap_candidate_index"], dtype=np.int32).reshape(-1, candidate_pose_top_k),
                    "candidate_axis_indices": np.asarray(pose_archive["candidate_axis_indices"], dtype=np.int16).reshape(-1, candidate_pose_top_k, 3),
                }
                if save_candidate_poses
                else {}
            ),
        )
        summary["pred_pose_archive"] = str(pred_pose_path)
    write_csv(output_dir / "per_sample.csv", rows)
    write_csv(output_dir / "summary.csv", [summary])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[AxisPose++] overall pairs={summary['samples']} "
        f"ACC@30={100.0 * summary['rot_acc_30deg']:.2f}% "
        f"median angular error={fmt_optional(summary['rot_deg_median'], 2)} deg"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate AxisPose++ on ShapeNet.")
    parser.add_argument("--config", default="configs/test_shapenet.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-batches", default=None)
    parser.add_argument("--max-samples", default=None)
    parser.add_argument("--max-relative-rot-deg", default=None)
    parser.add_argument("--model-points-root", default=None)
    args = parser.parse_args()

    cfg = load_merged_config(args.config)
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.checkpoint is not None:
        cfg["test"]["checkpoint"] = args.checkpoint
    if args.output_dir is not None:
        cfg["test"]["output_dir"] = args.output_dir
    if args.max_batches is not None:
        cfg["test"]["max_batches"] = int(args.max_batches)
    if args.max_samples is not None:
        cfg["test"]["max_samples"] = int(args.max_samples)
    if args.max_relative_rot_deg is not None:
        cfg["test"]["max_relative_rot_deg"] = float(args.max_relative_rot_deg)
    if args.model_points_root is not None:
        cfg.setdefault("model_points", {})["root"] = args.model_points_root
    run_test(cfg)


if __name__ == "__main__":
    main()

