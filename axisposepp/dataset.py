import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .geometry import open_image
from .heatmaps import make_center_heatmap, rgb_axis_to_soft_heatmap


def _split_axis_masks(axis_rgb, threshold=0.3):
    axis_rgb = axis_rgb.float()
    if axis_rgb.max() > 1.5:
        axis_rgb = axis_rgb / 255.0
    red, green, blue = axis_rgb
    return [
        (red > threshold) & (red > green) & (red > blue),
        (green > threshold) & (green > red) & (green > blue),
        (blue > threshold) & (blue > red) & (blue > green),
    ]


def _axis_endpoints(mask):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if xs.numel() < 2:
        return None
    points = torch.stack([xs.float(), ys.float()], dim=1)
    mean = points.mean(dim=0, keepdim=True)
    centered = points - mean
    covariance = centered.t().mm(centered) / max(float(points.shape[0] - 1), 1.0)
    _, vectors = torch.linalg.eigh(covariance)
    direction = vectors[:, -1]
    projection = centered @ direction
    return torch.stack([points[projection.argmin()], points[projection.argmax()]])


def estimate_axis_geometry(axis_rgb, threshold=0.3):
    endpoints = [_axis_endpoints(mask) for mask in _split_axis_masks(axis_rgb, threshold)]
    if any(points is None for points in endpoints):
        return None, None
    best_choice, best_score = None, None
    for choice in range(8):
        selected = torch.stack([endpoints[i][(choice >> i) & 1] for i in range(3)])
        center = selected.mean(dim=0)
        score = ((selected - center) ** 2).sum(dim=1).mean()
        if best_score is None or score < best_score:
            best_choice, best_score = choice, score
    center_points, directions = [], []
    for axis_index in range(3):
        center_index = (best_choice >> axis_index) & 1
        center_point = endpoints[axis_index][center_index]
        center_points.append(center_point)
        directions.append(endpoints[axis_index][1 - center_index] - center_point)
    center = torch.stack(center_points).mean(dim=0)
    directions = torch.nn.functional.normalize(torch.stack(directions), dim=-1, eps=1e-6)
    return center, directions


class ShapeNetPairs(Dataset):
    def __init__(self, root_dir, split, image_size=336, train_pair_file=None, test_pair_file=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = int(image_size)
        pair_file = train_pair_file if split == "train" else test_pair_file
        if pair_file is None:
            pair_file = "train_pairs.pkl" if split == "train" else "test_pairs.pkl"
        pair_file = Path(pair_file)
        self.pair_file = pair_file if pair_file.is_absolute() else self.root_dir / pair_file
        with self.pair_file.open("rb") as handle:
            self.samples = pickle.load(handle)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.axis_transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                antialias=False,
            ),
        ])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _flatten_pose(pose):
        pose = np.asarray(pose, dtype=np.float32)
        values = np.concatenate([pose[:3, :3].reshape(-1), pose[:3, 3]])
        return torch.from_numpy(values).float()

    def _path(self, prefix, suffix):
        return self.root_dir / f"{prefix}{suffix}"

    def __getitem__(self, index):
        sample = self.samples[index]
        ref_prefix, query_prefix = sample["ref"], sample["query"]
        ref_pose = np.loadtxt(self._path(ref_prefix, "_pose.txt")).astype(np.float32)
        query_pose = np.loadtxt(self._path(query_prefix, "_pose.txt")).astype(np.float32)
        ref_image = self.image_transform(open_image(self._path(ref_prefix, "_rot.png")))
        target_image = self.image_transform(open_image(self._path(query_prefix, "_rot.png")))
        ref_axis = self.axis_transform(open_image(self._path(ref_prefix, "_axisRot.png")))
        target_axis = self.axis_transform(open_image(self._path(query_prefix, "_axisRot.png")))
        axis_heatmap = rgb_axis_to_soft_heatmap(target_axis, method="dominant", threshold=0.3, kernel_size=9, sigma=2.0)
        center, directions = estimate_axis_geometry(target_axis, threshold=0.3)
        if center is None:
            center = torch.zeros(2, dtype=torch.float32)
            directions = torch.zeros(3, 2, dtype=torch.float32)
        center_heatmap = make_center_heatmap(center, self.image_size, self.image_size, sigma=2.0)
        center_norm = center / max(float(self.image_size - 1), 1.0)
        return {
            "ref_image": ref_image,
            "ref_axis": ref_axis,
            "ref_pose": self._flatten_pose(ref_pose),
            "target_image": target_image,
            "target_axis_heatmap": axis_heatmap,
            "target_center_heatmap": center_heatmap,
            "target_center_norm": center_norm,
            "target_directions": directions,
            "meta": {"ref": ref_prefix, "query": query_prefix},
        }
