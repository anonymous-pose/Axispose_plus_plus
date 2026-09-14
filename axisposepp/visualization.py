import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .losses import geometry_from_axis_heatmap


AXIS_COLORS = ((255, 0, 0), (0, 255, 0), (0, 80, 255))


def _line_image_from_px(height, width, center_px, directions, axis_length=None):
    if axis_length is None:
        axis_length = 0.42 * min(height, width)
    center = center_px.detach().float().cpu()
    directions = directions.detach().float().cpu()
    center = torch.stack(
        (
            center[0].clamp(0.0, float(width - 1)),
            center[1].clamp(0.0, float(height - 1)),
        )
    )
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    cx, cy = float(center[0]), float(center[1])
    for axis_index, color in enumerate(AXIS_COLORS):
        direction = directions[axis_index]
        direction = direction / direction.norm().clamp_min(1e-6)
        endpoint = center + direction * axis_length
        draw.line((cx, cy, float(endpoint[0]), float(endpoint[1])), fill=color, width=3)
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 0))
    return image


def _center_from_heatmap(axis_heatmap, center_heatmap=None):
    axis_heatmap = axis_heatmap.detach().float().cpu()
    _, height, width = axis_heatmap.shape
    if center_heatmap is not None:
        flat_index = int(center_heatmap.detach().float().cpu().squeeze(0).argmax())
        return torch.tensor([flat_index % width, flat_index // width], dtype=torch.float32)
    center, _ = geometry_from_axis_heatmap(axis_heatmap.unsqueeze(0))
    return center[0]


def _local_peak_candidates(channel, valid, top_k=32, nms_kernel=11):
    pooled = F.max_pool2d(channel.view(1, 1, *channel.shape), kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2).view_as(channel)
    scores = channel.masked_fill(~((channel >= pooled) & valid), float("-inf")).flatten()
    values, indices = torch.topk(scores, k=min(top_k, scores.numel()))
    finite = torch.isfinite(values)
    if finite.any():
        return values[finite], indices[finite]
    values, indices = torch.topk(channel.masked_fill(~valid, float("-inf")).flatten(), k=min(top_k, scores.numel()))
    return values[torch.isfinite(values)], indices[torch.isfinite(values)]


def _ray_response(channel, center, direction, min_distance=8, num_samples=96, axis_length=None):
    height, width = channel.shape
    extent = 0.46 * min(height, width) if axis_length is None else axis_length
    steps = torch.linspace(float(min_distance), float(extent), num_samples)
    points = center.view(1, 2) + steps.view(-1, 1) * direction.view(1, 2)
    xs = points[:, 0].round().long().clamp(0, width - 1)
    ys = points[:, 1].round().long().clamp(0, height - 1)
    return channel[ys, xs].mean()


def _geometry_guided_heatmap_geometry(axis_heatmap, center_heatmap=None, geometry_dirs=None, min_center_distance=8, top_k=32, geo_weight=0.5):
    axis_heatmap = axis_heatmap.detach().float().cpu()
    _, height, width = axis_heatmap.shape
    center = _center_from_heatmap(axis_heatmap, center_heatmap)
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    valid = (torch.stack([xs.float(), ys.float()], dim=-1) - center.view(1, 1, 2)).norm(dim=-1) >= float(min_center_distance)
    directions = []
    for axis_index, channel in enumerate(axis_heatmap):
        values, indices = _local_peak_candidates(channel, valid, top_k=top_k)
        global_direction = None
        if geometry_dirs is not None:
            global_direction = F.normalize(geometry_dirs[axis_index].detach().float().cpu(), dim=0, eps=1e-6)
        best_score, best_direction = None, None
        for value, flat_index in zip(values, indices):
            point = torch.tensor([int(flat_index) % width, int(flat_index) // width], dtype=torch.float32)
            direction = F.normalize(point - center, dim=0, eps=1e-6)
            ray_score = _ray_response(channel, center, direction, min_distance=min_center_distance)
            score = 0.6 * ray_score + 0.4 * value
            if global_direction is not None:
                score = score + geo_weight * ((direction * global_direction).sum().clamp(-1.0, 1.0) + 1.0) * 0.5
            if best_score is None or score > best_score:
                best_score, best_direction = score, direction
        directions.append(best_direction if best_direction is not None else torch.tensor([1.0, 0.0]))
    return center, torch.stack(directions)


def _geometry_guided_ray_search_geometry(axis_heatmap, center_heatmap=None, geometry_dirs=None, min_center_distance=8, num_angles=360, num_samples=128, geo_weight=0.5):
    axis_heatmap = axis_heatmap.detach().float().cpu()
    _, height, width = axis_heatmap.shape
    center = _center_from_heatmap(axis_heatmap, center_heatmap)
    angles = torch.linspace(0.0, 2.0 * torch.pi, num_angles + 1)[:-1]
    candidates = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    steps = torch.linspace(float(min_center_distance), 0.46 * min(height, width), num_samples)
    directions = []
    for axis_index, channel in enumerate(axis_heatmap):
        points = center.view(1, 1, 2) + candidates.view(num_angles, 1, 2) * steps.view(1, num_samples, 1)
        scores = channel[points[..., 1].round().long().clamp(0, height - 1), points[..., 0].round().long().clamp(0, width - 1)].mean(dim=1)
        if geometry_dirs is not None:
            global_direction = F.normalize(geometry_dirs[axis_index].detach().float().cpu(), dim=0, eps=1e-6)
            scores = scores + geo_weight * ((candidates * global_direction).sum(dim=-1).clamp(-1.0, 1.0) + 1.0) * 0.5
        directions.append(candidates[int(scores.argmax())])
    return center, torch.stack(directions)
