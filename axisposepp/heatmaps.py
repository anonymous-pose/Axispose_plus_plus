import torch
import torch.nn.functional as F


def gaussian_blur_masks(masks, kernel_size=9, sigma=2.0):
    """Blur binary masks into soft heatmaps and normalize each channel to [0, 1]."""
    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd")

    squeeze_batch = False
    if masks.dim() == 3:
        masks = masks.unsqueeze(0)
        squeeze_batch = True

    masks = masks.float()
    device = masks.device
    dtype = masks.dtype
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.repeat(masks.shape[1], 1, 1, 1)

    heatmap = F.conv2d(masks, kernel_2d, padding=kernel_size // 2, groups=masks.shape[1])
    max_val = heatmap.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    heatmap = heatmap / max_val
    return heatmap.squeeze(0) if squeeze_batch else heatmap


def rgb_axis_to_soft_heatmap(axis_rgb, method="dominant", threshold=0.3, kernel_size=9, sigma=2.0):
    """Convert RGB tri-axis renderings to 3 soft heatmaps: red=x, green=y, blue=z."""
    squeeze_batch = False
    if axis_rgb.dim() == 3:
        axis_rgb = axis_rgb.unsqueeze(0)
        squeeze_batch = True

    axis_rgb = axis_rgb.float()
    if axis_rgb.max() > 1.5:
        axis_rgb = axis_rgb / 255.0

    r = axis_rgb[:, 0:1]
    g = axis_rgb[:, 1:2]
    b = axis_rgb[:, 2:3]

    if method == "direct":
        x_mask = (r > threshold).float()
        y_mask = (g > threshold).float()
        z_mask = (b > threshold).float()
    elif method == "dominant":
        x_mask = ((r > threshold) & (r > g) & (r > b)).float()
        y_mask = ((g > threshold) & (g > r) & (g > b)).float()
        z_mask = ((b > threshold) & (b > r) & (b > g)).float()
    else:
        raise ValueError(f"Unknown axis RGB conversion method: {method}")

    masks = torch.cat([x_mask, y_mask, z_mask], dim=1)
    heatmap = gaussian_blur_masks(masks, kernel_size=kernel_size, sigma=sigma)
    return heatmap.squeeze(0) if squeeze_batch else heatmap


def make_center_heatmap(center_xy, height, width, sigma=2.0):
    """Create a Gaussian center heatmap from pixel coordinates [u, v]."""
    if center_xy.dim() == 1:
        center_xy = center_xy.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    device = center_xy.device
    dtype = center_xy.dtype
    ys = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
    xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
    u = center_xy[:, 0].view(-1, 1, 1)
    v = center_xy[:, 1].view(-1, 1, 1)
    heatmap = torch.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2 * sigma**2))
    heatmap = heatmap.unsqueeze(1)
    return heatmap.squeeze(0) if squeeze_batch else heatmap


def make_axis_line_heatmaps(center_xy, directions, height, width, axis_length=None, sigma=2.0):
    """Render three soft axis-line heatmaps from center and 2D directions."""
    if center_xy.dim() == 1:
        center_xy = center_xy.unsqueeze(0)
        directions = directions.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    if axis_length is None:
        axis_length = 0.45 * min(height, width)

    device = center_xy.device
    dtype = center_xy.dtype
    dirs = F.normalize(directions.float(), dim=-1, eps=1e-6).to(dtype=dtype)
    start = center_xy[:, None, :]
    end = start + dirs * axis_length

    ys = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
    xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
    px = torch.cat([xs.expand(center_xy.shape[0], 3, height, width).unsqueeze(-1),
                    ys.expand(center_xy.shape[0], 3, height, width).unsqueeze(-1)], dim=-1)

    seg = end - start
    rel = px - start[:, :, None, None, :]
    t = (rel * seg[:, :, None, None, :]).sum(-1) / (seg.square().sum(-1)[:, :, None, None] + 1e-6)
    t = t.clamp(0.0, 1.0)
    nearest = start[:, :, None, None, :] + t[..., None] * seg[:, :, None, None, :]
    dist = (px - nearest).square().sum(-1).sqrt()
    heatmap = torch.exp(-(dist**2) / (2 * sigma**2))
    return heatmap.squeeze(0) if squeeze_batch else heatmap
