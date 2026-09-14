import torch
import torch.nn as nn
import torch.nn.functional as F


AXIS_PAIRS = ((0, 1), (0, 2), (1, 2))


def weighted_bce_dice_loss(
    logits,
    target,
    threshold=0.05,
    fg_weight=20.0,
    bg_weight=1.0,
    dice_weight=1.0,
    eps=1e-6,
):
    target = target.float()
    logits = logits.float()
    weights = torch.where(
        target > threshold,
        torch.full_like(target, float(fg_weight)),
        torch.full_like(target, float(bg_weight)),
    )

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * weights).sum() / weights.sum().clamp_min(eps)

    pred = torch.sigmoid(logits)
    intersection = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * intersection + eps) / (union + eps)
    dice = dice.mean()
    return bce + dice_weight * dice


def fit_axis_lines(axis, eps=1e-6):
    """Fit one 2D line per axis heatmap with weighted PCA."""
    if axis.dim() != 4 or axis.shape[1] != 3:
        raise ValueError("axis must have shape B x 3 x H x W")

    axis = axis.float().clamp_min(0.0)
    b, c, h, w = axis.shape
    device = axis.device
    dtype = axis.dtype
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs.view(1, 1, h, w)
    ys = ys.view(1, 1, h, w)

    sum_w = axis.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    mean_x = (axis * xs).sum(dim=(2, 3), keepdim=True) / sum_w
    mean_y = (axis * ys).sum(dim=(2, 3), keepdim=True) / sum_w

    dx = xs - mean_x
    dy = ys - mean_y
    norm = sum_w.squeeze(-1).squeeze(-1)
    cov_xx = (axis * dx * dx).sum(dim=(2, 3)) / norm
    cov_xy = (axis * dx * dy).sum(dim=(2, 3)) / norm
    cov_yy = (axis * dy * dy).sum(dim=(2, 3)) / norm
    cov = torch.stack(
        (
            torch.stack((cov_xx, cov_xy), dim=-1),
            torch.stack((cov_xy, cov_yy), dim=-1),
        ),
        dim=-2,
    ).reshape(b * c, 2, 2)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    max_idx = eigvals.argmax(dim=-1).view(b * c, 1, 1).expand(-1, 2, 1)
    directions = torch.gather(eigvecs, dim=2, index=max_idx).squeeze(-1)
    directions = F.normalize(directions, dim=-1, eps=eps).view(b, c, 2)
    points = torch.stack((mean_x.squeeze(-1).squeeze(-1), mean_y.squeeze(-1).squeeze(-1)), dim=-1)
    return points, directions


def line_intersection(point1, direction1, point2, direction2, eps=1e-4):
    cross = direction1[:, 0] * direction2[:, 1] - direction1[:, 1] * direction2[:, 0]
    delta = point2 - point1
    numerator = delta[:, 0] * direction2[:, 1] - delta[:, 1] * direction2[:, 0]
    denom = cross + eps * torch.where(cross >= 0, torch.ones_like(cross), -torch.ones_like(cross))
    t = numerator / denom
    return point1 + t.unsqueeze(-1) * direction1


def geometry_from_axis_heatmap(axis, parallel_eps=0.05, eps=1e-6):
    """Compute center and signed directions from three GT/predicted axis heatmaps.

    The line orientation from PCA is sign-ambiguous. We resolve the sign by using
    the weighted line point: rendered axis segments have their weighted mean on
    the tip side of the center, so direction points from center toward that mean.
    """
    points, line_dirs = fit_axis_lines(axis, eps=eps)
    intersections = []
    weights = []
    for i, j in AXIS_PAIRS:
        intersections.append(line_intersection(points[:, i], line_dirs[:, i], points[:, j], line_dirs[:, j]))
        cross = line_dirs[:, i, 0] * line_dirs[:, j, 1] - line_dirs[:, i, 1] * line_dirs[:, j, 0]
        weights.append(cross.abs())
    intersections = torch.stack(intersections, dim=1)
    weights = torch.stack(weights, dim=1)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(parallel_eps)
    center = (intersections * weights.unsqueeze(-1)).sum(dim=1)

    center_to_mean = points - center[:, None, :]
    sign = torch.sign((line_dirs * center_to_mean).sum(dim=-1, keepdim=True)).clamp_min(0.0) * 2.0 - 1.0
    signed_dirs = F.normalize(line_dirs * sign, dim=-1, eps=eps)
    return center, signed_dirs


class AxisPosePPLoss(nn.Module):
    def __init__(
        self,
        axis_hm_weight=1.0,
        center_hm_weight=1.0,
        center_coord_weight=1.0,
        direction_weight=1.0,
        axis_fg_weight=20.0,
        axis_bg_weight=1.0,
        axis_threshold=0.05,
        axis_dice_weight=1.0,
        derive_geometry_from_axis=True,
        parallel_eps=0.05,
    ):
        super().__init__()
        self.axis_hm_weight = axis_hm_weight
        self.center_hm_weight = center_hm_weight
        self.center_coord_weight = center_coord_weight
        self.direction_weight = direction_weight
        self.axis_fg_weight = axis_fg_weight
        self.axis_bg_weight = axis_bg_weight
        self.axis_threshold = axis_threshold
        self.axis_dice_weight = axis_dice_weight
        self.derive_geometry_from_axis = derive_geometry_from_axis
        self.parallel_eps = parallel_eps

    def forward(self, outputs, batch):
        pred_hm = outputs["heatmap"]
        pred_axis = pred_hm[:, 0:3]
        pred_axis_logits = outputs["heatmap_logits"][:, 0:3]
        pred_center_hm = pred_hm[:, 3:4]

        target_axis = batch["target_axis_heatmap"].float()
        target_center_hm = batch["target_center_heatmap"].float()

        axis_loss = weighted_bce_dice_loss(
            pred_axis_logits,
            target_axis,
            threshold=self.axis_threshold,
            fg_weight=self.axis_fg_weight,
            bg_weight=self.axis_bg_weight,
            dice_weight=self.axis_dice_weight,
        )
        center_hm_loss = F.mse_loss(pred_center_hm, target_center_hm)

        if self.derive_geometry_from_axis or "target_center_norm" not in batch or "target_directions" not in batch:
            with torch.no_grad():
                target_center_px, target_dirs = geometry_from_axis_heatmap(
                    target_axis,
                    parallel_eps=self.parallel_eps,
                )
                h, w = target_axis.shape[-2:]
                target_center = target_center_px.clone()
                target_center[:, 0] = target_center[:, 0] / max(float(w - 1), 1.0)
                target_center[:, 1] = target_center[:, 1] / max(float(h - 1), 1.0)
                target_center = target_center.clamp(0.0, 1.0)
        else:
            target_center = batch["target_center_norm"].float()
            target_dirs = batch["target_directions"].float()

        center_coord_loss = F.smooth_l1_loss(outputs["center"], target_center)

        pred_dirs = F.normalize(outputs["directions"].float(), dim=-1, eps=1e-6)
        target_dirs = F.normalize(target_dirs.float(), dim=-1, eps=1e-6)

        direction_loss = (1.0 - (pred_dirs * target_dirs).sum(dim=-1)).mean()

        total = (
            self.axis_hm_weight * axis_loss
            + self.center_hm_weight * center_hm_loss
            + self.center_coord_weight * center_coord_loss
            + self.direction_weight * direction_loss
        )
        return {
            "loss": total,
            "axis_hm": axis_loss.detach(),
            "center_hm": center_hm_loss.detach(),
            "center_coord": center_coord_loss.detach(),
            "direction": direction_loss.detach(),
        }
