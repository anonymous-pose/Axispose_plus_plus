import itertools

import numpy as np
from numpy.linalg import norm

from .c2 import pose_frm_c2


DEFAULT_SIGN_CANDIDATES = tuple(itertools.product((1, -1), repeat=3))


def scale_intrinsics(K, src_size, dst_size):
    """Scale 3D camera intrinsics to resized image coordinates."""
    K = np.asarray(K, dtype=np.float64).copy()
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def normalize_directions(directions, eps=1e-8):
    directions = np.asarray(directions, dtype=np.float64)
    if directions.shape != (3, 2):
        raise ValueError(f"directions must have shape (3, 2), got {directions.shape}")
    denom = np.maximum(norm(directions, axis=1, keepdims=True), eps)
    return directions / denom


def axes_to_c2_points(center_px, directions, point_distance=128.0, signs=(1, 1, 1), swap_xy=False):
    """Convert a 2D tri-axis configuration to homogeneous C2 points."""
    center_px = np.asarray(center_px, dtype=np.float64).reshape(2)
    directions = normalize_directions(directions)
    signs = np.asarray(signs, dtype=np.float64).reshape(3)

    if swap_xy:
        directions = directions[[1, 0, 2]]
        signs = signs[[1, 0, 2]]

    endpoints = center_px[None, :] + directions * signs[:, None] * float(point_distance)
    points = np.vstack([center_px[None, :], endpoints])
    return np.vstack([points.T, np.ones(4, dtype=np.float64)])


def _to_real(array, imag_tol=1e-6, force_real=False):
    array = np.asarray(array)
    max_imag = 0.0
    if np.iscomplexobj(array):
        max_imag = float(np.max(np.abs(array.imag))) if array.size else 0.0
        if max_imag > imag_tol and not force_real:
            return array.real, False, max_imag
        return array.real, True, max_imag
    return array.astype(np.float64), True, max_imag


def _project_to_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def solve_c2_from_points(
    K,
    points_h,
    lamda0,
    flag=1,
    orthonormalize=True,
    imag_tol=1e-6,
    force_real=False,
):
    """Solve one C2 pose from a 3 x 4 homogeneous point matrix."""
    K = np.asarray(K, dtype=np.float64)
    points_h = np.asarray(points_h, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {K.shape}")
    if points_h.shape != (3, 4):
        raise ValueError(f"points_h must have shape (3, 4), got {points_h.shape}")

    X, R = pose_frm_c2(K.copy(), points_h.copy(), float(lamda0), int(flag))
    X, valid_X, max_imag_X = _to_real(X, imag_tol=imag_tol, force_real=force_real)
    R, valid_R, max_imag_R = _to_real(R, imag_tol=imag_tol, force_real=force_real)
    if orthonormalize:
        R = _project_to_rotation(R)

    t = X[:3, 0].reshape(3)
    RT = np.eye(4, dtype=np.float64)
    RT[:3, :3] = R
    RT[:3, 3] = t
    return {
        "RT": RT,
        "R": R,
        "t": t,
        "X": X,
        "valid": bool(valid_X and valid_R and np.isfinite(RT).all()),
        "valid_X": bool(valid_X),
        "valid_R": bool(valid_R),
        "max_imag_X": float(max_imag_X),
        "max_imag_R": float(max_imag_R),
        "force_real": bool(force_real),
    }


def solve_c2_candidates_from_axes(
    K,
    center_px,
    directions,
    lamda0,
    flag=1,
    point_distance=128.0,
    sign_candidates=None,
    include_xy_swap=False,
    orthonormalize=True,
    imag_tol=1e-6,
    force_real=False,
):
    """Generate C2 candidates from 2D tri-axes."""
    if sign_candidates is None:
        sign_candidates = DEFAULT_SIGN_CANDIDATES

    candidates = []
    swap_options = (False, True) if include_xy_swap else (False,)
    for swap_xy in swap_options:
        for signs in sign_candidates:
            points_h = axes_to_c2_points(
                center_px,
                directions,
                point_distance=point_distance,
                signs=signs,
                swap_xy=swap_xy,
            )
            try:
                solved = solve_c2_from_points(
                    K,
                    points_h,
                    lamda0=lamda0,
                    flag=flag,
                    orthonormalize=orthonormalize,
                    imag_tol=imag_tol,
                    force_real=force_real,
                )
            except Exception as exc:
                solved = {
                    "RT": None,
                    "R": None,
                    "t": None,
                    "X": None,
                    "valid": False,
                    "error": str(exc),
                    "valid_X": False,
                    "valid_R": False,
                    "max_imag_X": None,
                    "max_imag_R": None,
                    "force_real": bool(force_real),
                }
            solved.update(
                {
                    "points_h": points_h,
                    "signs": tuple(int(v) for v in signs),
                    "swap_xy": bool(swap_xy),
                    "flag": int(flag),
                    "lamda0": float(lamda0),
                    "point_distance": float(point_distance),
                }
            )
            candidates.append(solved)
    return candidates
