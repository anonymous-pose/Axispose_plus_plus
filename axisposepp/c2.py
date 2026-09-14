import numpy as np
from numpy.polynomial.polynomial import Polynomial


def pose_frm_c2(K, points_h, lamda0, branch):
    rays = np.asarray(points_h, dtype=np.float64).copy()
    rays = rays - np.tile(np.asarray(K, dtype=np.float64)[:, 2].reshape(-1, 1), (1, 4))
    rays[2, :] = abs(float(K[0, 0]))
    rays = rays / np.maximum(np.linalg.norm(rays, axis=0, keepdims=True), 1e-12)
    origin = float(lamda0) * rays[:, 0]
    axis_x, axis_y, axis_z = _solve_axes(rays, origin, branch)
    directions = np.column_stack([axis_x - origin, axis_y - origin, axis_z - origin])
    rotation = directions / np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-12)
    return np.column_stack([origin, axis_x, axis_y, axis_z]), rotation


def _solve_axes(rays, origin, branch):
    nx, ny, nz = rays[:, 1], rays[:, 2], rays[:, 3]
    a_xy, a_xz, a_yz = np.dot(nx, ny), np.dot(nx, nz), np.dot(ny, nz)
    b_x, b_y, b_z = np.dot(nx, origin), np.dot(ny, origin), np.dot(nz, origin)
    origin_norm_sq = np.dot(origin, origin)
    coefficient_a = a_yz * b_x**2 - a_xz * b_x * b_y - a_xy * b_x * b_z + a_xy * a_xz * origin_norm_sq
    coefficient_b = (-2 * a_yz * b_x * origin_norm_sq + (a_xz * origin_norm_sq + b_z * b_x) * b_y + (a_xy * origin_norm_sq + b_x * b_y) * b_z - (a_xy * b_z + a_xz * b_y) * origin_norm_sq)
    coefficient_c = a_yz * origin_norm_sq**2 - b_y * b_z * origin_norm_sq
    lambda_x = Polynomial([coefficient_c, coefficient_b, coefficient_a]).roots()
    lambda_y = (b_x * lambda_x - origin_norm_sq) / (a_xy * lambda_x - b_y)
    lambda_z = (b_x * lambda_x - origin_norm_sq) / (a_xz * lambda_x - b_z)
    index = int(branch) - 1
    return lambda_x[index] * nx, lambda_y[index] * ny, lambda_z[index] * nz
