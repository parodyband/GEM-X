"""Batched rotation math (numpy).

Convention: column vectors, so a world rotation is ``parent @ local`` and a
point maps as ``R @ p``. Maya matrices are row-vector; convert with
``.T`` at the boundary (see ``target.py``). Quaternions from gem-x.cpp are
XYZW.
"""

from __future__ import annotations

import numpy as np

# Maya rotateOrder enum -> axis sequence (first applied axis first).
ROTATE_ORDERS = {
    0: (0, 1, 2),  # xyz
    1: (1, 2, 0),  # yzx
    2: (2, 0, 1),  # zxy
    3: (0, 2, 1),  # xzy
    4: (1, 0, 2),  # yxz
    5: (2, 1, 0),  # zyx
}
_EVEN = {0, 1, 2}


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) XYZW -> (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) XYZW with w >= 0."""
    m = np.asarray(m, dtype=np.float64)
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cands = np.stack(
        [
            np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], 1 + tr], -1),
            np.stack([1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] - m[..., 1, 2]], -1),
            np.stack([m[..., 0, 1] + m[..., 1, 0], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1], m[..., 0, 2] - m[..., 2, 0]], -1),
            np.stack([m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], m[..., 1, 0] - m[..., 0, 1]], -1),
        ],
        axis=-2,
    )  # (..., 4 candidates, 4)
    norms = np.linalg.norm(cands, axis=-1)
    best = np.argmax(norms, axis=-1)
    q = np.take_along_axis(cands, best[..., None, None], axis=-2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., 3:4] < 0, -q, q)


def axis_rotation(axis: int, angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    m = np.zeros(angle.shape + (3, 3))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    m[..., axis, axis] = 1
    m[..., i, i] = c
    m[..., j, j] = c
    m[..., i, j] = -s
    m[..., j, i] = s
    return m


def euler_to_matrix(angles: np.ndarray, order: int = 0) -> np.ndarray:
    """(..., 3) radians (rx, ry, rz) in Maya rotate order -> (..., 3, 3)."""
    angles = np.asarray(angles, dtype=np.float64)
    i, j, k = ROTATE_ORDERS[order]
    return axis_rotation(k, angles[..., k]) @ axis_rotation(j, angles[..., j]) @ axis_rotation(i, angles[..., i])


def matrix_to_euler(m: np.ndarray, order: int = 0) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) radians (rx, ry, rz) for Maya rotate order."""
    m = np.asarray(m, dtype=np.float64)
    i, j, k = ROTATE_ORDERS[order]
    s = 1.0 if order in _EVEN else -1.0
    sb = np.clip(-s * m[..., k, i], -1.0, 1.0)
    b = np.arcsin(sb)
    cb = np.sqrt(np.maximum(0.0, 1.0 - sb * sb))
    a = np.arctan2(s * m[..., k, j], m[..., k, k])
    c = np.arctan2(s * m[..., j, i], m[..., i, i])
    lock = cb < 1e-7
    if np.any(lock):
        a = np.where(lock, np.arctan2(-s * m[..., j, k], m[..., j, j]), a)
        c = np.where(lock, 0.0, c)
    out = np.empty(m.shape[:-2] + (3,))
    out[..., i] = a
    out[..., j] = b
    out[..., k] = c
    return out


def _wrap_near(angle: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return angle + 2 * np.pi * np.round((ref - angle) / (2 * np.pi))


def closest_euler(angles: np.ndarray, previous: np.ndarray, order: int = 0) -> np.ndarray:
    """Pick the equivalent euler triple nearest ``previous`` (both (..., 3))."""
    i, j, k = ROTATE_ORDERS[order]
    alt = angles.copy()
    alt[..., i] = angles[..., i] + np.pi
    alt[..., j] = np.pi - angles[..., j]
    alt[..., k] = angles[..., k] + np.pi
    a = _wrap_near(angles, previous)
    b = _wrap_near(alt, previous)
    da = np.abs(a - previous).sum(-1, keepdims=True)
    db = np.abs(b - previous).sum(-1, keepdims=True)
    return np.where(db < da, b, a)


def euler_filter(angles: np.ndarray, order: int = 0, start: np.ndarray | None = None) -> np.ndarray:
    """Make an (N, 3) euler sequence continuous (like Maya's Euler Filter)."""
    out = np.array(angles, dtype=np.float64, copy=True)
    prev = start if start is not None else out[0]
    for n in range(len(out)):
        out[n] = closest_euler(out[n], prev, order)
        prev = out[n]
    return out


def orthonormalize(m: np.ndarray) -> np.ndarray:
    """Remove scale (and small shear) from (..., 3, 3) rotation-ish matrices."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    det = np.linalg.det(r)
    if np.any(det < 0):
        u = u.copy()
        u[det < 0, :, -1] *= -1
        r = u @ vt
    return r


def shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking direction ``a`` onto direction ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.eye(3)
    a, b = a / na, b / nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-9:  # opposite: rotate pi about any perpendicular axis
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Batched slerp; q0, q1 (..., 4), t broadcastable to (...,)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin = np.sin(theta)
    small = sin < 1e-6
    w0 = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1, sin))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1, sin))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)
