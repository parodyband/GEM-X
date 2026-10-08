"""World-space delta retargeting from SOMA-77 to an arbitrary joint hierarchy.

For a target joint j mapped to source joint s:

    W_j(t) = F @ G_s(t) @ C_j,    C_j = G_s0^T @ F^T @ A_j^T @ T0_j

F   aligns the source character frame (Y-up, facing +Z) with the target's.
G_s source world rotation; G_s0 its neutral rest value.
T0_j target rest world rotation.
A_j swings the source rest bone direction onto the target rest bone direction,
    so T-pose/A-pose differences between the two rest poses cancel out.

When the source is in its rest pose the target bone points along the source
bone; when the source matches the target's rest pose the target sits at T0.
Unmapped target joints keep their rest local rotation. Masked joints are
solved but not written, so their children keep the motion relative to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mathutil as mu
from . import soma
from .target import Character


@dataclass
class Options:
    translate_horizontal: bool = True
    translate_vertical: bool = True
    vertical_mode: str = "floor"  # floor | trajectory
    auto_scale: bool = True
    scale: float = 1.0  # used when auto_scale is False (target units per metre)
    facing_offset: float = 0.0  # degrees about the scene up axis

    @classmethod
    def from_dict(cls, d: dict) -> "Options":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Solution:
    rotations: dict[int, np.ndarray] = field(default_factory=dict)  # joint -> (N, 3) euler radians
    translations: dict[int, np.ndarray] = field(default_factory=dict)  # joint -> (N, 3) cm


def _rot_about(axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class Retargeter:
    def __init__(self, skeleton: soma.Skeleton, character: Character, options: Options | None = None):
        self.sk = skeleton
        self.ch = character
        self.opt = options or Options.from_dict(character.options)
        self.prepare()

    # ------------------------------------------------------------------ setup

    def prepare(self) -> None:
        sk, ch = self.sk, self.ch
        self.rest_world_rows, T0, P_t0 = ch.rest_world()
        self.T0, self.P_t0 = T0, P_t0
        self.src = {j: sk.index[s] for j, s in ch.mapping.items() if s in sk.index}
        self.src_to_tgt: dict[int, int] = {}
        for j in range(ch.count):  # first (top-most) target wins
            s = self.src.get(j)
            if s is not None and s not in self.src_to_tgt:
                self.src_to_tgt[s] = j

        # Rest local rotations (column) for unmapped joints.
        self.top_R = mu.orthonormalize(ch.top_parent_matrix()[:3, :3].T)
        self.rest_local_R = np.empty((ch.count, 3, 3))
        for j in range(ch.count):
            parent_R = T0[ch.parents[j]] if ch.parents[j] >= 0 else self.top_R
            self.rest_local_R[j] = parent_R.T @ T0[j]

        self.F = self._facing()
        self.A = self._alignments()
        G0 = sk.rest_world_R
        self.C = {j: G0[s].T @ self.F.T @ self.A[j].T @ T0[j] for j, s in self.src.items()}
        self.root_joint = self.src_to_tgt.get(sk.index.get(soma.HIPS, -1))
        self.scale = self._auto_scale() if self.opt.auto_scale else self.opt.scale
        self.JOt = np.transpose(ch.JO, (0, 2, 1))
        self.RAt = np.transpose(ch.RA, (0, 2, 1))

    def _pair(self, left: str, right: str):
        sk = self.sk
        li, ri = sk.index.get(left), sk.index.get(right)
        if li is None or ri is None or li not in self.src_to_tgt or ri not in self.src_to_tgt:
            return None
        src_lat = sk.rest_world_p[li] - sk.rest_world_p[ri]
        tgt_lat = self.P_t0[self.src_to_tgt[li]] - self.P_t0[self.src_to_tgt[ri]]
        return src_lat, tgt_lat

    def _facing(self) -> np.ndarray:
        up_t = self.ch.up
        up_s = np.array([0.0, 1.0, 0.0])
        pair = self._pair("LeftLeg", "RightLeg") or self._pair("LeftArm", "RightArm") or self._pair("LeftShoulder", "RightShoulder")
        src_lat, tgt_lat = pair if pair else (np.array([1.0, 0, 0]), np.array([1.0, 0, 0]))

        def basis(lat, up):
            x = lat - np.dot(lat, up) * up
            if np.linalg.norm(x) < 1e-6:
                x = np.cross(up, [0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0, 0])
            x /= np.linalg.norm(x)
            z = np.cross(x, up)
            return np.stack([x, up, z], axis=1)

        F = basis(tgt_lat, up_t) @ basis(src_lat, up_s).T
        if self.opt.facing_offset:
            F = _rot_about(up_t, self.opt.facing_offset) @ F
        return F

    def _is_descendant(self, a: int, b: int) -> bool:
        """True if target joint a is a strict descendant of b."""
        p = self.ch.parents[a]
        while p >= 0:
            if p == b:
                return True
            p = self.ch.parents[p]
        return False

    def _alignments(self) -> dict[int, np.ndarray]:
        sk, ch = self.sk, self.ch
        A: dict[int, np.ndarray] = {}
        for j in range(ch.count):  # parents first so leaves can inherit
            s = self.src.get(j)
            if s is None:
                continue
            found = None
            for sc in sk.primary_chain(s):
                tc = self.src_to_tgt.get(sc)
                if tc is not None and self._is_descendant(tc, j):
                    found = (sc, tc)
                    break
            if found is None:
                p = ch.parents[j]
                while p >= 0 and p not in A:
                    p = ch.parents[p]
                A[j] = A[p] if p >= 0 else np.eye(3)
                continue
            sc, tc = found
            d_src = self.F @ (sk.rest_world_p[sc] - sk.rest_world_p[s])
            d_tgt = self.P_t0[tc] - self.P_t0[j]
            A[j] = mu.shortest_arc(d_src, d_tgt)
        return A

    def _auto_scale(self) -> float:
        """Target units per source metre, from pelvis height above the feet."""
        sk, ch = self.sk, self.ch
        if self.root_joint is None:
            return 100.0
        feet = [self.src_to_tgt[sk.index[n]] for n in soma.FOOT_JOINTS if n in sk.index and sk.index[n] in self.src_to_tgt]
        hip = self.P_t0[self.root_joint]
        if feet:
            lowest = self.P_t0[feet][np.argmin(self.P_t0[feet] @ ch.up)]
            h = float(np.dot(hip - lowest, ch.up))
        else:
            h = float(np.dot(hip, ch.up))
        if h <= 1e-6:
            return 100.0
        return h / sk.rest_hip_height

    # ------------------------------------------------------------------ solve

    def source_world(self, local_q: np.ndarray):
        """local_q (N, 77, 4) XYZW -> world rotations, positions on the neutral skeleton."""
        R = mu.quat_xyzw_to_matrix(local_q)
        return self.sk.fk(R)

    def root_offset(self, P_s: np.ndarray, root: np.ndarray, root_ref: np.ndarray) -> np.ndarray:
        """Source pelvis displacement from its reference (N, 3) in metres, Y-up."""
        o = self.opt
        delta = np.asarray(root, dtype=np.float64) - np.asarray(root_ref, dtype=np.float64)
        if o.vertical_mode == "floor":
            hips = self.sk.index.get(soma.HIPS, 0)
            h = P_s[:, hips, 1] - P_s[:, self.sk.foot_indices, 1].min(axis=1)
            delta[:, 1] = h - self.sk.rest_hip_height
        if not o.translate_horizontal:
            delta[:, [0, 2]] = 0.0
        if not o.translate_vertical:
            delta[:, 1] = 0.0
        return delta

    def solve(self, local_q: np.ndarray, root: np.ndarray, root_ref: np.ndarray, joints=None) -> Solution:
        """local_q (N, 77, 4), root (N, 3) metres -> rotate/translate values for driven joints."""
        ch = self.ch
        G, P_s = self.source_world(local_q)
        n = G.shape[0]
        driven = set(ch.driven() if joints is None else joints)
        W = np.empty((n, ch.count, 3, 3))
        sol = Solution()
        for j in range(ch.count):
            p = ch.parents[j]
            Wp = W[:, p] if p >= 0 else self.top_R
            s = self.src.get(j)
            if s is not None:
                W[:, j] = self.F @ G[:, s] @ self.C[j]
            else:
                W[:, j] = Wp @ self.rest_local_R[j]
            if j in driven and s is not None:
                L = np.swapaxes(Wp, -1, -2) @ W[:, j]
                R = self.JOt[j] @ L @ self.RAt[j]
                sol.rotations[j] = mu.matrix_to_euler(R, int(ch.rotate_order[j]))

        r = self.root_joint
        if r is not None and (self.opt.translate_horizontal or self.opt.translate_vertical):
            delta = self.root_offset(P_s, root, root_ref)
            pos = self.P_t0[r] + self.scale * (delta @ self.F.T)  # (N, 3) target world
            parent = ch.parents[r]
            if parent >= 0:
                parent_world = self.rest_world_rows[parent]
            else:
                parent_world = ch.top_parent_matrix()
            inv = np.linalg.inv(parent_world)
            homo = np.concatenate([pos, np.ones((n, 1))], axis=1)
            sol.translations[r] = (homo @ inv)[:, :3]
        return sol
