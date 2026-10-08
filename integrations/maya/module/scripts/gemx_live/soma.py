"""The SOMA-77 source skeleton streamed by the GEM-X live server.

Native SOMA space: right-handed, Y-up, metres; a performer facing the camera
faces +Z and their left side is +X.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import mathutil as mu

_DATA = Path(__file__).resolve().parent / "data" / "soma77.json"

HIPS = "Hips"
FOOT_JOINTS = ("LeftFoot", "LeftToeBase", "LeftToeEnd", "RightFoot", "RightToeBase", "RightToeEnd")

# Mask groups by source joint. Fingers are matched by prefix below.
GROUPS = {
    "Root": ("Hips",),
    "Spine": ("Spine1", "Spine2", "Chest"),
    "Head": ("Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye"),
    "Left Arm": ("LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"),
    "Right Arm": ("RightShoulder", "RightArm", "RightForeArm", "RightHand"),
    "Left Leg": ("LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd"),
    "Right Leg": ("RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd"),
}
GROUP_ORDER = ("Root", "Spine", "Head", "Left Arm", "Right Arm", "Left Fingers", "Right Fingers", "Left Leg", "Right Leg")

# Child used to define a joint's bone direction when it has several children.
PRIMARY_CHILD = {
    "Hips": "Spine1",
    "Chest": "Neck1",
    "Head": "HeadEnd",
    "LeftHand": "LeftHandMiddle1",
    "RightHand": "RightHandMiddle1",
}


def group_of(name: str) -> str | None:
    for g, members in GROUPS.items():
        if name in members:
            return g
    if name.startswith("LeftHand") and name != "LeftHand":
        return "Left Fingers"
    if name.startswith("RightHand") and name != "RightHand":
        return "Right Fingers"
    return None


class Skeleton:
    """Topology and neutral rest pose of the source skeleton."""

    def __init__(self, names, parents, rest_local_rotations, rest_local_translations):
        self.names: list[str] = list(names)
        self.parents = np.asarray(parents, dtype=np.int64)
        self.count = len(self.names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.rest_local_q = np.asarray(rest_local_rotations, dtype=np.float64).reshape(self.count, 4)
        self.rest_local_t = np.asarray(rest_local_translations, dtype=np.float64).reshape(self.count, 3)
        self.rest_local_R = mu.quat_xyzw_to_matrix(self.rest_local_q)
        self.rest_world_R, self.rest_world_p = self.fk(self.rest_local_R[None])
        self.rest_world_R, self.rest_world_p = self.rest_world_R[0], self.rest_world_p[0]
        self.children: list[list[int]] = [[] for _ in range(self.count)]
        for j, p in enumerate(self.parents):
            if p >= 0:
                self.children[p].append(j)
        feet = [self.index[n] for n in FOOT_JOINTS if n in self.index]
        self.foot_indices = np.array(feet, dtype=np.int64)
        hips = self.index.get(HIPS, 0)
        # Pelvis height above the lowest foot point in the neutral pose.
        self.rest_hip_height = float(self.rest_world_p[hips, 1] - self.rest_world_p[self.foot_indices, 1].min())

    @classmethod
    def from_dict(cls, d: dict) -> "Skeleton":
        return cls(d["names"], d["parents"], d["rest_local_rotations"], d["rest_local_translations"])

    @classmethod
    def default(cls) -> "Skeleton":
        return cls.from_dict(json.loads(_DATA.read_text()))

    def to_dict(self) -> dict:
        return {
            "names": self.names,
            "parents": self.parents.tolist(),
            "rest_local_rotations": self.rest_local_q.reshape(-1).tolist(),
            "rest_local_translations": self.rest_local_t.reshape(-1).tolist(),
        }

    def fk(self, local_R: np.ndarray, local_t: np.ndarray | None = None):
        """Batched FK. local_R (N, J, 3, 3); local_t (N, J, 3) or rest.

        Returns world rotations (N, J, 3, 3) and positions (N, J, 3) with the
        root joint at its local translation (the origin for gem-x.cpp).
        """
        n = local_R.shape[0]
        if local_t is None:
            local_t = np.broadcast_to(self.rest_local_t, (n, self.count, 3))
        G = np.empty_like(local_R)
        P = np.empty((n, self.count, 3))
        for j in range(self.count):  # parents precede children in SOMA order
            p = self.parents[j]
            if p < 0:
                G[:, j] = local_R[:, j]
                P[:, j] = local_t[:, j]
            else:
                G[:, j] = G[:, p] @ local_R[:, j]
                P[:, j] = P[:, p] + np.einsum("nab,nb->na", G[:, p], local_t[:, j])
        return G, P

    def primary_chain(self, j: int):
        """Yield descendants along the primary bone chain starting below j."""
        while True:
            kids = self.children[j]
            if not kids:
                return
            name = PRIMARY_CHILD.get(self.names[j])
            j = self.index[name] if name in self.index else kids[0]
            yield j

    def is_ancestor(self, a: int, b: int) -> bool:
        """True if a is a strict ancestor of b."""
        p = self.parents[b]
        while p >= 0:
            if p == a:
                return True
            p = self.parents[p]
        return False
