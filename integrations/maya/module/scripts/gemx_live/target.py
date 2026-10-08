"""The target character: a joint hierarchy in the Maya scene.

All matrices read here are converted to column-vector convention and Maya
internal units (centimetres, radians).
"""

from __future__ import annotations

import json

import maya.api.OpenMaya as om
import maya.cmds as cmds
import numpy as np

from . import mapping as mp
from . import mathutil as mu

CONFIG_ATTR = "gemxLiveConfig"
CONFIG_VERSION = 1
_SKIP_TYPES = {"ikHandle", "ikEffector", "parentConstraint", "orientConstraint", "pointConstraint",
               "aimConstraint", "scaleConstraint", "poleVectorConstraint"}


def _dag(path: str) -> om.MDagPath:
    sel = om.MSelectionList()
    sel.add(path)
    return sel.getDagPath(0)


def _mmatrix_to_np(m: om.MMatrix) -> np.ndarray:
    return np.array(list(m), dtype=np.float64).reshape(4, 4)


def scene_up() -> np.ndarray:
    return np.array([0.0, 0.0, 1.0]) if cmds.upAxis(q=True, axis=True).lower() == "z" else np.array([0.0, 1.0, 0.0])


def find_skeleton_root(node: str) -> str | None:
    """Walk up from any node to the top-most joint of its hierarchy."""
    if not cmds.objExists(node):
        return None
    node = cmds.ls(node, long=True)[0]
    if cmds.nodeType(node) != "joint":
        joints = cmds.listRelatives(node, allDescendents=True, type="joint", fullPath=True) or []
        if not joints:
            return None
        node = min(joints, key=lambda p: p.count("|"))
    top = node
    while True:
        parent = cmds.listRelatives(top, parent=True, fullPath=True)
        if not parent or cmds.nodeType(parent[0]) != "joint":
            return top
        top = parent[0]


def _collect(root: str) -> list[str]:
    """Root plus every descendant joint/transform that leads to joints, parents first."""
    out = []

    def has_joints(node):
        return cmds.nodeType(node) == "joint" or bool(
            cmds.listRelatives(node, allDescendents=True, type="joint", fullPath=True)
        )

    def visit(node):
        out.append(node)
        for c in cmds.listRelatives(node, children=True, type="transform", fullPath=True) or []:
            if cmds.nodeType(c) in _SKIP_TYPES:
                continue
            if cmds.nodeType(c) in ("joint", "transform") and has_joints(c):
                visit(c)

    visit(cmds.ls(root, long=True)[0])
    return out


class Character:
    def __init__(self, root: str):
        root = find_skeleton_root(root)
        if root is None:
            raise ValueError("select a joint or a group containing a skeleton")
        self.root = root
        self.paths = _collect(root)
        self.count = len(self.paths)
        index = {p: i for i, p in enumerate(self.paths)}
        self.parents = [index.get(p.rsplit("|", 1)[0], -1) for p in self.paths]
        self.names = [p.rsplit("|", 1)[-1] for p in self.paths]
        self.short_names = [mp.strip_namespace(n) for n in self.names]
        self.is_joint = [cmds.nodeType(p) == "joint" for p in self.paths]
        self.up = scene_up()
        self._read_static()
        self.mapping: dict[int, str] = {}
        self.masked: set[int] = set()
        self.options: dict = {}
        self.preset_name = ""
        self.rest_local: np.ndarray | None = None  # (J, 4, 4) Maya row-vector local matrices
        if not self.load_config():
            self.capture_rest()

    # ------------------------------------------------------------------ scene data

    def _read_static(self) -> None:
        self.dag = [_dag(p) for p in self.paths]
        self.rotate_order = np.zeros(self.count, dtype=np.int64)
        self.JO = np.tile(np.eye(3), (self.count, 1, 1))
        self.RA = np.tile(np.eye(3), (self.count, 1, 1))
        self.rot_plugs, self.trans_plugs = [], []
        for i, d in enumerate(self.dag):
            fn = om.MFnDependencyNode(d.node())
            self.rotate_order[i] = fn.findPlug("rotateOrder", False).asInt()
            if self.is_joint[i]:
                jo = [fn.findPlug(a, False).asDouble() for a in ("jointOrientX", "jointOrientY", "jointOrientZ")]
                self.JO[i] = mu.euler_to_matrix(np.array(jo), 0)
            ra = [fn.findPlug(a, False).asDouble() for a in ("rotateAxisX", "rotateAxisY", "rotateAxisZ")]
            self.RA[i] = mu.euler_to_matrix(np.array(ra), 0)
            self.rot_plugs.append([fn.findPlug(a, False) for a in ("rotateX", "rotateY", "rotateZ")])
            self.trans_plugs.append([fn.findPlug(a, False) for a in ("translateX", "translateY", "translateZ")])

    def top_parent_matrix(self) -> np.ndarray:
        """World matrix (row-vector) of the root's parent, from the live scene."""
        return _mmatrix_to_np(self.dag[0].exclusiveMatrix())

    def capture_rest(self) -> None:
        """Use the current pose as the rest (reference) pose."""
        locals_ = []
        for d in self.dag:
            m = _mmatrix_to_np(d.inclusiveMatrix()) @ _mmatrix_to_np(d.exclusiveMatrixInverse())
            locals_.append(m)
        self.rest_local = np.stack(locals_)
        self.rest_values = self.read_values()

    def rest_world(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rest world matrices (row-vector, J x 4 x 4), rotations (column, J x 3 x 3), positions (J x 3)."""
        top = self.top_parent_matrix()
        world = np.empty((self.count, 4, 4))
        for j in range(self.count):
            parent = world[self.parents[j]] if self.parents[j] >= 0 else top
            world[j] = self.rest_local[j] @ parent
        rot = mu.orthonormalize(np.transpose(world[:, :3, :3], (0, 2, 1)))
        pos = world[:, 3, :3].copy()
        return world, rot, pos

    def read_values(self) -> np.ndarray:
        """Current rotate (radians) and translate (cm) values, (J, 6)."""
        return np.array([[p.asDouble() for p in self.rot_plugs[i] + self.trans_plugs[i]] for i in range(self.count)])

    # ------------------------------------------------------------------ writing poses

    def writable(self, j: int, translate: bool = False) -> bool:
        plugs = self.trans_plugs[j] if translate else self.rot_plugs[j]
        for p in plugs:
            if p.isLocked:
                return False
            if p.isDestination:
                src = p.source()
                if src.isNull or not src.node().hasFn(om.MFn.kAnimCurve):
                    return False
        return True

    def apply(self, rotations: dict[int, np.ndarray], translations: dict[int, np.ndarray] | None = None) -> None:
        """Set rotate (radians) / translate (cm) values directly (not undoable; for live preview)."""
        mod = om.MDGModifier()
        for j, euler in rotations.items():
            for plug, v in zip(self.rot_plugs[j], euler):
                mod.newPlugValueDouble(plug, float(v))
        for j, t in (translations or {}).items():
            for plug, v in zip(self.trans_plugs[j], t):
                mod.newPlugValueDouble(plug, float(v))
        mod.doIt()

    def restore(self, values: np.ndarray) -> None:
        rot = {j: values[j, :3] for j in range(self.count) if self.writable(j)}
        tr = {j: values[j, 3:] for j in range(self.count) if self.writable(j, True)}
        self.apply(rot, tr)

    # ------------------------------------------------------------------ config

    def driven(self) -> list[int]:
        return [j for j in self.mapping if j not in self.masked]

    def config_dict(self) -> dict:
        return {
            "version": CONFIG_VERSION,
            "preset": self.preset_name,
            "mapping": {self.short_names[j]: s for j, s in self.mapping.items()},
            "masked": sorted(self.short_names[j] for j in self.masked),
            "options": self.options,
            "rest": {self.short_names[j]: self.rest_local[j].reshape(-1).round(6).tolist() for j in range(self.count)},
        }

    def save_config(self) -> None:
        if not cmds.attributeQuery(CONFIG_ATTR, node=self.root, exists=True):
            cmds.addAttr(self.root, longName=CONFIG_ATTR, dataType="string")
        cmds.setAttr(f"{self.root}.{CONFIG_ATTR}", json.dumps(self.config_dict(), separators=(",", ":")), type="string")

    def load_config(self) -> bool:
        if not cmds.attributeQuery(CONFIG_ATTR, node=self.root, exists=True):
            return False
        raw = cmds.getAttr(f"{self.root}.{CONFIG_ATTR}")
        if not raw:
            return False
        try:
            data = json.loads(raw)
        except ValueError:
            return False
        by_name = {n: i for i, n in enumerate(self.short_names)}
        rest = data.get("rest", {})
        if set(rest) != set(self.short_names):
            return False  # hierarchy changed; caller re-captures
        self.rest_local = np.stack([np.array(rest[n], dtype=np.float64).reshape(4, 4) for n in self.short_names])
        self.rest_values = self.read_values()
        self.mapping = {by_name[t]: s for t, s in data.get("mapping", {}).items() if t in by_name}
        self.masked = {by_name[t] for t in data.get("masked", []) if t in by_name}
        self.options = dict(data.get("options", {}))
        self.preset_name = data.get("preset", "")
        return True

    def has_config(self) -> bool:
        return cmds.attributeQuery(CONFIG_ATTR, node=self.root, exists=True)
