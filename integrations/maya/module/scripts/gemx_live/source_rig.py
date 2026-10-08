"""Preview skeleton showing the raw SOMA-77 stream in the scene."""

from __future__ import annotations

import maya.api.OpenMaya as om
import maya.cmds as cmds
import numpy as np

from . import mathutil as mu
from . import soma

NAMESPACE = "gemxSource"
GROUP = f"{NAMESPACE}:GEMX_Source"
CM_PER_M = 100.0


def exists() -> bool:
    return cmds.objExists(GROUP)


def build(skeleton: soma.Skeleton, offset_cm: float = 0.0, radius: float = 1.5) -> str:
    """Create the preview skeleton (rest pose) under a group. Returns the group."""
    delete()
    if not cmds.namespace(exists=NAMESPACE):
        cmds.namespace(add=NAMESPACE)
    to_ui = lambda cm: om.MDistance(cm, om.MDistance.kCentimeters).asUnits(om.MDistance.uiUnit())
    grp = cmds.createNode("transform", name=GROUP)
    if cmds.upAxis(q=True, axis=True).lower() == "z":
        cmds.setAttr(f"{grp}.rotateX", 90)
    cmds.setAttr(f"{grp}.translateX", to_ui(offset_cm))
    joints = []
    for j, name in enumerate(skeleton.names):
        parent = joints[skeleton.parents[j]] if skeleton.parents[j] >= 0 else grp
        jnt = cmds.createNode("joint", name=f"{NAMESPACE}:{name}", parent=parent)
        t = skeleton.rest_local_t[j] * CM_PER_M
        cmds.setAttr(f"{jnt}.translate", *[to_ui(v) for v in t])
        jo = np.degrees(mu.matrix_to_euler(skeleton.rest_local_R[j], 0))
        cmds.setAttr(f"{jnt}.jointOrient", *jo)
        cmds.setAttr(f"{jnt}.radius", radius)
        joints.append(jnt)
    hips = joints[skeleton.index.get(soma.HIPS, 0)]
    cmds.setAttr(f"{hips}.translateY", to_ui(skeleton.rest_hip_height * CM_PER_M))
    cmds.setAttr(f"{grp}.overrideEnabled", 1)
    cmds.setAttr(f"{grp}.overrideColor", 17)  # yellow
    return grp


def delete() -> None:
    if cmds.objExists(GROUP):
        cmds.delete(GROUP)
    if cmds.namespace(exists=NAMESPACE) and not cmds.namespaceInfo(NAMESPACE, listNamespace=True):
        cmds.namespace(removeNamespace=NAMESPACE)


class Driver:
    """Writes stream poses onto the preview skeleton."""

    def __init__(self, skeleton: soma.Skeleton):
        self.sk = skeleton
        sel = om.MSelectionList()
        self.rot_plugs, self.hips_trans = [], None
        for name in skeleton.names:
            sel.clear()
            sel.add(f"{NAMESPACE}:{name}")
            fn = om.MFnDependencyNode(sel.getDependNode(0))
            self.rot_plugs.append([fn.findPlug(a, False) for a in ("rotateX", "rotateY", "rotateZ")])
            if name == soma.HIPS:
                self.hips_trans = [fn.findPlug(a, False) for a in ("translateX", "translateY", "translateZ")]
        self.JOt = np.transpose(skeleton.rest_local_R, (0, 2, 1))
        self.prev = np.zeros((skeleton.count, 3))

    def apply(self, local_q: np.ndarray, hips_position_m: np.ndarray) -> None:
        """local_q (77, 4) XYZW; hips_position_m (3,) Y-up metres above the floor."""
        R = mu.quat_xyzw_to_matrix(local_q)
        euler = mu.matrix_to_euler(self.JOt @ R, 0)
        euler = mu.closest_euler(euler, self.prev, 0)
        self.prev = euler
        mod = om.MDGModifier()
        for plugs, e in zip(self.rot_plugs, euler):
            for p, v in zip(plugs, e):
                mod.newPlugValueDouble(p, float(v))
        if self.hips_trans:
            for p, v in zip(self.hips_trans, hips_position_m * CM_PER_M):
                mod.newPlugValueDouble(p, float(v))
        mod.doIt()
