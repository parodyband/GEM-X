"""Test fixtures: a Mixamo-named skeleton built from the SOMA rest pose."""
import maya.cmds as cmds
from gemx_live import mapping as mp, soma


def build_mixamo(a_pose=True, namespace="mixamorig"):
    sk = soma.Skeleton.default()
    preset = mp.builtin_presets(sk.names)["Mixamo / HumanIK"]
    if not cmds.namespace(exists=namespace):
        cmds.namespace(add=namespace)
    grp = cmds.createNode("transform", name="Armature")
    joints = {}
    P = sk.rest_world_p * 100.0
    P[:, 1] += sk.rest_hip_height * 100.0
    for j, name in enumerate(sk.names):
        if name not in preset:
            continue
        p = sk.parents[j]
        while p >= 0 and sk.names[p] not in preset:
            p = sk.parents[p]
        cmds.select(joints[sk.names[p]] if p >= 0 else grp)
        joints[name] = cmds.joint(name=f"{namespace}:{preset[name]}", position=P[j].tolist())
    cmds.joint(joints["Hips"], edit=True, orientJoint="xyz", secondaryAxisOrient="yup", children=True, zeroScaleOrient=True)
    if a_pose:
        for side, sign in (("Left", -1), ("Right", 1)):
            cmds.rotate(0, 0, sign * 40, joints[f"{side}Arm"], relative=True, worldSpace=True)
    cmds.select(clear=True)
    return grp, joints
