"""Headless tests for the GEM-X Live Maya module. Run with mayapy."""
import sys, os, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module" / "scripts"))
import numpy as np
import maya.standalone
maya.standalone.initialize(name="python")
import maya.cmds as cmds
import maya.api.OpenMaya as om
from gemx_live import mathutil as mu, soma, mapping as mp, source_rig
from gemx_live.target import Character
from gemx_live.retarget import Retargeter, Options

FAILS = []
def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond: FAILS.append(msg)

# 1. Euler conversion matches Maya for every rotate order.
rng = np.random.default_rng(0)
worst = 0
for order in range(6):
    for _ in range(200):
        e = rng.uniform(-math.pi, math.pi, 3)
        ours = mu.euler_to_matrix(e, order)
        m = om.MEulerRotation(*e, order).asMatrix()
        theirs = np.array(list(m)).reshape(4, 4)[:3, :3].T  # Maya row-vector -> column
        worst = max(worst, np.abs(ours - theirs).max())
        back = mu.euler_to_matrix(mu.matrix_to_euler(ours, order), order)
        worst = max(worst, np.abs(back - ours).max())
check(worst < 1e-9, f"euler <-> matrix matches Maya for all rotate orders (max err {worst:.1e})")

q = mu.matrix_to_quat_xyzw(mu.euler_to_matrix(np.array([0.3, -1.2, 2.0]), 0))
check(np.abs(mu.quat_xyzw_to_matrix(q) - mu.euler_to_matrix(np.array([0.3, -1.2, 2.0]), 0)).max() < 1e-9, "quat round trip")

# 2. Build a Mixamo-named target from the SOMA rest pose, with Maya's default joint orients, in an A-pose.
sk = soma.Skeleton.default()
preset = mp.builtin_presets(sk.names)["Mixamo / HumanIK"]  # soma -> mixamo
cmds.file(new=True, force=True)
cmds.namespace(add="mixamorig")
grp = cmds.createNode("transform", name="Armature")
joints = {}
P = sk.rest_world_p * 100.0
P[:, 1] += sk.rest_hip_height * 100.0
for j, name in enumerate(sk.names):
    if name not in preset: continue
    p = sk.parents[j]
    while p >= 0 and sk.names[p] not in preset: p = sk.parents[p]
    cmds.select(joints[sk.names[p]] if p >= 0 else grp)
    joints[name] = cmds.joint(name="mixamorig:" + preset[name], position=P[j].tolist())
cmds.joint(joints["Hips"], edit=True, orientJoint="xyz", secondaryAxisOrient="yup", children=True, zeroScaleOrient=True)
for side, sign in (("Left", -1), ("Right", 1)):  # rotate arms down 40 deg -> A-pose rest
    cmds.rotate(0, 0, sign * 40, joints[f"{side}Arm"], relative=True, worldSpace=True)
cmds.setAttr(joints["Spine1"] + ".rotateOrder", 3)  # mix rotate orders
cmds.setAttr(joints["LeftForeArm"] + ".rotateAxis", 10, 20, 30)

ch = Character(joints["Hips"])
m, used = mp.auto_map(ch.names, ch.parents, sk.names)
ch.mapping = m
check(used == "Mixamo / HumanIK", f"auto-map picked preset: {used}")
check(len(m) == len(joints), f"auto-map mapped {len(m)}/{len(joints)} joints")
check(ch.mapping.get(ch.short_names.index("LeftUpLeg")) == "LeftLeg", "SOMA LeftLeg (thigh) -> Mixamo LeftUpLeg")

# 3. Retarget real GEM-X poses and compare bone directions with the source.
def synthetic_poses(n=150, seed=0, max_deg=50.0):
    """Random local rotations of up to max_deg around the SOMA rest pose."""
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(n, sk.count, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    half = np.radians(rng.uniform(0, max_deg, size=(n, sk.count, 1))) / 2
    delta = np.concatenate([axes * np.sin(half), np.cos(half)], axis=-1)
    return mu.matrix_to_quat_xyzw(sk.rest_local_R[None] @ mu.quat_xyzw_to_matrix(delta))


data = {"rot": synthetic_poses(), "disp": np.zeros((150, 3))}
rt = Retargeter(sk, ch, Options())
check(np.allclose(rt.F, np.eye(3), atol=1e-6), "facing alignment is identity for a +Z facing target")
frames = [0, 40, 80, 120, 148]
rest_vals = ch.read_values()
sol = rt.solve(data["rot"][frames], data["disp"][frames] * 0, np.zeros(3))
G, Ps = rt.source_world(data["rot"][frames])
worst_angle = 0
for k, f in enumerate(frames):
    ch.apply({j: e[k] for j, e in sol.rotations.items()}, {j: t[k] for j, t in sol.translations.items()})
    Pt = np.array([cmds.xform(p, q=True, ws=True, t=True) for p in ch.paths])
    for j, s in rt.src.items():
        for sc in sk.primary_chain(s):
            tc = rt.src_to_tgt.get(sc)
            if tc is not None and rt._is_descendant(tc, j):
                if sk.parents[sc] != s:
                    break  # an unmapped source joint sits between them; its rotation cannot show in this bone
                a = Pt[tc] - Pt[j]; b = rt.F @ (Ps[k, sc] - Ps[k, s])
                cosv = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
                ang_ = math.degrees(math.acos(np.clip(cosv, -1, 1)))
                if ang_ > 0.05: print(f"   frame {f} {ch.short_names[j]} -> {ch.short_names[tc]}: {ang_:.3f} deg")
                worst_angle = max(worst_angle, ang_)
                break
check(worst_angle < 0.05, f"target bone directions follow source (worst {worst_angle:.4f} deg)")
hip_y = cmds.xform(joints["Hips"], q=True, ws=True, t=True)[1]
check(abs(rt.scale - 100.0) < 2.0, f"auto scale ~100 cm/m for a same-size target ({rt.scale:.2f})")

# 4. Facing: rotate the whole character 180 deg; the solve must follow.
ch.restore(rest_vals)
cmds.setAttr(grp + ".rotateY", 180)
ch2 = Character(joints["Hips"]); ch2.mapping = m
ch2.capture_rest()
rt2 = Retargeter(sk, ch2, Options())
check(np.allclose(rt2.F @ np.array([0, 0, 1.0]), [0, 0, -1], atol=1e-6), "facing alignment follows a -Z facing target")
sol2 = rt2.solve(data["rot"][[80]], np.zeros((1, 3)), np.zeros(3))
ch2.apply({j: e[0] for j, e in sol2.rotations.items()}, {j: t[0] for j, t in sol2.translations.items()})
Pt = np.array([cmds.xform(p, q=True, ws=True, t=True) for p in ch2.paths])
G, Ps = rt2.source_world(data["rot"][[80]])
s, sc = sk.index["LeftArm"], sk.index["LeftForeArm"]
a = Pt[rt2.src_to_tgt[sc]] - Pt[rt2.src_to_tgt[s]]; b = rt2.F @ (Ps[0, sc] - Ps[0, s])
ang = math.degrees(math.acos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
check(ang < 0.05, f"rotated character: left upper arm direction matches ({ang:.4f} deg)")

# 5. Masking keeps a joint's local values.
cmds.setAttr(grp + ".rotateY", 0)
ch3 = Character(joints["Hips"]); ch3.mapping = m; ch3.capture_rest()
la = ch3.short_names.index("LeftArm"); ch3.masked = {la}
before = ch3.read_values()[la]
rt3 = Retargeter(sk, ch3, Options())
sol3 = rt3.solve(data["rot"][[80]], np.zeros((1, 3)), np.zeros(3))
check(la not in sol3.rotations, "masked joint is not solved for output")

# 6. Config round trip through the scene attribute.
ch3.options = Options(vertical_mode="trajectory").to_dict(); ch3.save_config()
ch4 = Character(joints["Hips"])
check(ch4.mapping == ch3.mapping and ch4.masked == ch3.masked and ch4.options["vertical_mode"] == "trajectory", "config persists on the root joint")

# 7. Preview rig builds and drives.
grp_src = source_rig.build(sk, offset_cm=150)
drv = source_rig.Driver(sk)
drv.apply(data["rot"][80], np.array([0, sk.rest_hip_height, 0]))
src_pos = np.array(cmds.xform("gemxSource:LeftForeArm", q=True, ws=True, t=True)) - np.array(cmds.xform("gemxSource:LeftArm", q=True, ws=True, t=True))
_, Ps = rt.source_world(data["rot"][[80]])
b = Ps[0, sk.index["LeftForeArm"]] - Ps[0, sk.index["LeftArm"]]
ang = math.degrees(math.acos(np.clip(np.dot(src_pos, b) / np.linalg.norm(src_pos) / np.linalg.norm(b), -1, 1)))
check(ang < 0.05, f"preview rig reproduces the source pose ({ang:.4f} deg)")

# 8. Playback: irregular arrivals become an evenly advancing display.
from gemx_live.playback import Playback
pb = Playback()
rng = np.random.default_rng(1)
t_src, base = 0.0, 100.0
for _ in range(40):
    t_src += 0.04  # 25 poses/s
    pb.add(t_src, {"v": t_src}, now=base + t_src + 0.03 + rng.uniform(0, 0.02))  # 30-50 ms transit jitter
check(abs(pb.delay - 0.06) < 0.005, f"display delay adapts to 1.5 sample intervals ({pb.delay * 1000:.0f} ms)")
shown = []
for k in range(45):  # display window inside the sampled range
    a, b, alpha = pb.at(base + 0.5 + k / 60.0)
    shown.append(a["v"] + alpha * (b["v"] - a["v"]))
steps = np.diff(shown)
check(np.all(steps > 0) and steps.std() < 1e-3, f"60 Hz display advances evenly ({steps.mean() * 1000:.1f} ms of source per tick)")
a, b, alpha = pb.at(base + 50.0)
check(a["v"] == t_src and alpha == 0.0, "holds the newest pose when the stream stops")

print("\n%d failure(s)" % len(FAILS))
maya.standalone.uninitialize()
os._exit(1 if FAILS else 0)
