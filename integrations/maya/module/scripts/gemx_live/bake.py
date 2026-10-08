"""Turn a take into keyframes on the target character.

``compute`` runs the retarget for every frame; ``KeyWriter`` writes the keys
with MAnimCurveChange / MDGModifier so the gemxBakeTake command can undo them.
"""

from __future__ import annotations

from dataclasses import dataclass

import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma
import numpy as np

from . import mathutil as mu
from . import soma
from .retarget import Options, Retargeter
from .takes import Take
from .target import Character


@dataclass
class Channel:
    plug: om.MPlug
    values: np.ndarray  # internal units (radians / cm)
    angular: bool


@dataclass
class BakeData:
    frames: np.ndarray  # (N,) scene frames
    channels: list[Channel]
    skipped: list[str]


def compute(take: Take, character: Character, start_frame: float, fps: float, options: Options | None = None) -> BakeData:
    skeleton = soma.Skeleton.from_dict(take.skeleton)
    rt = Retargeter(skeleton, character, options)
    times, rot, root = take.resample(fps)
    sol = rt.solve(rot, root, root[0])
    frames = start_frame + np.arange(len(times), dtype=np.float64)
    channels, skipped = [], []
    current = character.read_values()
    for j, euler in sol.rotations.items():
        if not character.writable(j):
            skipped.append(character.names[j])
            continue
        order = int(character.rotate_order[j])
        start = current[j, :3]
        euler = mu.euler_filter(euler, order, start=mu.closest_euler(euler[0], start, order))
        for axis, plug in enumerate(character.rot_plugs[j]):
            channels.append(Channel(plug, euler[:, axis], True))
    for j, tr in sol.translations.items():
        if not character.writable(j, translate=True):
            skipped.append(character.names[j] + ".translate")
            continue
        for axis, plug in enumerate(character.trans_plugs[j]):
            channels.append(Channel(plug, tr[:, axis], False))
    return BakeData(frames, channels, skipped)


class KeyWriter:
    """Undoable key writing. Call write() once, then undo()/redo()."""

    def __init__(self):
        self.change = oma.MAnimCurveChange()
        self.dgmod = om.MDGModifier()

    def write(self, data: BakeData) -> int:
        unit = om.MTime.uiUnit()
        times = om.MTimeArray([om.MTime(float(f), unit) for f in data.frames])
        first, last = data.frames[0], data.frames[-1]
        # Create curves for unkeyed channels in one modifier pass.
        curves = []
        for ch in data.channels:
            if ch.plug.isDestination:
                curves.append(ch.plug.source().node())
                continue
            node = self.dgmod.createNode("animCurveTA" if ch.angular else "animCurveTL")
            self.dgmod.connect(om.MFnDependencyNode(node).findPlug("output", False), ch.plug)
            curves.append(node)
        self.dgmod.doIt()
        count = 0
        for ch, node in zip(data.channels, curves):
            fn = oma.MFnAnimCurve(node)
            # Clear existing keys inside the baked range, keep the rest.
            for k in range(fn.numKeys - 1, -1, -1):
                f = fn.input(k).asUnits(unit)
                if first - 1e-4 <= f <= last + 1e-4:
                    fn.remove(k, self.change)
            fn.addKeys(
                times, om.MDoubleArray(ch.values.tolist()),
                oma.MFnAnimCurve.kTangentAuto, oma.MFnAnimCurve.kTangentAuto,
                True, self.change,
            )
            count += len(ch.values)
        return count

    def undo(self) -> None:
        self.change.undoIt()
        self.dgmod.undoIt()

    def redo(self) -> None:
        self.dgmod.doIt()
        self.change.redoIt()
