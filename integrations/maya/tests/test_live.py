"""End-to-end: plug-in + session + real capture server on a video. Run with mayapy."""
import os, sys, time, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / "module"
sys.path.insert(0, str(MODULE / "scripts")); sys.path.insert(0, str(HERE))
VIDEO = os.environ.get("GEMX_TEST_VIDEO", "")
if not os.path.isfile(VIDEO):
    print("SKIP: set GEMX_TEST_VIDEO to a video of one person (full body in frame)")
    sys.exit(0)
import maya.standalone
maya.standalone.initialize(name="python")
import maya.cmds as cmds
import numpy as np
from PySide6 import QtCore
app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
cmds.undoInfo(state=True, infinity=True)
cmds.loadPlugin(str(MODULE / "plug-ins" / "gemxLive.py"))

from gemx_live import prefs
from gemx_live.session import Session
from helpers import build_mixamo

FAILS = []
def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond: FAILS.append(msg)

def pump(until, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        if until(): return True
        time.sleep(0.005)
    return False

tmp = Path(tempfile.mkdtemp(prefix="gemx_test_"))
prefs.takes_folder = lambda: tmp / "takes"
prefs.set("gemxRoot", str(MODULE.parents[2] / "third_party" / "gem-x.cpp"))
prefs.set("serverPython", str(MODULE.parent / "server" / ".venv" / "Scripts" / "python.exe"))
prefs.set("serverScript", str(MODULE.parent / "server" / "gemx_live_server.py"))
prefs.set("port", 47821)
prefs.set("sourceKind", "video"); prefs.set("videoPath", VIDEO); prefs.set("videoMode", "all")
prefs.set("autoApply", 1); prefs.set("startAtCurrent", 0); prefs.set("startFrame", 10.0)
prefs.set("driveLive", 1); prefs.set("showSource", 0)

cmds.file(new=True, force=True)
grp, joints = build_mixamo()
arm = joints["LeftArm"]
cmds.setKeyframe(arm, attribute="rotateX", time=1, value=5)
cmds.setKeyframe(arm, attribute="rotateX", time=900, value=-5)

s = Session()
logs = []
s.log.connect(lambda m: logs.append(m))
s.launch_server()
check(pump(lambda: s.client.is_connected and bool(s.server_info), 90), "server launched and connected")
s.set_character(joints["Hips"])
check(s.retargeter is not None and len(s.character.mapping) == len(joints), f"character mapped ({len(s.character.mapping)} joints)")
rest = s.character.read_values()

frames = []
s.stats.connect(lambda m: frames.append(m))
s.start_capture()
check(pump(lambda: sum(1 for f in frames if f.get("outcome") == "pose") >= 30, 60), "receiving poses")
pump(lambda: False, 0.3)
live = s.character.read_values()
moved = np.abs(live[:, :3] - rest[:, :3]).max()
check(moved > 0.05, f"live drive moves the character (max change {np.degrees(moved):.1f} deg)")

s.start_recording("TestTake", countdown=0, duration=0)
n0 = len(frames)
pump(lambda: len(frames) - n0 >= 60, 60)
path = s.stop_recording()
check(path is not None and path.exists(), f"take saved ({path.name if path else None})")
from gemx_live.takes import Take
take = Take.load(path)
check(take.frame_count >= 55, f"take has {take.frame_count} poses over {take.duration:.2f}s")

curve = cmds.listConnections(arm + ".rotateX", source=True, type="animCurve")
times = cmds.keyframe(arm + ".rotateX", q=True, timeChange=True) or []
fps = 24.0 if cmds.currentUnit(q=True, time=True) == "film" else 30.0
expected = int(np.floor(take.duration * fps + 1e-6)) + 1
inside = [t for t in times if 10 <= t <= 10 + expected - 1]
check(len(inside) == expected, f"baked {len(inside)} keys on LeftArm.rotateX (expected {expected})")
check(1.0 in times and 900.0 in times, "keys outside the baked range are preserved")
check(not s.drive_live, "live drive paused after keying so the animation shows")

cmds.undo()
times_after_undo = cmds.keyframe(arm + ".rotateX", q=True, timeChange=True) or []
check(sorted(times_after_undo) == [1.0, 900.0], f"undo restores the original keys ({times_after_undo})")
cmds.redo()
check(len(cmds.keyframe(arm + ".rotateX", q=True, timeChange=True) or []) == len(times), "redo re-applies the bake")

# Masked joints are not keyed by the command.
s.set_driven([s.character.short_names.index("RightForeArm")], False)
fore = joints["RightForeArm"]
cmds.cutKey(fore, clear=True)
n = cmds.gemxBakeTake(take=str(path), root=joints["Hips"], start=200)
n = n[0] if isinstance(n, list) else n
check(not cmds.keyframe(fore, q=True, keyframeCount=True), "masked joint gets no keys from gemxBakeTake")
check(n > 0, f"gemxBakeTake returns key count ({n})")

s.stop_capture()
pump(lambda: not s.stream.get("running"), 10)
print('before shutdown: connected', s.client.is_connected, 'server running', s.server.running, 'state', s.client._sock.state(), flush=True); t_sd = time.time(); s.shutdown(stop_server=True); print(f'shutdown took {time.time()-t_sd:.2f}s', flush=True)
check(pump(lambda: not s.server.running, 15), "server process exits on shutdown")
print("\n".join("  log: " + l for l in logs if "ggml" not in l and "load_backend" not in l))
print("\n%d failure(s)" % len(FAILS))
maya.standalone.uninitialize()
os._exit(1 if FAILS else 0)
