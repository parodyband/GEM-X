"""Smoke-test an installed release bundle (no dev venv, no path overrides). Run with mayapy.

    mayapy tests/test_bundle.py <install dir> <video>
"""
import os, sys, time, tempfile
from pathlib import Path
BUNDLE = Path(sys.argv[1]).resolve()
VIDEO = sys.argv[2]
sys.path.insert(0, str(BUNDLE / "module" / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import maya.standalone
maya.standalone.initialize(name="python")
import maya.cmds as cmds
from PySide6 import QtCore
app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
cmds.loadPlugin(str(BUNDLE / "module" / "plug-ins" / "gemxLive.py"))
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

check(Path(prefs.server_python()).name.lower() == "mayapy.exe", f"server runs on Maya's Python: {prefs.server_python()}")
check(Path(prefs.gemx_root()) == BUNDLE / "runtime", f"runtime resolved inside the bundle: {prefs.gemx_root()}")
check(not prefs.missing_models(), f"models found in {prefs.models_dir()}")
prefs.takes_folder = lambda: Path(tempfile.mkdtemp(prefix="gemx_bundle_takes_"))
prefs.set("port", 47823); prefs.set("sourceKind", "video"); prefs.set("videoPath", VIDEO); prefs.set("videoMode", "all")
prefs.set("countdown", 0); prefs.set("autoApply", 1); prefs.set("startAtCurrent", 1)

cmds.file(new=True, force=True)
grp, joints = build_mixamo()
s = Session()
logs = []; s.log.connect(logs.append)
s.launch_server()
check(pump(lambda: s.client.is_connected and bool(s.server_info), 90), "bundled server launches and connects")
s.set_character(joints["Hips"])
poses = []; s.stats.connect(lambda m: poses.append(m) if m.get("outcome") == "pose" else None)
s.start_capture()
check(pump(lambda: len(poses) >= 30, 60), f"poses stream from the bundled runtime ({poses[-1]['fps'] if poses else 0:.1f} fps)")
s.start_recording("BundleTake", 0, 0); n = len(poses)
pump(lambda: len(poses) - n >= 30, 30)
path = s.stop_recording()
keys = cmds.keyframe(joints["LeftArm"] + ".rotateX", q=True, keyframeCount=True) or 0
check(path is not None and keys > 10, f"take recorded and keyed ({keys} keys)")
s.shutdown(stop_server=True)
check(pump(lambda: not s.server.running, 15), "server exits")
if FAILS: print("\n".join(logs))
print("\n%d failure(s)" % len(FAILS))
maya.standalone.uninitialize()
os._exit(1 if FAILS else 0)
