"""Drive the GEM-X Live window headless (Qt offscreen) against a real server. Run with mayapy."""
import os, sys, time, tempfile
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
from pathlib import Path
HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / "module"
sys.path.insert(0, str(MODULE / "scripts")); sys.path.insert(0, str(HERE))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="gemx_ui_"))
OUT.mkdir(parents=True, exist_ok=True)
from PySide6 import QtCore, QtWidgets
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
VIDEO = os.environ.get("GEMX_TEST_VIDEO", "")
if not os.path.isfile(VIDEO):
    print("SKIP: set GEMX_TEST_VIDEO to a video of one person (full body in frame)")
    sys.exit(0)
import maya.standalone
maya.standalone.initialize(name="python")
import maya.cmds as cmds
cmds.undoInfo(state=True, infinity=True)
cmds.loadPlugin(str(MODULE / "plug-ins" / "gemxLive.py"))
from gemx_live import prefs, ui
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
def snap(win, name):
    win.repaint(); app.processEvents()
    p = OUT / f"{name}.png"; win.grab().save(str(p)); print("  image", p)

tmp = Path(tempfile.mkdtemp(prefix="gemx_ui_takes_"))
prefs.takes_folder = lambda: tmp
prefs.set("gemxRoot", str(MODULE.parents[2] / "third_party" / "gem-x.cpp"))
prefs.set("serverPython", str(MODULE.parent / "server" / ".venv" / "Scripts" / "python.exe"))
prefs.set("serverScript", str(MODULE.parent / "server" / "gemx_live_server.py"))
prefs.set("port", 47822)
prefs.set("sourceKind", "video"); prefs.set("videoPath", VIDEO); prefs.set("videoMode", "realtime")
prefs.set("autoApply", 1); prefs.set("startAtCurrent", 1); prefs.set("driveLive", 1); prefs.set("countdown", 0); prefs.set("duration", 0.0)

cmds.file(new=True, force=True)
grp, joints = build_mixamo()

win = ui.LivePanel(ui.get_session())
win.resize(470, 860)
win.show()
s = win.s
snap(win, "0_capture_disconnected")

# Character tab: pick the rig.
win.tabs.setCurrentIndex(1)
cmds.select(joints["LeftForeArm"])
win._on_use_selected()
check(s.character is not None and s.character.names[0].endswith("Hips"), "Use Selected walks up to the skeleton root")
check(len(win._tree_items) == s.character.count == len(joints), f"tree lists {len(win._tree_items)} joints")
check(win._combos[s.character.short_names.index("LeftUpLeg")].currentText() == "LeftLeg", "tree shows SOMA LeftLeg on LeftUpLeg")
check("Mixamo" in win.char_info.text(), f"info: {win.char_info.text()}")
b = win._group_buttons["Left Fingers"]
b.click()
check(not b.isChecked() and all(j in s.character.masked for j in s.group_joints("Left Fingers")), "group button masks the left fingers")
b.click()
check(not any(j in s.character.masked for j in s.group_joints("Left Fingers")), "group button drives them again")
jn = s.character.short_names.index("Neck")
win._combos[jn].setCurrentText("Neck2")
check(s.character.mapping.get(jn) == "Neck2", "changing a Source dropdown remaps the joint")
win._combos[jn].setCurrentText("—")
check(jn not in s.character.mapping, "choosing — unmaps it")
win._combos[jn].setCurrentText("Neck1")
win.opt_vmode.setCurrentIndex(1)
check(s.character.options.get("vertical_mode") == "trajectory", "root option persists to the character")
win.opt_vmode.setCurrentIndex(0)
item = win._tree_items[s.character.short_names.index("Head")]
item.setCheckState(2, QtCore.Qt.CheckState.Unchecked)
check(s.character.short_names.index("Head") in s.character.masked, "unchecking a joint's Drive box masks it")
item.setCheckState(2, QtCore.Qt.CheckState.Checked)
snap(win, "1_character")

# Capture tab: launch server, stream video.
win.tabs.setCurrentIndex(0)
win.btn_launch.click()
check(pump(lambda: s.client.is_connected and bool(s.server_info), 90), "Launch Server connects")
check(win.btn_start.isEnabled(), "Start enabled once connected")
win.btn_start.click()
check(pump(lambda: s.latest is not None and win.preview.has_image, 60), "poses and preview images arrive")
previews = []
s.preview_jpeg.connect(lambda d: previews.append(time.time()))
frames_seen = []
s.stats.connect(lambda m: frames_seen.append(time.time()))
writes = []
orig_apply = s.character.apply
s.character.apply = lambda *a, **k: (writes.append(time.time()), orig_apply(*a, **k))
pump(lambda: False, 3.0)
s.character.apply = orig_apply
check(len(previews) / 3.0 > 20, f"camera preview streams independently of inference ({len(previews)/3.0:.1f} images/s)")
check(len(frames_seen) / 3.0 > 15, f"inference rate {len(frames_seen)/3.0:.1f} poses/s")
check(len(writes) / 3.0 > 40, f"smooth playback writes the character at display rate ({len(writes)/3.0:.0f}/s)")
check("pose" in win.stats_label.text(), f"stats: {win.stats_label.text()}")
check("fps" in win.fps_label.text(), f"header: {win.fps_label.text()}")
snap(win, "2_capture_streaming")

# Record tab: record ~2 s, auto-key.
win.tabs.setCurrentIndex(2)
check(win.btn_record.isEnabled(), "Record enabled while streaming")
win.take_name.setText("UiTake")
win.btn_record.click()
check(pump(lambda: s.recording, 5), "recording starts")
pump(lambda: s.recorder is not None and len(s.recorder) >= 30, 30)
snap(win, "3_recording")
win.btn_record.click()
check(pump(lambda: not s.recording, 5), "recording stops")
check(win.takes_tree.topLevelItemCount() == 1, "take appears in the list")
keyed = cmds.keyframe(joints["LeftArm"] + ".rotateX", q=True, keyframeCount=True) or 0
check(keyed > 10, f"take keyed onto the character on stop ({keyed} keys)")
check(not win.chk_drive.isChecked(), "Drive checkbox reflects the pause after keying")
snap(win, "4_takes")

win.btn_launch.click()  # Stop Server
check(pump(lambda: not s.server.running, 15), "Stop Server ends the process")
ui.close(shutdown=True)
print("\n%d failure(s)" % len(FAILS))
maya.standalone.uninitialize()
os._exit(1 if FAILS else 0)
