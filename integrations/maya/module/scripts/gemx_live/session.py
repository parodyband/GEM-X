"""Live session: connection, character, live drive and recording (no widgets)."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import maya.api.OpenMaya as om
import maya.cmds as cmds
import numpy as np
from PySide6 import QtCore

from . import mapping as mp
from . import mathutil as mu
from . import prefs, soma, source_rig
from .client import ServerClient, ServerProcess
from .playback import Playback
from .retarget import Options, Retargeter
from .takes import Recorder, Take, EXTENSION, list_takes, next_take_name, safe_name
from .target import Character


class Session(QtCore.QObject):
    log = QtCore.Signal(str)
    connection_changed = QtCore.Signal(bool)
    server_process_changed = QtCore.Signal(bool)
    stream_state = QtCore.Signal(dict)
    stats = QtCore.Signal(dict)
    preview_jpeg = QtCore.Signal(bytes)
    character_changed = QtCore.Signal()
    drive_changed = QtCore.Signal(bool)
    recording_changed = QtCore.Signal(bool)
    countdown = QtCore.Signal(int)
    record_progress = QtCore.Signal(float, int)
    takes_changed = QtCore.Signal()
    models_progress = QtCore.Signal(str, float)  # file, 0..1
    models_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client = ServerClient(self)
        self.server = ServerProcess(self)
        self.skeleton = soma.Skeleton.default()
        self.skeleton_dict = self.skeleton.to_dict()
        self.server_info: dict = {}
        self.stream: dict = {}
        self.character: Character | None = None
        self.retargeter: Retargeter | None = None
        self.drive_live = bool(prefs.get("driveLive"))
        self.show_source = bool(prefs.get("showSource"))
        self.latest: tuple[float, np.ndarray, np.ndarray] | None = None  # t, rot (77, 4), root (3,)
        self.root_ref: np.ndarray | None = None
        self._prev_euler: dict[int, np.ndarray] = {}
        self._saved_pose: np.ndarray | None = None
        self._source_driver: source_rig.Driver | None = None
        self.smooth_playback = bool(prefs.get("smoothPlayback"))
        self._playback = Playback()
        self._live_joints: list[int] = []  # writable driven joints, in payload order
        self._live_trans: list[int] = []
        self._last_euler: np.ndarray | None = None
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._render_timer.setInterval(16)  # ~60 Hz display
        self._render_timer.timeout.connect(self._render)
        self.recorder: Recorder | None = None
        self.recording = False
        self._record_name = ""
        self._record_duration = 0.0
        self._record_wall0 = 0.0
        self._countdown_left = 0
        self._countdown_timer = QtCore.QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._tick_countdown)

        c = self.client
        c.connected.connect(lambda: (self.connection_changed.emit(True), self.log.emit("Connected to capture server")))
        c.disconnected.connect(self._on_disconnected)
        c.hello.connect(self._on_hello)
        c.state.connect(self._on_state)
        c.frames.connect(self._on_frames)
        c.preview.connect(self._on_preview)
        c.error.connect(lambda m: self.log.emit(f"Server: {m}"))
        s = self.server
        s.ready.connect(lambda host, port: self.client.connect_to(host, port))
        s.output.connect(lambda line: self.log.emit(line))
        s.finished.connect(self._on_server_finished)

    # ------------------------------------------------------------------ connection

    def launch_server(self) -> None:
        if self.server.running:
            return
        missing = prefs.missing_models()
        if missing:
            raise RuntimeError(f"models not downloaded yet ({', '.join(missing)}); press Download Models first")
        self.server.start(
            prefs.server_python(), prefs.server_script(), prefs.gemx_root(),
            prefs.get("host"), int(prefs.get("port")),
            ["--detect-interval", str(prefs.get("detectInterval")), "--tracking", prefs.get("tracking")]
            + (["--flip-test"] if prefs.get("flipTest") else []),
        )
        self.server_process_changed.emit(True)
        self.log.emit("Starting capture server (loading models)...")

    def stop_server(self) -> None:
        self.client.send({"cmd": "shutdown"})
        self.client.flush()
        self.server.stop()

    # ------------------------------------------------------------------ models

    def download_models(self) -> None:
        """Fetch the ~4 GB of GGUF models in a background process."""
        if getattr(self, "_fetch", None) is not None and self._fetch.state() != QtCore.QProcess.ProcessState.NotRunning:
            return
        proc = QtCore.QProcess(self)
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        for key in ("PYTHONPATH", "PYTHONHOME"):
            env.remove(key)
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        script = str(prefs.SERVER_DIR / "fetch_models.py")
        buffer = bytearray()

        def on_output():
            buffer.extend(bytes(proc.readAllStandardOutput()))
            while b"\n" in buffer:
                line, _, rest = bytes(buffer).partition(b"\n")
                buffer[:] = rest
                text = line.decode(errors="replace").strip()
                if text.startswith("PROGRESS "):
                    _, name, done, total = text.split()
                    self.models_progress.emit(name, int(done) / max(int(total), 1))
                elif text and text not in ("DONE", "FAILED"):
                    self.log.emit(text)

        def on_finished(code, _status):
            ok = code == 0 and not prefs.missing_models()
            self.log.emit("Models ready" if ok else "Model download failed; press Download Models to retry")
            self.models_finished.emit(ok)

        proc.readyReadStandardOutput.connect(on_output)
        proc.finished.connect(on_finished)
        proc.start(prefs.server_python(), [script, "--dest", str(prefs.models_dir()), "--machine"])
        self._fetch = proc
        self.log.emit(f"Downloading models to {prefs.models_dir()} (about {prefs.MODELS_SIZE_GB:.0f} GB)...")

    def _on_server_finished(self, code: int) -> None:
        self.server_process_changed.emit(False)
        self.log.emit(f"Capture server exited ({code})")

    def connect_server(self) -> None:
        self.client.connect_to(prefs.get("host"), int(prefs.get("port")))

    def disconnect_server(self) -> None:
        self.client.close()

    def _on_disconnected(self) -> None:
        self.connection_changed.emit(False)
        self.stream = {}
        if self.recording:
            self.stop_recording()
        self.log.emit("Disconnected from capture server")

    def _on_hello(self, msg: dict) -> None:
        self.server_info = msg.get("server", {})
        sk = msg.get("skeleton")
        if sk and sk.get("names") != self.skeleton.names:
            self.skeleton = soma.Skeleton.from_dict(sk)
            self.skeleton_dict = self.skeleton.to_dict()
            self._rebuild_retargeter()
        self._on_state(msg.get("state", {}))
        self.log.emit(f"Server ready on {self.server_info.get('device', 'unknown device')}")

    def _on_state(self, msg: dict) -> None:
        self.stream = msg
        self.stream_state.emit(msg)
        if msg.get("error"):
            self.log.emit(f"Server error: {msg['error']}")
        if msg.get("message") == "finished" and self.recording:
            self.stop_recording()

    # ------------------------------------------------------------------ capture control

    def start_capture(self) -> None:
        kind = prefs.get("sourceKind")
        self.configure_server()
        if kind == "video":
            path = prefs.get("videoPath")
            if not path:
                raise ValueError("choose a video file first")
            cmd = {"cmd": "start", "source": "video", "path": path, "mode": prefs.get("videoMode")}
        else:
            w, h = (int(v) for v in prefs.get("resolution").split("x"))
            cmd = {"cmd": "start", "source": "camera", "camera": int(prefs.get("camera")), "width": w, "height": h}
        self.root_ref = None
        self.client.send(cmd)

    def stop_capture(self) -> None:
        self.client.send({"cmd": "stop"})

    def reset_tracking(self) -> None:
        self.root_ref = None
        self.client.send({"cmd": "reset"})

    def recenter(self) -> None:
        if self.latest is not None:
            self.root_ref = self.latest[2].copy()

    def configure_server(self) -> None:
        self.client.send({
            "cmd": "configure",
            "detect_interval": int(prefs.get("detectInterval")),
            "tracking": prefs.get("tracking"),
            "mirror": bool(prefs.get("mirror")),
        })

    def set_preview_wanted(self, wanted: bool) -> None:
        """Only stream camera images while someone is looking at them."""
        self.client.send({"cmd": "configure", "preview": bool(wanted)})

    # ------------------------------------------------------------------ frames

    def _on_frames(self, frames: list) -> None:
        last = frames[-1]
        self.stats.emit(last)
        for msg in frames:
            if msg.get("outcome") != "pose" or "rot" not in msg:
                continue
            rot = np.asarray(msg["rot"], dtype=np.float64).reshape(-1, 4)
            root = np.asarray(msg["root"], dtype=np.float64)
            self.latest = (float(msg["t"]), rot, root)
            if self.root_ref is None:
                self.root_ref = root.copy()
            if self.recording and self.recorder is not None:
                self.recorder.add(msg["t"], rot, root)
            self._push_live(float(msg["t"]), rot, root)
        if self.recording and self.recorder is not None:
            elapsed = time.perf_counter() - self._record_wall0
            self.record_progress.emit(elapsed, len(self.recorder))
            if self._record_duration > 0 and elapsed >= self._record_duration:
                self.stop_recording()
        if self._playback.samples:
            if self.smooth_playback:
                if not self._render_timer.isActive():
                    self._render_timer.start()
            else:
                self._render(newest=True)

    def _on_preview(self, msg: dict) -> None:
        data = msg.get("jpeg")
        if data:
            self.preview_jpeg.emit(base64.b64decode(data))

    def _push_live(self, t: float, rot: np.ndarray, root: np.ndarray) -> None:
        """Retarget one pose (once) and queue it for display."""
        driving = self.drive_live and self.retargeter is not None
        showing = self.show_source and self._source_driver is not None
        if not (driving or showing):
            return
        payload = {}
        try:
            if driving:
                sol = self.retargeter.solve(rot[None], root[None], self.root_ref)
                ch = self.character
                euler = np.array([sol.rotations[j][0] for j in self._live_joints]).reshape(-1, 3)
                if self._last_euler is not None and self._last_euler.shape == euler.shape:
                    for k, j in enumerate(self._live_joints):  # keep successive samples interpolable
                        euler[k] = mu.closest_euler(euler[k], self._last_euler[k], int(ch.rotate_order[j]))
                self._last_euler = euler
                payload["euler"] = euler
                payload["trans"] = np.array([sol.translations[j][0] for j in self._live_trans]).reshape(-1, 3)
            if showing:
                payload["rot"] = rot
                payload["hips"] = self._source_hips(rot, root)
        except Exception as e:  # keep streaming; stop driving the character
            self.log.emit(f"Live update failed: {e}")
            self.set_drive_live(False)
            return
        self._playback.add(t, payload)

    def _render(self, newest: bool = False) -> None:
        """Show the interpolated pose for now (timer) or the newest sample."""
        pb = self._playback
        if newest:
            if not pb.samples:
                return
            a = b = pb.samples[-1][1]
            alpha = 0.0
        else:
            sample = pb.at()
            if sample is None:
                self._render_timer.stop()
                return
            a, b, alpha = sample
            if pb.stale(seconds=0.5):
                self._render_timer.stop()  # stream paused; hold the last pose
        try:
            if "euler" in a and "euler" in b and self.drive_live and self.character is not None:
                if self._saved_pose is None:
                    self._saved_pose = self.character.read_values()
                euler = a["euler"] + alpha * (b["euler"] - a["euler"])
                trans = a["trans"] + alpha * (b["trans"] - a["trans"])
                self.character.apply(dict(zip(self._live_joints, euler)), dict(zip(self._live_trans, trans)))
            if "rot" in a and "rot" in b and self.show_source and self._source_driver is not None:
                rot = mu.slerp_xyzw(a["rot"], b["rot"], alpha)
                self._source_driver.apply(rot, a["hips"] + alpha * (b["hips"] - a["hips"]))
        except Exception as e:
            self.log.emit(f"Live update failed: {e}")
            self.set_drive_live(False)

    def set_smooth_playback(self, on: bool) -> None:
        self.smooth_playback = on
        prefs.set("smoothPlayback", on)
        if not on:
            self._render_timer.stop()

    def _source_hips(self, rot: np.ndarray, root: np.ndarray) -> np.ndarray:
        _, P = self.skeleton.fk(mu.quat_xyzw_to_matrix(rot)[None])
        hips = self.skeleton.index.get(soma.HIPS, 0)
        h = P[0, hips, 1] - P[0, self.skeleton.foot_indices, 1].min()
        ref = self.root_ref if self.root_ref is not None else root
        return np.array([root[0] - ref[0], h, root[2] - ref[2]])

    # ------------------------------------------------------------------ character

    def set_character(self, node: str) -> None:
        self._release_character()
        ch = Character(node)
        if not ch.mapping:
            ch.mapping, ch.preset_name = mp.auto_map(ch.names, ch.parents, self.skeleton.names)
            ch.options = Options().to_dict()
            ch.save_config()
        self.character = ch
        self._rebuild_retargeter()
        self.log.emit(f"Character {ch.names[0]}: {len(ch.mapping)}/{ch.count} joints mapped ({ch.preset_name or 'saved mapping'})")

    def clear_character(self) -> None:
        self._release_character()
        self.character = None
        self.retargeter = None
        self.character_changed.emit()

    def _rebuild_retargeter(self) -> None:
        self._prev_euler.clear()
        self._reset_live()
        if self.character is None:
            self.retargeter = None
        else:
            try:
                self.retargeter = Retargeter(self.skeleton, self.character, Options.from_dict(self.character.options))
            except Exception as e:
                self.retargeter = None
                self.log.emit(f"Retarget setup failed: {e}")
        if self.retargeter is not None:
            ch = self.character
            self._live_joints = [j for j in ch.driven() if j in self.retargeter.src and ch.writable(j)]
            r = self.retargeter.root_joint
            o = self.retargeter.opt
            moves = o.translate_horizontal or o.translate_vertical
            self._live_trans = [r] if (r is not None and moves and r in ch.driven() and ch.writable(r, True)) else []
        self.character_changed.emit()

    def _reset_live(self) -> None:
        self._playback.clear()
        self._last_euler = None
        self._live_joints, self._live_trans = [], []

    def _character_edited(self) -> None:
        self.character.save_config()
        self._rebuild_retargeter()

    def set_mapping(self, joint: int, source: str | None) -> None:
        if source:
            self.character.mapping[joint] = source
        else:
            self.character.mapping.pop(joint, None)
        self._character_edited()

    def set_driven(self, joints, driven: bool) -> None:
        for j in joints:
            if driven:
                self.character.masked.discard(j)
            else:
                self.character.masked.add(j)
        self._character_edited()

    def group_joints(self, group: str) -> list[int]:
        return [j for j, s in self.character.mapping.items() if soma.group_of(s) == group]

    def set_options(self, **kw) -> None:
        opts = Options.from_dict(self.character.options)
        for k, v in kw.items():
            setattr(opts, k, v)
        self.character.options = opts.to_dict()
        self._character_edited()

    def auto_map(self) -> None:
        ch = self.character
        ch.mapping, ch.preset_name = mp.auto_map(ch.names, ch.parents, self.skeleton.names)
        self._character_edited()
        self.log.emit(f"Auto-mapped {len(ch.mapping)} joints ({ch.preset_name})")

    def apply_builtin_preset(self, name: str) -> None:
        ch = self.character
        preset = mp.builtin_presets(self.skeleton.names)[name]
        ch.mapping = mp.apply_preset(preset, ch.names)
        ch.preset_name = name
        self._character_edited()

    def clear_mapping(self) -> None:
        self.character.mapping = {}
        self.character.masked = set()
        self._character_edited()

    def capture_rest(self) -> None:
        self._release_character()
        self.character.capture_rest()
        self._character_edited()
        self.log.emit("Rest pose captured from the current pose")

    def save_mapping(self, path: str) -> None:
        ch = self.character
        mp.save_preset(path, {ch.short_names[j]: s for j, s in ch.mapping.items()}, [ch.short_names[j] for j in ch.masked])

    def load_mapping(self, path: str) -> None:
        ch = self.character
        mapping, masked = mp.load_preset(path)
        ch.mapping = mp.remap_by_short_name(mapping, ch.names)
        by_name = {n: i for i, n in enumerate(ch.short_names)}
        ch.masked = {by_name[n] for n in masked if n in by_name}
        ch.preset_name = Path(path).stem
        self._character_edited()

    # ------------------------------------------------------------------ live drive

    def set_drive_live(self, on: bool) -> None:
        if on != self.drive_live:
            self.drive_live = on
            prefs.set("driveLive", on)
            self.drive_changed.emit(on)
        if not on:
            self._release_character()

    def _release_character(self) -> None:
        """Stop driving and put the character back the way it was before live drive."""
        self._prev_euler.clear()
        self._playback.clear()
        self._last_euler = None
        if self.character is not None and self._saved_pose is not None:
            self.character.restore(self._saved_pose)
            cmds.currentTime(cmds.currentTime(q=True), update=True)  # re-evaluate keyed channels
        self._saved_pose = None

    def set_show_source(self, on: bool) -> None:
        self.show_source = on
        prefs.set("showSource", on)
        self._playback.clear()
        if on:
            if not source_rig.exists():
                source_rig.build(self.skeleton, offset_cm=-150.0)
            self._source_driver = source_rig.Driver(self.skeleton)
        else:
            self._source_driver = None
            source_rig.delete()

    # ------------------------------------------------------------------ recording

    def default_take_name(self) -> str:
        return next_take_name(prefs.takes_folder())

    def start_recording(self, name: str, countdown: int, duration: float) -> None:
        if self.recording or self._countdown_timer.isActive():
            return
        if not self.client.is_connected or not self.stream.get("running"):
            raise RuntimeError("start a capture source before recording")
        self._record_name = safe_name(name) or self.default_take_name()
        self._record_duration = max(0.0, float(duration))
        self._countdown_left = int(countdown)
        if self._countdown_left > 0:
            self.countdown.emit(self._countdown_left)
            self._countdown_timer.start()
        else:
            self._begin_recording()

    def _tick_countdown(self) -> None:
        self._countdown_left -= 1
        if self._countdown_left > 0:
            self.countdown.emit(self._countdown_left)
        else:
            self._countdown_timer.stop()
            self.countdown.emit(0)
            self._begin_recording()

    def _begin_recording(self) -> None:
        self.recorder = Recorder()
        self.recording = True
        self._record_wall0 = time.perf_counter()
        self.recording_changed.emit(True)
        self.log.emit(f"Recording {self._record_name}")

    def cancel_countdown(self) -> None:
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
            self.countdown.emit(0)

    def stop_recording(self) -> Path | None:
        self.cancel_countdown()
        if not self.recording:
            return None
        self.recording = False
        self.recording_changed.emit(False)
        rec, self.recorder = self.recorder, None
        if rec is None or len(rec) < 2:
            self.log.emit("Recording stopped: no poses captured")
            return None
        meta = {"source": self.stream.get("source"), "device": self.server_info.get("device")}
        take = rec.to_take(self._record_name, self.skeleton_dict, meta)
        folder = prefs.takes_folder()
        path = folder / (self._record_name + EXTENSION)
        if path.exists():
            path = folder / (next_take_name(folder, self._record_name) + EXTENSION)
        take.save(path)
        self.takes_changed.emit()
        self.log.emit(f"Saved {take.name}: {take.duration:.2f}s, {take.frame_count} poses -> {path}")
        if prefs.get("autoApply") and self.character is not None:
            self.apply_take(path)
        return path

    def takes(self) -> list[Path]:
        return list_takes(prefs.takes_folder())

    def start_frame(self) -> float:
        return float(cmds.currentTime(q=True)) if prefs.get("startAtCurrent") else float(prefs.get("startFrame"))

    def apply_take(self, path: Path, start: float | None = None) -> int:
        if self.character is None:
            raise RuntimeError("set a target character first")
        self.set_drive_live(False)  # pause live drive so the baked animation shows
        start = self.start_frame() if start is None else start
        take = Take.load(path)
        fps = om.MTime(1.0, om.MTime.kSeconds).asUnits(om.MTime.uiUnit())
        end = start + int(np.floor(take.duration * fps + 1e-6))
        cmds.undoInfo(openChunk=True, chunkName=f"GEM-X Key {take.name}")
        try:
            keys = cmds.gemxBakeTake(take=str(path), root=self.character.root, start=start)
            if cmds.playbackOptions(q=True, maxTime=True) < end:
                cmds.playbackOptions(maxTime=end)
            cmds.currentTime(start, update=True)
        finally:
            cmds.undoInfo(closeChunk=True)
        keys = keys[0] if isinstance(keys, (list, tuple)) else keys
        self.log.emit(f"Applied {take.name} to {self.character.names[0]} at frame {start:g} ({keys} keys)")
        return keys

    # ------------------------------------------------------------------ teardown

    def shutdown(self, stop_server: bool = True) -> None:
        self._render_timer.stop()
        fetch = getattr(self, "_fetch", None)
        if fetch is not None and fetch.state() != QtCore.QProcess.ProcessState.NotRunning:
            fetch.kill()
        self.cancel_countdown()
        if self.recording:
            self.stop_recording()
        self._release_character()
        if self.show_source:
            self._source_driver = None
        if self.client.is_connected:
            if stop_server and self.server.running:
                self.client.send({"cmd": "shutdown"})
                self.client.flush()
            self.client.close()
        if stop_server:
            self.server.stop()
