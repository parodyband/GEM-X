"""GEM-X live capture server.

Reads a webcam or video file, runs gem-x.cpp's live pipeline (YOLOX -> ViTPose
-> GEM denoiser) and streams SOMA-77 poses to connected clients (the Maya
plug-in) as newline-delimited JSON over TCP.

    python gemx_live_server.py --gemx C:/dev/gem-x.cpp

Protocol (one JSON object per line, UTF-8):

  client -> server
    {"cmd": "hello"}
    {"cmd": "start", "source": "camera", "camera": 0, "width": 1280, "height": 720}
    {"cmd": "start", "source": "video", "path": "clip.mp4", "mode": "realtime" | "all"}
    {"cmd": "stop"}
    {"cmd": "reset"}
    {"cmd": "configure", "detect_interval": 5, "mirror": false, "preview": true, ...}
    {"cmd": "shutdown"}

  server -> client
    {"type": "hello", "protocol": ..., "skeleton": {...}, "server": {...}, "state": {...}}
    {"type": "state", "running": bool, "source": ..., "message": ..., "error": ...}
    {"type": "frame", "seq": n, "t": seconds, "outcome": "pose" | "warmup" | "lost" | "ambiguous",
     "rot": [77*4 parent-local xyzw], "trans": [77*3 parent-local metres], "root": [x, y, z], ...}
    {"type": "preview", "jpeg": base64, "width": w, "height": h}

Coordinates are SOMA native: right-handed, Y-up, metres; a performer facing
the camera faces +Z. "root" integrates gem-x.cpp's per-frame root displacement
into a continuous Y-up trajectory that only "reset" returns to the origin.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import socket
import socketserver
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if (_HERE / "site-packages").is_dir():  # release installs put OpenCV here for mayapy
    sys.path.insert(0, str(_HERE / "site-packages"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import gemx_native as gx  # noqa: E402

PROTOCOL = "gemx-live/1"
DEFAULT_PORT = 47811
SERVER_VERSION = "1.0.0"

log = logging.getLogger("gemx-live")


@dataclass
class Config:
    gemx_root: str
    backend: str = "Vulkan"
    device: int = 0
    threads: int = 8
    window: int = 30
    detect_interval: int = 5
    selection: str = "largest"  # largest | continuity
    precision: str = "fast"  # fast | strict
    tracking: str = "keypoints"  # keypoints: detector only (re)acquires | detector: every N frames
    mirror: bool = False
    preview: bool = True
    preview_width: int = 640
    preview_fps: float = 30.0

    # Fields that need the native pipeline rebuilt when they change.
    PIPELINE_KEYS = ("backend", "device", "threads", "window", "detect_interval", "selection", "precision")

    MODEL_FILES = ("gem-x-contact-f32.gguf", "vitpose-f32.gguf", "yolox-f32.gguf")

    def models_dir(self) -> Path:
        """Release bundles keep models in runtime/models; a gem-x.cpp checkout in generated/reference."""
        root = Path(self.gemx_root)
        for d in (root / "models", root / "generated" / "reference"):
            if (d / self.MODEL_FILES[0]).is_file():
                return d
        return root / "models" if (root / "gemx.dll").is_file() else root / "generated" / "reference"

    def models(self) -> tuple[str, str, str]:
        d = self.models_dir()
        return tuple(str(d / name) for name in self.MODEL_FILES)

    def build_dir(self) -> Path:
        """Release bundles: gemx.dll at the root. Checkouts: build/win-<preset> or build/<preset>."""
        sub = "vulkan" if self.backend == "Vulkan" else "release"
        root = Path(self.gemx_root)
        lib = "gemx.dll" if os.name == "nt" else "libgemx.so"
        for candidate in (root, root / "build" / f"win-{sub}", root / "build" / sub):
            if (candidate / lib).is_file():
                return candidate
        raise FileNotFoundError(f"no gem-x.cpp build found under {root}")


# --------------------------------------------------------------------------- sources


class CameraSource:
    """Grabs webcam frames on a thread and keeps only the newest one."""

    is_live = True

    def __init__(self, index: int, width: int, height: int, fps: int = 30):
        self.description = f"camera:{index}"
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if os.name == "nt" else [cv2.CAP_ANY]
        self._cap = None
        for api in backends:
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                self._cap = cap
                break
            cap.release()
        if self._cap is None:
            raise RuntimeError(f"cannot open camera {index}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame, not a queue of old ones
        self._cond = threading.Condition()
        self._latest: tuple[int, int, np.ndarray] | None = None  # (counter, t_ns, bgr)
        self._counter = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            t_ns = time.perf_counter_ns()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._counter += 1
                self._latest = (self._counter, t_ns, bgr)
                self._cond.notify_all()

    def next(self, last_counter: int, timeout: float = 1.0):
        """Return (counter, capture_ns, bgr) newer than last_counter, or None."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._stop.is_set() or (self._latest is not None and self._latest[0] > last_counter),
                timeout,
            )
            if not ok or self._stop.is_set():
                return None
            return self._latest

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2)
        self._cap.release()


class VideoSource:
    """Reads a video file.

    realtime: paced to the file's frame rate, skipping frames inference cannot
              keep up with (behaves like a camera).
    all:      every frame in order, timestamped from the file's frame rate.
    """

    is_live = False

    def __init__(self, path: str, mode: str = "realtime"):
        self.description = f"video:{Path(path).name}"
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.mode = mode
        self._index = -1
        self._start_ns = None
        self.finished = False

    def next(self, last_counter: int, timeout: float = 1.0):
        if self.mode == "realtime":
            now = time.perf_counter_ns()
            if self._start_ns is None:
                self._start_ns = now
            target = int((now - self._start_ns) * 1e-9 * self.fps)
            target = max(target, self._index + 1)
            while self._index < target - 1:  # drop late frames
                if not self._cap.grab():
                    self.finished = True
                    return None
                self._index += 1
        ok, bgr = self._cap.read()
        if not ok:
            self.finished = True
            return None
        self._index += 1
        t_ns = int(self._index * 1e9 / self.fps)
        if self.mode == "realtime":
            wait = self._start_ns + t_ns - time.perf_counter_ns()
            if wait > 0:
                time.sleep(wait * 1e-9)
            capture_ns = time.perf_counter_ns()
        else:
            capture_ns = time.perf_counter_ns()
        return self._index + 1, t_ns, bgr, capture_ns

    def close(self) -> None:
        self._cap.release()


# --------------------------------------------------------------------------- math


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def zup_to_yup(v: np.ndarray) -> np.ndarray:
    # gem-x.cpp uses YtoZ(x, y, z) = (x, -z, y); this is its inverse.
    return np.array([v[0], v[2], -v[1]])


class KeypointTracker:
    """Crop box for the next frame from this frame's 2D keypoints.

    Top-down pose tracking: once someone is found, their own keypoints place
    the next crop, so the person detector (~40 ms) only runs to (re)acquire.
    """

    MIN_CONF = 0.35
    MIN_POINTS = 12
    SMOOTHING = 0.6  # weight of the new box; damps crop jitter

    def __init__(self):
        self.box: np.ndarray | None = None

    def reset(self) -> None:
        self.box = None

    def update(self, keypoints: np.ndarray | None, width: int, height: int) -> None:
        if keypoints is None:
            self.box = None
            return
        good = keypoints[:, 2] >= self.MIN_CONF
        if int(good.sum()) < self.MIN_POINTS:
            self.box = None
            return
        pts = keypoints[good, :2]
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        bw, bh = x1 - x0, y1 - y0
        if bw < 4 or bh < 16:
            self.box = None
            return
        # Joint centres sit inside the silhouette; pad to a detector-like box.
        px, py = 0.12 * bw + 0.04 * bh, 0.08 * bh
        box = np.array([x0 - px, y0 - py, x1 + px, y1 + py], dtype=np.float64)
        if self.box is not None:
            box = self.SMOOTHING * box + (1.0 - self.SMOOTHING) * self.box
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        self.box = box if (box[2] - box[0] >= 4 and box[3] - box[1] >= 16) else None


# --------------------------------------------------------------------------- engine


@dataclass
class Stats:
    fps: float = 0.0
    infer_ms: float = 0.0
    latency_ms: float = 0.0
    frames: int = 0
    poses: int = 0


class Engine:
    def __init__(self, config: Config):
        self.config = config
        self._lib = gx.load_library(config.build_dir())
        self._pipe: gx.LivePipeline | None = None
        self._pipe_key: tuple | None = None
        self._lock = threading.RLock()  # guards pipeline/source/config
        self._clients: set["ClientHandler"] = set()
        self._clients_lock = threading.Lock()
        self._source = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._reset_requested = threading.Event()
        self._rebuild_requested = threading.Event()
        self.root = np.zeros(3)
        self.stats = Stats()
        self.error: str | None = None
        self._skeleton: dict | None = None
        self._overlay: tuple | None = None  # (keypoints, box, outcome) from the newest inference
        self._published = threading.Condition()  # video frames handed to the preview thread
        self._published_frame: tuple[int, np.ndarray] | None = None
        self._preview_thread: threading.Thread | None = None
        if config.precision == "strict":
            for k in gx.STRICT_ENV:
                os.environ[k] = "1"
        self._ensure_pipeline()

    # ---- pipeline

    def _ensure_pipeline(self) -> gx.LivePipeline:
        c = self.config
        key = tuple(getattr(c, k) for k in Config.PIPELINE_KEYS)
        if self._pipe is not None and key == self._pipe_key:
            return self._pipe
        if self._pipe is not None:
            self._pipe.close()
            self._pipe = None
        gem, pose, det = c.models()
        missing = [Path(p).name for p in (gem, pose, det) if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                f"models missing in {c.models_dir()}: {', '.join(missing)}. "
                "Download them from the GEM-X Live window or run fetch_models.py."
            )
        t0 = time.perf_counter()
        self._pipe = gx.LivePipeline(
            self._lib, gem, pose, det,
            gx.default_module(c.build_dir(), c.backend), backend=c.backend, device=c.device,
            threads=c.threads, window=c.window, detect_interval=c.detect_interval,
            selection=gx.CONTINUITY if c.selection == "continuity" else gx.PARITY,
            precision=gx.STRICT_F32 if c.precision == "strict" else gx.BACKEND_DEFAULT,
        )
        self._pipe_key = key
        definition = self._pipe.definition
        soma = definition["soma77"]
        self._skeleton = {
            "schema": soma["schema"],
            "names": soma["joint_names"],
            "parents": soma["parents"],
            "rest_local_rotations": soma["rest_local_rotations"],
            "rest_local_translations": soma["rest_local_translations"],
            "units": "metres",
            "up": "y",
            "facing": "+z toward camera",
            "quaternion_order": "xyzw",
        }
        self.device_name = definition["config"].get("device", "")
        log.info("pipeline ready in %.2fs on %s", time.perf_counter() - t0, self.device_name)
        return self._pipe

    # ---- clients

    def add_client(self, client: "ClientHandler") -> None:
        with self._clients_lock:
            self._clients.add(client)

    def remove_client(self, client: "ClientHandler") -> None:
        with self._clients_lock:
            self._clients.discard(client)

    def broadcast(self, msg: dict) -> None:
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        with self._clients_lock:
            clients = list(self._clients)
        preview = msg.get("type") == "preview"
        for c in clients:
            if preview and not c.wants_preview:
                continue
            c.send_raw(line, preview)

    def hello(self) -> dict:
        return {
            "type": "hello",
            "protocol": PROTOCOL,
            "skeleton": self._skeleton,
            "server": {
                "version": SERVER_VERSION,
                "device": self.device_name,
                "config": {k: v for k, v in asdict(self.config).items() if k != "gemx_root"},
            },
            "state": self.state(),
        }

    def state(self, message: str | None = None) -> dict:
        running = self._thread is not None and self._thread.is_alive()
        s = {
            "type": "state",
            "running": running,
            "source": self._source.description if (running and self._source) else None,
            "stats": asdict(self.stats),
            "error": self.error,
        }
        if message:
            s["message"] = message
        return s

    # ---- commands

    def handle(self, cmd: dict, client: "ClientHandler") -> None:
        name = cmd.get("cmd")
        try:
            if name == "hello":
                client.wants_preview = bool(cmd.get("preview", True))
                client.send(self.hello())
            elif name == "ping":
                client.send({"type": "pong", "t": time.time()})
            elif name == "start":
                self.start(cmd)
            elif name == "stop":
                self.stop()
            elif name == "reset":
                self._reset_requested.set()
                if not self.running:
                    self._do_reset()
                self.broadcast(self.state("reset"))
            elif name == "configure":
                self.configure(cmd, client)
            elif name == "shutdown":
                log.info("shutdown requested")
                self.stop()
                threading.Thread(target=self.server_shutdown, daemon=True).start()
            else:
                client.send({"type": "error", "message": f"unknown command {name!r}"})
        except Exception as e:  # report to the client instead of dropping the connection
            log.exception("command %s failed", name)
            self.error = str(e)
            client.send({"type": "error", "message": str(e)})
            self.broadcast(self.state())

    server_shutdown = staticmethod(lambda: None)  # replaced by main()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def configure(self, cmd: dict, client: "ClientHandler") -> None:
        if "preview" in cmd:
            client.wants_preview = bool(cmd["preview"])
        changed_pipeline = False
        with self._lock:
            for k, v in cmd.items():
                if k in ("cmd", "gemx_root", "preview") or not hasattr(self.config, k):
                    continue
                cur = getattr(self.config, k)
                v = type(cur)(v)
                if v != cur:
                    setattr(self.config, k, v)
                    changed_pipeline |= k in Config.PIPELINE_KEYS
        if changed_pipeline:
            if self.running:
                self._rebuild_requested.set()  # swapped in by the inference loop; the camera stays open
            else:
                with self._lock:
                    self._ensure_pipeline()
        self.broadcast({"type": "config", "config": {k: v for k, v in asdict(self.config).items() if k != "gemx_root"}})

    @staticmethod
    def _source_key(cmd: dict) -> tuple:
        if cmd.get("source", "camera") == "camera":
            return ("camera", int(cmd.get("camera", 0)), int(cmd.get("width", 1280)), int(cmd.get("height", 720)))
        return ("video", cmd.get("path"), cmd.get("mode", "realtime"))

    def start(self, cmd: dict) -> None:
        if self.running:
            if self._source_key(cmd) == self._source_key(getattr(self, "_last_start", {})):
                self.broadcast(self.state("already running"))
                return
            self.stop()
        source = cmd.get("source", "camera")
        if source == "camera":
            src = CameraSource(int(cmd.get("camera", 0)), int(cmd.get("width", 1280)), int(cmd.get("height", 720)))
        elif source == "video":
            src = VideoSource(cmd["path"], cmd.get("mode", "realtime"))
        else:
            raise ValueError(f"unknown source {source!r}")
        self._last_start = dict(cmd)
        with self._lock:
            self._ensure_pipeline()
            self._source = src
            self.error = None
            self._do_reset()
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
            self._thread.start()
            self._preview_thread = threading.Thread(target=self._preview_loop, args=(src,), name="preview", daemon=True)
            self._preview_thread.start()
        log.info("started %s", src.description)
        self.broadcast(self.state("started"))

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5)
        self._thread = None
        with self._published:
            self._published.notify_all()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=2)
            self._preview_thread = None
        if self._source is not None:
            self._source.close()
            self._source = None
        self.broadcast(self.state("stopped"))

    def _do_reset(self) -> None:
        with self._lock:
            if self._pipe is not None:
                self._pipe.reset()
            self.root = np.zeros(3)
            self.stats = Stats()
        self._reset_requested.clear()

    # ---- inference loop

    def _run(self) -> None:
        src = self._source
        pipe = self._pipe
        last_counter = 0
        seq = 0
        t0_ns = None
        tracker = KeypointTracker()
        fps_ema = None
        last_done = None
        try:
            while not self._stop.is_set():
                if self._reset_requested.is_set():
                    self._do_reset()
                    tracker.reset()
                    t0_ns = None
                    seq = 0
                if self._rebuild_requested.is_set():
                    self._rebuild_requested.clear()
                    with self._lock:
                        pipe = self._ensure_pipeline()
                    tracker.reset()
                got = src.next(last_counter)
                if got is None:
                    if getattr(src, "finished", False):
                        break
                    continue
                if len(got) == 4:  # video: (counter, t_ns, bgr, capture_ns)
                    last_counter, t_ns, bgr, capture_ns = got
                else:  # camera: (counter, t_ns, bgr)
                    last_counter, t_ns, bgr = got
                    capture_ns = t_ns
                if t0_ns is None:
                    t0_ns = t_ns
                if not src.is_live:
                    self._publish(last_counter, bgr)
                if self.config.mirror:
                    bgr = cv2.flip(bgr, 1)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                seq += 1
                source_us = (t_ns - t0_ns) // 1000
                box = tracker.box if self.config.tracking == "keypoints" else None
                t_infer = time.perf_counter()
                with self._lock:
                    frame = pipe.submit(rgb, seq, source_us, box=box, subject_id=1 if box is not None else 0)
                infer_ms = (time.perf_counter() - t_infer) * 1000
                h, w = rgb.shape[:2]
                tracker.update(frame.keypoints, w, h)
                self._overlay = (frame.keypoints, frame.box, frame.outcome_name)
                done = time.perf_counter()
                if last_done is not None:
                    inst = 1.0 / max(done - last_done, 1e-6)
                    fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
                last_done = done
                st = self.stats
                st.frames += 1
                st.fps = round(fps_ema or 0.0, 2)
                st.infer_ms = round(infer_ms, 2)
                st.latency_ms = round((time.perf_counter_ns() - capture_ns) / 1e6, 2)
                self.broadcast(self._frame_message(frame, source_us))
        except Exception as e:
            log.exception("inference loop failed")
            self.error = str(e)
        finally:
            finished = getattr(src, "finished", False)
            msg = self.state("finished" if finished else "stopped")
            msg["running"] = False
            msg["finished"] = finished
            self.broadcast(msg)

    def _frame_message(self, f: gx.Frame, source_us: int) -> dict:
        msg = {
            "type": "frame",
            "seq": f.sequence,
            "t": source_us / 1e6,
            "outcome": f.outcome_name,
            "epoch": f.epoch,
            "track": f.track_epoch,
            "flags": f.flags,
            "fps": self.stats.fps,
            "infer_ms": self.stats.infer_ms,
            "latency_ms": self.stats.latency_ms,
        }
        if f.outcome != gx.POSE:
            return msg
        self.stats.poses += 1
        reset = bool(f.flags & gx.FLAG_RESET)
        if f.root_displacement is not None and f.smpl_anchor is not None and not reset:
            d = zup_to_yup(quat_rotate_wxyz(f.smpl_anchor.astype(np.float64), f.root_displacement.astype(np.float64)))
            if np.all(np.isfinite(d)):
                self.root = self.root + d
        msg["rot"] = np.round(f.rotations.reshape(-1), 6).tolist()
        msg["root"] = np.round(self.root, 5).tolist()
        if f.keypoints is not None:
            msg["kp_conf"] = round(float(f.keypoints[:, 2].mean()), 3)
        return msg

    def _publish(self, counter: int, bgr: np.ndarray) -> None:
        with self._published:
            self._published_frame = (counter, bgr)
            self._published.notify_all()

    def _wants_preview(self) -> bool:
        with self._clients_lock:
            return self.config.preview and any(c.wants_preview for c in self._clients)

    def _preview_loop(self, src) -> None:
        """Stream the camera to clients at camera rate, independent of inference."""
        last = 0
        next_due = 0.0
        while not self._stop.is_set():
            if src.is_live:
                got = src.next(last, timeout=0.5)
                if got is None:
                    continue
                last, bgr = got[0], got[2]
            else:
                with self._published:
                    self._published.wait_for(
                        lambda: self._stop.is_set()
                        or (self._published_frame is not None and self._published_frame[0] > last),
                        0.5,
                    )
                    frame = self._published_frame
                if self._stop.is_set() or frame is None or frame[0] <= last:
                    continue
                last, bgr = frame
            now = time.perf_counter()
            if now < next_due or not self._wants_preview():
                continue
            next_due = now + 1.0 / max(self.config.preview_fps, 1.0)
            if self.config.mirror:
                bgr = cv2.flip(bgr, 1)
            try:
                self.broadcast(self._preview_message(bgr, self._overlay))
            except Exception:
                log.exception("preview failed")

    def _preview_message(self, bgr: np.ndarray, overlay: tuple | None) -> dict:
        h, w = bgr.shape[:2]
        pw = min(self.config.preview_width, w)
        scale = pw / w
        img = cv2.resize(bgr, (pw, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        outcome = ""
        if overlay is not None:
            kp, box, outcome = overlay
            if box is not None:
                x0, y0, x1, y1 = (box * scale).astype(int)
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 178, 102), 1)
            if kp is not None and self._skeleton is not None:
                for j, p in enumerate(self._skeleton["parents"]):
                    if p < 0 or kp[j, 2] < 0.3 or kp[p, 2] < 0.3:
                        continue
                    a = (int(kp[j, 0] * scale), int(kp[j, 1] * scale))
                    b = (int(kp[p, 0] * scale), int(kp[p, 1] * scale))
                    cv2.line(img, a, b, (80, 220, 120), 2, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return {
            "type": "preview",
            "outcome": outcome,
            "width": img.shape[1],
            "height": img.shape[0],
            "jpeg": base64.b64encode(jpg.tobytes()).decode() if ok else "",
        }

    def close(self) -> None:
        self.stop()
        if self._pipe is not None:
            self._pipe.close()
            self._pipe = None


# --------------------------------------------------------------------------- network


class ClientHandler(socketserver.StreamRequestHandler):
    engine: Engine  # set on the subclass in main()

    MAX_QUEUED = 1800  # ~60 s of frames; beyond that the oldest are dropped

    def setup(self) -> None:
        super().setup()
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.wants_preview = True
        self.alive = True
        self.dropped = 0
        # Outgoing messages go through a queue drained by a writer thread, so a
        # client that stops reading (Maya busy baking) never blocks inference or
        # command handling. Only the newest preview image is kept.
        self._queue: deque[tuple[bool, bytes]] = deque()
        self._cond = threading.Condition()
        self._writer = threading.Thread(target=self._write_loop, name="client-writer", daemon=True)
        self._writer.start()
        self.engine.add_client(self)
        log.info("client connected %s:%d", *self.client_address)

    def send_raw(self, line: bytes, preview: bool = False) -> None:
        if not self.alive:
            return
        with self._cond:
            if preview:
                self._queue = deque(item for item in self._queue if not item[0])
            elif len(self._queue) >= self.MAX_QUEUED:
                self._queue.popleft()
                self.dropped += 1
            self._queue.append((preview, line))
            self._cond.notify()

    def _write_loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._queue or not self.alive)
                if not self.alive:
                    return
                _, line = self._queue.popleft()
            try:
                self.wfile.write(line)
                self.wfile.flush()
            except OSError:
                self.alive = False
                return

    def send(self, msg: dict) -> None:
        self.send_raw((json.dumps(msg, separators=(",", ":")) + "\n").encode())

    def handle(self) -> None:
        try:
            for raw in self.rfile:
                if not self.alive:
                    break
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    cmd = json.loads(raw)
                except json.JSONDecodeError:
                    self.send({"type": "error", "message": "invalid JSON"})
                    continue
                self.engine.handle(cmd, self)
        except (ConnectionError, OSError):
            pass  # client went away

    def finish(self) -> None:
        self.engine.remove_client(self)
        with self._cond:
            self.alive = False
            self._cond.notify_all()
        log.info("client disconnected %s:%d", *self.client_address)
        try:
            super().finish()
        except OSError:
            pass


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def watch_parent(pid: int, on_exit) -> None:
    """Call on_exit when process pid ends (the server should not outlive Maya)."""

    def alive() -> bool:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if not handle:
                return False
            try:
                return kernel32.WaitForSingleObject(handle, 0) != 0  # WAIT_OBJECT_0 = exited
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def run():
        while alive():
            time.sleep(1.0)
        log.info("parent process %d exited; shutting down", pid)
        on_exit()

    threading.Thread(target=run, name="parent-watch", daemon=True).start()


def default_gemx_root() -> str:
    env = os.environ.get("GEMX_CPP_ROOT")
    if env:
        return env
    runtime = _HERE.parent / "runtime"  # release bundle
    if (runtime / "gemx.dll").is_file():
        return str(runtime)
    return str(_HERE.parents[2] / "third_party" / "gem-x.cpp")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gemx", default=default_gemx_root(), help="gem-x.cpp checkout with build/ and generated/reference/")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--backend", choices=["Vulkan", "CPU"], default="Vulkan")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--detect-interval", type=int, default=5)
    ap.add_argument("--selection", choices=["largest", "continuity"], default="largest")
    ap.add_argument("--precision", choices=["fast", "strict"], default="fast")
    ap.add_argument("--tracking", choices=["keypoints", "detector"], default="keypoints",
                    help="keypoints: follow the person from their own keypoints, detector only to acquire")
    ap.add_argument("--flip-test", action="store_true",
                    help="average ViTPose with a mirrored pass: slightly steadier keypoints, about half the speed")
    ap.add_argument("--camera", type=int, default=None, help="start capturing this camera immediately")
    ap.add_argument("--video", default=None, help="start streaming this video immediately")
    ap.add_argument("--parent-pid", type=int, default=None, help="exit when this process exits")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["GEMX_VITPOSE_FLIP"] = "1" if a.flip_test else "0"  # read once by gem-x.cpp
    config = Config(
        gemx_root=a.gemx, backend=a.backend, device=a.device, threads=a.threads, window=a.window,
        detect_interval=a.detect_interval, selection=a.selection, precision=a.precision, tracking=a.tracking,
    )
    engine = Engine(config)
    handler = type("Handler", (ClientHandler,), {"engine": engine})
    server = Server((a.host, a.port), handler)
    engine.server_shutdown = server.shutdown
    if a.parent_pid:
        watch_parent(a.parent_pid, server.shutdown)
    log.info("listening on %s:%d (protocol %s)", a.host, a.port, PROTOCOL)
    print(f"GEMX_LIVE_READY {a.host}:{a.port}", flush=True)  # parsed by the Maya launcher
    if a.camera is not None:
        engine.start({"source": "camera", "camera": a.camera})
    elif a.video:
        engine.start({"source": "video", "path": a.video})
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("shutting down")
        engine.close()
        server.server_close()
        log.info("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
