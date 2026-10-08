"""GEM-X Live window (dockable workspaceControl)."""

from __future__ import annotations

import functools
import json
import shutil
import traceback
from pathlib import Path

import maya.api.OpenMaya as om
import maya.cmds as cmds
import numpy as np
from maya import OpenMayaUI as omui
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin
from PySide6 import QtCore, QtGui, QtWidgets

from . import __version__, mapping as mp, prefs, soma
from .session import Session
from .takes import EXTENSION, Take, safe_name

WINDOW_NAME = "gemxLiveWindow"
CONTROL_NAME = WINDOW_NAME + "WorkspaceControl"

Qt = QtCore.Qt
CHECKED = Qt.CheckState.Checked
UNCHECKED = Qt.CheckState.Unchecked
NO_SOURCE = "—"

_session: Session | None = None
_window: "LiveWindow | None" = None


def get_session() -> Session:
    global _session
    if _session is None:
        _session = Session()
    return _session


def show() -> "LiveWindow":
    global _window
    if _window is not None:
        try:
            _window.show(dockable=True)
            _window.raise_()
            return _window
        except RuntimeError:  # underlying C++ object deleted
            _window = None
    if cmds.workspaceControl(CONTROL_NAME, exists=True):
        cmds.deleteUI(CONTROL_NAME)
    _window = LiveWindow(get_session())
    _window.show(dockable=True, floating=True, width=470, height=820,
                 uiScript="import gemx_live.ui as gemx_ui; gemx_ui.restore()")
    return _window


def restore() -> None:
    """Called by Maya to rebuild the window inside a saved workspace layout."""
    global _window
    parent = omui.MQtUtil.getCurrentParent()
    _window = LiveWindow(get_session())
    ptr = omui.MQtUtil.findControl(_window.objectName())
    omui.MQtUtil.addWidgetToMayaLayout(int(ptr), int(parent))


def close(shutdown: bool = False) -> None:
    global _window, _session
    if _window is not None:
        try:
            _window.release()
            _window.close()
        except RuntimeError:
            pass
        _window = None
    if cmds.workspaceControl(CONTROL_NAME, exists=True):
        cmds.deleteUI(CONTROL_NAME)
    if shutdown and _session is not None:
        _session.shutdown(stop_server=True)
        _session = None


def guarded(fn):
    """Report exceptions from UI actions instead of letting Qt swallow them."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            self.set_status(str(e), error=True)
            om.MGlobal.displayWarning(f"GEM-X Live: {e}")
            traceback.print_exc()

    return wrapper


class PreviewView(QtWidgets.QWidget):
    """Live camera view: decodes JPEG frames and paints them aspect-fit.

    Painting instead of QLabel.setPixmap avoids relayouts on every frame.
    Frames that arrive while the view is hidden are not decoded.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QtGui.QImage | None = None
        self.setMinimumHeight(200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    @property
    def has_image(self) -> bool:
        return self._image is not None

    def set_jpeg(self, data: bytes) -> None:
        if not self.isVisible():
            return
        img = QtGui.QImage.fromData(data, "JPG")
        if not img.isNull():
            self._image = img
            self.update()

    def clear(self) -> None:
        self._image = None
        self.update()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#1b1b1b"))
        if self._image is None:
            p.setPen(QtGui.QColor("#666"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No video")
            return
        size = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - size.width()) // 2
        y = (self.height() - size.height()) // 2
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QtCore.QRect(x, y, size.width(), size.height()), self._image)


def _hline() -> QtWidgets.QFrame:
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    f.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    return f


class LiveWindow(MayaQWidgetDockableMixin, QtWidgets.QWidget):
    """Dockable host for the panel."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(WINDOW_NAME)
        self.setWindowTitle("GEM-X Live")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.panel = LivePanel(session, self)
        lay.addWidget(self.panel)

    def release(self) -> None:
        self.panel.release()

    def dockCloseEventTriggered(self):
        self.release()

    def closeEvent(self, event):
        self.release()
        super().closeEvent(event)


class LivePanel(QtWidgets.QWidget):
    """Capture / Character / Record tabs bound to a Session."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setObjectName("gemxLivePanel")
        self.s = session
        self._tree_items: dict[int, QtWidgets.QTreeWidgetItem] = {}
        self._combos: dict[int, QtWidgets.QComboBox] = {}
        self._group_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._updating = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addLayout(self._build_header())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_capture_tab(), "Capture")
        self.tabs.addTab(self._build_character_tab(), "Character")
        self.tabs.addTab(self._build_record_tab(), "Record")
        root.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(lambda _i: self._update_preview_wanted())
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status)

        s = self.s
        s.log.connect(lambda m: self.set_status(m))
        s.connection_changed.connect(lambda _c: (self._refresh_connection(), self._update_preview_wanted()))
        s.server_process_changed.connect(lambda _r: self._refresh_connection())
        s.stream_state.connect(lambda _m: self._refresh_connection())
        s.stats.connect(self._on_stats)
        s.preview_jpeg.connect(lambda data: self.preview.set_jpeg(data))
        s.character_changed.connect(self._refresh_character_info)
        s.drive_changed.connect(self._on_drive_changed)
        s.recording_changed.connect(self._on_recording_changed)
        s.countdown.connect(self._on_countdown)
        s.record_progress.connect(self._on_record_progress)
        s.takes_changed.connect(self._refresh_takes)
        s.models_progress.connect(self._on_models_progress)
        s.models_finished.connect(self._on_models_finished)

        self._refresh_connection()
        self._rebuild_tree()
        self._refresh_takes()
        self.take_name.setText(self.s.default_take_name())

    # ------------------------------------------------------------------ header / status

    def _build_header(self):
        row = QtWidgets.QHBoxLayout()
        self.dot = QtWidgets.QLabel("●")
        self.dot.setFixedWidth(14)
        title = QtWidgets.QLabel(f"<b>GEM-X Live</b> <span style='color:#888'>v{__version__}</span>")
        self.conn_label = QtWidgets.QLabel()
        self.fps_label = QtWidgets.QLabel()
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.dot)
        row.addWidget(title)
        row.addWidget(self.conn_label, 1)
        row.addWidget(self.fps_label)
        return row

    def set_status(self, text: str, error: bool = False) -> None:
        color = "#e36c5c" if error else "#bbbbbb"
        self.status.setText(f"<span style='color:{color}'>{text}</span>")

    def _set_dot(self, color: str, tip: str) -> None:
        self.dot.setStyleSheet(f"color: {color}; font-size: 14px;")
        self.dot.setToolTip(tip)

    # ------------------------------------------------------------------ capture tab

    def _build_capture_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        server = QtWidgets.QGroupBox("Capture Server")
        g = QtWidgets.QGridLayout(server)
        self.host = QtWidgets.QLineEdit(prefs.get("host"))
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(int(prefs.get("port")))
        self.host.editingFinished.connect(lambda: prefs.set("host", self.host.text().strip()))
        self.port.valueChanged.connect(lambda v: prefs.set("port", int(v)))
        self.btn_launch = QtWidgets.QPushButton("Launch Server")
        self.btn_launch.setToolTip("Start the local GEM-X server process (loads the models, ~2 s)")
        self.btn_launch.clicked.connect(self._on_launch)
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.setToolTip("Connect to a server that is already running")
        self.btn_connect.clicked.connect(self._on_connect)
        g.addWidget(QtWidgets.QLabel("Host"), 0, 0)
        g.addWidget(self.host, 0, 1)
        g.addWidget(QtWidgets.QLabel("Port"), 0, 2)
        g.addWidget(self.port, 0, 3)
        g.addWidget(self.btn_launch, 1, 0, 1, 2)
        g.addWidget(self.btn_connect, 1, 2, 1, 2)

        self.paths_toggle = QtWidgets.QToolButton()
        self.paths_toggle.setText("Server paths")
        self.paths_toggle.setCheckable(True)
        self.paths_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.paths_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.paths_toggle.setStyleSheet("QToolButton { border: none; }")
        g.addWidget(self.paths_toggle, 2, 0, 1, 4)
        self.paths = QtWidgets.QWidget()
        pg = QtWidgets.QGridLayout(self.paths)
        pg.setContentsMargins(0, 0, 0, 0)
        for row, (label, key, is_dir, auto) in enumerate((
            ("Python", "serverPython", False, prefs.server_python),
            ("Script", "serverScript", False, prefs.server_script),
            ("Runtime", "gemxRoot", True, prefs.gemx_root),
        )):
            edit = QtWidgets.QLineEdit(prefs.get(key))
            edit.setPlaceholderText(f"automatic: {auto()}")
            edit.setToolTip("Leave empty to use the automatic location")
            edit.editingFinished.connect(functools.partial(self._path_changed, key, edit))
            browse = QtWidgets.QToolButton()
            browse.setText("...")
            browse.clicked.connect(functools.partial(self._browse_path, key, edit, is_dir))
            pg.addWidget(QtWidgets.QLabel(label), row, 0)
            pg.addWidget(edit, row, 1)
            pg.addWidget(browse, row, 2)
        self.paths.setVisible(False)
        self.paths_toggle.toggled.connect(self._toggle_paths)
        g.addWidget(self.paths, 3, 0, 1, 4)

        # Shown until the models are on disk.
        self.models_row = QtWidgets.QWidget()
        mr = QtWidgets.QHBoxLayout(self.models_row)
        mr.setContentsMargins(0, 0, 0, 0)
        self.models_label = QtWidgets.QLabel(f"Models not downloaded ({prefs.MODELS_SIZE_GB:.0f} GB, one time)")
        self.models_label.setStyleSheet("color: #e0b341;")
        self.models_bar = QtWidgets.QProgressBar()
        self.models_bar.setRange(0, 1000)
        self.models_bar.setTextVisible(True)
        self.models_bar.setVisible(False)
        self.btn_models = QtWidgets.QPushButton("Download Models")
        self.btn_models.clicked.connect(self._on_download_models)
        mr.addWidget(self.models_label, 1)
        mr.addWidget(self.models_bar, 1)
        mr.addWidget(self.btn_models)
        g.addWidget(self.models_row, 4, 0, 1, 4)
        self.models_row.setVisible(bool(prefs.missing_models()))
        lay.addWidget(server)

        src = QtWidgets.QGroupBox("Source")
        sg = QtWidgets.QGridLayout(src)
        self.rb_camera = QtWidgets.QRadioButton("Camera")
        self.rb_video = QtWidgets.QRadioButton("Video")
        (self.rb_video if prefs.get("sourceKind") == "video" else self.rb_camera).setChecked(True)
        self.rb_camera.toggled.connect(lambda on: prefs.set("sourceKind", "camera" if on else "video"))
        self.camera = QtWidgets.QSpinBox()
        self.camera.setRange(0, 15)
        self.camera.setValue(int(prefs.get("camera")))
        self.camera.setToolTip("Camera index (0 is the default webcam)")
        self.camera.valueChanged.connect(lambda v: prefs.set("camera", int(v)))
        self.resolution = QtWidgets.QComboBox()
        self.resolution.addItems(["640x480", "1280x720", "1920x1080"])
        self.resolution.setCurrentText(prefs.get("resolution"))
        self.resolution.currentTextChanged.connect(lambda t: prefs.set("resolution", t))
        self.video_path = QtWidgets.QLineEdit(prefs.get("videoPath"))
        self.video_path.setPlaceholderText("video file")
        self.video_path.editingFinished.connect(lambda: prefs.set("videoPath", self.video_path.text().strip()))
        video_browse = QtWidgets.QToolButton()
        video_browse.setText("...")
        video_browse.clicked.connect(self._browse_video)
        self.video_mode = QtWidgets.QComboBox()
        self.video_mode.addItem("Real-time", "realtime")
        self.video_mode.addItem("Every frame", "all")
        self.video_mode.setToolTip("Real-time drops frames like a live camera; Every frame processes the whole clip")
        self.video_mode.setCurrentIndex(max(0, self.video_mode.findData(prefs.get("videoMode"))))
        self.video_mode.currentIndexChanged.connect(lambda _i: prefs.set("videoMode", self.video_mode.currentData()))
        sg.addWidget(self.rb_camera, 0, 0)
        sg.addWidget(self.camera, 0, 1)
        sg.addWidget(self.resolution, 0, 2, 1, 2)
        sg.addWidget(self.rb_video, 1, 0)
        vrow = QtWidgets.QHBoxLayout()
        vrow.addWidget(self.video_path, 1)
        vrow.addWidget(video_browse)
        sg.addLayout(vrow, 1, 1, 1, 2)
        sg.addWidget(self.video_mode, 1, 3)

        self.tracking = QtWidgets.QComboBox()
        self.tracking.addItem("Follow keypoints (fast)", "keypoints")
        self.tracking.addItem("Detector every N frames", "detector")
        self.tracking.setToolTip(
            "Follow keypoints: after finding you once, each frame's crop comes from your own pose, so the\n"
            "person detector only runs to re-find you. Detector: re-detects every N frames (slower)."
        )
        self.tracking.setCurrentIndex(max(0, self.tracking.findData(prefs.get("tracking"))))
        self.tracking.currentIndexChanged.connect(self._on_tracking_changed)
        self.detect = QtWidgets.QSpinBox()
        self.detect.setRange(1, 30)
        self.detect.setPrefix("N = ")
        self.detect.setValue(int(prefs.get("detectInterval")))
        self.detect.setToolTip("Detector mode: run the person detector every N frames")
        self.detect.valueChanged.connect(self._on_detect_changed)
        self.detect.setEnabled(prefs.get("tracking") == "detector")
        self.mirror = QtWidgets.QCheckBox("Mirror")
        self.mirror.setToolTip("Flip the image so the character moves like a mirror reflection")
        self.mirror.setChecked(bool(prefs.get("mirror")))
        self.mirror.toggled.connect(self._on_mirror)
        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(QtWidgets.QLabel("Tracking"))
        orow.addWidget(self.tracking, 1)
        orow.addWidget(self.detect)
        orow.addWidget(self.mirror)
        sg.addLayout(orow, 2, 0, 1, 4)
        self.flip_test = QtWidgets.QCheckBox("Steadier keypoints (flip test, about half the speed)")
        self.flip_test.setToolTip("Averages each pose with a mirrored pass. Takes effect when the server is (re)launched.")
        self.flip_test.setChecked(bool(prefs.get("flipTest")))
        self.flip_test.toggled.connect(self._on_flip_test)
        sg.addWidget(self.flip_test, 3, 0, 1, 4)

        brow = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(lambda: self.s.stop_capture())
        self.btn_reset = QtWidgets.QPushButton("Reset Tracking")
        self.btn_reset.setToolTip("Clear the temporal window and return the root trajectory to the origin")
        self.btn_reset.clicked.connect(lambda: self.s.reset_tracking())
        brow.addWidget(self.btn_start)
        brow.addWidget(self.btn_stop)
        brow.addWidget(self.btn_reset)
        sg.addLayout(brow, 4, 0, 1, 4)
        lay.addWidget(src)

        self.preview = PreviewView()
        lay.addWidget(self.preview, 1)
        self.stats_label = QtWidgets.QLabel(" ")
        self.stats_label.setStyleSheet("color: #999;")
        lay.addWidget(self.stats_label)
        return w

    def _toggle_paths(self, on: bool) -> None:
        self.paths.setVisible(on)
        self.paths_toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    def _path_changed(self, key, edit) -> None:
        prefs.set(key, edit.text().strip())
        self.models_row.setVisible(bool(prefs.missing_models()))

    @guarded
    def _on_download_models(self, *_):
        self.btn_models.setEnabled(False)
        self.models_bar.setVisible(True)
        self.models_bar.setValue(0)
        self.s.download_models()

    def _on_models_progress(self, name: str, fraction: float) -> None:
        index = prefs.MODEL_FILES.index(name) if name in prefs.MODEL_FILES else 0
        self.models_bar.setFormat(f"{name}  %p%")
        self.models_bar.setValue(int(fraction * 1000))
        self.models_label.setText(f"Downloading model {index + 1} of {len(prefs.MODEL_FILES)}...")

    def _on_models_finished(self, ok: bool) -> None:
        self.btn_models.setEnabled(True)
        self.models_bar.setVisible(False)
        missing = prefs.missing_models()
        self.models_row.setVisible(bool(missing))
        if missing:
            self.models_label.setText("Download incomplete. Press Download Models to resume.")

    def _browse_path(self, key, edit, is_dir) -> None:
        start = edit.text() or str(Path.home())
        if is_dir:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", start)
        if path:
            edit.setText(path)
            prefs.set(key, path)

    def _browse_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video", self.video_path.text() or str(Path.home()),
            "Video (*.mp4 *.mov *.avi *.mkv *.webm);;All files (*)",
        )
        if path:
            self.video_path.setText(path)
            prefs.set("videoPath", path)
            self.rb_video.setChecked(True)

    @guarded
    def _on_launch(self, *_):
        if self.s.server.running:
            self.s.stop_server()
        else:
            prefs.set("host", self.host.text().strip())
            self.s.launch_server()
        self._refresh_connection()

    @guarded
    def _on_connect(self, *_):
        if self.s.client.is_connected:
            self.s.disconnect_server()
        else:
            prefs.set("host", self.host.text().strip())
            self.s.connect_server()

    @guarded
    def _on_start(self, *_):
        prefs.set("videoPath", self.video_path.text().strip())
        self.s.start_capture()

    def _on_detect_changed(self, v: int) -> None:
        prefs.set("detectInterval", int(v))
        # The server rebuilds its pipeline for this; wait until the value settles.
        if not hasattr(self, "_detect_timer"):
            self._detect_timer = QtCore.QTimer(self)
            self._detect_timer.setSingleShot(True)
            self._detect_timer.setInterval(600)
            self._detect_timer.timeout.connect(lambda: self.s.client.is_connected and self.s.configure_server())
        self._detect_timer.start()

    def _on_tracking_changed(self, _index: int) -> None:
        mode = self.tracking.currentData()
        prefs.set("tracking", mode)
        self.detect.setEnabled(mode == "detector")
        if self.s.client.is_connected:
            self.s.configure_server()

    def _on_flip_test(self, on: bool) -> None:
        prefs.set("flipTest", on)
        if self.s.server.running or self.s.client.is_connected:
            self.set_status("Flip test changes when the server is relaunched (Stop Server, then Launch Server).")

    def _update_preview_wanted(self) -> None:
        wanted = self.isVisible() and self.tabs.currentIndex() == 0
        if self.s.client.is_connected:
            self.s.set_preview_wanted(wanted)
        if not wanted:
            self.preview.clear()

    def _on_mirror(self, on: bool) -> None:
        prefs.set("mirror", on)
        if self.s.client.is_connected:
            self.s.configure_server()

    def _refresh_connection(self) -> None:
        s = self.s
        connected = s.client.is_connected
        running = bool(s.stream.get("running"))
        self.btn_launch.setText("Stop Server" if s.server.running else "Launch Server")
        self.btn_connect.setText("Disconnect" if connected else "Connect")
        self.btn_start.setEnabled(connected)
        self.btn_stop.setEnabled(connected and running)
        self.btn_reset.setEnabled(connected)
        if not connected:
            self._set_dot("#666666", "Not connected")
            self.conn_label.setText("<span style='color:#888'>not connected</span>")
            self.fps_label.setText("")
        elif s.recording:
            self._set_dot("#e0473a", "Recording")
        elif running:
            self._set_dot("#5ec269", "Streaming")
            self.conn_label.setText(f"<span style='color:#aaa'>{s.stream.get('source') or ''}</span>")
        else:
            self._set_dot("#e0b341", "Connected, idle")
            device = s.server_info.get("device", "")
            self.conn_label.setText(f"<span style='color:#aaa'>idle · {device}</span>")
        self._refresh_record_enabled()

    def _on_stats(self, msg: dict) -> None:
        outcome = msg.get("outcome", "")
        color = {"pose": "#5ec269", "warmup": "#e0b341"}.get(outcome, "#e36c5c")
        self.fps_label.setText(f"<span style='color:#aaa'>{msg.get('fps', 0):.1f} fps</span>")
        self.stats_label.setText(
            f"<span style='color:{color}'>{outcome}</span> · inference {msg.get('infer_ms', 0):.0f} ms"
            f" · latency {msg.get('latency_ms', 0):.0f} ms · frame {msg.get('seq', 0)}"
        )

    # ------------------------------------------------------------------ character tab

    def _build_character_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        row = QtWidgets.QHBoxLayout()
        self.root_field = QtWidgets.QLineEdit()
        self.root_field.setReadOnly(True)
        self.root_field.setPlaceholderText("select a joint of the target skeleton")
        use_sel = QtWidgets.QPushButton("Use Selected")
        use_sel.setToolTip("Use the skeleton of the selected joint (or a group containing it) as the target")
        use_sel.clicked.connect(self._on_use_selected)
        sel_btn = QtWidgets.QToolButton()
        sel_btn.setText("Select")
        sel_btn.clicked.connect(lambda: self.s.character and cmds.select(self.s.character.root))
        row.addWidget(QtWidgets.QLabel("Target"))
        row.addWidget(self.root_field, 1)
        row.addWidget(use_sel)
        row.addWidget(sel_btn)
        lay.addLayout(row)
        self.char_info = QtWidgets.QLabel(" ")
        self.char_info.setStyleSheet("color: #999;")
        self.char_info.setWordWrap(True)
        lay.addWidget(self.char_info)

        tools = QtWidgets.QHBoxLayout()
        self.btn_auto = QtWidgets.QPushButton("Auto Map")
        self.btn_auto.clicked.connect(self._on_auto_map)
        self.btn_presets = QtWidgets.QToolButton()
        self.btn_presets.setText("Presets")
        self.btn_presets.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_presets.setMenu(self._build_preset_menu())
        self.btn_rest = QtWidgets.QPushButton("Capture Rest Pose")
        self.btn_rest.setToolTip("Use the character's current pose as its rest pose (T-pose or A-pose both work)")
        self.btn_rest.clicked.connect(self._on_capture_rest)
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.clicked.connect(self._on_clear_mapping)
        tools.addWidget(self.btn_auto)
        tools.addWidget(self.btn_presets)
        tools.addWidget(self.btn_rest)
        tools.addWidget(self.btn_clear)
        lay.addLayout(tools)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Joint", "Source (SOMA)", "Drive"])
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(12)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.tree.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        lay.addWidget(self.tree, 1)

        mask = QtWidgets.QGroupBox("Drive (unchecked groups are masked)")
        mg = QtWidgets.QGridLayout(mask)
        mg.setSpacing(3)
        for i, group in enumerate(soma.GROUP_ORDER):
            b = QtWidgets.QPushButton(group)
            b.setCheckable(True)
            b.setChecked(True)
            b.clicked.connect(lambda checked=False, g=group: self._on_group_clicked(g, checked))
            self._group_buttons[group] = b
            mg.addWidget(b, i // 3, i % 3)
        lay.addWidget(mask)

        rootbox = QtWidgets.QGroupBox("Root Motion")
        rg = QtWidgets.QGridLayout(rootbox)
        self.opt_horizontal = QtWidgets.QCheckBox("Horizontal travel")
        self.opt_vertical = QtWidgets.QCheckBox("Vertical")
        self.opt_vmode = QtWidgets.QComboBox()
        self.opt_vmode.addItem("Floor contact", "floor")
        self.opt_vmode.addItem("Trajectory", "trajectory")
        self.opt_vmode.setToolTip("Floor contact keeps the lowest foot on the ground; Trajectory follows the estimated path (can drift)")
        self.opt_autoscale = QtWidgets.QCheckBox("Auto scale")
        self.opt_autoscale.setToolTip("Scale root travel by the ratio of hip heights")
        self.opt_scale = QtWidgets.QDoubleSpinBox()
        self.opt_scale.setRange(0.001, 100000.0)
        self.opt_scale.setDecimals(3)
        self.opt_scale.setSuffix(" units/m")
        self.opt_facing = QtWidgets.QComboBox()
        for deg in (0, 90, 180, -90):
            self.opt_facing.addItem(f"{deg:+d}°" if deg else "0°", float(deg))
        self.opt_facing.setToolTip("Extra rotation of the performance about the up axis")
        rg.addWidget(self.opt_horizontal, 0, 0)
        rg.addWidget(self.opt_vertical, 0, 1)
        rg.addWidget(self.opt_vmode, 0, 2)
        rg.addWidget(self.opt_autoscale, 1, 0)
        rg.addWidget(self.opt_scale, 1, 1)
        facing = QtWidgets.QHBoxLayout()
        facing.addWidget(QtWidgets.QLabel("Facing"))
        facing.addWidget(self.opt_facing, 1)
        rg.addLayout(facing, 1, 2)
        for wdg, sig in ((self.opt_horizontal, "toggled"), (self.opt_vertical, "toggled"), (self.opt_autoscale, "toggled"),
                         (self.opt_vmode, "currentIndexChanged"), (self.opt_facing, "currentIndexChanged")):
            getattr(wdg, sig).connect(self._on_options_changed)
        self.opt_scale.editingFinished.connect(self._on_options_changed)
        lay.addWidget(rootbox)

        live = QtWidgets.QGroupBox("Live")
        lg = QtWidgets.QHBoxLayout(live)
        self.chk_drive = QtWidgets.QCheckBox("Drive character")
        self.chk_drive.setChecked(self.s.drive_live)
        self.chk_drive.setToolTip("Pose the character from the live stream. Its original pose comes back when this is turned off.")
        self.chk_drive.toggled.connect(self._on_drive_toggled)
        self.chk_smooth = QtWidgets.QCheckBox("Smooth playback")
        self.chk_smooth.setToolTip("Interpolate between poses at display rate (adds about one frame of delay)")
        self.chk_smooth.setChecked(self.s.smooth_playback)
        self.chk_smooth.toggled.connect(lambda on: self.s.set_smooth_playback(on))
        self.chk_source = QtWidgets.QCheckBox("Show source skeleton")
        self.chk_source.setChecked(self.s.show_source)
        self.chk_source.toggled.connect(self._on_source_toggled)
        recenter = QtWidgets.QPushButton("Recenter")
        recenter.setToolTip("Make the performer's current position the character's rest position")
        recenter.clicked.connect(lambda: self.s.recenter())
        lg.addWidget(self.chk_drive)
        lg.addWidget(self.chk_smooth)
        lg.addWidget(self.chk_source)
        lg.addStretch(1)
        lg.addWidget(recenter)
        lay.addWidget(live)
        return w

    def _build_preset_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        for name in mp.builtin_presets(self.s.skeleton.names):
            menu.addAction(name, functools.partial(self._on_builtin_preset, name))
        menu.addSeparator()
        menu.addAction("Load Mapping...", self._on_load_mapping)
        menu.addAction("Save Mapping...", self._on_save_mapping)
        return menu

    @guarded
    def _on_use_selected(self, *_):
        sel = cmds.ls(selection=True, long=True)
        if not sel:
            raise RuntimeError("select a joint of the target skeleton")
        self.s.set_character(sel[0])
        self._rebuild_tree()

    @guarded
    def _on_auto_map(self, *_):
        self._require_character()
        self.s.auto_map()
        self._rebuild_tree()

    @guarded
    def _on_builtin_preset(self, name):
        self._require_character()
        self.s.apply_builtin_preset(name)
        self._rebuild_tree()
        self.set_status(f"Applied preset {name}: {len(self.s.character.mapping)} joints mapped")

    @guarded
    def _on_load_mapping(self):
        self._require_character()
        folder = prefs.mappings_folder()
        folder.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load mapping", str(folder), "Mapping (*.json)")
        if path:
            self.s.load_mapping(path)
            self._rebuild_tree()

    @guarded
    def _on_save_mapping(self):
        self._require_character()
        folder = prefs.mappings_folder()
        folder.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save mapping", str(folder / "mapping.json"), "Mapping (*.json)")
        if path:
            self.s.save_mapping(path)
            self.set_status(f"Saved mapping to {path}")

    @guarded
    def _on_capture_rest(self, *_):
        self._require_character()
        self.s.capture_rest()

    @guarded
    def _on_clear_mapping(self, *_):
        self._require_character()
        if QtWidgets.QMessageBox.question(self, "GEM-X Live", "Clear the whole mapping?") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.s.clear_mapping()
        self._rebuild_tree()

    def _require_character(self):
        if self.s.character is None:
            raise RuntimeError("choose a target skeleton first (Character tab → Use Selected)")

    def _rebuild_tree(self) -> None:
        self._updating = True
        self.tree.clear()
        self._tree_items.clear()
        self._combos.clear()
        ch = self.s.character
        if ch is not None:
            sources = [NO_SOURCE] + list(self.s.skeleton.names)
            for j in range(ch.count):
                parent = self._tree_items.get(ch.parents[j])
                item = QtWidgets.QTreeWidgetItem(parent if parent is not None else self.tree)
                item.setText(0, ch.short_names[j])
                item.setToolTip(0, ch.paths[j])
                item.setData(0, Qt.ItemDataRole.UserRole, j)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if not ch.is_joint[j]:
                    item.setForeground(0, QtGui.QBrush(QtGui.QColor("#888")))
                self._tree_items[j] = item
                combo = QtWidgets.QComboBox()
                combo.addItems(sources)
                combo.setMaxVisibleItems(25)
                combo.setCurrentText(ch.mapping.get(j, NO_SOURCE))
                combo.currentTextChanged.connect(lambda text, j=j: self._on_combo_changed(j, text))
                self.tree.setItemWidget(item, 1, combo)
                self._combos[j] = combo
                self._style_row(j)
            self.tree.expandAll()
        self._updating = False
        self._refresh_character_info()

    def _style_row(self, j: int) -> None:
        ch = self.s.character
        item = self._tree_items[j]
        mapped = j in ch.mapping
        driven = mapped and j not in ch.masked
        item.setCheckState(2, CHECKED if driven else UNCHECKED)
        if not mapped:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(2, Qt.ItemDataRole.CheckStateRole, None)
        else:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        color = "#ddd" if driven else ("#c99a4a" if mapped else "#777")
        if ch.is_joint[j]:
            item.setForeground(0, QtGui.QBrush(QtGui.QColor(color)))

    @guarded
    def _on_combo_changed(self, j: int, text: str):
        if self._updating:
            return
        self.s.set_mapping(j, None if text == NO_SOURCE else text)
        self._updating = True
        self._style_row(j)
        self._updating = False

    @guarded
    def _on_item_changed(self, item, column):
        if self._updating or column != 2:
            return
        j = item.data(0, Qt.ItemDataRole.UserRole)
        if j is None or j not in self.s.character.mapping:
            return
        driven = item.checkState(2) == CHECKED
        targets = [j]
        selected = [i.data(0, Qt.ItemDataRole.UserRole) for i in self.tree.selectedItems()]
        if j in selected:  # apply to the whole selection
            targets = [k for k in selected if k in self.s.character.mapping]
        self.s.set_driven(targets, driven)
        self._updating = True
        for k in targets:
            self._style_row(k)
        self._updating = False

    def _on_tree_selection(self) -> None:
        ch = self.s.character
        if ch is None or self._updating:
            return
        nodes = [ch.paths[i.data(0, Qt.ItemDataRole.UserRole)] for i in self.tree.selectedItems()]
        if nodes:
            cmds.select(nodes, replace=True)

    def _tree_menu(self, pos) -> None:
        if self.s.character is None:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Drive selected", lambda: self._set_selected_driven(True))
        menu.addAction("Mask selected", lambda: self._set_selected_driven(False))
        menu.addSeparator()
        menu.addAction("Unmap selected", self._unmap_selected)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    @guarded
    def _set_selected_driven(self, driven: bool):
        ch = self.s.character
        targets = [i.data(0, Qt.ItemDataRole.UserRole) for i in self.tree.selectedItems()]
        targets = [j for j in targets if j in ch.mapping]
        self.s.set_driven(targets, driven)
        self._updating = True
        for j in targets:
            self._style_row(j)
        self._updating = False

    @guarded
    def _unmap_selected(self):
        for i in self.tree.selectedItems():
            j = i.data(0, Qt.ItemDataRole.UserRole)
            self._combos[j].setCurrentText(NO_SOURCE)

    @guarded
    def _on_group_clicked(self, group: str, checked: bool):
        self._require_character()
        joints = self.s.group_joints(group)
        if not joints:
            self.set_status(f"No target joints are mapped to {group}")
            return
        self.s.set_driven(joints, checked)
        self._updating = True
        for j in joints:
            self._style_row(j)
        self._updating = False

    @guarded
    def _on_options_changed(self, *_):
        if self._updating or self.s.character is None:
            return
        self.s.set_options(
            translate_horizontal=self.opt_horizontal.isChecked(),
            translate_vertical=self.opt_vertical.isChecked(),
            vertical_mode=self.opt_vmode.currentData(),
            auto_scale=self.opt_autoscale.isChecked(),
            scale=float(self.opt_scale.value()),
            facing_offset=float(self.opt_facing.currentData()),
        )

    def _refresh_character_info(self) -> None:
        ch = self.s.character
        self._updating = True
        enabled = ch is not None
        for wdg in (self.btn_auto, self.btn_presets, self.btn_rest, self.btn_clear, self.opt_horizontal,
                    self.opt_vertical, self.opt_vmode, self.opt_autoscale, self.opt_scale, self.opt_facing):
            wdg.setEnabled(enabled)
        if ch is None:
            self.root_field.setText("")
            self.char_info.setText("No target. Select a joint of your character and press Use Selected.")
            self._updating = False
            self._refresh_record_enabled()
            return
        self.root_field.setText(ch.names[0])
        rt = self.s.retargeter
        o = rt.opt if rt else None
        if o:
            self.opt_horizontal.setChecked(o.translate_horizontal)
            self.opt_vertical.setChecked(o.translate_vertical)
            self.opt_vmode.setCurrentIndex(max(0, self.opt_vmode.findData(o.vertical_mode)))
            self.opt_autoscale.setChecked(o.auto_scale)
            self.opt_scale.setValue(rt.scale)
            self.opt_scale.setEnabled(not o.auto_scale)
            self.opt_facing.setCurrentIndex(max(0, self.opt_facing.findData(float(o.facing_offset))))
        for group, b in self._group_buttons.items():
            joints = self.s.group_joints(group)
            b.setEnabled(bool(joints))
            b.setChecked(bool(joints) and all(j not in ch.masked for j in joints))
        driven = len(ch.driven())
        root = ch.names[rt.root_joint] if rt and rt.root_joint is not None else "none"
        up = "Z" if ch.up[2] > 0.5 else "Y"
        warn = "" if rt else " <span style='color:#e36c5c'>retarget setup failed</span>"
        self.char_info.setText(
            f"{len(ch.mapping)}/{ch.count} mapped · {driven} driven · root {root} · {up}-up"
            f"{(' · ' + ch.preset_name) if ch.preset_name else ''}{warn}"
        )
        self._updating = False
        self._refresh_record_enabled()

    def _on_drive_toggled(self, on: bool) -> None:
        if not self._updating:
            self.s.set_drive_live(on)

    def _on_drive_changed(self, on: bool) -> None:
        self._updating = True
        self.chk_drive.setChecked(on)
        self._updating = False

    @guarded
    def _on_source_toggled(self, on: bool):
        self.s.set_show_source(on)

    # ------------------------------------------------------------------ record tab

    def _build_record_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QGridLayout()
        self.take_name = QtWidgets.QLineEdit()
        renew = QtWidgets.QToolButton()
        renew.setText("↻")
        renew.setToolTip("Next free take name")
        renew.clicked.connect(lambda: self.take_name.setText(self.s.default_take_name()))
        self.countdown_spin = QtWidgets.QSpinBox()
        self.countdown_spin.setRange(0, 30)
        self.countdown_spin.setSuffix(" s")
        self.countdown_spin.setValue(int(prefs.get("countdown")))
        self.countdown_spin.valueChanged.connect(lambda v: prefs.set("countdown", int(v)))
        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0, 3600)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setSpecialValueText("until stopped")
        self.duration_spin.setValue(float(prefs.get("duration")))
        self.duration_spin.valueChanged.connect(lambda v: prefs.set("duration", float(v)))
        self.start_current = QtWidgets.QCheckBox("Current time")
        self.start_current.setChecked(bool(prefs.get("startAtCurrent")))
        self.start_frame = QtWidgets.QDoubleSpinBox()
        self.start_frame.setRange(-1e6, 1e6)
        self.start_frame.setDecimals(0)
        self.start_frame.setValue(float(prefs.get("startFrame")))
        self.start_frame.setEnabled(not self.start_current.isChecked())
        self.start_current.toggled.connect(lambda on: (prefs.set("startAtCurrent", on), self.start_frame.setEnabled(not on)))
        self.start_frame.valueChanged.connect(lambda v: prefs.set("startFrame", float(v)))
        self.auto_apply = QtWidgets.QCheckBox("Key the character when recording stops")
        self.auto_apply.setChecked(bool(prefs.get("autoApply")))
        self.auto_apply.toggled.connect(lambda on: prefs.set("autoApply", on))

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(self.take_name, 1)
        name_row.addWidget(renew)
        form.addWidget(QtWidgets.QLabel("Take"), 0, 0)
        form.addLayout(name_row, 0, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("Countdown"), 1, 0)
        form.addWidget(self.countdown_spin, 1, 1)
        form.addWidget(QtWidgets.QLabel("Duration"), 1, 2)
        form.addWidget(self.duration_spin, 1, 3)
        form.addWidget(QtWidgets.QLabel("Start frame"), 2, 0)
        form.addWidget(self.start_current, 2, 1)
        form.addWidget(self.start_frame, 2, 2, 1, 2)
        form.addWidget(self.auto_apply, 3, 0, 1, 4)
        lay.addLayout(form)

        self.btn_record = QtWidgets.QPushButton("●  Record")
        self.btn_record.setMinimumHeight(44)
        self.btn_record.clicked.connect(self._on_record)
        self._style_record(False)
        lay.addWidget(self.btn_record)
        self.record_info = QtWidgets.QLabel(" ")
        self.record_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.record_info.font()
        font.setPointSize(font.pointSize() + 3)
        self.record_info.setFont(font)
        lay.addWidget(self.record_info)
        lay.addWidget(_hline())

        self.takes_tree = QtWidgets.QTreeWidget()
        self.takes_tree.setHeaderLabels(["Take", "Length", "Poses", "Recorded"])
        self.takes_tree.setRootIsDecorated(False)
        self.takes_tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.takes_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.takes_tree.itemDoubleClicked.connect(lambda *_: self._on_apply_take())
        lay.addWidget(self.takes_tree, 1)

        grid = QtWidgets.QGridLayout()
        buttons = (
            ("Key Character", self._on_apply_take, "Retarget the selected take onto the character (undoable)"),
            ("Rename", self._on_rename_take, ""),
            ("Delete", self._on_delete_take, ""),
            ("Import...", self._on_import_take, ""),
            ("Export...", self._on_export_take, ""),
            ("Show Folder", self._on_show_folder, ""),
        )
        for i, (label, fn, tip) in enumerate(buttons):
            b = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            grid.addWidget(b, i // 3, i % 3)
        lay.addLayout(grid)
        self.takes_folder_label = QtWidgets.QLabel()
        self.takes_folder_label.setStyleSheet("color: #888;")
        self.takes_folder_label.setWordWrap(True)
        lay.addWidget(self.takes_folder_label)
        return w

    def _style_record(self, recording: bool) -> None:
        if recording:
            self.btn_record.setText("■  Stop")
            self.btn_record.setStyleSheet("QPushButton { background: #b8322a; color: white; font-weight: bold; }")
        else:
            self.btn_record.setText("●  Record")
            self.btn_record.setStyleSheet("QPushButton { background: #5a2a27; color: #ffb3ad; font-weight: bold; }")

    def _refresh_record_enabled(self) -> None:
        ok = self.s.client.is_connected and bool(self.s.stream.get("running"))
        self.btn_record.setEnabled(ok or self.s.recording)
        if not ok and not self.s.recording:
            self.btn_record.setToolTip("Start a capture source on the Capture tab first")
        else:
            self.btn_record.setToolTip("")

    @guarded
    def _on_record(self, *_):
        s = self.s
        if s.recording or s._countdown_timer.isActive():
            s.stop_recording()
            return
        name = self.take_name.text().strip() or s.default_take_name()
        s.start_recording(name, self.countdown_spin.value(), self.duration_spin.value())

    def _on_countdown(self, n: int) -> None:
        if n > 0:
            self.record_info.setText(f"<span style='color:#e0b341'>Recording in {n}...</span>")
            self._style_record(True)
        elif not self.s.recording:
            self.record_info.setText(" ")
            self._style_record(False)

    def _on_recording_changed(self, on: bool) -> None:
        self._style_record(on)
        self._refresh_connection()
        if not on:
            self.record_info.setText(" ")
            self.take_name.setText(self.s.default_take_name())

    def _on_record_progress(self, elapsed: float, poses: int) -> None:
        m, sec = divmod(elapsed, 60)
        self.record_info.setText(f"<span style='color:#e0473a'>● REC</span> {int(m):02d}:{sec:04.1f} · {poses} poses")

    def _refresh_takes(self) -> None:
        self.takes_tree.clear()
        folder = prefs.takes_folder()
        self.takes_folder_label.setText(str(folder))
        for path in self.s.takes():
            try:
                with np.load(path, allow_pickle=False) as z:
                    times = z["times"]
                    header = json.loads(str(z["header"]))
            except Exception:
                continue
            length = float(times[-1] - times[0]) if len(times) > 1 else 0.0
            item = QtWidgets.QTreeWidgetItem([
                header.get("name", path.stem), f"{length:.1f} s", str(len(times)), header.get("created", "").replace("T", " "),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(0, str(path))
            self.takes_tree.addTopLevelItem(item)
        for c in range(1, 4):
            self.takes_tree.resizeColumnToContents(c)

    def _selected_takes(self) -> list[Path]:
        return [Path(i.data(0, Qt.ItemDataRole.UserRole)) for i in self.takes_tree.selectedItems()]

    @guarded
    def _on_apply_take(self, *_):
        paths = self._selected_takes()
        if not paths:
            raise RuntimeError("select a take")
        self._require_character()
        start = self.s.start_frame()
        for path in paths:
            self.s.apply_take(path, start)

    @guarded
    def _on_rename_take(self, *_):
        paths = self._selected_takes()
        if len(paths) != 1:
            raise RuntimeError("select one take to rename")
        take = Take.load(paths[0])
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename take", "Name", text=take.name)
        name = safe_name(name) if ok else ""
        if not name or name == take.name:
            return
        new_path = paths[0].with_name(name + EXTENSION)
        if new_path.exists():
            raise RuntimeError(f"a take named {name} already exists")
        take.name = name
        take.save(new_path)
        paths[0].unlink()
        self._refresh_takes()

    @guarded
    def _on_delete_take(self, *_):
        paths = self._selected_takes()
        if not paths:
            return
        names = ", ".join(p.name[: -len(EXTENSION)] for p in paths)
        if QtWidgets.QMessageBox.question(self, "Delete takes", f"Delete {names}? This cannot be undone.") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for p in paths:
            p.unlink(missing_ok=True)
        self._refresh_takes()

    @guarded
    def _on_import_take(self, *_):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Import takes", str(Path.home()), f"GEM-X takes (*{EXTENSION})")
        folder = prefs.takes_folder()
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, folder / Path(f).name)
        self._refresh_takes()

    @guarded
    def _on_export_take(self, *_):
        paths = self._selected_takes()
        if not paths:
            raise RuntimeError("select takes to export")
        dest = QtWidgets.QFileDialog.getExistingDirectory(self, "Export takes to", str(Path.home()))
        if dest:
            for p in paths:
                shutil.copy2(p, Path(dest) / p.name)
            self.set_status(f"Exported {len(paths)} take(s) to {dest}")

    @guarded
    def _on_show_folder(self, *_):
        folder = prefs.takes_folder()
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    # ------------------------------------------------------------------ lifecycle

    def release(self) -> None:
        """Stop driving the character when the window goes away."""
        try:
            self.s.set_drive_live(False)
        except Exception:
            traceback.print_exc()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_preview_wanted()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._update_preview_wanted()
