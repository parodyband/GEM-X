"""Connection to the GEM-X live server, on Qt's event loop (no threads in Maya)."""

from __future__ import annotations

import json
import os
import subprocess

from PySide6 import QtCore, QtNetwork


class ServerClient(QtCore.QObject):
    """Newline-delimited JSON over TCP. Emits one signal per message type."""

    connected = QtCore.Signal()
    disconnected = QtCore.Signal()
    hello = QtCore.Signal(dict)
    state = QtCore.Signal(dict)
    frames = QtCore.Signal(list)  # all frames from one socket read, in order
    preview = QtCore.Signal(dict)
    config = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sock = QtNetwork.QTcpSocket(self)
        self._sock.setSocketOption(QtNetwork.QAbstractSocket.SocketOption.LowDelayOption, 1)
        self._sock.connected.connect(self._on_connected)
        self._sock.disconnected.connect(self.disconnected)
        self._sock.readyRead.connect(self._on_ready_read)
        self._sock.errorOccurred.connect(self._on_error)
        self._buffer = bytearray()

    @property
    def is_connected(self) -> bool:
        return self._sock.state() == QtNetwork.QAbstractSocket.SocketState.ConnectedState

    def connect_to(self, host: str, port: int) -> None:
        if self._sock.state() != QtNetwork.QAbstractSocket.SocketState.UnconnectedState:
            self._sock.abort()
        self._buffer.clear()
        self._sock.connectToHost(host, int(port))

    def close(self) -> None:
        self._sock.disconnectFromHost()

    def send(self, msg: dict) -> None:
        if self.is_connected:
            self._sock.write((json.dumps(msg) + "\n").encode())

    def flush(self, timeout_ms: int = 1000) -> None:
        """Push queued messages out now (for use before blocking the event loop)."""
        if self.is_connected and self._sock.bytesToWrite():
            self._sock.waitForBytesWritten(timeout_ms)

    def _on_connected(self) -> None:
        self.send({"cmd": "hello", "preview": True})
        self.connected.emit()

    def _on_error(self, _err) -> None:
        self.error.emit(self._sock.errorString())

    def _on_ready_read(self) -> None:
        self._buffer += bytes(self._sock.readAll())
        frames = []
        while True:
            nl = self._buffer.find(b"\n")
            if nl < 0:
                break
            line = bytes(self._buffer[:nl])
            del self._buffer[: nl + 1]
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            kind = msg.get("type")
            if kind == "frame":
                frames.append(msg)
            elif kind == "preview":
                self.preview.emit(msg)
            elif kind == "hello":
                self.hello.emit(msg)
            elif kind == "state":
                self.state.emit(msg)
            elif kind == "config":
                self.config.emit(msg.get("config", {}))
            elif kind == "error":
                self.error.emit(msg.get("message", "server error"))
        if frames:
            self.frames.emit(frames)


class ServerProcess(QtCore.QObject):
    """Launches the server as a child process and reports when it listens."""

    ready = QtCore.Signal(str, int)
    finished = QtCore.Signal(int)
    output = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = QtCore.QProcess(self)
        self._proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(lambda code, _status: self.finished.emit(code))

    @property
    def running(self) -> bool:
        return self._proc.state() != QtCore.QProcess.ProcessState.NotRunning

    def start(self, python: str, script: str, gemx_root: str, host: str, port: int, extra: list[str] | None = None) -> None:
        if self.running:
            return
        for path, what in ((python, "server Python"), (script, "server script")):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{what} not found: {path}")
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        for key in ("PYTHONPATH", "PYTHONHOME", "MAYA_LOCATION"):  # keep Maya's Python out of the server
            env.remove(key)
        self._proc.setProcessEnvironment(env)
        self._proc.setWorkingDirectory(os.path.dirname(script))
        if os.name == "nt":
            def no_console(args):
                try:
                    args.flags |= subprocess.CREATE_NO_WINDOW
                except Exception:
                    pass

            try:
                self._proc.setCreateProcessArgumentsModifier(no_console)
            except Exception:
                pass  # Qt already hides consoles for GUI parents
        args = [script, "--gemx", gemx_root, "--host", host, "--port", str(port),
                "--parent-pid", str(os.getpid())] + (extra or [])
        self._proc.start(python, args)

    def stop(self, grace_ms: int = 5000) -> None:
        """Wait for a requested shutdown, then end the whole process tree.

        venv launchers on Windows run the real interpreter as a child, so
        killing only the launcher would leave the server running.
        """
        if not self.running:
            return
        # Keep Qt's event loop turning so queued socket writes (the shutdown request) go out.
        timer = QtCore.QElapsedTimer()
        timer.start()
        while self.running and timer.elapsed() < grace_ms:
            QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
            self._proc.waitForFinished(30)
        if not self.running:
            return
        pid = self._proc.processId()
        if os.name == "nt" and pid:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            self._proc.kill()
        self._proc.waitForFinished(2000)

    def _on_output(self) -> None:
        text = bytes(self._proc.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            if line.startswith("GEMX_LIVE_READY"):
                host, port = line.split()[1].rsplit(":", 1)
                self.ready.emit(host, int(port))
            elif line.strip():
                self.output.emit(line)
