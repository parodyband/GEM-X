"""User preferences stored as Maya optionVars (prefixed gemxLive_)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import maya.cmds as cmds

PREFIX = "gemxLive_"
MODULE_ROOT = Path(__file__).resolve().parents[2]  # .../module
SERVER_DIR = MODULE_ROOT.parent / "server"
RUNTIME_DIR = MODULE_ROOT.parent / "runtime"  # present in release bundles
MODEL_FILES = ("gem-x-contact-f32.gguf", "vitpose-f32.gguf", "yolox-f32.gguf")
MODELS_SIZE_GB = 4.0

DEFAULTS = {
    "host": "127.0.0.1",
    "port": 47811,
    "serverPython": "",  # empty = automatic (see server_python)
    "serverScript": "",
    "gemxRoot": "",
    "sourceKind": "camera",
    "camera": 0,
    "resolution": "1280x720",
    "videoPath": "",
    "videoMode": "realtime",
    "detectInterval": 5,
    "tracking": "keypoints",
    "flipTest": 0,
    "smoothPlayback": 1,
    "mirror": 0,
    "preview": 1,
    "driveLive": 1,
    "showSource": 0,
    "countdown": 3,
    "duration": 0.0,
    "autoApply": 1,
    "startAtCurrent": 1,
    "startFrame": 1.0,
}


def get(key: str):
    name = PREFIX + key
    default = DEFAULTS[key]
    if not cmds.optionVar(exists=name):
        return default
    value = cmds.optionVar(q=name)
    if isinstance(default, bool) or (isinstance(default, int) and not isinstance(default, bool)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return value


def set(key: str, value) -> None:  # noqa: A001 - mirrors optionVar semantics
    name = PREFIX + key
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        cmds.optionVar(intValue=(name, value))
    elif isinstance(value, float):
        cmds.optionVar(floatValue=(name, value))
    else:
        cmds.optionVar(stringValue=(name, str(value)))


# ---------------------------------------------------------------- resolved paths
# An explicit preference wins; otherwise a development checkout (server/.venv,
# third_party/gem-x.cpp) or a release bundle (runtime/, Maya's own mayapy).


def mayapy() -> str:
    location = os.environ.get("MAYA_LOCATION") or str(Path(sys.executable).resolve().parents[1])
    return str(Path(location) / "bin" / ("mayapy.exe" if os.name == "nt" else "mayapy"))


def server_python() -> str:
    explicit = get("serverPython")
    if explicit and os.path.isfile(explicit):
        return explicit
    venv = SERVER_DIR / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(venv) if venv.is_file() else mayapy()


def server_script() -> str:
    explicit = get("serverScript")
    return explicit if explicit and os.path.isfile(explicit) else str(SERVER_DIR / "gemx_live_server.py")


def gemx_root() -> str:
    explicit = get("gemxRoot") or os.environ.get("GEMX_CPP_ROOT", "")
    if explicit and os.path.isdir(explicit):
        return explicit
    if (RUNTIME_DIR / "gemx.dll").is_file():
        return str(RUNTIME_DIR)
    return str(MODULE_ROOT.parents[2] / "third_party" / "gem-x.cpp")


def models_dir() -> Path:
    root = Path(gemx_root())
    for d in (root / "models", root / "generated" / "reference"):
        if (d / MODEL_FILES[0]).is_file():
            return d
    return root / "models" if (root / "gemx.dll").is_file() else root / "generated" / "reference"


def missing_models() -> list[str]:
    d = models_dir()
    return [name for name in MODEL_FILES if not (d / name).is_file()]


def takes_folder() -> Path:
    root = cmds.workspace(q=True, rootDirectory=True) or os.path.expanduser("~")
    data = cmds.workspace(fileRuleEntry="data") or "data"
    return Path(root) / data / "gemx_takes"


def mappings_folder() -> Path:
    return Path(cmds.internalVar(userPrefDir=True)) / "gemx_live" / "mappings"
