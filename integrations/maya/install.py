"""Install the GEM-X Live Maya module.

Drag this file into a Maya viewport, or run it with any Python (mayapy works):

    python install.py                 # writes <Documents>/maya/modules/gemx_live.mod
    python install.py --uninstall

The .mod file points Maya at ./module. On the next start Maya adds the
gemx_live package, the gemxLive plug-in (auto-loaded by the module's
userSetup.py) and its icons. When dropped into a running Maya, the plug-in is
loaded right away.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODULE_NAME = "GEMXLive"
MODULE_VERSION = "1.0.0"
MOD_FILE = "gemx_live.mod"
HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE / "module"


def documents_dir() -> Path:
    """The user's Documents folder, following OneDrive/known-folder redirection."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD), ("d4", ctypes.c_ubyte * 8)]

        # FOLDERID_Documents {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
        fid = GUID(0xFDD39AD0, 0x238F, 0x46AF, (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7))
        path = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(fid), 0, None, ctypes.byref(path)) == 0:
            try:
                return Path(path.value)
            finally:
                ctypes.windll.ole32.CoTaskMemFree(path)
        return Path(os.environ.get("USERPROFILE", Path.home())) / "Documents"
    return Path.home()


def maya_app_dir() -> Path:
    env = os.environ.get("MAYA_APP_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        return documents_dir() / "maya"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Preferences" / "Autodesk" / "maya"
    return Path.home() / "maya"


def mod_text() -> str:
    # Maya 2025+ (Python 3, PySide6). Paths below are relative to the module root.
    return (
        f"+ {MODULE_NAME} {MODULE_VERSION} {MODULE_ROOT.as_posix()}\n"
        "PYTHONPATH +:= scripts\n"
        "MAYA_PLUG_IN_PATH +:= plug-ins\n"
        "XBMLANGPATH +:= icons\n"
    )


def install(modules_dir: Path | None = None) -> Path:
    modules_dir = modules_dir or maya_app_dir() / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    path = modules_dir / MOD_FILE
    path.write_text(mod_text())
    return path


def uninstall(modules_dir: Path | None = None) -> bool:
    path = (modules_dir or maya_app_dir() / "modules") / MOD_FILE
    if path.exists():
        path.unlink()
        return True
    return False


def onMayaDroppedPythonFile(*_args):
    """Maya calls this when the file is dropped into a viewport."""
    import maya.cmds as cmds

    path = install(Path(cmds.internalVar(userAppDir=True)) / "modules")
    scripts = str(MODULE_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.append(scripts)
    plugin = str(MODULE_ROOT / "plug-ins" / "gemxLive.py")
    if not cmds.pluginInfo("gemxLive", q=True, loaded=True):
        cmds.loadPlugin(plugin)
    cmds.pluginInfo(plugin, edit=True, autoload=True)
    cmds.gemxLive()
    cmds.confirmDialog(
        title="GEM-X Live",
        message=(
            f"Installed.\n\nModule file: {path}\n\n"
            "If the Capture tab says the models are missing, press Download Models (about 4 GB, one time)."
        ),
        button=["OK"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Install the GEM-X Live Maya module")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--modules-dir", type=Path, default=None, help="default: <Documents>/maya/modules")
    a = ap.parse_args()
    if a.uninstall:
        print("removed" if uninstall(a.modules_dir) else "not installed")
    else:
        print(f"wrote {install(a.modules_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
