"""Assemble the Windows release bundle for GEM-X Live.

    python tools/make_release.py            # -> dist/GEMX-Live-Maya-win64.zip

Needs a Vulkan build of gem-x.cpp (third_party/gem-x.cpp/scripts/build_windows.bat).
The models are not bundled (4 GB); the installer downloads them.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAYA_DIR = HERE.parent  # integrations/maya
REPO = MAYA_DIR.parents[1]  # GEM-X checkout
GEMX = REPO / "third_party" / "gem-x.cpp"
NAME = "GEMX-Live"
ASSET = "GEMX-Live-Maya-win64.zip"
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "site-packages")


def vc_redist_dlls() -> list[Path]:
    """App-local copies of the MSVC runtime the binaries link against."""
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    vs = subprocess.check_output([str(vswhere), "-latest", "-property", "installationPath"], text=True).strip()
    redist = sorted(glob.glob(os.path.join(vs, "VC", "Redist", "MSVC", "14.*", "x64")))[-1]
    names = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll")
    found = [Path(p) for p in glob.glob(os.path.join(redist, "Microsoft.VC*.*", "*.dll")) if Path(p).name.lower() in names]
    missing = set(names) - {p.name.lower() for p in found}
    if missing:
        raise SystemExit(f"MSVC redistributable DLLs not found: {sorted(missing)}")
    return found


def build(out: Path) -> Path:
    build_dir = GEMX / "build" / "win-vulkan"
    if not (build_dir / "gemx.dll").is_file():
        raise SystemExit(f"build gem-x.cpp first: {GEMX / 'scripts' / 'build_windows.bat'}")
    stage = out / NAME
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for f in ("README.md", "install.py", "install.ps1", "install.bat"):
        shutil.copy2(MAYA_DIR / f, stage / f)
    shutil.copy2(REPO / "LICENSE", stage / "LICENSE")
    shutil.copytree(MAYA_DIR / "module", stage / "module", ignore=SKIP)
    (stage / "server").mkdir()
    for f in ("gemx_live_server.py", "gemx_native.py", "fetch_models.py", "requirements.txt"):
        shutil.copy2(MAYA_DIR / "server" / f, stage / "server" / f)

    runtime = stage / "runtime"
    (runtime / "bin").mkdir(parents=True)
    shutil.copy2(build_dir / "gemx.dll", runtime / "gemx.dll")
    for dll in sorted((build_dir / "bin").glob("*.dll")):
        shutil.copy2(dll, runtime / "bin" / dll.name)
    for dll in vc_redist_dlls():
        shutil.copy2(dll, runtime / dll.name)
    licenses = runtime / "licenses"
    shutil.copytree(GEMX / "LICENSES", licenses)
    shutil.copy2(GEMX / "LICENSE", licenses / "gem-x.cpp-LICENSE.txt")
    shutil.copy2(GEMX / "NOTICE", licenses / "gem-x.cpp-NOTICE.txt")
    shutil.copy2(GEMX / "ggml" / "LICENSE", licenses / "GGML-MIT.txt")
    (runtime / "models").mkdir()
    (runtime / "models" / "README.txt").write_text(
        "The installer (or Download Models in the GEM-X Live window) puts the GEM-X GGUF models here.\n"
        "Source: https://huggingface.co/LocalAI-io/GEM-X-GGUF (SHA-256 checked).\n"
    )

    zip_path = out / ASSET
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out).as_posix())
    shutil.copy2(MAYA_DIR / "install.ps1", out / "install.ps1")  # released next to the zip
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=MAYA_DIR / "dist")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    zip_path = build(a.out)
    print(f"{zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"{a.out / 'install.ps1'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
