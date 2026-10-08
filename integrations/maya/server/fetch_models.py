"""Download the gem-x.cpp live models and verify them against SHA256SUMS.

    python fetch_models.py --dest ../../../third_party/gem-x.cpp/generated/reference
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/LocalAI-io/GEM-X-GGUF/resolve/main"
FILES = ("gem-x-contact-f32.gguf", "yolox-f32.gguf", "vitpose-f32.gguf")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, machine: bool) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            done += len(chunk)
            if not total:
                continue
            if machine:
                pct = int(done * 100 / total)
                if pct != shown:
                    print(f"PROGRESS {dest.name} {done} {total}", flush=True)
                    shown = pct
            else:
                print(f"\r  {dest.name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    if not machine:
        print()
    tmp.replace(dest)


def default_dest() -> Path:
    here = Path(__file__).resolve().parent
    runtime = here.parent / "runtime"  # release bundle
    if (runtime / "gemx.dll").is_file():
        return runtime / "models"
    return here.parents[2] / "third_party" / "gem-x.cpp" / "generated" / "reference"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=default_dest())
    ap.add_argument("--machine", action="store_true", help="one PROGRESS line per percent (for the Maya UI)")
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    sums = urllib.request.urlopen(f"{REPO}/SHA256SUMS").read().decode()
    expected = {line.split()[-1].lstrip("*"): line.split()[0] for line in sums.splitlines() if line.strip()}
    ok = True
    for name in FILES:
        path = a.dest / name
        if not path.exists():
            print(f"downloading {name}")
            download(f"{REPO}/{name}", path, a.machine)
        got = sha256(path)
        if got != expected.get(name):
            print(f"{name}: SHA-256 mismatch ({got}); deleting it, run again to re-download")
            path.unlink()
            ok = False
        else:
            print(f"{name}: ok")
    print("DONE" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
