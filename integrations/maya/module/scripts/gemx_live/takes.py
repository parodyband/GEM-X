"""Recorded takes: raw SOMA-77 source motion, saved per take as .npz.

Takes keep the source motion, not retargeted keys, so a take can be applied
again after the mapping, mask or root options change.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import mathutil as mu

EXTENSION = ".gemxtake.npz"


@dataclass
class Take:
    name: str
    times: np.ndarray  # (N,) seconds, strictly increasing
    rotations: np.ndarray  # (N, J, 4) parent-local XYZW
    root: np.ndarray  # (N, 3) integrated root trajectory, Y-up metres
    skeleton: dict  # soma.Skeleton.to_dict()
    created: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def frame_count(self) -> int:
        return len(self.times)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"format": "gemx-live-take/1", "name": self.name, "created": self.created,
                  "meta": self.meta, "skeleton": self.skeleton}
        with open(path, "wb") as f:  # np.savez appends .npz to bare string paths
            np.savez_compressed(
                f, header=np.array(json.dumps(header)), times=self.times.astype(np.float64),
                rotations=self.rotations.astype(np.float32), root=self.root.astype(np.float32),
            )
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Take":
        with np.load(Path(path), allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            return cls(
                name=header["name"], times=z["times"].astype(np.float64),
                rotations=z["rotations"].astype(np.float64), root=z["root"].astype(np.float64),
                skeleton=header["skeleton"], created=header.get("created", ""), meta=header.get("meta", {}),
                path=Path(path),
            )

    def resample(self, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample at a fixed rate from the first pose. Returns (times, rotations, root)."""
        t0, t1 = self.times[0], self.times[-1]
        count = int(np.floor((t1 - t0) * fps + 1e-6)) + 1
        t = t0 + np.arange(count) / fps
        i = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 1)
        i1 = np.minimum(i + 1, len(self.times) - 1)
        span = self.times[i1] - self.times[i]
        a = np.where(span > 0, (t - self.times[i]) / np.where(span > 0, span, 1), 0.0)
        rot = mu.slerp_xyzw(self.rotations[i], self.rotations[i1], a[:, None])
        root = self.root[i] + (self.root[i1] - self.root[i]) * a[:, None]
        return t, rot, root


class Recorder:
    """Accumulates pose frames while recording."""

    def __init__(self):
        self.times: list[float] = []
        self.rotations: list[np.ndarray] = []
        self.root: list[np.ndarray] = []

    def add(self, t: float, rotations: np.ndarray, root: np.ndarray) -> None:
        if self.times and t <= self.times[-1]:
            return  # duplicates / restarts
        self.times.append(float(t))
        self.rotations.append(np.asarray(rotations, dtype=np.float32).reshape(-1, 4))
        self.root.append(np.asarray(root, dtype=np.float32).reshape(3))

    def __len__(self) -> int:
        return len(self.times)

    def to_take(self, name: str, skeleton: dict, meta: dict | None = None) -> Take:
        return Take(
            name=name, times=np.array(self.times), rotations=np.stack(self.rotations),
            root=np.stack(self.root), skeleton=skeleton, meta=meta or {},
        )


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_") or "Take"


def next_take_name(folder: Path, base: str = "Take") -> str:
    existing = {p.name[: -len(EXTENSION)] for p in folder.glob("*" + EXTENSION)} if folder.is_dir() else set()
    n = 1
    while f"{base}_{n:03d}" in existing:
        n += 1
    return f"{base}_{n:03d}"


def list_takes(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*" + EXTENSION), key=lambda p: p.stat().st_mtime, reverse=True)
