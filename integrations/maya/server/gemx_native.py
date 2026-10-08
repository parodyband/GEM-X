"""ctypes binding for the gem-x.cpp live streaming ABI (include/gemx_stream.h).

Only the live (absent-image) pipeline is wrapped. Every call is synchronous;
ctypes releases the GIL for its duration, so a capture thread keeps running
while the GPU works.
"""

from __future__ import annotations

import ctypes as C
import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# gemx_status
OK = 0

# Selection / precision / temporal policy
PARITY = 0
CONTINUITY = 1
STRICT_F32 = 0
BACKEND_DEFAULT = 1
FRAME_INDEX = 0

# Outcomes
WARMUP = 0
POSE = 1
LOST = 2
AMBIGUOUS = 3
OUTCOME_NAMES = {WARMUP: "warmup", POSE: "pose", LOST: "lost", AMBIGUOUS: "ambiguous"}

# Flags
FLAG_RESET = 1
FLAG_CROP_REUSED = 2
FLAG_FULL_IMAGE = 4
FLAG_DETECTOR_RAN = 8
FLAG_CALLER_BOX = 16

# Result float channels
POSITIONS = 0
ROTATIONS = 1
TRANSLATIONS = 2
SMPL_JOINTS = 3
SMPL_ANCHOR = 4
ROOT_AXIS_ANGLE = 5
ROOT_TRANSLATION = 6
CAMERA_POSITIONS = 7
CAMERA_TRANSLATION = 8
KEYPOINTS = 9
BOX = 10
METRICS = 11
IDENTITY = 12
SCALES = 13
ROOT_DISPLACEMENT = 14

STRICT_ENV = ("GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2")

_ERR_CAP = 1024


class GemxError(RuntimeError):
    pass


class _Live(C.Structure):
    pass


class _Result(C.Structure):
    pass


_LiveP = C.POINTER(_Live)
_ResultP = C.POINTER(_Result)


def _bind(lib: C.CDLL) -> None:
    u32, u64, i64 = C.c_uint32, C.c_uint64, C.c_int64
    cstr, err = C.c_char_p, C.c_char_p

    lib.gemx_live_create.argtypes = [
        cstr, cstr, cstr, cstr, cstr, u32, cstr,
        u32, u32, u32, u32, u32, u32, i64,
        C.POINTER(_LiveP), err, u64,
    ]
    lib.gemx_live_destroy.argtypes = [_LiveP]
    lib.gemx_live_destroy.restype = None
    lib.gemx_live_definition.argtypes = [_LiveP, C.c_char_p, u64, C.POINTER(u64), err, u64]
    lib.gemx_live_reset.argtypes = [_LiveP, err, u64]
    lib.gemx_live_submit.argtypes = [
        _LiveP, C.c_void_p, u64, u32, u32, u64, u64, i64,
        C.POINTER(C.c_float), u64, u64, C.POINTER(_ResultP), err, u64,
    ]
    lib.gemx_live_result_destroy.argtypes = [_ResultP]
    lib.gemx_live_result_destroy.restype = None
    lib.gemx_live_result_info.argtypes = [
        _ResultP, C.POINTER(u64), C.POINTER(i64), C.POINTER(u64), C.POINTER(u64),
        C.POINTER(u32), C.POINTER(u32), err, u64,
    ]
    lib.gemx_live_result_interval_start.argtypes = [_ResultP, C.POINTER(i64), err, u64]
    lib.gemx_live_result_copy.argtypes = [_ResultP, u32, C.POINTER(C.c_float), u64, C.POINTER(u64), err, u64]
    for name in (
        "gemx_live_create", "gemx_live_definition", "gemx_live_reset", "gemx_live_submit",
        "gemx_live_result_info", "gemx_live_result_interval_start", "gemx_live_result_copy",
    ):
        getattr(lib, name).restype = C.c_int


def load_library(build_dir: str | os.PathLike) -> C.CDLL:
    """Load gemx.dll / libgemx.so from a gem-x.cpp build directory."""
    build_dir = Path(build_dir).resolve()
    if os.name == "nt":
        for d in (build_dir, build_dir / "bin"):
            if d.is_dir():
                os.add_dll_directory(str(d))
        lib = C.CDLL(str(build_dir / "gemx.dll"))
    else:
        lib = C.CDLL(str(build_dir / "libgemx.so"))
    _bind(lib)
    return lib


def default_module(build_dir: str | os.PathLike, backend: str) -> str:
    build_dir = Path(build_dir).resolve()
    if backend == "CPU":
        return str(build_dir / "bin")
    name = "ggml-vulkan.dll" if os.name == "nt" else "libggml-vulkan.so"
    return str(build_dir / "bin" / name)


@dataclass
class Frame:
    """One decoded live result. Arrays are None unless outcome == POSE."""

    sequence: int
    source_time_us: int
    interval_start_us: int | None
    epoch: int
    track_epoch: int
    outcome: int
    flags: int
    rotations: np.ndarray | None = None  # (77, 4) parent-local XYZW
    translations: np.ndarray | None = None  # (77, 3) parent-local metres
    positions: np.ndarray | None = None  # (77, 3) Y-up, pelvis at origin
    root_axis_angle: np.ndarray | None = None  # (3,)
    root_displacement: np.ndarray | None = None  # (3,) metres, SMPL anchor-local, Z-up
    smpl_anchor: np.ndarray | None = None  # (4,) WXYZ
    keypoints: np.ndarray | None = None  # (77, 3) pixel x, y, confidence
    box: np.ndarray | None = None  # (4,) xyxy
    metrics: np.ndarray | None = None

    @property
    def outcome_name(self) -> str:
        return OUTCOME_NAMES.get(self.outcome, str(self.outcome))


class LivePipeline:
    """Resident detector + ViTPose + GEM denoiser with a rolling window."""

    def __init__(
        self,
        lib: C.CDLL,
        gem_model: str,
        pose_model: str,
        detector_model: str,
        module: str,
        backend: str = "Vulkan",
        device: int = 0,
        threads: int = 8,
        window: int = 30,
        detect_interval: int = 5,
        selection: int = CONTINUITY,
        precision: int = BACKEND_DEFAULT,
        max_gap_us: int = 2_000_000,
    ):
        self._lib = lib
        self._err = C.create_string_buffer(_ERR_CAP)
        handle = _LiveP()
        self._check(
            lib.gemx_live_create(
                gem_model.encode(), pose_model.encode(), detector_model.encode(),
                module.encode(), backend.encode(), device, None,
                threads, window, detect_interval, selection, precision, FRAME_INDEX, max_gap_us,
                C.byref(handle), self._err, _ERR_CAP,
            )
        )
        self._handle = handle
        self.definition = self._read_definition()

    def _check(self, status: int) -> None:
        if status != OK:
            raise GemxError(self._err.value.decode(errors="replace") or f"gemx status {status}")

    def _read_definition(self) -> dict:
        required = C.c_uint64()
        self._check(self._lib.gemx_live_definition(self._handle, None, 0, C.byref(required), self._err, _ERR_CAP))
        buf = C.create_string_buffer(required.value)
        self._check(
            self._lib.gemx_live_definition(self._handle, buf, required.value, C.byref(required), self._err, _ERR_CAP)
        )
        return json.loads(buf.value.decode())

    def reset(self) -> None:
        self._check(self._lib.gemx_live_reset(self._handle, self._err, _ERR_CAP))

    def submit(self, rgb: np.ndarray, sequence: int, source_time_us: int,
               box: np.ndarray | None = None, subject_id: int = 0) -> Frame:
        """Run one RGB frame (H, W, 3 uint8, RGB order) through the pipeline.

        With ``box`` (xyxy pixels) the person detector is skipped and that box
        is used as the crop. Switching between boxed and automatic frames, or
        changing ``subject_id``, starts a new temporal context.
        """
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected an (H, W, 3) uint8 RGB image")
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        box_ptr, box_count = None, 0
        if box is not None:
            box_arr = np.ascontiguousarray(box, dtype=np.float32).reshape(4)
            box_ptr, box_count = box_arr.ctypes.data_as(C.POINTER(C.c_float)), 4
        result = _ResultP()
        self._check(
            self._lib.gemx_live_submit(
                self._handle, rgb.ctypes.data, rgb.nbytes, w, h, rgb.strides[0],
                sequence, source_time_us, box_ptr, box_count, subject_id, C.byref(result), self._err, _ERR_CAP,
            )
        )
        try:
            return self._decode(result)
        finally:
            self._lib.gemx_live_result_destroy(result)

    def _decode(self, result: _ResultP) -> Frame:
        lib, err = self._lib, self._err
        seq, epoch, track = C.c_uint64(), C.c_uint64(), C.c_uint64()
        t_us = C.c_int64()
        outcome, flags = C.c_uint32(), C.c_uint32()
        self._check(
            lib.gemx_live_result_info(
                result, C.byref(seq), C.byref(t_us), C.byref(epoch), C.byref(track),
                C.byref(outcome), C.byref(flags), err, _ERR_CAP,
            )
        )
        frame = Frame(seq.value, t_us.value, None, epoch.value, track.value, outcome.value, flags.value)
        if outcome.value != POSE:
            # Warm-up frames still carry the 2D observation used for tracking.
            frame.keypoints = self._copy(result, KEYPOINTS, (77, 3))
            frame.box = self._copy(result, BOX, (4,))
            return frame
        start = C.c_int64()
        self._check(lib.gemx_live_result_interval_start(result, C.byref(start), err, _ERR_CAP))
        frame.interval_start_us = start.value

        copy = functools.partial(self._copy, result)
        frame.rotations = copy(ROTATIONS, (77, 4))
        frame.translations = copy(TRANSLATIONS, (77, 3))
        frame.positions = copy(POSITIONS, (77, 3))
        frame.root_axis_angle = copy(ROOT_AXIS_ANGLE, (3,))
        frame.root_displacement = copy(ROOT_DISPLACEMENT, (3,))
        frame.smpl_anchor = copy(SMPL_ANCHOR, (4,))
        frame.keypoints = copy(KEYPOINTS, (77, 3))
        frame.box = copy(BOX, (4,))
        frame.metrics = copy(METRICS, ())
        return frame

    def _copy(self, result: _ResultP, channel: int, shape: tuple[int, ...]) -> np.ndarray | None:
        lib, err = self._lib, self._err
        required = C.c_uint64()
        self._check(lib.gemx_live_result_copy(result, channel, None, 0, C.byref(required), err, _ERR_CAP))
        if required.value == 0:
            return None
        out = np.empty(required.value, dtype=np.float32)
        self._check(
            lib.gemx_live_result_copy(
                result, channel, out.ctypes.data_as(C.POINTER(C.c_float)), out.size,
                C.byref(required), err, _ERR_CAP,
            )
        )
        return out.reshape(shape) if shape else out

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.gemx_live_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "LivePipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
