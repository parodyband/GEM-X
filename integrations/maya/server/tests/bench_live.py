"""Run a video through the live pipeline and report per-frame timing."""
import argparse, json, os, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gemx_native as gx

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--gemx", default=str(Path(__file__).resolve().parents[4] / "third_party" / "gem-x.cpp"))
ap.add_argument("--precision", choices=["fast", "strict"], default="fast")
ap.add_argument("--frames", type=int, default=150)
ap.add_argument("--detect", type=int, default=5)
ap.add_argument("--selection", choices=["parity", "continuity"], default="parity")
ap.add_argument("--track", action="store_true", help="keypoint tracking with caller boxes")
ap.add_argument("--dump", default=None, help="write definition + poses to this .npz/.json prefix")
a = ap.parse_args()
if a.precision == "strict":
    for k in gx.STRICT_ENV: os.environ[k] = "1"
root = Path(a.gemx); models = root / "generated/reference"
build = next(d for d in (root / "build/win-vulkan", root / "build/vulkan") if (d / "gemx.dll").exists())
lib = gx.load_library(build)
t0 = time.perf_counter()
pipe = gx.LivePipeline(lib, str(models/"gem-x-contact-f32.gguf"), str(models/"vitpose-f32.gguf"), str(models/"yolox-f32.gguf"),
    gx.default_module(build, "Vulkan"), detect_interval=a.detect,
    precision=gx.STRICT_F32 if a.precision == "strict" else gx.BACKEND_DEFAULT,
    selection=gx.PARITY if a.selection == "parity" else gx.CONTINUITY)
print(f"load {time.perf_counter()-t0:.2f}s  device: {pipe.definition.get('config',{})}")
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30
times, outs, poses = [], {}, []
from gemx_live_server import KeypointTracker
tracker = KeypointTracker()
for i in range(a.frames):
    ok, bgr = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    box = tracker.box if a.track else None
    t = time.perf_counter(); f = pipe.submit(rgb, i + 1, int(i * 1e6 / fps), box=box, subject_id=1 if box is not None else 0); dt = time.perf_counter() - t
    tracker.update(f.keypoints, rgb.shape[1], rgb.shape[0])
    times.append(dt); outs[f.outcome_name] = outs.get(f.outcome_name, 0) + 1
    if f.outcome == gx.POSE: poses.append(f)
print(f"{len(times)} frames {rgb.shape[1]}x{rgb.shape[0]}  outcomes {outs}")
ts = np.array(times[10:]) * 1000
print(f"ms/frame  mean {ts.mean():.1f}  p50 {np.median(ts):.1f}  p95 {np.percentile(ts,95):.1f}  -> {1000/ts.mean():.1f} fps")
if poses: print("metrics(last):", np.round(poses[-1].metrics, 2).tolist())
if a.dump:
    Path(a.dump + "_definition.json").write_text(json.dumps(pipe.definition, indent=1))
    np.savez(a.dump + "_poses.npz", rot=np.stack([p.rotations for p in poses]), trans=np.stack([p.translations for p in poses]),
        pos=np.stack([p.positions for p in poses]), root_aa=np.stack([p.root_axis_angle for p in poses]),
        disp=np.stack([p.root_displacement if p.root_displacement is not None else np.zeros(3) for p in poses]),
        anchor=np.stack([p.smpl_anchor for p in poses]), t=np.array([p.source_time_us for p in poses]))
    print("dumped", a.dump)
pipe.close()
