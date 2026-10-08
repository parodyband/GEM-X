# GEM-X Live for Maya

Markerless motion capture from a webcam or video, streamed live onto any
skeleton in Maya and recorded as keyframes.

```
webcam / video ──► capture server ──────────────► Maya plug-in
                   gem-x.cpp (C++/Vulkan)  TCP     map · mask · drive live (60 Hz) · record · key
                   ViTPose → GEM, YOLOX            SOMA-77 poses, 30-frame window
                   only to find the person         + camera preview at camera rate
```

- **Live:** about 25–28 poses/s from your webcam on an RTX 4070 Ti SUPER,
  shown smoothly at display rate. The camera preview streams at camera rate.
- **Any skeleton:** auto-mapping for Mixamo / HumanIK, Unreal 5 and 4, Character
  Creator 3/4, Advanced Skeleton and SOMA names. Unknown rigs fall back to
  name and hierarchy heuristics. Every joint can be remapped by hand.
- **Masking:** turn off any joint or group (spine, head, arms, fingers, legs,
  root) for both live drive and keying.
- **Recording:** countdown, fixed or open-ended takes, and a take library. Takes
  store the raw source motion, so you can key them again after changing the
  mapping or mask.
- **Standard Maya tooling:** Plug-in Manager plug-in, `.mod` module, GEM-X menu,
  dockable workspace window, optionVar preferences, an undoable
  `gemxBakeTake` command, and the character setup saved in the scene.

## Install (Windows)

You need Maya 2025 or later and a GPU with a Vulkan driver (any recent NVIDIA,
AMD or Intel card, ~6 GB free VRAM). No compiler, SDK or separate Python is
needed.

**One line.** Paste into PowerShell:

```powershell
irm https://github.com/parodyband/GEM-X/releases/latest/download/install.ps1 | iex
```

It installs to `%LOCALAPPDATA%\GEMX-Live`, adds OpenCV to Maya's own Python,
downloads the models (~4 GB, one time, SHA-256 checked) and registers the Maya
module. Start Maya: the **GEM-X** menu is there. Open **Live Capture**, press
**Launch Server**, then **Start**.

**Or download the zip:** get
[GEMX-Live-Maya-win64.zip](https://github.com/parodyband/GEM-X/releases/latest/download/GEMX-Live-Maya-win64.zip),
extract it anywhere and double-click `install.bat`. You can also drag
`install.py` into a Maya viewport, then press **Download Models** in the window.

<details>
<summary>Installer options and uninstalling</summary>

```powershell
& ([scriptblock]::Create((irm https://github.com/parodyband/GEM-X/releases/latest/download/install.ps1))) -InstallDir D:\Tools\GEMX-Live -SkipModels
```

| option | |
|---|---|
| `-InstallDir` | where to install (default `%LOCALAPPDATA%\GEMX-Live`) |
| `-SkipModels` | skip the 4 GB download; use **Download Models** in the window later |
| `-Mayapy` | a specific `mayapy.exe` (default: the newest Maya 2025+) |

Running the installer again updates in place and keeps your downloaded models.

To uninstall, delete the install folder and `Documents\maya\modules\gemx_live.mod`.
</details>

## Build from source

Requirements: Visual Studio 2022 C++ tools, CMake 3.24+, Ninja, the
[Vulkan SDK](https://vulkan.lunarg.com/), Git and Python 3.10+.

```bat
git clone --recursive https://github.com/parodyband/GEM-X.git
cd GEM-X\integrations\maya
setup_windows.bat
```

`setup_windows.bat`:

- builds [gem-x.cpp](https://github.com/parodyband/gem-x.cpp) from
  `third_party/gem-x.cpp` into `build/win-vulkan`;
- creates the server's virtualenv;
- downloads the models from
  [LocalAI-io/GEM-X-GGUF](https://huggingface.co/LocalAI-io/GEM-X-GGUF) and
  checks their SHA-256;
- writes `gemx_live.mod` into your Maya modules folder.

The module's `userSetup.py` loads the plug-in when Maya starts.
`tools/make_release.py` builds the release zip from a source build.

## Use

Open **GEM-X → Live Capture...**

### 1. Capture

1. **Launch Server.** This starts the capture server and loads the models
   (~2 s). If you run the server yourself, use **Connect** instead.
2. Choose **Camera** (index and resolution) or **Video**. *Real-time* plays
   the clip like a camera; *Every frame* processes all of it.
3. **Start.** The preview shows the tracked person and their 2D keypoints.

Keep the camera still and your whole body in frame. *Mirror* makes the
character copy you like a mirror.

- **Tracking → Follow keypoints** (default): the person detector finds you
  once. After that, each frame's crop comes from your own keypoints, so the
  ~40 ms detector stops running. If tracking is lost, it re-detects.
  *Detector every N frames* is the slower alternative.
- **Steadier keypoints (flip test)** averages each pose with a mirrored pass.
  It is slightly steadier and about half the speed. This applies the next time
  the server launches.

### 2. Character

1. Pose your character in its rest pose. T-pose and A-pose both work.
2. Select any of its joints and press **Use Selected**. The skeleton is
   auto-mapped; the info line shows which preset matched.
3. Check the **Source (SOMA)** column. Change any joint's source from its
   drop-down, or right-click for *Unmap selected*.
4. **Drive** checkboxes and the group buttons mask joints. Masked joints keep
   their own values, and their children keep moving relative to them.
5. **Root Motion:**
   - *Horizontal travel* and *Vertical* toggle the two components.
   - *Floor contact* keeps the lowest foot on the ground; *Trajectory* follows
     the estimated path, which can drift.
   - *Auto scale* scales travel by hip height.
   - *Facing* rotates the performance about the up axis.
6. **Drive character** poses the rig live. Turning it off, or closing the
   window, puts the rig back as it was. **Smooth playback** interpolates
   between poses at display rate, adding about one pose (~40 ms) of delay;
   turn it off to see each pose as it arrives. **Recenter** makes your
   current position the character's start position.

The mapping, mask, options and rest pose are saved on the skeleton's root joint
(`gemxLiveConfig` attribute) and come back with the scene. Use
**Presets → Save Mapping...** to reuse a mapping on other characters.

### 3. Record

1. Set the take name, countdown and duration (0 = until stopped).
2. **Record.** Recording starts after the countdown; press **Stop**, or let
   the duration end it.
3. With *Key the character when recording stops* on, the take is retargeted
   and keyed at the start frame, and live drive pauses so you see the result.
   One **Undo** removes it.

Takes are saved to `<project>/data/gemx_takes/*.gemxtake.npz`. Select a take
and press **Key Character** to key it again, for example after changing the
mapping or onto another character.

### Scripting

```python
import maya.cmds as cmds
cmds.loadPlugin("gemxLive.py")
cmds.gemxLive()                                   # open the window
cmds.gemxBakeTake(take="C:/proj/data/gemx_takes/Take_001.gemxtake.npz",
                  root="mixamorig:Hips", start=1)  # undoable; returns the key count
```

`gemxBakeTake` uses the mapping, mask and root options stored on the
character, so set the character up once in the window first.

## How the retarget works

Every source joint's world rotation is transferred as a world-space delta from
its rest pose:

```
W_target(t) = F · G_source(t) · G_source_restᵀ · Fᵀ · A_jᵀ · T_target_rest
```

- `F` aligns the performer's frame with the character's (up axis, facing),
  measured from the hips.
- `A_j` swings each source bone onto the matching target bone in the two rest
  poses, so T-pose versus A-pose differences cancel out.

Target joints with no mapping keep their rest local rotation, so extra twist
or spine joints are fine. Rotations are written through `jointOrient` and
`rotateAxis` in each joint's own rotate order, then Euler-filtered. The tests
check bone directions on an A-pose Mixamo rig with mixed rotate orders and
rotate axes, and the error is zero.

## Capture server

`server/gemx_live_server.py` wraps the gem-x.cpp live API
(`include/gemx_stream.h`) with ctypes and serves newline-delimited JSON over
TCP (default `127.0.0.1:47811`):

| client → server | |
|---|---|
| `{"cmd":"start","source":"camera","camera":0,"width":1280,"height":720}` | webcam |
| `{"cmd":"start","source":"video","path":"clip.mp4","mode":"realtime"\|"all"}` | file |
| `{"cmd":"stop"}`, `{"cmd":"reset"}`, `{"cmd":"shutdown"}` | |
| `{"cmd":"configure","detect_interval":5,"mirror":false,"preview":true}` | |

| server → client | |
|---|---|
| `hello` | protocol version, SOMA-77 skeleton (names, parents, rest pose), device |
| `frame` | `outcome`, `t` (s), `rot` 77×4 parent-local XYZW, `trans` 77×3, `root` (integrated, Y-up m), fps, latency |
| `preview` | JPEG of the camera frame with the tracked box and keypoints |
| `state` | running / source / errors |

SOMA space is right-handed, Y-up and in metres; a performer facing the camera
faces +Z. Any client can use the stream; the Maya plug-in is one example.

## Performance

RTX 4070 Ti SUPER, Vulkan, 812×720 input:

| mode | per pose | rate |
|---|---|---|
| **follow keypoints, no flip (default)** | **~37 ms** | **~27 poses/s** |
| follow keypoints, flip test | ~55 ms | ~18 poses/s |
| detector every frame, flip test | ~85–120 ms | 8–11 poses/s |

Stage costs:

- **ViTPose** (DINOv3 ViT-H): ~25 ms single view, ~47 ms with the flip test.
- **YOLOX-X:** ~41 ms, run only to find the person.
- **GEM:** ~4 ms.

In Maya, retargeting costs ~1.7 ms per pose and writing a 65-joint pose
~0.4 ms. A TensorRT FP16 ViTPose would be the next big speed-up.

## Tests

From `integrations/maya`, with Maya's Python:

```bat
set MAYAPY="C:\Program Files\Autodesk\Maya2027\bin\mayapy.exe"
set GEMX_TEST_VIDEO=C:\path\to\one_person.mp4
%MAYAPY% tests\test_core.py      :: rotation math, mapping, retarget accuracy, playback
%MAYAPY% tests\test_live.py      :: plug-in + server + record + undoable bake
%MAYAPY% tests\test_ui.py        :: the window, headless, against a live server
%MAYAPY% tests\test_bundle.py %LOCALAPPDATA%\GEMX-Live %GEMX_TEST_VIDEO%   :: an installed release
```

`test_live.py` and `test_ui.py` need a built server and a clip of one person
with their full body in frame. They skip if `GEMX_TEST_VIDEO` is not set.

## Licenses

The integration code is Apache-2.0, like the rest of this repository.

The models are not:

- **GEM-X** weights are under the
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
  which allows commercial use. Keep its attribution notice.
- **ViTPose** (a DINOv3 ViT-H backbone) is under the
  [DINOv3 License](../../third_party/gem-x.cpp/LICENSES/DINOv3.md). It is
  royalty-free and allows commercial use. Redistributing the weights requires
  shipping the agreement with them. Section 1.b.v forbids trade-control
  restricted end uses, and its wording includes military uses and "the
  development or use of guns or illegal weapons". Read it for your own use case.
- **YOLOX-X** is Apache-2.0.

This is a summary, not legal advice; see `third_party/gem-x.cpp/docs/LICENSING.md`
and the full license texts.
