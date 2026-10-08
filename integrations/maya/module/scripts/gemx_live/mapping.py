"""Source (SOMA-77) -> target joint mapping: presets and auto-mapping.

A mapping is ``{target_joint_short_name: soma_joint_name}``. Target names are
compared without namespaces, so ``mixamorig:Hips`` matches ``Hips``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SIDES = (("Left", "L"), ("Right", "R"))
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def _sided(table: dict[str, str], fmt_src: str, fmt_dst, extra: dict | None = None) -> dict[str, str]:
    """Expand a per-side table. fmt_dst is a callable (side_long, side_short, value)."""
    out = dict(extra or {})
    for long, short in SIDES:
        for src, dst in table.items():
            out[fmt_src.format(side=long, name=src)] = fmt_dst(long, short, dst)
    return out


def _mixamo() -> dict[str, str]:
    body = {"Shoulder": "Shoulder", "Arm": "Arm", "ForeArm": "ForeArm", "Hand": "Hand",
            "Leg": "UpLeg", "Shin": "Leg", "Foot": "Foot", "ToeBase": "ToeBase", "ToeEnd": "Toe_End"}
    m = _sided(body, "{side}{name}", lambda l, s, v: f"{l}{v}", {
        "Hips": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Chest": "Spine2",
        "Neck1": "Neck", "Head": "Head", "HeadEnd": "HeadTop_End",
    })
    for long, _ in SIDES:
        for i, n in enumerate(("1", "2", "3", "End"), start=1):
            m[f"{long}HandThumb{n}"] = f"{long}HandThumb{i}"
        for f in FINGERS[1:]:
            for i, n in enumerate(("2", "3", "4", "End"), start=1):
                m[f"{long}Hand{f}{n}"] = f"{long}Hand{f}{i}"
    return m


def _unreal(ue5: bool) -> dict[str, str]:
    m = {"Hips": "pelvis", "Head": "head", "Neck1": "neck_01"}
    if ue5:
        m.update({"Spine1": "spine_01", "Spine2": "spine_03", "Chest": "spine_05", "Neck2": "neck_02"})
    else:
        m.update({"Spine1": "spine_01", "Spine2": "spine_02", "Chest": "spine_03"})
    for long, short in SIDES:
        s = short.lower()
        m.update({
            f"{long}Shoulder": f"clavicle_{s}", f"{long}Arm": f"upperarm_{s}", f"{long}ForeArm": f"lowerarm_{s}",
            f"{long}Hand": f"hand_{s}", f"{long}Leg": f"thigh_{s}", f"{long}Shin": f"calf_{s}",
            f"{long}Foot": f"foot_{s}", f"{long}ToeBase": f"ball_{s}",
        })
        for i in (1, 2, 3):
            m[f"{long}HandThumb{i}"] = f"thumb_0{i}_{s}"
        for f in FINGERS[1:]:
            low = f.lower()
            if ue5:
                m[f"{long}Hand{f}1"] = f"{low}_metacarpal_{s}"
            for i in (1, 2, 3):
                m[f"{long}Hand{f}{i + 1}"] = f"{low}_0{i}_{s}"
    return m


def _character_creator() -> dict[str, str]:
    m = {"Hips": "CC_Base_Hip", "Spine1": "CC_Base_Waist", "Spine2": "CC_Base_Spine01", "Chest": "CC_Base_Spine02",
         "Neck1": "CC_Base_NeckTwist01", "Neck2": "CC_Base_NeckTwist02", "Head": "CC_Base_Head"}
    cc_finger = {"Thumb": "Thumb", "Index": "Index", "Middle": "Mid", "Ring": "Ring", "Pinky": "Pinky"}
    for long, short in SIDES:
        p = f"CC_Base_{short}_"
        m.update({
            f"{long}Shoulder": p + "Clavicle", f"{long}Arm": p + "Upperarm", f"{long}ForeArm": p + "Forearm",
            f"{long}Hand": p + "Hand", f"{long}Leg": p + "Thigh", f"{long}Shin": p + "Calf",
            f"{long}Foot": p + "Foot", f"{long}ToeBase": p + "ToeBase",
        })
        for i in (1, 2, 3):
            m[f"{long}HandThumb{i}"] = f"{p}Thumb{i}"
        for f in FINGERS[1:]:
            for i in (1, 2, 3):
                m[f"{long}Hand{f}{i + 1}"] = f"{p}{cc_finger[f]}{i}"
    return m


def _advanced_skeleton() -> dict[str, str]:
    m = {"Hips": "Root_M", "Spine1": "Spine1_M", "Spine2": "Spine2_M", "Chest": "Chest_M",
         "Neck1": "Neck_M", "Head": "Head_M", "HeadEnd": "HeadEnd_M"}
    for long, short in SIDES:
        m.update({
            f"{long}Shoulder": f"Scapula_{short}", f"{long}Arm": f"Shoulder_{short}", f"{long}ForeArm": f"Elbow_{short}",
            f"{long}Hand": f"Wrist_{short}", f"{long}Leg": f"Hip_{short}", f"{long}Shin": f"Knee_{short}",
            f"{long}Foot": f"Ankle_{short}", f"{long}ToeBase": f"Toes_{short}", f"{long}ToeEnd": f"ToesEnd_{short}",
        })
        for i in (1, 2, 3):
            m[f"{long}HandThumb{i}"] = f"ThumbFinger{i}_{short}"
        for f in FINGERS[1:]:
            for i in (1, 2, 3):
                m[f"{long}Hand{f}{i + 1}"] = f"{f}Finger{i}_{short}"
    return m


def _soma(names) -> dict[str, str]:
    return {n: n for n in names}


def builtin_presets(source_names) -> dict[str, dict[str, str]]:
    """Preset name -> {soma_joint: target_joint}."""
    return {
        "Mixamo / HumanIK": _mixamo(),
        "Unreal Engine 5 (Manny/Quinn)": _unreal(True),
        "Unreal Engine 4 (Mannequin)": _unreal(False),
        "Character Creator 3/4": _character_creator(),
        "Advanced Skeleton": _advanced_skeleton(),
        "SOMA (GEM-X)": _soma(source_names),
    }


# --------------------------------------------------------------------------- matching


def strip_namespace(name: str) -> str:
    return name.rsplit("|", 1)[-1].rsplit(":", 1)[-1]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", strip_namespace(name).lower())


def _match_targets(candidate: str, targets_norm: list[str]) -> int | None:
    """Index of the target matching a preset name (exact, else short prefix)."""
    c = _norm(candidate)
    best, best_prefix = None, 99
    for i, t in enumerate(targets_norm):
        if t == c:
            return i
        if len(c) >= 4 and t.endswith(c):
            prefix = len(t) - len(c)
            if prefix <= 12 and prefix < best_prefix:
                best, best_prefix = i, prefix
    return best


def apply_preset(preset: dict[str, str], target_names: list[str]) -> dict[int, str]:
    """Map target indices to source names using a {source: target} preset."""
    targets_norm = [_norm(n) for n in target_names]
    out: dict[int, str] = {}
    for src, dst in preset.items():
        i = _match_targets(dst, targets_norm)
        if i is not None and i not in out:
            out[i] = src
    return out


# Heuristic keyword table for unknown rigs: source -> (side, required words, forbidden words).
_HEURISTICS = [
    ("Hips", None, ("hips",), ()),
    ("Hips", None, ("pelvis",), ()),
    ("Head", None, ("head",), ("end", "top", "nub", "twist")),
    ("Shoulder", "side", ("clavicle",), ()),
    ("Shoulder", "side", ("collar",), ()),
    ("Shoulder", "side", ("shoulder",), ()),
    ("ForeArm", "side", ("forearm",), ("twist", "roll")),
    ("ForeArm", "side", ("lowerarm",), ("twist", "roll")),
    ("ForeArm", "side", ("elbow",), ()),
    ("Arm", "side", ("upperarm",), ("twist", "roll")),
    ("Arm", "side", ("arm",), ("fore", "lower", "twist", "roll")),
    ("Hand", "side", ("hand",), ("thumb", "index", "middle", "ring", "pinky", "finger", "end")),
    ("Hand", "side", ("wrist",), ()),
    ("Leg", "side", ("thigh",), ("twist", "roll")),
    ("Leg", "side", ("upleg",), ("twist", "roll")),
    ("Leg", "side", ("upperleg",), ("twist", "roll")),
    ("Shin", "side", ("calf",), ("twist", "roll")),
    ("Shin", "side", ("shin",), ()),
    ("Shin", "side", ("knee",), ()),
    ("Shin", "side", ("lowerleg",), ("twist", "roll")),
    ("Shin", "side", ("leg",), ("up", "upper", "twist", "roll")),
    ("Foot", "side", ("foot",), ("end", "toe")),
    ("Foot", "side", ("ankle",), ()),
    ("ToeBase", "side", ("toe",), ("end", "nub")),
    ("ToeBase", "side", ("ball",), ()),
]


def _tokens(name: str) -> tuple[str | None, str]:
    """Detect side and return (side, normalised name without side markers)."""
    raw = strip_namespace(name)
    spaced = re.sub(r"([a-z])([A-Z])", r"\1_\2", raw)
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", spaced) if p]
    side = None
    keep = []
    for p in parts:
        low = p.lower()
        if low in ("l", "left", "lf", "lft"):
            side = "Left"
        elif low in ("r", "right", "rt", "rgt"):
            side = "Right"
        else:
            keep.append(low)
    return side, "".join(keep)


def heuristic_map(target_names: list[str], parents: list[int], taken: dict[int, str]) -> dict[int, str]:
    out = dict(taken)
    used_src = set(out.values())
    info = [_tokens(n) for n in target_names]
    for src, side_rule, req, forbid in _HEURISTICS:
        for side in (("Left", "Right") if side_rule else (None,)):
            name = f"{side}{src}" if side else src
            if name in used_src:
                continue
            for i, (tside, word) in enumerate(info):
                if i in out or (side_rule and tside != side) or (not side_rule and tside is not None):
                    continue
                if all(r in word for r in req) and not any(f in word for f in forbid):
                    out[i] = name
                    used_src.add(name)
                    break
    _map_chain(out, target_names, parents, "Hips", ("Spine1", "Spine2", "Chest"), stop_words=("neck", "head", "clavicle", "shoulder"))
    _map_chain(out, target_names, parents, "Chest", ("Neck1", "Neck2"), stop_words=("head",), require_word="neck")
    return out


def _map_chain(out, names, parents, start_src, chain_src, stop_words, require_word=None):
    """Distribute a SOMA chain over the target joints between two mapped joints."""
    rev = {v: k for k, v in out.items()}
    if start_src not in rev or any(c in rev for c in chain_src):
        return
    start = rev[start_src]
    children = {}
    for i, p in enumerate(parents):
        children.setdefault(p, []).append(i)
    chain = []
    j = start
    while True:
        kids = [k for k in children.get(j, []) if k not in out]
        kids = [k for k in kids if not any(w in _tokens(names[k])[1] for w in stop_words)]
        if require_word:
            kids = [k for k in kids if require_word in _tokens(names[k])[1]]
        kids = [k for k in kids if _tokens(names[k])[0] is None]
        if not kids:
            break
        j = kids[0]
        chain.append(j)
    if not chain:
        return
    n = len(chain_src)
    if len(chain) >= n:
        picks = [chain[round(k * (len(chain) - 1) / max(n - 1, 1))] for k in range(n)]
        for src, tgt in zip(chain_src, picks):
            out.setdefault(tgt, src)
    else:
        for src, tgt in zip(chain_src[: len(chain) - 1] + chain_src[-1:], chain):
            out.setdefault(tgt, src)


def auto_map(target_names: list[str], parents: list[int], source_names: list[str]) -> tuple[dict[int, str], str]:
    """Best preset plus heuristics. Returns ({target_index: source}, preset name)."""
    best, best_name = {}, "Heuristic"
    for name, preset in builtin_presets(source_names).items():
        m = apply_preset(preset, target_names)
        if len(m) > len(best):
            best, best_name = m, name
    if len(best) < 5:
        best, best_name = {}, "Heuristic"
    return heuristic_map(target_names, parents, best), best_name


# --------------------------------------------------------------------------- user presets


def save_preset(path: str | Path, mapping: dict[str, str], masked: list[str], name: str = "") -> None:
    data = {"format": "gemx-live-mapping/1", "name": name or Path(path).stem, "mapping": mapping, "masked": sorted(masked)}
    Path(path).write_text(json.dumps(data, indent=2))


def load_preset(path: str | Path) -> tuple[dict[str, str], list[str]]:
    data = json.loads(Path(path).read_text())
    return dict(data.get("mapping", {})), list(data.get("masked", []))


def remap_by_short_name(saved: dict[str, str], target_names: list[str]) -> dict[int, str]:
    """Apply a saved {target_short: source} mapping to the current joints."""
    by_name = {strip_namespace(n): i for i, n in enumerate(target_names)}
    return {by_name[t]: s for t, s in saved.items() if t in by_name}
