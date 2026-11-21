"""animation-quality skill: shared helpers for the pixel-measuring tools (head_profile, register, edge_profile, limb_track).

Not a CLI. The tools put their own folder on sys.path explicitly (python -I does not), so `python -I scripts/<tool>.py`
works from any working directory. Only numpy + opencv (the same as diffmap.py / crops.py).

Conventions shared by every tool
- Sources are LABEL=PATH (a video, frame n counted from 0, or a still used for every frame). The first source is the
  REFERENCE; every other source is compared with it.
- Boxes are x0,y0,x1,y1 in pixels of the source frame, or 0..1 frame fractions when every value is <= 1.
  A per-frame box file is JSON {"<frame>": [x0,y0,x1,y1], ...}; a missing frame uses the nearest listed frame.
- L*, a*, b* are CIELAB from OpenCV's 8-bit conversion: L* = L8 * 100/255 (0..100), a* = a8 - 128, b* = b8 - 128.
- Angles are screen angles counter-clockwise from screen-right (90 = up, 180 = left, 270 = down).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------------------------------------------- sources and frames
def parse_sources(items: list[str], default_labels=("ref", "ours")) -> list[tuple[str, Path]]:
    out = []
    for i, s in enumerate(items):
        if "=" in s and not Path(s).exists():
            k, p = s.split("=", 1)
        else:
            k, p = (default_labels[i] if i < len(default_labels) else f"src{i}"), s
        if not Path(p).exists():
            sys.exit(f"source not found: {p}")
        out.append((k, Path(p)))
    if len(out) < 1:
        sys.exit("need at least one source (the reference first)")
    return out


def parse_frames(s: str) -> list[int]:
    """'40,50,60' or '40-60' or '40-60:5' (step) or a mix: '10,40-60:10'."""
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            rng, _, step = part.partition(":")
            a, b = (int(v) for v in rng.split("-"))
            out += list(range(a, b + 1, int(step) if step else 1))
        else:
            out.append(int(part))
    return sorted(set(out))


DECODER = "ffmpeg"   # tools set this from --decoder


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,color_space,color_range", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"][0]


def colour_filter(info: dict) -> str:
    """YUV -> RGB with the stream's own matrix (untagged HD = bt709, as players assume); the same rule video players use."""
    m = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601"}.get(info.get("color_space"))
    if m is None:
        m = "bt709" if int(info["height"]) >= 720 else "bt601"
    rg = "pc" if info.get("color_range") == "pc" else "tv"
    return f"scale={info['width']}:{info['height']}:in_color_matrix={m}:in_range={rg}:flags=bicubic"


def read_frames(path: Path, frames: list[int], decoder: str | None = None) -> dict[int, np.ndarray]:
    """BGR uint8 frames at exact frame numbers (counted from 0, decoded sequentially, no seeking). A still is used for every frame.
    decoder "ffmpeg" (default): the stream's own colour matrix (bt709 for HD). "cv2": OpenCV's VideoCapture, which converts
    bt709 video with the bt601 matrix: colours shift by a few levels (mean ~3, up to ~20 on saturated pixels). Use "cv2" only to
    reproduce numbers from older notes that were measured that way; never mix decoders inside one comparison."""
    decoder = decoder or DECODER
    if path.suffix.lower() in IMG:
        im = cv2.imread(str(path))
        if im is None:
            sys.exit(f"cannot read image {path}")
        return {f: im for f in frames}
    want, got, last = set(frames), {}, max(frames)
    i = 0
    if decoder == "cv2":
        cap = cv2.VideoCapture(str(path))
        while i <= last:
            ok, im = cap.read()
            if not ok:
                break
            if i in want:
                got[i] = im
            i += 1
        cap.release()
    else:
        info = probe(path)
        w, h = int(info["width"]), int(info["height"])
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-vf", colour_filter(info),
               "-frames:v", str(last + 1), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        n = w * h * 3
        while i <= last:
            b = pr.stdout.read(n)
            if len(b) < n:
                break
            if i in want:
                got[i] = np.frombuffer(b, np.uint8).reshape(h, w, 3).copy()
            i += 1
        pr.stdout.close()
        pr.wait()
    missing = sorted(want - set(got))
    if missing:
        print(f"warning: {path.name} has no frame(s) {missing[:8]}{'...' if len(missing) > 8 else ''} (decoded {i})", file=sys.stderr)
    return got


def load_all(srcs: list[tuple[str, Path]], frames: list[int]) -> dict[str, dict[int, np.ndarray]]:
    """frames of every source; a source whose size differs from the reference (the first) is resized to it (INTER_AREA)
    with a warning, so px boxes, points and masks mean the same place in every source."""
    data = {k: read_frames(p, frames) for k, p in srcs}
    ref = next((im for im in data[srcs[0][0]].values()), None)
    if ref is None:
        return data
    H, W = ref.shape[:2]
    for k, _ in srcs[1:]:
        for f, im in list(data[k].items()):
            if im.shape[:2] != (H, W):
                if f == min(data[k]):
                    print(f"warning: {k} is {im.shape[1]}x{im.shape[0]}, the reference {W}x{H}: {k} resized to the reference size",
                          file=sys.stderr)
                data[k][f] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA if im.shape[1] > W else cv2.INTER_LINEAR)
    return data


def add_decoder_arg(ap) -> None:
    ap.add_argument("--decoder", choices=["ffmpeg", "cv2"], default="ffmpeg",
                    help="ffmpeg = the stream's colour matrix (default); cv2 = OpenCV VideoCapture (bt601: only to reproduce old notes)")


# ---------------------------------------------------------------- boxes
def parse_box(s: str) -> list[float]:
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        sys.exit(f"box must be x0,y0,x1,y1: {s}")
    return v


def box_px(box: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    if max(abs(v) for v in box) <= 1.0:
        x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        sys.exit(f"box {box} is empty inside a {w}x{h} frame")
    return x0, y0, x1, y1


class Boxes:
    """--box (one box for every frame) or --boxes FILE.json ({"frame": box}); nearest listed frame when missing."""

    def __init__(self, box: str | None, boxes_file: str | None, required=True):
        self.one = parse_box(box) if box else None
        self.per: dict[int, list[float]] = {}
        if boxes_file:
            raw = json.loads(Path(boxes_file).read_text(encoding="utf-8"))
            self.per = {int(k): [float(x) for x in v] for k, v in raw.items()}
        if required and self.one is None and not self.per:
            sys.exit("give --box x0,y0,x1,y1 or --boxes FILE.json")

    def get(self, f: int) -> list[float] | None:
        if self.per:
            k = f if f in self.per else min(self.per, key=lambda q: abs(q - f))
            return self.per[k]
        return self.one


# ---------------------------------------------------------------- colour
def lab_planes(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128


def rgb_to_lab(rgb) -> tuple[float, float, float]:
    px = np.uint8([[[int(rgb[2]), int(rgb[1]), int(rgb[0])]]])
    L, a, b = lab_planes(px)
    return float(L[0, 0]), float(a[0, 0]), float(b[0, 0])


def condition_mask(bgr: np.ndarray, where: list[dict] | dict) -> np.ndarray:
    """OR over a list of AND-conditions, each {"L": [lo, hi], "a": [...], "b": [...], "R": [...], "G": [...], "B": [...],
    "R-B": [...], "R-G": [...], "G-B": [...], "absa": hi, "absb": hi}; bounds are exclusive (lo < v < hi), null = open."""
    L, a, b = lab_planes(bgr)
    ch = {"L": L, "a": a, "b": b, "R": bgr[..., 2].astype(np.float32), "G": bgr[..., 1].astype(np.float32),
          "B": bgr[..., 0].astype(np.float32), "absa": np.abs(a), "absb": np.abs(b)}
    ch["R-B"], ch["R-G"], ch["G-B"] = ch["R"] - ch["B"], ch["R"] - ch["G"], ch["G"] - ch["B"]
    conds = where if isinstance(where, list) else [where]
    out = np.zeros(L.shape, bool)
    for c in conds:
        m = np.ones(L.shape, bool)
        for k, rng in c.items():
            if k not in ch:
                sys.exit(f"unknown channel {k!r} in a mask condition (use L a b R G B R-B R-G G-B absa absb)")
            lo, hi = (None, rng) if k in ("absa", "absb") and not isinstance(rng, list) else rng
            if lo is not None:
                m &= ch[k] > lo
            if hi is not None:
                m &= ch[k] < hi
        out |= m
    return out


def parse_ranges(s: str) -> dict:
    """'L=66:,a=-6:30,b=2:45' -> {"L": [66, None], "a": [-6, 30], "b": [2, 45]} (bounds exclusive, empty = open).
    absa / absb take one upper bound: 'absa=14' (or 'absa=:14') -> {"absa": 14} = |a*| < 14."""
    out = {}
    for part in s.split(","):
        k, _, r = part.partition("=")
        lo, _, hi = r.partition(":")
        k = k.strip()
        if k in ("absa", "absb"):
            out[k] = float(hi) if hi else float(lo)
        else:
            out[k] = [float(lo) if lo else None, float(hi) if hi else None]
    return out


# ---------------------------------------------------------------- masks
def fill_holes(m: np.ndarray) -> np.ndarray:
    """binary hole filling (background reached from the border with 4-connectivity stays background)."""
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m.astype(np.uint8)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2, flags=4)
    return (ff[1:-1, 1:-1] != 2)


def edt(m: np.ndarray) -> np.ndarray:
    """exact Euclidean distance of each mask pixel to the nearest non-mask pixel (0 outside)."""
    return cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def morph(m: np.ndarray, op: int, k: int) -> np.ndarray:
    if k <= 1:
        return m.astype(bool)
    return cv2.morphologyEx(m.astype(np.uint8), op, np.ones((k, k), np.uint8)).astype(bool)


def pick_component(m: np.ndarray, how: str = "largest", seed: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """largest component, or the one with most pixels inside seed=(x0,y0,x1,y1) (crop coordinates), or all pixels."""
    if how == "all":
        return m.astype(bool)
    n, lb, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n < 2:
        return np.zeros(m.shape, bool)
    if how == "seed" and seed is not None:
        x0, y0, x1, y1 = seed
        cnt = np.bincount(lb[y0:y1, x0:x1].ravel(), minlength=n)
        cnt[0] = 0
        k = int(np.argmax(cnt))
    else:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lb == k


def angle_map(shape, cx: float, cy: float) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (np.degrees(np.arctan2(-(yy - cy), xx - cx)) + 360) % 360


def sector(ang: np.ndarray, centre: float, width: float) -> np.ndarray:
    return ((ang - centre + width / 2) % 360) < width


def nanmed(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def fmt(v, nd=1) -> str:
    return "  nan" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def dump_json(path: str | None, obj) -> None:
    if not path:
        return

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, np.ndarray):
            return [conv(x) for x in o.tolist()]
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        return o

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(conv(obj), indent=1), encoding="utf-8")
    print(f"wrote {path}")
