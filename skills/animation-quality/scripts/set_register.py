"""animation-quality skill: REGISTER THE SET ALONE. Is the background's error a camera aim (pan / tilt / zoom), or the set's
layout / proportions? Search the camera as an image warp of character-hidden stills; no render per candidate.

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/set_register.py --ref ref.mp4 --normal "stills/Comp_f{frame}_C.png"
         --hidden "stills/Comp_f{frame}_H.png" [--wide "stills/Comp_f{frame}_HW.png" --wide-scale 0.74]
         [--frames 124,132,140] [--char-mask chars.png] [--char-thr 6] [--pad 24]
         [--box x0,y0,x1,y1] [--mask M] [--mask-boxes out|in] [--mask-tiles 60]
         [--shift 200] [--shift-step 40] [--scale 0.9:1.35] [--scale-step 0.05] [--min-shift 10] [--no-scale]
         [--centre cx,cy] [--strips 2] [--stretch 1.0:1.4] [--stretch-step 0.05] [--baseline-y 1500[,1560] | 1400:1600:50]
         [--pool] [--noise 0.3] [--fov-v DEG] [--decoder ffmpeg|diffmap|cv2] [--save DIR] [--json out.json]

  --ref     the reference: a video (frame n from 0), one still for every frame, or a path with {frame}.
  --normal  OUR still at each frame, character visible (C). --hidden: the SAME camera with the character hidden (H).
  --wide    optional: the character-hidden still rendered ZOOMED OUT (same camera position and aim, only the zoom / focal
            length changed). --wide-scale F = how big the set looks in it relative to the normal still (< 1; a log-zoom knob
            of -0.3 gives F = exp(-0.3) = 0.741). With --wide the search samples the wide still, so shifts and zoom-outs
            see real set beyond the normal frame's edge; without it the hidden still is warped (border replicated).
            The paths may contain {frame}. Stills of another size are resized to the reference's size (warned).
  --char-mask  the character's pixels as an image (nonzero = character), instead of |C - H| > --char-thr (default 6, max
            over RGB, then dilated 2 px). Needed when --hidden is not given.
  --pad     scored pixels stay this many reference px away from OUR character (default 24). Put the REFERENCE's
            characters (and captions, a ground band, ...) in --mask (error_split's rules: boxes JSON EXCLUDED by default,
            or a black / white image, nonzero = scored) or limit the region with --box.
  --shift / --shift-step  coarse grid of set shifts, +-shift reference px in steps (default 200 / 40); then refined
            by halving down to --min-shift (default 10 px).
  --scale lo:hi / --scale-step  zoom about the principal point (--centre, default the frame centre), coarse grid in log
            steps (0.05 = ~5 %), refined by halving. --no-scale = shifts only.
  --strips N  also report the best warp for N vertical strips of the region on their own (default 2 = left / right;
            1 = whole region only). Every candidate's error map is scored for the whole region and every strip at once.
  --stretch lo:hi + --baseline-y Y[,Y2,...|lo:hi:step]  also search a vertical stretch of the set about a ground line
            (everything above y = Y grows by k, the line and everything below stay): "taller above the same base line".
            Tried with the camera at identity, per strip as well; k on a log grid (--stretch-step, default 0.05), refined
            per baseline. Give the baseline you measured on the reference (where the row meets the ground) or a range.
  --pool    one warp for all frames together (mean score); default = a search per frame.
  --noise   gains below this (diffmap units, default 0.3) count as "no gain" in the verdict. Not calibrated on many
            shots: compare it with frame-to-frame and render noise in your project.
  --fov-v   the camera's vertical field of view in degrees: shifts are also printed as approximate pan / tilt degrees.
  --save    write the best composite per frame (whole region) as PNG, to look at next to the reference.

Model (the same as the round that found it): the composite of a candidate = C's character pixels (the mask) over the
set warped by (shift d, scale s about centre c, optional stretch k about y = Y); set pixel q is sampled from the wide
still at c + F * (stretch^-1(q) - c - d) / s. The character stays where it is on screen, so this models a camera move
that moves only the SET (true when the character is screen-pinned, e.g. a shot whose character is placed in screen space,
or a set-only camera bump). The score is diffmap.py's metric unchanged (270x480 analysis grid, CIELAB of a sigma 1.5
blur, dE76 + 0.5 x local contrast difference), mean over the scored set pixels: error_split.py's `base`.

Signs: shift +x = the set moves RIGHT on screen, +y = DOWN; to get it with the camera, pan / tilt the camera the opposite
way (pan left / tilt up). Scale > 1 = the set looks bigger (zoom in / move closer).
Read the gains against sim(identity) (the composite rebuilt from the wide still at no warp): resampling a wide still is
softer than the normal still, so comparing with today's real still (`actual`) mixes that loss into the gain.
Verdict (outermost strips): opposite shifts = layout / proportion, NOT the camera (move or reshape set pieces); both
want the same shift / scale = a camera aim error a set-only camera bump fixes; a stretch that beats every zoom = the set
stands taller (or shorter) above the same base line: a height stretch about the row's ground, not a zoom.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep the skill folder free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import error_split as ES  # noqa: E402  (diffmap's metric, analysis size, decoders, mask)
import measurelib as M  # noqa: E402


# ---------------------------------------------------------------- inputs
def frame_path(pat: str, f: int) -> Path:
    return Path(pat.format(frame=f)) if "{frame}" in pat else Path(pat)


def read_rgb(p: Path, sw: int, sh: int, label: str, warned: set) -> np.ndarray:
    if not p.exists():
        sys.exit(f"{label} still not found: {p}")
    im = cv2.imread(str(p))
    if im is None:
        sys.exit(f"cannot read {p}")
    if im.shape[:2] != (sh, sw):
        if label not in warned:
            print(f"warning: {label} is {im.shape[1]}x{im.shape[0]}, the reference {sw}x{sh}: resized to the reference size",
                  file=sys.stderr)
            warned.add(label)
        im = cv2.resize(im, (sw, sh), interpolation=cv2.INTER_AREA if im.shape[1] > sw else cv2.INTER_LINEAR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_ref(ref: str, frames: list[int], W: int, H: int, decoder: str) -> dict[int, np.ndarray]:
    if "{frame}" in ref:
        out = {}
        for f in frames:
            p = Path(ref.format(frame=f))
            if not p.exists():
                sys.exit(f"reference still missing: {p}")
            out[f] = ES.load(p, [f], W, H, "diffmap" if decoder == "diffmap" else "ffmpeg")[f]
        return out
    return ES.load(Path(ref), frames, W, H, decoder)


def rng(lo: float, hi: float, step: float) -> list[float]:
    n = int(math.floor((hi - lo) / step + 1e-9))
    return [lo + i * step for i in range(n + 1)]


def pair(s: str, what: str) -> tuple[float, float]:
    v = [float(x) for x in s.split(":")]
    if len(v) != 2 or v[0] > v[1]:
        sys.exit(f"{what} must be lo:hi with lo <= hi: {s}")
    return v[0], v[1]


# ---------------------------------------------------------------- one frame's data
class Frame:
    def __init__(self, f, R, C, src, kw, char, scored_units, sw, sh, W, H, centre):
        self.f, self.R, self.C, self.src, self.kw, self.char = f, R, C, src, kw, char
        self.units = scored_units  # list of analysis-grid bool masks: whole, strip 1..N
        self.sw, self.sh, self.W, self.H = sw, sh, W, H
        self.cx, self.cy = centre
        self.ys, self.xs = None, None

    def set_image(self, s: float, dx: float, dy: float, k: float = 1.0, yb: float | None = None):
        """the set as a camera variant would see it, plus the share of pixels sampled outside the source still."""
        a = self.kw / s
        bx, by = self.cx - a * (self.cx + dx), self.cy - a * (self.cy + dy)
        if k == 1.0 or yb is None:
            Mx = np.array([[a, 0, bx], [0, a, by]], np.float32)
            out = cv2.warpAffine(self.src, Mx, (self.sw, self.sh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
            sx0, sx1 = bx, a * (self.sw - 1) + bx
            sy0, sy1 = by, a * (self.sh - 1) + by
            ox = max(0.0, -sx0) + max(0.0, sx1 - (self.sw - 1))
            oy = max(0.0, -sy0) + max(0.0, sy1 - (self.sh - 1))
            inside = max(0.0, 1 - ox / max(1e-9, sx1 - sx0)) * max(0.0, 1 - oy / max(1e-9, sy1 - sy0))
            return out, 1.0 - inside
        if self.ys is None:
            self.ys, self.xs = np.mgrid[0:self.sh, 0:self.sw].astype(np.float32)
        py = np.where(self.ys < yb, yb - (yb - self.ys) / k, self.ys)
        mx = (a * self.xs + bx).astype(np.float32)
        my = (a * py + by).astype(np.float32)
        out = cv2.remap(self.src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        outside = float(((mx < 0) | (mx > self.sw - 1) | (my < 0) | (my > self.sh - 1)).mean())
        return out, outside

    def composite(self, S: np.ndarray) -> np.ndarray:
        return np.where(self.char[..., None], self.C, S)

    def scores(self, full_rgb: np.ndarray) -> np.ndarray:
        e = ES.error_map(self.R, cv2.resize(full_rgb, (self.W, self.H), interpolation=cv2.INTER_AREA))
        return np.array([float(e[u].mean()) if u.any() else float("nan") for u in self.units])


# ---------------------------------------------------------------- the search
class Search:
    def __init__(self, frames: list[Frame]):
        self.FR = frames
        self.cache: dict[tuple, tuple[np.ndarray, float]] = {}

    def eval(self, s: float, dx: float, dy: float, k: float = 1.0, yb: float | None = None) -> tuple[np.ndarray, float]:
        key = (round(math.log(s), 6), round(dx, 3), round(dy, 3), round(k, 6), yb)
        if key not in self.cache:
            sc, out = [], []
            for fr in self.FR:
                S, o = fr.set_image(s, dx, dy, k, yb)
                sc.append(fr.scores(fr.composite(S)))
                out.append(o)
            self.cache[key] = (np.nanmean(np.stack(sc), 0), float(np.mean(out)))
        return self.cache[key]

    def grid(self, cands, nunits):
        best = [(math.inf, None)] * nunits
        for c in cands:
            v, _ = self.eval(*c)
            for u in range(nunits):
                if v[u] < best[u][0]:
                    best[u] = (float(v[u]), c)
        return best

    def refine(self, u: int, c0, shift_step: float, min_shift: float, zstep: float, use_scale: bool):
        """pattern search around c0 for unit u: shift step and log-scale step halve each stage."""
        best_v, best = float(self.eval(*c0)[0][u]), c0
        st, zs = shift_step / 2, zstep / 2
        while st >= min_shift - 1e-9:
            moved = True
            while moved:
                moved = False
                s, dx, dy = best[0], best[1], best[2]
                zz = (-zs, 0.0, zs) if use_scale else (0.0,)
                for dz in zz:
                    for ex in (-st, 0.0, st):
                        for ey in (-st, 0.0, st):
                            c = (round(s * math.exp(dz), 6), dx + ex, dy + ey)
                            v = float(self.eval(*c)[0][u])
                            if v < best_v - 1e-9:
                                best_v, best, moved = v, c, True
            st, zs = st / 2, zs / 2
        return best_v, best


# ---------------------------------------------------------------- report
def deg(px: float, fov_v: float | None, sh: int) -> str:
    if not fov_v:
        return ""
    fpx = (sh / 2) / math.tan(math.radians(fov_v) / 2)
    return f" (~{math.degrees(math.atan(abs(px) / fpx)):.2f} deg)"


def describe(c, fov_v, sh) -> str:
    s, dx, dy = c[0], c[1], c[2]
    return (f"shift x {dx:+5.0f}{deg(dx, fov_v, sh)} y {dy:+5.0f}{deg(dy, fov_v, sh)} px, scale {s:.3f} (log {math.log(s):+.3f})")


def verdict(units: list[dict], noise: float, step: float, zstep: float, stretch: dict | None) -> list[str]:
    out = []
    whole = units[0]
    if stretch:
        sg = stretch["units"][0]["gain_sim"]
        if sg >= noise and sg > whole["gain_sim"]:
            out.append(f"STRETCH beats every camera warp ({sg:.2f} vs {whole['gain_sim']:.2f}): the set stands "
                       f"{'taller' if stretch['units'][0]['k'] > 1 else 'shorter'} above the same base line -> a height "
                       "stretch of the row about its own ground, not a zoom (a zoom moves the base line). The strip "
                       "verdict below is about camera warps only and does not apply.")
    if len(units) >= 3 and (units[1]["warp"] is None or units[-1]["warp"] is None):
        out.append("A STRIP HAS NO SCORED PIXELS on these frames (fully masked or outside the box): no strip verdict.")
    elif len(units) >= 3:
        L, R = units[1], units[-1]
        gl, gr = L["gain_sim"], R["gain_sim"]
        (sl, xl, yl), (sr, xr, yr) = L["warp"], R["warp"]
        opp = lambda a, b: a * b < 0 and min(abs(a), abs(b)) >= step  # noqa: E731
        close = lambda a, b: abs(a - b) <= max(step, 0.35 * max(abs(a), abs(b)))  # noqa: E731
        if gl < noise and gr < noise:
            out.append(f"NEITHER SIDE GAINS (both < {noise:g}): the set is registered, or its difference is not a warp "
                       "(colour, detail, layout). Try error_split.py.")
        elif (opp(xl, xr) or opp(yl, yr)) and gl >= noise and gr >= noise:
            out.append("OPPOSITE: the left and right strips want opposite shifts -> layout / proportion, NOT the camera "
                       "(move, widen or reshape set pieces; a camera move would fix one side and break the other).")
        elif close(xl, xr) and close(yl, yr) and abs(math.log(sl / sr)) <= 2 * zstep + 1e-9 and whole["gain_sim"] >= noise:
            out.append("AGREE: both strips want the same shift / scale -> a camera aim error. A set-only camera bump of "
                       f"{describe(whole['warp'], None, 0)} fixes it when the character is screen-pinned.")
        else:
            out.append("MIXED: the strips neither agree nor oppose clearly; look at the per-strip rows and crops.")
    elif whole["gain_sim"] >= noise:
        out.append(f"WHOLE REGION gains {whole['gain_sim']:.2f}: a camera warp helps (run with --strips 2 to tell camera from layout).")
    else:
        out.append(f"no warp gains {noise:g} or more on the whole region.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--normal", required=True)
    ap.add_argument("--hidden")
    ap.add_argument("--wide")
    ap.add_argument("--wide-scale", type=float)
    ap.add_argument("--frames", default="0")
    ap.add_argument("--char-mask")
    ap.add_argument("--char-thr", type=float, default=6)
    ap.add_argument("--pad", type=float, default=24)
    ap.add_argument("--box")
    ap.add_argument("--mask")
    ap.add_argument("--mask-boxes", choices=["out", "in"], default="out")
    ap.add_argument("--mask-tiles", type=int, default=0)
    ap.add_argument("--shift", type=float, default=200)
    ap.add_argument("--shift-step", type=float, default=40)
    ap.add_argument("--min-shift", type=float, default=10)
    ap.add_argument("--scale", default="0.9:1.35")
    ap.add_argument("--scale-step", type=float, default=0.05)
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--centre")
    ap.add_argument("--strips", type=int, default=2)
    ap.add_argument("--stretch")
    ap.add_argument("--stretch-step", type=float, default=0.05)
    ap.add_argument("--baseline-y")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--fov-v", type=float)
    ap.add_argument("--save")
    ap.add_argument("--json")
    ap.add_argument("--decoder", choices=["ffmpeg", "diffmap", "cv2"], default="ffmpeg",
                    help="reference decode: ffmpeg = the stream's colour matrix (default); diffmap = exactly diffmap.py's decode; cv2 = OpenCV")
    a = ap.parse_args()

    if not a.hidden and not a.wide:
        sys.exit("give --hidden (the character-hidden still, same camera) and / or --wide")
    if a.wide and not a.wide_scale:
        sys.exit("--wide needs --wide-scale (how big the set looks in the wide still vs the normal one, e.g. 0.741)")
    if not a.hidden and not a.char_mask:
        sys.exit("without --hidden the character's pixels must come from --char-mask")
    if a.stretch and not a.baseline_y:
        sys.exit("--stretch needs --baseline-y (the set's ground line, reference px)")
    frames = M.parse_frames(a.frames)
    ref0 = frame_path(a.ref, frames[0])
    if not ref0.exists():
        sys.exit(f"reference not found: {ref0}")
    sw, sh = ES.src_size(ref0)
    W, H = ES.analysis_size(sw, sh)
    sx, sy = sw / W, sh / H
    centre = tuple(float(v) for v in a.centre.split(",")) if a.centre else ((sw - 1) / 2, (sh - 1) / 2)
    bx0, by0, bx1, by1 = M.box_px(M.parse_box(a.box), sw, sh) if a.box else (0, 0, sw, sh)
    region = np.zeros((H, W), bool)
    ax0, ay0, ax1, ay1 = (int(round(bx0 / sx)), int(round(by0 / sy)), int(round(bx1 / sx)), int(round(by1 / sy)))
    region[ay0:ay1, ax0:ax1] = True
    nstrip = max(1, a.strips)
    edges = [int(round(ax0 + (ax1 - ax0) * i / nstrip)) for i in range(nstrip + 1)]
    names = ["whole"] + ([("left", "right")[i] if nstrip == 2 else f"strip{i + 1}" for i in range(nstrip)] if nstrip > 1 else [])
    umask = ES.Mask(a.mask, a.mask_boxes, a.mask_tiles, sw, sh, W, H)
    R = load_ref(a.ref, frames, W, H, a.decoder)
    cmask_img = None
    if a.char_mask:
        im = cv2.imread(a.char_mask, cv2.IMREAD_GRAYSCALE)
        if im is None:
            sys.exit(f"cannot read --char-mask {a.char_mask}")
        cmask_img = cv2.resize(im, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    pad_it = max(0, int(round(a.pad / sx)))
    warned: set = set()
    FR, actual = [], {}
    for f in frames:
        if f not in R:
            sys.exit(f"the reference has no frame {f}")
        C = read_rgb(frame_path(a.normal, f), sw, sh, "normal", warned)
        Hd = read_rgb(frame_path(a.hidden, f), sw, sh, "hidden", warned) if a.hidden else None
        if cmask_img is not None:
            char = cmask_img
        else:
            char = np.abs(C.astype(np.int16) - Hd.astype(np.int16)).max(2) > a.char_thr
            char = cv2.dilate(char.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        cs = cv2.resize(char.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) > 0.01
        near = cv2.dilate(cs.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=pad_it) > 0 if pad_it else cs
        scored = ~near & region & umask.at(f, (bx0, by0))
        units = [scored]
        if nstrip > 1:
            for i in range(nstrip):
                u = scored.copy()
                u[:, :edges[i]] = False
                u[:, edges[i + 1]:] = False
                units.append(u)
        if not scored.any():
            sys.exit(f"frame {f}: no scored set pixels (character, --pad, --mask and --box exclude everything)")
        src, kw = (read_rgb(frame_path(a.wide, f), sw, sh, "wide", warned), a.wide_scale) if a.wide else (Hd, 1.0)
        fr = Frame(f, R[f], C, src, kw, char, units, sw, sh, W, H, centre)
        actual[f] = fr.scores(C)
        FR.append(fr)

    lo, hi = pair(a.scale, "--scale")
    zs = a.scale_step  # log grid anchored at 0 (scale 1 = today's camera is always a candidate)
    zgrid = [0.0] if a.no_scale else [n * zs for n in range(math.ceil(math.log(lo) / zs - 1e-9), math.floor(math.log(hi) / zs + 1e-9) + 1)]
    sg = rng(-a.shift, a.shift, a.shift_step) if a.shift > 0 else [0.0]
    yb_list = []
    for part in (a.baseline_y or "").split(","):
        if ":" in part:  # lo:hi:step
            b0, b1, bs = (float(v) for v in part.split(":"))
            yb_list += rng(b0, b1, bs)
        elif part.strip():
            yb_list.append(float(part))
    print(f"reference {sw}x{sh}, analysis {W}x{H}; frames {frames} ({'pooled' if a.pool else 'searched one by one'}); source "
          f"{'wide still, set scale ' + format(a.wide_scale, 'g') if a.wide else 'hidden still (border replicated)'}; centre "
          f"{centre[0]:g},{centre[1]:g}; region {bx0},{by0},{bx1},{by1}; shifts +-{a.shift:g} step {a.shift_step:g} -> "
          f"{a.min_shift:g} px; scale {'off' if a.no_scale else f'{lo:g}..{hi:g} log step {a.scale_step:g}'}; "
          f"{len(zgrid) * len(sg) ** 2} coarse candidates; decoder {a.decoder}")
    print(f"scored set px per frame (analysis grid): " + ", ".join(f"f{fr.f} {int(fr.units[0].sum())}" for fr in FR))

    groups = [FR] if a.pool else [[fr] for fr in FR]
    results = {}
    for g in groups:
        tag = "pooled" if a.pool else f"f{g[0].f}"
        S = Search(g)
        nu = len(g[0].units)
        act = np.nanmean(np.stack([actual[fr.f] for fr in g]), 0)
        sim0, _ = S.eval(1.0, 0.0, 0.0)
        cands = [(round(math.exp(z), 6), dx, dy) for z in zgrid for dx in sg for dy in sg]
        coarse = S.grid(cands, nu)
        units = []
        for u in range(nu):
            if coarse[u][1] is None:  # no scored pixels in this unit on any frame (e.g. a band fully masked): report, don't crash
                print(f"warning: unit {names[u]} has no scored pixels on these frames; skipped", file=sys.stderr)
                units.append({"name": names[u], "warp": None, "score": float("nan"), "actual": float(act[u]),
                              "sim0": float(sim0[u]), "gain_sim": float("nan"), "gain_actual": float("nan"),
                              "whole_at_this_warp": float("nan"), "outside": float("nan")})
                continue
            v, c = S.refine(u, coarse[u][1], a.shift_step if a.shift > 0 else 0.0, a.min_shift, a.scale_step, not a.no_scale)
            allv, outside = S.eval(*c)
            units.append({"name": names[u], "warp": c, "score": v, "actual": float(act[u]), "sim0": float(sim0[u]),
                          "gain_sim": float(sim0[u] - v), "gain_actual": float(act[u] - v), "whole_at_this_warp": float(allv[0]),
                          "outside": outside})
        stretch = None
        if a.stretch:
            klo, khi = pair(a.stretch, "--stretch")
            st_ = a.stretch_step
            ks = [n * st_ for n in range(math.ceil(math.log(klo) / st_ - 1e-9), math.floor(math.log(khi) / st_ + 1e-9) + 1)]
            sb = [(math.inf, None)] * nu
            for yb in yb_list:  # per baseline: the log-k grid, then k refined (step halved twice) for every unit
                loc = [(math.inf, None)] * nu
                for z in ks:
                    v, _ = S.eval(1.0, 0.0, 0.0, round(math.exp(z), 6), yb)
                    for u in range(nu):
                        if v[u] < loc[u][0]:
                            loc[u] = (float(v[u]), round(math.exp(z), 6))
                for u in range(nu):
                    h = st_ / 2
                    for _ in range(2):
                        for sgn in (-1, 1):
                            k2 = round(loc[u][1] * math.exp(sgn * h), 6)
                            if not klo - 1e-6 <= k2 <= khi + 1e-6:
                                continue
                            v2 = float(S.eval(1.0, 0.0, 0.0, k2, yb)[0][u])
                            if v2 < loc[u][0]:
                                loc[u] = (v2, k2)
                        h /= 2
                    if loc[u][0] < sb[u][0]:
                        sb[u] = (loc[u][0], (loc[u][1], yb))
            stretch = {"units": [{"name": names[u], "k": sb[u][1][0], "baseline_y": sb[u][1][1], "score": sb[u][0],
                                  "gain_sim": float(sim0[u] - sb[u][0])} for u in range(nu)]}
        vd = verdict(units, a.noise, a.shift_step, a.scale_step, stretch)
        print(f"\n{tag}: actual still {act[0]:.2f} | sim(identity) {sim0[0]:.2f} (resampling loss {sim0[0] - act[0]:+.2f})"
              f"   [{len(S.cache)} warps scored]")
        for r in units:
            if r["warp"] is None:
                print(f"  {r['name']:7s} no scored pixels on these frames")
                continue
            warn =f"  ! {100 * r['outside']:.0f} % of the frame sampled past the still's edge (replicated)" if r["outside"] > 0.02 else ""
            print(f"  {r['name']:7s} best {describe(r['warp'], a.fov_v, sh)}: {r['score']:6.2f}  gain vs sim {r['gain_sim']:+6.2f}"
                  f"  vs actual {r['gain_actual']:+6.2f}" + (f"  (whole region at this warp {r['whole_at_this_warp']:.2f})"
                                                              if r["name"] != "whole" else "") + warn)
        if stretch:
            for r in stretch["units"]:
                print(f"  stretch {r['name']:7s} k {r['k']:.3f} about y {r['baseline_y']:g}: {r['score']:6.2f}  gain vs sim "
                      f"{r['gain_sim']:+6.2f}")
        for line in vd:
            print(f"  -> {line}")
        if a.save and units[0]["warp"] is not None:
            Path(a.save).mkdir(parents=True, exist_ok=True)
            c = units[0]["warp"]
            for fr in g:
                img, _ = fr.set_image(*c)
                p = Path(a.save) / f"setreg_f{fr.f}_best.png"
                cv2.imwrite(str(p), cv2.cvtColor(fr.composite(img), cv2.COLOR_RGB2BGR))
            print(f"  wrote best composites to {a.save}")
        results[tag] = {"frames": [fr.f for fr in g], "actual": float(act[0]), "sim0": float(sim0[0]),
                        "units": [{**r, "warp": None if r["warp"] is None else
                                   {"scale": r["warp"][0], "dx": r["warp"][1], "dy": r["warp"][2]}} for r in units],
                        "stretch": stretch, "verdict": vd}
    M.dump_json(a.json, {"frames": frames, "analysis": [W, H], "region_px": [bx0, by0, bx1, by1], "centre": centre,
                         "wide_scale": a.wide_scale, "decoder": a.decoder, "strips": names, "results": results,
                         "signs": "dx + = set moves right on screen, dy + = down; scale > 1 = set bigger"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
