"""animation-quality skill: SOLVE surface input colours through the render's own transfer (light, grade, veil, blur) so the
frame scores best against a reference on diffmap's metric, instead of picking colours off the reference by eye.

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/colour_solve.py --ref ref.mp4 --manifest toggles.json [--frames 4,20]
         [--box x0,y0,x1,y1] [--mask chars.json] [--mask-boxes out|in] [--mask-tiles 60]
         [--fix wall=#506868,...] [--bounds trim=#000000:#ffffff,...] [--current colours.json] [--init colours.json]
         [--step 8] [--iters 3] [--refine 2] [--decoder ffmpeg|diffmap|cv2] [--per-frame]
         [--colours solved.json] [--json out.json]

  --ref       the reference: a video (frame n counted from 0), one still used for every frame, or a path containing
              {frame} (one still per frame, e.g. ref_f{frame}.png).
  --manifest  JSON describing the probe stills (relative paths resolve against the manifest's folder). Pattern form:
                {"levels": {"low": "#202020", "base": "#808080", "high": "#e0e0e0"},
                 "frames": [4, 20],
                 "surfaces": ["wall", "trim", "roof"],
                 "base":   "stills/Comp_f{frame}_G.png",
                 "toggle": "stills/Comp_f{frame}_T{surface}{level}.png",
                 "level_tags": {"low": "d", "high": "l"},
                 "fix": {"roof": "#403830"}, "bounds": {"trim": ["#000000", "#ffffff"]}}
              or the explicit form (any file names):
                {"levels": {...}, "stills": {"4": {"base": "a.png", "toggles": {"wall": {"low": "b.png", "high": "c.png"}}}}}
              base = every surface of interest at the base level; each toggle = that ONE surface at low / high, all
              others at base. Levels are a grey (0..255) or a hex per channel; defaults #202020 / #808080 / #e0e0e0.
              "surface_levels": {"wall": {"low": "#304050", "base": "#607080", "high": "#90a0b0"}} gives one surface its
              own levels (its base / toggle stills rendered at those): use it to RE-LINEARISE around a first solve.
  --frames    frames to solve over (default: the manifest's); every frame needs its base and every toggle.
  --box       score only this region (px of the reference frame, or 0..1); default the whole frame.
  --mask      exclude characters etc.: a JSON of boxes per frame {"<frame>": [[x0,y0,x1,y1], ...]} (EXCLUDED; --mask-boxes in
              = only those scored) or a black / white image (nonzero = scored). Same rules as error_split.py.
  --fix       hold these surfaces at a given input colour (also manifest "fix"); --bounds limit a surface's input per
              channel (lo:hi hex; also manifest "bounds").
  --current   a {surface: "#hex"} JSON of today's colours: its predicted score is printed next to the solved one.
  --init      start the descent from these colours (default: every surface at the base level).
  --step / --iters  phase 1: every surface and channel on its own tries 0..255 in --step levels (default 8), --iters
              rounds (default 3). --refine N: phase 2, up to N rounds (default 6, stops when nothing moves) of a pattern
              search per surface from --step down to 1 level along R, G, B, grey and the opponent directions.
  --colours   write the solved {surface: "#hex"} map (feed it to the probe that renders the check still).

Model (one per-channel piecewise-linear response per surface, measured, not assumed):
  r_s,lo = toggle_lo - base and r_s,hi = toggle_hi - base per pixel and channel give what the render does to the frame when
  surface s goes from the base level to low / high (light, shadow, grade, veil, fog, depth of field, antialiasing all
  included). predicted(in) = base + sum_s f_s(in_s), f_s,c = r_s,lo,c * (base_c - in_c) / (base_c - low_c) for in_c <= base_c,
  else r_s,hi,c * (in_c - base_c) / (high_c - base_c); extrapolated linearly outside [low, high]; clipped to 0..255 at the end.
  Assumptions: surfaces add (no surface lights another), each output channel follows the same input channel (a grade that
  mixes channels or shifts hue breaks this), and the response is close to linear between the three levels.
The score is diffmap.py's metric unchanged (270x480 / 480x270 analysis, CIELAB of a sigma 1.5 blur, dE76 + 0.5 x local
contrast difference), mean over the scored pixels and the frames: the same number error_split.py calls `base`.
Every candidate is scored exactly (the metric's 14 px reach is recomputed around the surface's support only).

ALWAYS render the solved colours and score the real still: in the round this was built in, the prediction ran about
0.4 diffmap units optimistic (predicted lower than rendered). The prediction ranks candidates; the render decides.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep the skill folder free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import error_split as ES  # noqa: E402  (diffmap's metric, analysis size, decoders, mask)
import measurelib as M  # noqa: E402

REACH = 14  # px (analysis grid): sigma 1.5 blur (radius 6) then sigma 2 blur (radius 8) in error_map
WARN = ("prediction only: in the round this method was built in, the predicted score ran ~0.4 diffmap units better than "
        "the rendered still. Render the solved colours (one still batch) and score the real stills before keeping anything.")


def hex3(s) -> np.ndarray:
    if isinstance(s, (int, float)):
        return np.full(3, float(s), np.float32)
    if isinstance(s, (list, tuple)):
        return np.array([float(x) for x in s], np.float32)
    s = str(s).strip()
    if not s.startswith("#") or len(s) != 7:
        sys.exit(f"colour must be #rrggbb, a grey 0..255 or [r,g,b]: {s!r}")
    return np.array([int(s[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def tohex(v) -> str:
    return "#" + "".join(f"{int(min(255, max(0, round(float(x))))):02x}" for x in v)


def kv_list(s: str | None) -> dict[str, str]:
    out = {}
    for part in (s or "").split(","):
        if part.strip():
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------- manifest
def read_manifest(path: Path, frames_arg: str | None):
    man = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    lv = man.get("levels", {})
    glob_lv = {k: lv.get(k, d) for k, d in (("low", "#202020"), ("base", "#808080"), ("high", "#e0e0e0"))}
    rp = lambda p: (root / p) if not Path(p).is_absolute() else Path(p)  # noqa: E731
    stills: dict[int, dict] = {}
    if "stills" in man:
        for f, d in man["stills"].items():
            stills[int(f)] = {"base": rp(d["base"]),
                              "toggles": {s: {lv_: rp(t[lv_]) for lv_ in ("low", "high")} for s, t in d["toggles"].items()}}
        frames = sorted(stills)
    else:
        for k in ("frames", "surfaces", "base", "toggle"):
            if k not in man:
                sys.exit(f"manifest needs '{k}' (pattern form) or 'stills' (explicit form)")
        tags = man.get("level_tags", {"low": "low", "high": "high"})
        frames = [int(f) for f in man["frames"]]
        for f in frames:
            stills[f] = {"base": rp(man["base"].format(frame=f)),
                         "toggles": {s: {lv_: rp(man["toggle"].format(frame=f, surface=s, level=tags[lv_])) for lv_ in ("low", "high")}
                                     for s in man["surfaces"]}}
    if frames_arg:
        frames = M.parse_frames(frames_arg)
    surfaces = sorted({s for f in frames if f in stills for s in stills[f]["toggles"]}, key=lambda s: (
        list(man["surfaces"]).index(s) if "surfaces" in man and s in man["surfaces"] else 10 ** 6, s))
    missing = []
    for f in frames:
        if f not in stills:
            sys.exit(f"frame {f} is not in the manifest")
        need = [stills[f]["base"]] + [p for s in surfaces for p in stills[f]["toggles"].get(s, {}).values()]
        if len(stills[f]["toggles"]) != len(surfaces):
            sys.exit(f"frame {f} does not list every surface ({sorted(set(surfaces) - set(stills[f]['toggles']))} missing)")
        missing += [str(p) for p in need if not p.exists()]
    levels = {}  # surface -> [low, base, high] x channel; "surface_levels" overrides the global "levels" per surface
    for sf in surfaces:
        d = {**glob_lv, **man.get("surface_levels", {}).get(sf, {})}
        L = np.stack([hex3(d["low"]), hex3(d["base"]), hex3(d["high"])])
        if np.any(L[1] - L[0] <= 0) or np.any(L[2] - L[1] <= 0):
            sys.exit(f"levels of {sf} must be low < base < high in every channel: {d}")
        levels[sf] = L
    if missing:
        sys.exit(f"{len(missing)} probe still(s) missing, e.g.\n  " + "\n  ".join(missing[:6]))
    return man, levels, frames, surfaces, stills


def read_still(p: Path, W: int, H: int) -> np.ndarray:
    im = cv2.imread(str(p))
    if im is None:
        sys.exit(f"cannot read {p}")
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)


def load_ref(ref: str, frames: list[int], W: int, H: int, decoder: str) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    if "{frame}" in ref:
        out = {}
        for f in frames:
            p = Path(ref.format(frame=f))
            if not p.exists():
                sys.exit(f"reference still missing: {p}")
            out[f] = ES.load(p, [f], W, H, "diffmap" if decoder == "diffmap" else "ffmpeg")[f]
        return out, ES.src_size(Path(ref.format(frame=frames[0])))
    p = Path(ref)
    if not p.exists():
        sys.exit(f"reference not found: {ref}")
    return ES.load(p, frames, W, H, decoder), ES.src_size(p)


# ---------------------------------------------------------------- the model
class Frame:
    def __init__(self, f, R, G, lo, hi, m, lv):
        """lv: (surfaces, [low, base, high], channels)."""
        self.f, self.R, self.G, self.lo, self.hi, self.m = f, R, G, lo, hi, m
        self.b, self.dl, self.dh = lv[:, 1], lv[:, 1] - lv[:, 0], lv[:, 2] - lv[:, 1]
        H, W = R.shape[:2]
        self.windows = []  # per surface: (compute window, evaluation slice inside it, eval window) or None
        for k in range(lo.shape[0]):
            act = (np.abs(lo[k]).max(-1) > 1.0) | (np.abs(hi[k]).max(-1) > 1.0)
            if not act.any():
                self.windows.append(None)
                continue
            ys, xs = np.nonzero(act)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            ey0, ey1, ex0, ex1 = max(0, y0 - REACH), min(H, y1 + REACH), max(0, x0 - REACH), min(W, x1 + REACH)
            cy0, cy1, cx0, cx1 = max(0, y0 - 2 * REACH), min(H, y1 + 2 * REACH), max(0, x0 - 2 * REACH), min(W, x1 + 2 * REACH)
            self.windows.append(((slice(cy0, cy1), slice(cx0, cx1)), (slice(ey0 - cy0, ey1 - cy0), slice(ex0 - cx0, ex1 - cx0)),
                                 (slice(ey0, ey1), slice(ex0, ex1)), int(act.sum())))
        self.n = int(m.sum())

    def contrib(self, k: int, c: int, v: float, win=None) -> np.ndarray:
        lo, hi = (self.lo[k, ..., c], self.hi[k, ..., c]) if win is None else (self.lo[k][win][..., c], self.hi[k][win][..., c])
        b = self.b[k, c]
        return lo * ((b - v) / self.dl[k, c]) if v <= b else hi * ((v - b) / self.dh[k, c])

    def predict(self, sol: np.ndarray) -> np.ndarray:
        P = self.G.copy()
        for k in range(sol.shape[0]):
            for c in range(3):
                P[..., c] += self.contrib(k, c, sol[k, c])
        return P

    def err_sum(self, P: np.ndarray) -> float:
        return float(ES.error_map(self.R, np.clip(P, 0, 255).astype(np.uint8))[self.m].sum())


def score(frames: list[Frame], sol: np.ndarray) -> tuple[float, dict[int, float]]:
    per = {fr.f: fr.err_sum(fr.predict(sol)) / max(1, fr.n) for fr in frames}
    return float(np.mean(list(per.values()))), per


def descend(frames: list[Frame], sol: np.ndarray, free: list[int], bounds: dict[int, tuple[np.ndarray, np.ndarray]],
            step: float, iters: int, refine: int, log) -> np.ndarray:
    """exact-metric search per surface. Phase 1 (--iters rounds): every channel on its own over 0..255 in --step levels (a
    global look, so a far-off start cannot trap it). Phase 2 (--refine rounds): a pattern search from --step down to 1
    level along R, G, B, grey and the opponent directions (diffmap's CIELAB couples the channels, so axis moves alone stall
    in a valley)."""
    P = [fr.predict(sol) for fr in frames]
    full = (np.zeros(3), np.full(3, 255.0))

    def setup(k):
        """per frame: (window, eval slice, eval mask, frame without surface k inside the window)."""
        out = []
        for fr, Q in zip(frames, P):
            w = fr.windows[k]
            if w is None:
                continue
            cw, ev, ew, _ = w
            rest = Q[cw].copy()
            for c in range(3):
                rest[..., c] -= fr.contrib(k, c, sol[k, c], cw)
            out.append((fr, Q, cw, ev, fr.m[ew], rest))
        return out

    def cost(st, v) -> float:
        tot = 0.0
        for fr, Q, cw, ev, mw, rest in st:
            T = rest.copy()
            for c in range(3):
                T[..., c] += fr.contrib(k_cur[0], c, v[c], cw)
            tot += float(ES.error_map(fr.R[cw], np.clip(T, 0, 255).astype(np.uint8))[ev][mw].sum()) / max(1, fr.n)
        return tot

    def commit(st, v):
        for fr, Q, cw, ev, mw, rest in st:
            T = rest.copy()
            for c in range(3):
                T[..., c] += fr.contrib(k_cur[0], c, v[c], cw)
            Q[cw] = T

    k_cur = [0]
    for rnd in range(iters):
        for k in free:
            k_cur[0] = k
            lo_b, hi_b = bounds.get(k, full)
            for c in range(3):
                st = setup(k)
                cur = sol[k].copy()
                best_v, best_e = cur.copy(), cost(st, cur)
                for x in np.unique(np.clip(np.arange(0.0, 256.0, step), lo_b[c], hi_b[c])):
                    v = cur.copy()
                    v[c] = x
                    e = cost(st, v)
                    if e < best_e - 1e-9:
                        best_v, best_e = v, e
                if not np.array_equal(best_v, cur):
                    commit(st, best_v)
                    sol[k] = best_v
        log(f"round {rnd} (each channel over 0..255, step {step:g})", P)
    dirs = [np.array(d, float) for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, -1, 0), (0, 1, -1), (1, 0, -1))]
    for rnd in range(refine):
        moved = 0
        for k in free:
            k_cur[0] = k
            lo_b, hi_b = bounds.get(k, full)
            st = setup(k)
            cur = sol[k].copy()
            e_cur = cost(st, cur)
            s = float(step)
            while s >= 1.0:
                improved = False
                for d in dirs:
                    for sg in (1, -1):
                        v = np.clip(cur + sg * s * d, lo_b, hi_b)
                        if np.array_equal(v, cur):
                            continue
                        e = cost(st, v)
                        if e < e_cur - 1e-9:
                            cur, e_cur, improved = v, e, True
                if not improved:
                    s /= 2
            if not np.array_equal(cur, sol[k]):
                commit(st, cur)
                sol[k] = cur
                moved += 1
        log(f"refine {rnd} (pattern search, step {step:g} -> 1; {moved} surface(s) moved)", P)
        if not moved:
            break
    return sol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frames")
    ap.add_argument("--box")
    ap.add_argument("--mask")
    ap.add_argument("--mask-boxes", choices=["out", "in"], default="out")
    ap.add_argument("--mask-tiles", type=int, default=0)
    ap.add_argument("--fix")
    ap.add_argument("--bounds")
    ap.add_argument("--current")
    ap.add_argument("--init")
    ap.add_argument("--step", type=float, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--refine", type=int, default=6)
    ap.add_argument("--per-frame", action="store_true")
    ap.add_argument("--colours")
    ap.add_argument("--json")
    ap.add_argument("--decoder", choices=["ffmpeg", "diffmap", "cv2"], default="ffmpeg",
                    help="reference decode: ffmpeg = the stream's colour matrix (default); diffmap = exactly diffmap.py's decode; cv2 = OpenCV")
    a = ap.parse_args()

    man, levels, frames, surfaces, stills = read_manifest(Path(a.manifest), a.frames)
    ref0 = Path(a.ref.format(frame=frames[0])) if "{frame}" in a.ref else Path(a.ref)
    if not ref0.exists():
        sys.exit(f"reference not found: {ref0}")
    sw, sh = ES.src_size(ref0)
    W, H = ES.analysis_size(sw, sh)
    R, _ = load_ref(a.ref, frames, W, H, a.decoder)
    bw, bh = ES.src_size(stills[frames[0]]["base"])
    if abs(bw / bh - sw / sh) > 0.01:
        print(f"warning: probe stills are {bw}x{bh}, the reference {sw}x{sh}: aspect differs, both squeezed to {W}x{H}", file=sys.stderr)
    bx0, by0, bx1, by1 = M.box_px(M.parse_box(a.box), sw, sh) if a.box else (0, 0, sw, sh)
    sx, sy = sw / W, sh / H
    rmask = np.zeros((H, W), bool)
    rmask[int(round(by0 / sy)):int(round(by1 / sy)), int(round(bx0 / sx)):int(round(bx1 / sx))] = True
    mask = ES.Mask(a.mask, a.mask_boxes, a.mask_tiles, sw, sh, W, H)

    idx = {s: k for k, s in enumerate(surfaces)}
    fix = {**{k: v for k, v in man.get("fix", {}).items()}, **kv_list(a.fix)}
    bnd_raw = {k: v for k, v in man.get("bounds", {}).items()}
    for k, v in kv_list(a.bounds).items():
        bnd_raw[k] = v.split(":")
    for nm in list(fix) + list(bnd_raw):
        if nm not in idx:
            sys.exit(f"--fix/--bounds names an unknown surface {nm!r} (surfaces: {', '.join(surfaces)})")
    bounds = {idx[k]: (hex3(v[0]), hex3(v[1])) for k, v in bnd_raw.items()}

    LV = np.stack([levels[sf] for sf in surfaces]).astype(np.float64)
    lvs = sorted({" / ".join(tohex(x) for x in levels[sf]) for sf in surfaces})
    print(f"analysis {W}x{H}; {len(surfaces)} surface(s), {len(frames)} frame(s) {frames}; levels low / base / high "
          f"{lvs[0] if len(lvs) == 1 else 'per surface'}; region {bx0},{by0},{bx1},{by1} px; decoder {a.decoder}")
    FR: list[Frame] = []
    for f in frames:
        if f not in R:
            sys.exit(f"the reference has no frame {f}")
        G = read_still(stills[f]["base"], W, H)
        lo = np.stack([read_still(stills[f]["toggles"][s]["low"], W, H) - G for s in surfaces])
        hi = np.stack([read_still(stills[f]["toggles"][s]["high"], W, H) - G for s in surfaces])
        m = mask.at(f, (bx0, by0)) & rmask
        if not m.any():
            sys.exit(f"frame {f}: no scored pixels (box / mask exclude everything)")
        FR.append(Frame(f, R[f], G, lo, hi, m, LV))

    # per-surface response: where it acts, and what an input of low / base / high renders as there (median over its support)
    resp = {}
    for s, k in idx.items():
        npx = [fr.windows[k][3] if fr.windows[k] else 0 for fr in FR]
        if not any(npx):
            print(f"warning: surface {s} changes no pixel in any frame (toggle = base): it cannot be solved", file=sys.stderr)
        resp[s] = {"px": npx}
    sol = LV[:, 1].copy()
    if a.init:
        for s, v in json.loads(Path(a.init).read_text(encoding="utf-8")).items():
            if s in idx:
                sol[idx[s]] = hex3(v)
    for s, v in fix.items():
        sol[idx[s]] = hex3(v)
    free = [k for s, k in idx.items() if s not in fix and any(resp[s]["px"])]
    for k in free:  # an --init / base start outside a surface's --bounds is clamped into them (the descent only checks moves)
        if k in bounds:
            lo_b, hi_b = bounds[k]
            sol[k] = np.clip(sol[k], lo_b, hi_b)

    base_score, base_pf = score(FR, LV[:, 1].copy())
    print(f"all surfaces at base (= the base still): {base_score:.3f}")
    cur_score = None
    if a.current:
        curj = json.loads(Path(a.current).read_text(encoding="utf-8"))
        cur = LV[:, 1].copy()
        unknown = [s for s in curj if s not in idx]
        if unknown:
            print(f"warning: --current lists surfaces not in the manifest (ignored): {unknown}", file=sys.stderr)
        for s, v in curj.items():
            if s in idx:
                cur[idx[s]] = hex3(v)
        cur_score, cur_pf = score(FR, cur)
        print(f"current colours (predicted): {cur_score:.3f}")
    start, _ = score(FR, sol)
    print(f"start: {start:.3f}")

    def log(what, P):
        s = float(np.mean([fr.err_sum(Q) / max(1, fr.n) for fr, Q in zip(FR, P)]))
        print(f"{what}: {s:.3f}", flush=True)

    sol = descend(FR, sol, free, bounds, a.step, a.iters, a.refine, log)
    final, final_pf = score(FR, sol)

    print(f"\nsolved (predicted, diffmap units, lower is better): {final:.3f}   [all at base {base_score:.3f}"
          + (f", current {cur_score:.3f}" if cur_score is not None else "") + "]")
    print(f"{'surface':16s} {'input':8s} {'px/frame':>9s}   ref median there   predicted there   (support = where the toggle moves "
          f"the frame by > 25 % of its max)")
    for s, k in idx.items():
        refm, prdm = [], []
        for fr in FR:
            amp = np.abs(fr.hi[k]).max(-1) + np.abs(fr.lo[k]).max(-1)
            if amp.max() <= 1:
                continue
            sup = (amp > 0.25 * amp.max()) & fr.m
            if sup.any():
                P = fr.predict(sol)
                refm.append(np.median(fr.R[sup].astype(np.float32), 0))
                prdm.append(np.median(np.clip(P[sup], 0, 255), 0))
        rm = tohex(np.mean(refm, 0)) if refm else "-"
        pm = tohex(np.mean(prdm, 0)) if prdm else "-"
        tag = " (fixed)" if s in fix else (" (no response)" if k not in free and s not in fix else "")
        resp[s].update({"input": tohex(sol[k]), "ref_median": rm, "pred_median": pm, "fixed": s in fix})
        print(f"{s:16s} {tohex(sol[k]):8s} {int(np.mean(resp[s]['px'])):9d}   {rm:16s}   {pm:15s}{tag}")
    if a.per_frame:
        for f in frames:
            print(f"  f{f}: base {base_pf[f]:.3f}" + (f"  current {cur_pf[f]:.3f}" if cur_score is not None else "")
                  + f"  solved {final_pf[f]:.3f}")
    hit = [s for s, k in idx.items() if k in free and any(sol[k, c] in (0.0, 255.0) for c in range(3))]
    if hit:
        print(f"note: {', '.join(hit)} solved to 0 or 255 in some channel: the render cannot reach the reference there "
              f"(a light / grade / geometry limit, or the surface covers the wrong pixels). Check 1:1 crops before using it.")
    print(f"\nWARNING: {WARN}")
    colours = {s: tohex(sol[k]) for s, k in idx.items()}
    if a.colours:
        M.dump_json(a.colours, colours)
    M.dump_json(a.json, {"frames": frames, "analysis": [W, H], "region_px": [bx0, by0, bx1, by1], "decoder": a.decoder,
                         "levels": {sf: [tohex(x) for x in levels[sf]] for sf in surfaces}, "colours": colours,
                         "score": {"all_base": base_score, "current": cur_score, "start": start, "solved": final},
                         "per_frame": {str(f): {"all_base": base_pf[f], "solved": final_pf[f]} for f in frames},
                         "surfaces": resp, "warning": WARN})
    return 0


if __name__ == "__main__":
    sys.exit(main())
