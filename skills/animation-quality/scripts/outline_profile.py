"""animation-quality skill: the OUTLINE (ink line) WIDTH and INK COLOUR around a filled shape, by angle sector, reference vs ours.

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/outline_profile.py REF=ref.mp4 OURS=ours.mp4 --frames 40-160:20
         (--box x0,y0,x1,y1 | --boxes boxes.json) [--fill warm|pale|"L=62:,absa=14,b=-22:17"|JSON]
         [--step 30] [--out-max 0.12] [--min-step 12] [--close 15] [--key-deg D] [--per-frame]
         [--fit] [--skip 60-120,...] [--w0 PX] [--json out.json]

  sources   LABEL=PATH, reference first (videos: exact frame n from 0; a still is used for every frame).
  --box     a SEARCH box around the shape (px or 0..1 fractions); --boxes JSON {"frame": box} for a moving shape.
            Make it a little bigger than the shape INCLUDING its outline; the mask is the largest fill component inside it.
  --fill    the colour test for the shape's FILL (not the ink), CIELAB / RGB, bounds exclusive.
            warm = L>66, -6<a<30, 2<b<45 (tan / peach skin); pale = L>58, |a|<14, -30<b<12 (white / grey-pink skin).
            Custom: "L=60:,a=-10:20,b=-5:30" (absa / absb take one upper bound: "absa=14"), or a measurelib JSON
            condition ({"L": [60, null], ...} or a list of them = OR), e.g. for a coloured vest.
  --step    sector width in degrees (sectors centred on 0, step, 2 step ...).
  --out-max how far out each normal is scanned, in mask widths (default 0.12 W).
  --min-step  minimum L* drop from the darker side to the line's core; less = no line there (point rejected).
  --key-deg the key light direction (screen degrees it comes FROM): prints the width on the lit side and on the far side.
  --per-frame  also print each frame's sector widths and cos fit.
  --fit     also fit a directional-ink model (a line thick on one side, thin on the other) to the pooled sectors:
            width(theta) = thin + (thick - thin) * smoothstep(lo, hi, cos(theta - pole)); grid search pole (5 deg steps),
            lo -0.9..0.45, hi lo+0.2..1.0, least squares for thin / thick, sectors weighted by sqrt(points read).
  --skip    sectors left out of the --fit model fit, e.g. "60-120" (under a hat brim). Their rows are still printed;
            the cos fit always uses every read sector (dropping one side leaves it unconstrained).
  --w0      the outline rule's base width in px at this size: --fit then also prints thick / thin as factors on it.

What it measures (per source):
  mask = largest component of the fill test in the box, opened 3, closed --close, holes filled = the INNER edge of the ink
  (the shape without its outline). W = mask width (px). Its contour is smoothed (2 % of W); every 2nd contour point
  samples L* along the OUTWARD normal (bilinear, 0.25 px steps) from -4 px to out-max x W. The ink run = the contiguous
  samples around the darkest point (searched from -2 px to 0.06 W) darker than half way between that minimum and the
  darker side (inside = median L* 3-6 px in; outside = the brightest L* within 0.05 W + 8 px past the minimum); width =
  the distance between the two sub-sample half-way crossings. A point is rejected when the step is < --min-step or the
  run touches the scan end (a dark hat / hair / background behind it: no readable outline).
  Ink colour = median RGB of the run's middle third. Angles = the outward normal's screen angle (CCW from screen-right,
  90 = up, 270 = down).
  Pooled table: every frame's widths are divided by that frame's W and scaled to the median W, then per sector: points
  read (n), median width px [q25-q75], width / W, the ink's L* minimum (median) and RGB. Sectors with < 3 points: "--".
  Cos fit over the read sectors: width = a + b cos(theta - phi): phi = the THICK pole, thick a+b, thin a-b on the far
  side, ratio = thick / thin.

Read it like this: a hand-inked reference usually has ONE thick pole on the side away from the key and a thin lit side
(ratio 2-4); a ratio near 1 is a uniform ring. Compare the ink RGB as well as the width: a thinner line in a lighter ink
is less ink than a reference's dark line. Register the shape first (register.py).

Caveats: sectors under a hat brim read the brim, not the head (skip them in the fit); a sector where something touches
the shape (a collar, a straw, a hand) can be bimodal (wide IQR, two widths); a sector with a handful of points (n < ~10)
is not a reading; shapes under ~80 px wide are noisy (the
line is 1-4 px); a body / clothing mask takes the largest component only (one panel of a vest, not both).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep the skill folder free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import measurelib as M  # noqa: E402

PRESETS = {"warm": {"L": [66, None], "a": [-6, 30], "b": [2, 45]},
           "pale": {"L": [58, None], "a": [-14, 14], "b": [-30, 12]}}


def parse_fill(s: str):
    if s in PRESETS:
        return PRESETS[s]
    if s.strip()[:1] in "{[":
        return json.loads(s)
    out: dict = {}
    for part in s.split(","):
        k, _, r = part.partition("=")
        lo, _, hi = r.partition(":")
        k = k.strip()
        if k in ("absa", "absb"):
            out[k] = float(hi) if hi else float(lo)
        else:
            out[k] = [float(lo) if lo else None, float(hi) if hi else None]
    return out


def parse_skip(s: str | None) -> list[tuple[float, float]]:
    out = []
    for r in (s or "").split(","):
        if r.strip():
            a, b = r.split("-")
            out.append((float(a), float(b)))
    return out


def skipped(deg: float, skip) -> bool:
    return any(a <= deg <= b for a, b in skip)


def shape_mask(crop, fill, close):
    m = M.morph(M.condition_mask(crop, fill), cv2.MORPH_OPEN, 3)
    m = M.pick_component(m)
    if m.sum() < 200:
        return None
    return M.fill_holes(M.morph(m, cv2.MORPH_CLOSE, close))


def profile(img, box, fill, out_max, min_step, close):
    H, Wd = img.shape[:2]
    x0, y0, x1, y1 = M.box_px(box, Wd, H)
    m = shape_mask(img[y0:y1, x0:x1], fill, close)
    if m is None:
        return None
    L = M.lab_planes(img)[0].astype(np.float32)
    rgb = img[..., ::-1].astype(np.float32)
    ys, xs = np.nonzero(m)
    W = float(xs.max() - xs.min() + 1)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0, :].astype(np.float64)
    n = len(c)
    k = max(2, int(round(0.02 * W)))
    ker = np.ones(2 * k + 1) / (2 * k + 1)  # circular smoothing of the contour
    sm = np.stack([np.convolve(np.r_[c[-k:, i], c[:, i], c[:k, i]], ker, "valid") for i in (0, 1)], 1)
    t = np.roll(sm, -1, 0) - np.roll(sm, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([t[:, 1], -t[:, 0]], 1)
    probe = np.round(sm + 4 * nrm).astype(int)  # outward = a step along the normal leaves the mask
    inside = m[np.clip(probe[:, 1], 0, m.shape[0] - 1), np.clip(probe[:, 0], 0, m.shape[1] - 1)]
    nrm[inside] *= -1
    ds = np.arange(-4.0, out_max * W, 0.25)
    srch = (ds >= -2) & (ds <= 0.06 * W)
    inner = (ds >= -6) & (ds <= -3)
    rows = []
    for i in range(0, n, 2):
        px, py = sm[i, 0] + x0, sm[i, 1] + y0
        X = (px + nrm[i, 0] * ds).astype(np.float32)[None, :]
        Y = (py + nrm[i, 1] * ds).astype(np.float32)[None, :]
        prof = cv2.remap(L, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
        j = int(np.argmin(np.where(srch, prof, 1e9)))
        lmin = float(prof[j])
        lin = float(np.median(prof[inner]))
        far = (ds > ds[j]) & (ds <= ds[j] + 0.05 * W + 8)
        lout = float(prof[far].max()) if far.any() else lmin
        lside = min(lin, lout)
        if lside - lmin < min_step:
            continue
        thr = lmin + 0.5 * (lside - lmin)
        a = b = j
        while a > 0 and prof[a - 1] < thr:
            a -= 1
        while b < len(ds) - 1 and prof[b + 1] < thr:
            b += 1
        if b >= len(ds) - 2 or a == 0:
            continue

        def cross(i0, i1):
            p0, p1 = prof[i0], prof[i1]
            f = 0.0 if p1 == p0 else (thr - p0) / (p1 - p0)
            return ds[i0] + f * (ds[i1] - ds[i0])
        ain, bout = cross(a - 1, a), cross(b, b + 1)
        w = float(bout - ain)
        core = (ds >= ain + 0.33 * w) & (ds <= bout - 0.33 * w)
        if not core.any():
            core = np.zeros_like(ds, bool)
            core[j] = True
        cx_ = (px + nrm[i, 0] * ds[core]).astype(np.float32)[None, :]
        cy_ = (py + nrm[i, 1] * ds[core]).astype(np.float32)[None, :]
        col = np.median(cv2.remap(rgb, cx_, cy_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0], 0)
        ang = (math.degrees(math.atan2(-nrm[i, 1], nrm[i, 0])) + 360) % 360
        rows.append((ang, w, lmin, col))
    return dict(W=W, rows=rows, contour=n, centre=[float(xs.mean() + x0), float(ys.mean() + y0)])


def sectors(rows, step, W):
    out = []
    for cdeg in range(0, 360, step):
        sel = [r for r in rows if ((r[0] - cdeg + step / 2) % 360) < step]
        if len(sel) < 3:
            out.append(dict(deg=cdeg, n=len(sel)))
            continue
        w = np.array([r[1] for r in sel])
        out.append(dict(deg=cdeg, n=len(sel), px=float(np.median(w)), q25=float(np.percentile(w, 25)),
                        q75=float(np.percentile(w, 75)), perW=float(np.median(w) / W),
                        lmin=float(np.median([r[2] for r in sel])),
                        rgb=[int(round(v)) for v in np.median(np.array([r[3] for r in sel]), 0)]))
    return out


def cos_fit(secs, skip=()):
    ok = [s for s in secs if "px" in s and not skipped(s["deg"], skip)]
    if len(ok) < 4:
        return None
    th = np.radians([s["deg"] for s in ok])
    A = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)
    sol, *_ = np.linalg.lstsq(A, np.array([s["px"] for s in ok]), rcond=None)
    a, p, q = (float(v) for v in sol)
    b = math.hypot(p, q)
    return dict(a=a, b=b, phi=(math.degrees(math.atan2(q, p)) + 360) % 360, thick=a + b, thin=a - b,
                ratio=(a + b) / max(1e-6, a - b))


def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def model_fit(secs, skip=()):
    """a directional-ink model: thin + (thick - thin) * smoothstep(lo, hi, cos(theta - pole))."""
    S = [s for s in secs if "px" in s and not skipped(s["deg"], skip)]
    if len(S) < 4:
        return None
    th = np.radians([s["deg"] for s in S])
    y = np.array([s["px"] for s in S])
    wt = np.sqrt([s["n"] for s in S])
    best = None
    for pole in range(0, 360, 5):
        c = np.cos(th - math.radians(pole))
        for lo in np.arange(-0.9, 0.5, 0.05):
            for hi in np.arange(lo + 0.2, 1.01, 0.05):
                s = smoothstep(lo, hi, c)
                A = np.stack([1 - s, s], 1) * wt[:, None]
                sol, *_ = np.linalg.lstsq(A, y * wt, rcond=None)
                e = float(np.sum((A @ sol - y * wt) ** 2) / np.sum(wt ** 2))
                if best is None or e < best[0]:
                    best = (e, pole, float(lo), float(hi), float(sol[0]), float(sol[1]))
    e, pole, lo, hi, thin, thick = best
    pred = thin + (thick - thin) * smoothstep(lo, hi, np.cos(th - math.radians(pole)))
    if thick < thin:  # the same curve seen from the other side: report the pole at the THICK side
        thin, thick, pole, lo, hi = thick, thin, (pole + 180) % 360, -hi, -lo
    return dict(thick=thick, thin=thin, pole=pole, lo=round(lo, 2), hi=round(hi, 2), rms=math.sqrt(e),
                per_sector=[dict(deg=S[i]["deg"], px=float(y[i]), model=float(pred[i])) for i in range(len(S))])


def ang_dist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--box")
    ap.add_argument("--boxes")
    ap.add_argument("--fill", default="warm")
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--out-max", type=float, default=0.12)
    ap.add_argument("--min-step", type=float, default=12)
    ap.add_argument("--close", type=int, default=15)
    ap.add_argument("--key-deg", type=float)
    ap.add_argument("--per-frame", action="store_true")
    ap.add_argument("--fit", action="store_true", help="fit a directional-ink model (thick/thin/pole/lo/hi)")
    ap.add_argument("--skip", help="sectors left out of the --fit model fit, e.g. 60-120 (hat brim)")
    ap.add_argument("--w0", type=float, help="outline rule base width px: --fit prints thick/thin as factors on it")
    ap.add_argument("--json")
    M.add_decoder_arg(ap)
    a = ap.parse_args()
    M.DECODER = a.decoder
    srcs = M.parse_sources(a.sources)
    frames = M.parse_frames(a.frames)
    boxes = M.Boxes(a.box, a.boxes)
    fill = parse_fill(a.fill)
    skip = parse_skip(a.skip)
    data = M.load_all(srcs, frames)
    res: dict = {}
    for lab, _ in srcs:
        allrows, Ws, per = [], [], {}
        for f in frames:
            im = data[lab].get(f)
            if im is None:
                continue
            r = profile(im, boxes.get(f), fill, a.out_max, a.min_step, a.close)
            if r is None:
                print(f"{lab} f{f}: no fill mask in the box")
                continue
            Ws.append(r["W"])
            allrows += [(x[0], x[1] / r["W"], x[2], x[3]) for x in r["rows"]]  # widths in W, rescaled after pooling
            s = sectors(r["rows"], a.step, r["W"])
            per[str(f)] = dict(W=r["W"], read=len(r["rows"]), centre=r["centre"], sectors=s, fit=cos_fit(s))
            if a.per_frame:
                ft = per[str(f)]["fit"]
                print(f"{lab} f{f} W {r['W']:.0f} read {len(r['rows'])} " + " ".join(f"{x['deg']}:{M.fmt(x.get('px'))}" for x in s)
                      + (f"  fit thick {ft['thick']:.1f} @ {ft['phi']:.0f} thin {ft['thin']:.1f}" if ft else ""))
        if not Ws:
            continue
        Wm = float(np.median(Ws))
        s = sectors([(x[0], x[1] * Wm) + tuple(x[2:]) for x in allrows], a.step, Wm)
        ft = cos_fit(s)
        out = dict(frames=len(Ws), Wmed=Wm, sectors=s, fit=ft, per_frame=per)
        print(f"\n== {lab}: {len(Ws)} frames, mask W median {Wm:.0f} px (widths scaled to it) ==")
        print(" sector   n   width px [q25-q75]   /W     ink L*  ink RGB")
        for x in s:
            tag = "  (skipped in the model fit)" if skipped(x["deg"], skip) else ""
            if "px" not in x:
                print(f"  {x['deg']:3d}   {x['n']:4d}   --{tag}")
                continue
            print(f"  {x['deg']:3d}   {x['n']:4d}   {x['px']:5.1f} [{x['q25']:4.1f}-{x['q75']:4.1f}]  {x['perW']:.4f}  "
                  f"{x['lmin']:5.1f}  {tuple(x['rgb'])}{tag}")
        if ft:
            print(f" cos fit: thick {ft['thick']:.1f} px at {ft['phi']:.0f} deg, thin {ft['thin']:.1f} px at {(ft['phi'] + 180) % 360:.0f} deg,"
                  f" ratio {ft['ratio']:.2f}, mean {ft['a']:.1f} px ({ft['a'] / Wm:.4f} W)")
        else:
            print(" cos fit: fewer than 4 read sectors")
        if a.key_deg is not None:
            def near(d):
                c = [x for x in s if "px" in x and ang_dist(x["deg"], d) <= a.step / 2 + 0.1]
                return c[0]["px"] if c else float("nan")
            lit, far = near(a.key_deg % 360), near((a.key_deg + 180) % 360)
            out["key"] = dict(deg=a.key_deg, lit_px=lit, far_px=far)
            print(f" key {a.key_deg:.0f}: lit side {M.fmt(lit)} px, far side {M.fmt(far)} px")
        if a.fit:
            mf = model_fit(s, skip)
            out["model_fit"] = mf
            if mf is None:
                print(" model fit: fewer than 4 read sectors")
            else:
                print(f" model fit: thick {mf['thick']:.1f} px thin {mf['thin']:.1f} px pole {mf['pole']} lo {mf['lo']:.2f} hi {mf['hi']:.2f}"
                      f" rms {mf['rms']:.2f} px" + (f"  -> factors on w0 {a.w0:.1f}: thick {mf['thick'] / a.w0:.3f} thin {mf['thin'] / a.w0:.3f}"
                                                   if a.w0 else ""))
                print("   deg:measured/model " + " ".join(f"{int(p['deg'])}:{p['px']:.1f}/{p['model']:.1f}" for p in mf["per_sector"]))
        res[lab] = out
    M.dump_json(a.json, dict(frames=frames, sources={k: str(p) for k, p in srcs}, fill=fill, step=a.step, skip=skip, results=res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
