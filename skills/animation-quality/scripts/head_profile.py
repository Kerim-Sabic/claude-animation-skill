"""animation-quality skill: the SHADE PROFILE of a face / head (or any smooth skin-like shape) on pixels, reference vs ours.

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/head_profile.py REF=ref.mp4 OURS=ours.mp4 --frames 40-160:20
         (--box x0,y0,x1,y1 | --boxes boxes.json) [--skin warm|pale|"L=66:,a=-6:30,b=2:45"] [--use-min-l 55]
         [--step 15] [--sectors 165-375] [--core-d 0.2] [--core-min-l L] [--polar] [--vprof] [--json out.json]

  sources   LABEL=PATH, reference first (videos: exact frame n from 0; a still is used for every frame).
  --box     a SEARCH box around the head (px or 0..1 fractions); --boxes JSON {"frame": box} for a moving head.
            Make it a little bigger than the head; the mask is the largest skin component inside it.
  --skin    the skin colour test (CIELAB, bounds exclusive). warm = L>66, -6<a<30, 2<b<45 (tan / peach faces);
            pale = L>58, |a|<14, -30<b<12 (white / grey-pink faces). Custom: "L=60:,a=-10:20,b=-5:30".
  --use-min-l  only pixels of the mask brighter than this enter the medians (warm default 55: drops ink and pupils).
  --sectors    restrict the crescent fit to these screen angles (e.g. 165-375 = the lower half incl. the right edge).
  --core-min-l only pixels brighter than this count as the lit core (use it when a hat-brim shadow band or a second
               shade sits inside the mask and would pull the core down; e.g. 92 for a near-white face).

What it measures (per source, per frame):
  mask = largest skin component (opened 3x3) -> closed 15x15 -> holes filled (eyes, mouth, marks) = the head's outer shape.
  W = mask width (px); d = distance from the outer contour / W (so numbers compare across head sizes);
  core = median L* at d > core-d (the lit middle of the face).
  Per 15-deg sector (centred on 0, 15, 30 ...): the L*(d) profile in 0.01 W bins;
     plateau = median L* at d 0.01-0.05 (the shade right inside the outline), depth = core - plateau,
     width = first d where L* rises past (plateau + core)/2, t10-90 = the terminator's 10-90 % rise in W.
  Crescent fit over sectors with depth > 2.5: width(theta) = i + o cos(theta - phi): phi = where the shade is widest
  (the shade pole), o = how one-sided it is (o ~ 0 = a ring all round, i.e. an outline glow, not a light).
  plateau = median L* in the arc sectors at d 0.01-0.05 (deepest = core - the deepest sector's depth);
  arc = the sectors whose depth > 0.7 x the deepest; plateau RGB (those sectors, d 0.01-0.06) vs core RGB (d > core-d)
  and their ratio = the shade colour as a multiplier of the lit colour.
  --polar: L* per 45-deg sector x d bins 0/.015/.03/.05/.07/.09/.12/.16/.22/.32. --vprof: L* per 10 % of the mask height
  (centre third of its width), top to bottom.

Read it like this: a reference with a flat plateau to ~0.09 W and a crisp terminator wants a two-tone toon edge; one with
L* rising steadily from the outline wants a soft gradient; a sector with no depth on the lit side means NO edge ring there.
Register the head first (register.py): a correct shade profile drawn on a head that sits 20 px off still scores worse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep the skill folder free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import measurelib as M  # noqa: E402

PRESETS = {"warm": ({"L": [66, None], "a": [-6, 30], "b": [2, 45]}, 55.0),
           "pale": ({"L": [58, None], "a": [-14, 14], "b": [-30, 12]}, 0.0)}
BINS = [0, .015, .03, .05, .07, .09, .12, .16, .22, .32]
DB = np.arange(0, .3, .01)


def head_mask(crop: np.ndarray, skin: dict, use_min_l: float):
    L, _, _ = M.lab_planes(crop)
    sk = M.morph(M.condition_mask(crop, skin), cv2.MORPH_OPEN, 3)
    m = M.pick_component(sk)
    if m.sum() < 50:
        return None
    fill = M.fill_holes(M.morph(m, cv2.MORPH_CLOSE, 15))
    ys, xs = np.nonzero(fill)
    W = int(xs.max() - xs.min() + 1)
    cx, cy = float(xs.mean()), float(ys.mean())
    d = M.edt(fill) / W
    ang = M.angle_map(fill.shape, cx, cy)
    use = m & (L > use_min_l)
    return dict(L=L, m=m, fill=fill, use=use, d=d, ang=ang, W=W, cx=cx, cy=cy,
                bb=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))


def measure(img: np.ndarray, box, skin, use_min_l, step, sect, core_d, polar, vprof, core_min_l=None):
    x0, y0, x1, y1 = M.box_px(box, img.shape[1], img.shape[0])
    c = img[y0:y1, x0:x1]
    h = head_mask(c, skin, use_min_l)
    if h is None:
        return None
    L, use, d, ang = h["L"], h["use"], h["d"], h["ang"]
    corem = use & (d > core_d)
    if core_min_l is not None:
        corem &= L > core_min_l
    core = float(np.median(L[corem])) if corem.sum() else float("nan")
    rows = []
    for s in range(0, 360, step):
        sel = use & M.sector(ang, s, step)
        prof = np.array([np.median(L[q]) if (q := sel & (d >= a) & (d < a + .01)).sum() > 8 else np.nan for a in DB])
        pl = float(np.nanmedian(prof[1:5])) if np.isfinite(prof[1:5]).sum() >= 2 else float("nan")
        inner = M.nanmed(prof[12:20])
        depth = core - pl
        wid = t1090 = float("nan")
        if depth > 2.5:
            ok = np.isfinite(prof) & (DB > .01)
            i = np.where(ok & (prof > (pl + core) / 2))[0]
            wid = float(DB[i[0]]) if len(i) else float("nan")
            t10 = np.where(ok & (prof > pl + .1 * depth))[0]
            t90 = np.where(ok & (prof > pl + .9 * depth))[0]
            if len(t10) and len(t90):
                t1090 = float(DB[t90[0]] - DB[t10[0]])
        inside = sect is None or any(((s - a) % 360) <= (b - a) for a, b in [sect])
        rows.append(dict(deg=s, plateau=pl, inner=inner, depth=depth, width=wid, t1090=t1090, fit=inside))
    R = np.array([[r["deg"], r["depth"], r["width"], r["t1090"]] for r in rows], float)
    okfit = np.isfinite(R[:, 2]) & np.array([r["fit"] for r in rows])
    phi = o = i0 = float("nan")
    if okfit.sum() >= 3:
        th = np.radians(R[okfit, 0])
        A = np.c_[np.ones(okfit.sum()), np.cos(th), np.sin(th)]
        p = np.linalg.lstsq(A, R[okfit, 2], rcond=None)[0]
        i0, o, phi = float(p[0]), float(np.hypot(p[1], p[2])), float(np.degrees(np.arctan2(p[2], p[1])) % 360)
    infit = np.array([r["fit"] for r in rows])
    dep = np.where(infit, R[:, 1], np.nan)
    mx = float(np.nanmax(dep)) if np.isfinite(dep).any() else float("nan")
    arc_secs = R[np.isfinite(dep) & (dep > 0.7 * mx), 0] if np.isfinite(mx) and mx > 2.5 else np.array([])
    rgbP = ratio = None
    plateau = float("nan")
    rgbC = np.median(c[corem], 0)[::-1] if corem.sum() else None
    if len(arc_secs):
        insec = np.zeros(use.shape, bool)
        for s in arc_secs:
            insec |= M.sector(ang, s, step)
        q = use & insec & (d > .01) & (d < .06)
        if q.sum():
            rgbP = np.median(c[q], 0)[::-1]
            plateau = float(np.median(L[q & (d < .05)])) if (q & (d < .05)).sum() else float("nan")
            if rgbC is not None:
                ratio = rgbP / np.maximum(rgbC, 1)
    t_arc = M.nanmed([r["t1090"] for r in rows if r["deg"] in set(arc_secs.tolist())]) if len(arc_secs) else float("nan")
    out = dict(box=[x0, y0, x1, y1], W=h["W"], centre=[h["cx"] + x0, h["cy"] + y0],
               mask_bbox=[h["bb"][0] + x0, h["bb"][1] + y0, h["bb"][2] + x0, h["bb"][3] + y0],
               core=core, plateau=plateau, plateau_min=core - mx, maxdepth=mx, arc=[float(arc_secs.min()), float(arc_secs.max())] if len(arc_secs) else None,
               t1090_arc=t_arc, fit=dict(phi=phi, o=o, i=i0),
               rgb_plateau=rgbP, rgb_core=rgbC, shade_ratio=ratio, sectors=rows)
    if polar:
        pol = {}
        for s in range(8):
            sel = use & M.sector(ang, s * 45, 45)
            pol[s * 45] = [float(np.median(L[q])) if (q := sel & (d >= BINS[i]) & (d < BINS[i + 1])).sum() > 15 else float("nan")
                           for i in range(len(BINS) - 1)]
        out["polar"] = pol
    if vprof:
        bx0, by0, bx1, by1 = h["bb"]
        cx0, cx1 = bx0 + (bx1 - bx0) // 3, bx1 - (bx1 - bx0) // 3
        vp = []
        for i in range(10):
            r0, r1 = by0 + (by1 - by0) * i // 10, by0 + (by1 - by0) * (i + 1) // 10
            q = np.zeros(use.shape, bool)
            q[r0:r1, cx0:cx1] = True
            q &= use
            vp.append(dict(L=float(np.median(L[q])) if q.sum() > 10 else float("nan"),
                           rgb=np.median(c[q], 0)[::-1].astype(int).tolist() if q.sum() > 10 else None))
        out["vprof"] = vp
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--box")
    ap.add_argument("--boxes")
    ap.add_argument("--skin", default="warm")
    ap.add_argument("--use-min-l", type=float)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--sectors", help="a-b screen degrees used for the fit / arc, e.g. 165-375")
    ap.add_argument("--core-d", type=float, default=0.2)
    ap.add_argument("--core-min-l", type=float)
    ap.add_argument("--polar", action="store_true")
    ap.add_argument("--vprof", action="store_true")
    ap.add_argument("--json")
    M.add_decoder_arg(ap)
    a = ap.parse_args()
    M.DECODER = a.decoder
    if a.skin in PRESETS:
        skin, uml = PRESETS[a.skin]
    else:
        skin, uml = M.parse_ranges(a.skin), 0.0
    if a.use_min_l is not None:
        uml = a.use_min_l
    sect = None
    if a.sectors:
        s0, s1 = (float(v) for v in a.sectors.split("-"))
        sect = (s0 % 360, s0 % 360 + (s1 - s0))
    srcs = M.parse_sources(a.sources)
    frames = M.parse_frames(a.frames)
    boxes = M.Boxes(a.box, a.boxes)
    data = M.load_all(srcs, frames)
    res: dict = {}
    for f in frames:
        for k, _ in srcs:
            im = data[k].get(f)
            if im is None:
                continue
            r = measure(im, boxes.get(f), skin, uml, a.step, sect, a.core_d, a.polar, a.vprof, a.core_min_l)
            res.setdefault(str(f), {})[k] = r
            if r is None:
                print(f"f{f} {k:6s} no skin mask in the box"); continue
            rp = tuple(int(v) for v in r["rgb_plateau"]) if r["rgb_plateau"] is not None else None
            rc = tuple(int(v) for v in r["rgb_core"]) if r["rgb_core"] is not None else None
            rt = tuple(round(float(v), 3) for v in r["shade_ratio"]) if r["shade_ratio"] is not None else None
            print(f"f{f} {k:6s} W{r['W']} c({r['centre'][0]:.0f},{r['centre'][1]:.0f}) core {M.fmt(r['core'])} plateau {M.fmt(r['plateau'])} (deepest {M.fmt(r['plateau_min'])}) "
                  f"maxdepth {M.fmt(r['maxdepth'])} arc {r['arc']} t10-90 {M.fmt(r['t1090_arc'], 3)} "
                  f"fit phi {M.fmt(r['fit']['phi'], 0)} o {M.fmt(r['fit']['o'], 3)} i {M.fmt(r['fit']['i'], 3)} "
                  f"P{rp} C{rc} P/C {rt}")
            print("        deg:plateau/depth/width/t10-90  " + " ".join(
                f"{s['deg']}:{M.fmt(s['plateau'], 0)}/{M.fmt(s['depth'], 1)}/{M.fmt(s['width'], 2)}/{M.fmt(s['t1090'], 2)}"
                for s in r["sectors"] if np.isfinite(s["depth"]) and s["depth"] > 2.5 and s["fit"]))
            if a.polar:
                print("        polar d-bins " + " ".join(f"{b:.3f}" for b in BINS[:-1]))
                for s, row in r["polar"].items():
                    print(f"          {s:3d}  " + " ".join(f"{M.fmt(v):>5s}" for v in row))
            if a.vprof:
                print("        vprof L top->bottom " + " ".join(M.fmt(v["L"], 0) for v in r["vprof"]))
    M.dump_json(a.json, dict(frames=frames, sources={k: str(p) for k, p in srcs}, skin=skin, use_min_l=uml, results=res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
