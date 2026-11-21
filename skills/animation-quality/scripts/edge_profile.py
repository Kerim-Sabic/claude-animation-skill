"""animation-quality skill: EDGE SPREAD (how soft an edge is) at chosen points, reference vs ours, on VIDEO frames.

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/edge_profile.py REF=ref.mp4 OURS=ours.mp4
         (--points points.json | --point F,X,Y[,NORMAL_DEG[,LABEL]] ...) [--tol 0.20] [--search 10] [--profile] [--json out.json]

  sources   LABEL=PATH, reference first. Use the rendered VIDEO for ours, not a still: a soft edge measured on a still and on
            the encoded video of the same frame disagreed by ~0.1 in ratio (and a jpeg + x264 "encode simulation" of the
            still did not predict the video either). Stills are fine for ranking variants, not for passing a gate.
  --points  JSON list [{"f": 70, "pt": [x, y], "normal_deg": 0, "label": "wall edge"}, ...]; px of the reference frame.
            normal_deg = the direction ACROSS the edge (screen angle from +x towards +y DOWN, the image convention: 0 = a
            vertical edge measured left->right, 90 = a horizontal edge measured top->bottom). Omit it to use the local
            gradient direction on the reference.
  --point   the same inline: F,X,Y[,NORMAL_DEG[,LABEL]] (repeatable).
  --tol     PASS when |ours / ref - 1| <= tol.

Method: the luma (OpenCV BGR->gray) profile across the edge, sampled every 0.5 px over +-34 px along the normal (bilinear),
averaged over 7 parallel lines +-6 px along the edge, smoothed by a 3-tap box; the edge is the steepest point within
+-search px; it extends while the slope stays above 5 % of the peak; spread = contrast / peak slope (px; = the 10-90 % rise
of a Gaussian blur, ~2.5 sigma). Rejected (no edge): contrast < 20 levels, or the slope changes sign inside the edge (a
thin line, not a step). The point is first moved onto the reference's own edge centre (search +-14, then +-4 px), and ours
is looked for within +-search px of that same place, so a slightly misplaced edge in ours is still measured.
Prints ref / ours spread, ratio, PASS / FAIL, where ours' edge sits relative to the reference's (px along the normal),
and both contrasts. --profile also prints the two raw profiles (every 2 px) so you can see a step vs a thin line.
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


def bilinear(G: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """order-1 interpolation, coordinates clamped to the image (edge pixels repeat)."""
    h, w = G.shape
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = xs - x0, ys - y0
    top = G[y0, x0] * (1 - fx) + G[y0, x1] * fx
    bot = G[y1, x0] * (1 - fx) + G[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def sample_profile(G, pt, ang, half=34.0, step=0.5, avg=6.0):
    nx, ny = math.cos(ang), math.sin(ang)
    tx, ty = -ny, nx
    offs = np.arange(-half, half + 1e-6, step)
    ks = np.arange(-avg, avg + 1e-6, 2.0)
    acc = np.zeros_like(offs)
    for k in ks:
        acc += bilinear(G, pt[0] + k * tx + offs * nx, pt[1] + k * ty + offs * ny)
    return offs, acc / len(ks)


def grad_angle(G, pt, r=10):
    x, y = int(round(pt[0])), int(round(pt[1]))
    x0, y0, x1, y1 = max(1, x - r), max(1, y - r), min(G.shape[1] - 1, x + r + 1), min(G.shape[0] - 1, y + r + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return 0.0
    w = G[y0:y1, x0:x1]
    gx = cv2.Sobel(w, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(w, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    return 0.5 * math.atan2(2 * jxy, jxx - jyy)


def measure_edge(G, pt, ang, search=10.0):
    offs, p = sample_profile(G, pt, ang)
    ps = np.convolve(p, np.ones(3) / 3, "same")
    g = np.gradient(ps, offs)
    c = len(offs) // 2
    lo, hi = c - int(search / 0.5), c + int(search / 0.5)
    k = lo + int(np.argmax(np.abs(g[lo:hi + 1])))
    if k <= lo + 1 or k >= hi - 1:
        return None
    gs = g * (1.0 if g[k] > 0 else -1.0)
    i = k
    while i > 1 and gs[i - 1] > 0.05 * gs[k]:
        i -= 1
    j = k
    while j < len(p) - 2 and gs[j + 1] > 0.05 * gs[k]:
        j += 1
    contrast = abs(ps[j] - ps[i])
    if contrast < 20.0 or j - i < 1:
        return None
    if np.any(gs[i:j + 1] < -0.15 * gs[k]):
        return None
    return dict(width=float(contrast / gs[k]), contrast=float(contrast), centre=float(offs[k]))


def recentre(G, pt, ang):
    m = measure_edge(G, pt, ang, search=14.0)
    if m is None:
        return None, None
    pt2 = (pt[0] + m["centre"] * math.cos(ang), pt[1] + m["centre"] * math.sin(ang))
    return pt2, (measure_edge(G, pt2, ang, search=4.0) or m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--points")
    ap.add_argument("--point", action="append", default=[])
    ap.add_argument("--tol", type=float, default=0.20)
    ap.add_argument("--search", type=float, default=10.0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--json")
    M.add_decoder_arg(ap)
    a = ap.parse_args()
    M.DECODER = a.decoder
    pts = json.loads(Path(a.points).read_text(encoding="utf-8")) if a.points else []
    for s in a.point:
        v = s.split(",", 4)
        pts.append(dict(f=int(v[0]), pt=[float(v[1]), float(v[2])], normal_deg=float(v[3]) if len(v) > 3 and v[3] != "" else None,
                        label=v[4] if len(v) > 4 else f"f{v[0]}@({v[1]},{v[2]})"))
    if not pts:
        sys.exit("give --points FILE.json or --point F,X,Y[,NORMAL_DEG[,LABEL]]")
    srcs = M.parse_sources(a.sources)
    frames = sorted({int(p["f"]) for p in pts})
    data = M.load_all(srcs, frames)
    gray = {k: {f: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for f, im in fr.items()} for k, fr in data.items()}
    ref = srcs[0][0]
    rows = []
    for p in pts:
        f = int(p["f"])
        label = p.get("label", f"f{f}")
        Gr = gray[ref].get(f)
        if Gr is None:
            continue
        ang = math.radians(p["normal_deg"]) if p.get("normal_deg") is not None else grad_angle(Gr, p["pt"])
        pt2, ra = recentre(Gr, p["pt"], ang)
        for k, _ in srcs[1:]:
            Go = gray[k].get(f)
            row = dict(label=label, f=f, pt=p["pt"], normal_deg=math.degrees(ang), source=k)
            if ra is None:
                row.update(status="UNSCORED", why="no clean edge on the reference at this point")
            elif Go is None:
                row.update(status="UNSCORED", why=f"{k} has no frame {f}")
            else:
                mo = measure_edge(Go, pt2, ang, search=a.search)
                if mo is None:
                    row.update(ref_px=ra["width"], ref_contrast=ra["contrast"], status="FAIL",
                               why=f"no matching edge in {k} within +-{a.search:g} px")
                else:
                    r = mo["width"] / max(ra["width"], 1e-6)
                    row.update(ref_px=ra["width"], ours_px=mo["width"], ratio=r, ours_offset_px=mo["centre"],
                               ref_contrast=ra["contrast"], ours_contrast=mo["contrast"],
                               status="PASS" if abs(r - 1) <= a.tol else "FAIL")
            rows.append(row)
            if "ratio" in row:
                print(f"f{f:<4d} {label[:46]:46s} ref {row['ref_px']:5.2f} {k} {row['ours_px']:5.2f} ratio {row['ratio']:4.2f} {row['status']}"
                      f"  ({k} edge at {row['ours_offset_px']:+.1f} px, contrast {row['ours_contrast']:.0f}; ref contrast {row['ref_contrast']:.0f})")
            else:
                print(f"f{f:<4d} {label[:46]:46s} {row['status']}: {row['why']}")
            if a.profile and ra is not None and Go is not None:
                for nm, G in ((ref, Gr), (k, Go)):
                    offs, pr = sample_profile(G, pt2, ang)
                    print(f"      {nm:6s} " + " ".join(f"{v:.0f}" for v in pr[::4]))
    sc = [r for r in rows if r["status"] != "UNSCORED"]
    print(f"{sum(r['status'] == 'PASS' for r in sc)} PASS / {sum(r['status'] == 'FAIL' for r in sc)} FAIL / "
          f"{len(rows) - len(sc)} UNSCORED (tol +-{a.tol * 100:.0f} %)")
    M.dump_json(a.json, dict(tol=a.tol, rows=rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
