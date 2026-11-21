"""animation-quality skill: thin DARK SHAPES (legs, arms, poles, outlines of limbs) in a box: where they are, how wide, what colour,
reference vs ours, per frame.

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/limb_track.py REF=ref.mp4 OURS=ours.mp4 --frames 150-250:5
         (--box x0,y0,x1,y1 | --boxes boxes.json) [--row-step 8] [--dark 20] [--width 6,60] [--at-y Y] [--band 12]
         [--json out.json]

  --box     where the limbs are (px or 0..1 fractions); scanned row by row every --row-step px.
  --dark    a pixel is "dark" when its L* is more than this below the row's median L* inside the box (the background).
  --width   keep dark runs whose width (px) is in this range: drops single ink pixels and big dark areas.
  --at-y    also report the limb CENTRES (x) on the rows within +-band px of this y: runs are clustered along x (a gap of more
            than half the max width starts a new limb), one median centre per limb. Reference limbs are paired with ours
            by nearest centre and the offset ours - ref (px) printed: that is the number to put into a per-frame shift.

Prints per source and frame: number of runs, width median [IQR], core colour (median RGB of the middle half of each run)
with its L*, a*, b*, and the median darkest L* of the runs; with --at-y the limb centres and offsets.
Pick the box so it holds only the limbs (and plain background): a dark shadow, outline or prop in the box becomes a "limb".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep the skill folder free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import measurelib as M  # noqa: E402


def runs(img: np.ndarray, box, step: int, dark: float, wmin: int, wmax: int):
    x0, y0, x1, y1 = M.box_px(box, img.shape[1], img.shape[0])
    L, _, _ = M.lab_planes(img[y0:y1, x0:x1])
    out = []
    for yy in range(0, y1 - y0, step):
        row = L[yy]
        dk = row < np.median(row) - dark
        x = 0
        while x < len(row):
            if dk[x]:
                e = x
                while e < len(row) and dk[e]:
                    e += 1
                w = e - x
                if wmin <= w <= wmax:
                    a, b = x + w // 4, e - w // 4
                    out.append(dict(y=y0 + yy, x0=x0 + x, x1=x0 + e - 1, centre=x0 + (x + e - 1) / 2, width=w,
                                    core=img[y0 + yy, x0 + a:x0 + b].reshape(-1, 3), minL=float(row[x:e].min())))
                x = e
            else:
                x += 1
    return out


def limbs_at(rs, y, band, wmax):
    xs = sorted(r["centre"] for r in rs if abs(r["y"] - y) <= band)
    if not xs:
        return []
    groups, cur = [], [xs[0]]
    for v in xs[1:]:
        if v - cur[-1] > wmax / 2:
            groups.append(cur)
            cur = [v]
        else:
            cur.append(v)
    groups.append(cur)
    return [float(np.median(g)) for g in groups]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--box")
    ap.add_argument("--boxes")
    ap.add_argument("--row-step", type=int, default=8)
    ap.add_argument("--dark", type=float, default=20.0)
    ap.add_argument("--width", default="6,60")
    ap.add_argument("--at-y", type=float)
    ap.add_argument("--band", type=float, default=12.0)
    ap.add_argument("--json")
    M.add_decoder_arg(ap)
    a = ap.parse_args()
    M.DECODER = a.decoder
    wmin, wmax = (int(v) for v in a.width.split(","))
    srcs = M.parse_sources(a.sources)
    frames = M.parse_frames(a.frames)
    boxes = M.Boxes(a.box, a.boxes)
    data = M.load_all(srcs, frames)
    ref = srcs[0][0]
    res: dict = {}
    for f in frames:
        cen = {}
        for k, _ in srcs:
            im = data[k].get(f)
            if im is None:
                continue
            rs = runs(im, boxes.get(f), a.row_step, a.dark, wmin, wmax)
            if not rs:
                print(f"f{f} {k:6s} no dark runs"); res.setdefault(str(f), {})[k] = None
                continue
            core = np.median(np.concatenate([r["core"] for r in rs]), 0)[::-1]
            Lc, ac, bc = M.rgb_to_lab(core)
            wid = np.array([r["width"] for r in rs])
            r = dict(runs=len(rs), width_median=float(np.median(wid)), width_iqr=[float(np.percentile(wid, 25)), float(np.percentile(wid, 75))],
                     core_rgb=core.astype(int).tolist(), core_lab=[Lc, ac, bc], min_L=float(np.median([q["minL"] for q in rs])))
            if a.at_y is not None:
                cen[k] = r["centres_at_y"] = limbs_at(rs, a.at_y, a.band, wmax)
            res.setdefault(str(f), {})[k] = r
            print(f"f{f} {k:6s} runs {r['runs']} width {r['width_median']:.1f} [{r['width_iqr'][0]:.0f}-{r['width_iqr'][1]:.0f}] "
                  f"core RGB {tuple(r['core_rgb'])} L {Lc:.1f} a {ac:+.0f} b {bc:+.0f} darkest L {r['min_L']:.1f}"
                  + (f" | centres at y {a.at_y:.0f}: " + ", ".join(f"{v:.0f}" for v in r["centres_at_y"]) if a.at_y is not None else ""))
        if a.at_y is not None and ref in cen:
            for k, _ in srcs[1:]:
                if k not in cen or not cen[k] or not cen[ref]:
                    continue
                pairs = [(c, min(cen[k], key=lambda v: abs(v - c))) for c in cen[ref]]
                res[str(f)][k]["offsets_vs_ref"] = [dict(ref=c, ours=o, dx=o - c) for c, o in pairs]
                print(f"      {k} - {ref} at y {a.at_y:.0f}: " + ", ".join(f"ref {c:.0f} -> {o:.0f} ({o - c:+.0f})" for c, o in pairs))
    M.dump_json(a.json, dict(frames=frames, results=res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
