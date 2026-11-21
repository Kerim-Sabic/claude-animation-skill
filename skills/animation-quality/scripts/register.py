"""animation-quality skill: REGISTER a character on the reference's pixels: per frame and per part (face, hat, body ...), where
does ours sit and how big is it compared with the reference? Run this BEFORE any shading / colour test on that character.

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/register.py REF=ref.mp4 OURS=ours.mp4 --frames 45-90:5 --parts parts.json
         [--scale 0.88,1.20,0.02] [--shift 24] [--json out.json]

  sources   LABEL=PATH, reference first; every other source is registered onto the reference.
  --parts   JSON {"part name": SPEC, ...}. SPEC:
              "box":   [x0,y0,x1,y1] px or 0..1 fractions, or {"frame": box, ...} per frame (nearest frame used).
                       ONE box for both sources: make it big enough to hold the part in the reference AND in ours.
              "where": colour test, one {...} or a list of {...} OR'ed together; each {...} ANDs channel bounds
                       (exclusive): "L","a","b" (CIELAB, L 0..100), "R","G","B" (0..255), "R-B","R-G","G-B" as [lo, hi] (null = open),
                       "absa"/"absb" as a single upper bound on |a| / |b|.
              "pick":  "largest" (default) | "all" | "seed" (the component with most pixels inside "seed": [x0,y0,x1,y1]
                       in the same units as box);
              "open": k, "close": k (square kernels, px; defaults 5 / 0), "fill": true|false (fill holes; default true).
              "rel":   the name of an EARLIER part: "box" is then in units of that part's bbox IN THE SAME SOURCE
                       (x = x0 + u*W, y = y0 + v*H), e.g. a hat box above the face, a body box below it. Use this when
                       ours and the reference sit far apart: each source gets its own box around its own character.
            Example (a pale face, a dark hat above it, a red sweater below; replace numbers with yours):
              {"face":    {"box": [150,650,560,960], "where": {"L": [62,null], "absa": 14, "b": [-35,12]}, "close": 15},
               "hat":     {"rel": "face", "box": [-0.8,-1.3,1.8,0.1], "where": {"L": [null,40]}, "open": 1},
               "sweater": {"rel": "face", "box": [-0.3,1.0,1.3,2.6], "pick": "all", "fill": false, "close": 9,
                           "where": [{"R": [150,null], "G": [null,110], "B": [null,120]}]}}
  --scale   lo,hi,step for sx and sy (scale of ours about its own centroid); --shift  search +-px around the centroid offset (default 40).

Prints per frame and part: the reference's and ours' bbox and centroid, ours - ref (centroid and bbox centre, px), the size
ratio ours/ref (bbox W and H), the mask IoU as it is, and the best scale + shift that maps OURS ONTO THE REFERENCE
(sx, sy about ours' centroid, then dx, dy: negative dx = move ours left) with the IoU it reaches. Search: every scale pair
on the grid with every integer shift within +-shift px of the centroid offset (one cross-correlation per scale pair, full
resolution), then a 0.005 scale grid +-0.02 around the best pair.

How to use the numbers: a consistent offset over a hold (e.g. ours +20 px right, -20 px high) is a placement fix (card or
camera knob), a scale > 1 means ours is smaller, sy != sx means a different head shape / proportion. A best IoU that stays
low (< 0.75) means the masks do not describe the same thing: check them before trusting the shift. Re-run after the fix:
the offset should be within 2-3 px and the IoU high. Only then test shading on that part.
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

import measurelib as M  # noqa: E402


def part_box(spec, f, w, h, done):
    """the part's search box in px for this source; "rel" boxes are in units of another part's bbox in THE SAME source."""
    b = spec["box"]
    if isinstance(b, dict):
        per = {int(k): v for k, v in b.items()}
        b = per[f if f in per else min(per, key=lambda q: abs(q - f))]
    if spec.get("rel"):
        anc = done.get(spec["rel"])
        if anc is None or not anc.any():
            return None
        ys, xs = np.nonzero(anc)
        ax0, ay0, aw, ah = xs.min(), ys.min(), np.ptp(xs) + 1, np.ptp(ys) + 1
        b = [ax0 + b[0] * aw, ay0 + b[1] * ah, ax0 + b[2] * aw, ay0 + b[3] * ah]
        return M.box_px([max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])] if max(b) > 1 else b, w, h)
    return M.box_px(b, w, h)


def part_mask(img: np.ndarray, spec: dict, f: int, done: dict) -> np.ndarray:
    h, w = img.shape[:2]
    full = np.zeros((h, w), bool)
    bx = part_box(spec, f, w, h, done)
    if bx is None:
        return full
    x0, y0, x1, y1 = bx
    c = img[y0:y1, x0:x1]
    m = M.condition_mask(c, spec["where"])
    m = M.morph(m, cv2.MORPH_OPEN, int(spec.get("open", 5)))
    seed = None
    if spec.get("pick") == "seed":
        sx0, sy0, sx1, sy1 = M.box_px(spec["seed"], w, h)
        seed = (max(0, sx0 - x0), max(0, sy0 - y0), max(1, sx1 - x0), max(1, sy1 - y0))
    m = M.pick_component(m, spec.get("pick", "largest"), seed)
    m = M.morph(m, cv2.MORPH_CLOSE, int(spec.get("close", 0)))
    if spec.get("fill", True):
        m = M.fill_holes(m)
    full[y0:y1, x0:x1] = m
    return full


def iou(a, b) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def warp(m: np.ndarray, sx, sy, dx, dy, c) -> np.ndarray:
    A = np.float32([[sx, 0, c[0] * (1 - sx) + dx], [0, sy, c[1] * (1 - sy) + dy]])
    return cv2.warpAffine(m.astype(np.uint8), A, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST).astype(bool)


def search(A: np.ndarray, B: np.ndarray, scales, shift: int):
    """best (iou, sx, sy, dx, dy) mapping B onto A: B scaled by (sx, sy) about its own centroid, then moved by (dx, dy).
    For each scale pair every integer shift within +-shift px of the centroid offset is scored at once with a
    cross-correlation (intersection counts); IoU = I / (|A| + |B'| - I). Then a finer scale grid around the best pair."""
    ya, xa = np.nonzero(A)
    yb, xb = np.nonzero(B)
    S = int(shift)
    smax = max(scales) + 0.03
    pad = int(S + (smax - 1) * max(np.ptp(xb), np.ptp(yb)) + abs(xa.mean() - xb.mean()) + abs(ya.mean() - yb.mean()) + 8)
    X0, Y0 = max(0, min(xa.min(), xb.min()) - pad), max(0, min(ya.min(), yb.min()) - pad)
    X1, Y1 = min(A.shape[1], max(xa.max(), xb.max()) + pad + 1), min(A.shape[0], max(ya.max(), yb.max()) + pad + 1)
    a = np.zeros((Y1 - Y0 + 2 * S, X1 - X0 + 2 * S), np.float32)
    a[S:-S or None, S:-S or None] = A[Y0:Y1, X0:X1]
    b = B[Y0:Y1, X0:X1]
    c = (xb.mean() - X0, yb.mean() - Y0)
    d0 = (round(xa.mean() - xb.mean()), round(ya.mean() - yb.mean()))
    nA = float(A.sum())

    def score(sx, sy):
        bw = warp(b, sx, sy, d0[0], d0[1], c)
        ys, xs = np.nonzero(bw)
        if not len(xs):
            return (0.0, sx, sy, 0.0, 0.0)
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        tpl = bw[by0:by1, bx0:bx1].astype(np.float32)
        img = a[by0:by1 + 2 * S, bx0:bx1 + 2 * S]
        inter = cv2.matchTemplate(img, tpl, cv2.TM_CCORR)
        iu = inter / np.maximum(nA + float(tpl.sum()) - inter, 1)
        j, i = np.unravel_index(int(np.argmax(iu)), iu.shape)
        return (float(iu[j, i]), float(sx), float(sy), float(d0[0] + i - S), float(d0[1] + j - S))

    best = max(score(sx, sy) for sx in scales for sy in scales)
    _, bsx, bsy, _, _ = best
    fine = np.arange(-0.02, 0.0201, 0.005)
    best = max([best] + [score(bsx + u, bsy + v) for u in fine for v in fine])
    return best


def stats(m: np.ndarray) -> dict:
    ys, xs = np.nonzero(m)
    return dict(bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], W=int(np.ptp(xs) + 1), H=int(np.ptp(ys) + 1),
                centroid=[float(xs.mean()), float(ys.mean())], bbox_centre=[(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2],
                area=int(m.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--parts", required=True, help="JSON file or inline JSON")
    ap.add_argument("--scale", default="0.88,1.20,0.02")
    ap.add_argument("--shift", type=int, default=40)
    ap.add_argument("--min-area", type=int, default=500)
    ap.add_argument("--json")
    M.add_decoder_arg(ap)
    a = ap.parse_args()
    M.DECODER = a.decoder
    parts = json.loads(Path(a.parts).read_text(encoding="utf-8") if Path(a.parts).exists() else a.parts)
    lo, hi, st = (float(v) for v in a.scale.split(","))
    scales = [float(v) for v in np.arange(lo, hi + st / 2, st)]
    srcs = M.parse_sources(a.sources)
    frames = M.parse_frames(a.frames)
    data = M.load_all(srcs, frames)
    ref = srcs[0][0]
    res: dict = {}
    for f in frames:
        if f not in data[ref]:
            continue
        done: dict = {k: {} for k, _ in srcs}
        for name, spec in parts.items():
            A = done[ref][name] = part_mask(data[ref][f], spec, f, done[ref])
            for k, _ in srcs[1:]:
                if f not in data[k]:
                    continue
                B = done[k][name] = part_mask(data[k][f], spec, f, done[k])
                row = res.setdefault(str(f), {}).setdefault(name, {})
                if A.sum() < a.min_area or B.sum() < a.min_area:
                    row[k] = dict(status="empty", ref_area=int(A.sum()), area=int(B.sum()))
                    print(f"f{f} {name:10s} {k}: mask too small (ref {A.sum()} px, {k} {B.sum()} px): check the box / colour test")
                    continue
                sa, sb = stats(A), stats(B)
                v, sx, sy, dx, dy = search(A, B, scales, a.shift)
                off = [sb["centroid"][0] - sa["centroid"][0], sb["centroid"][1] - sa["centroid"][1]]
                offb = [sb["bbox_centre"][0] - sa["bbox_centre"][0], sb["bbox_centre"][1] - sa["bbox_centre"][1]]
                row[k] = dict(ref=sa, ours=sb, offset_centroid=off, offset_bbox_centre=offb, size_ratio=[sb["W"] / sa["W"], sb["H"] / sa["H"]],
                              iou=iou(A, B), best=dict(iou=v, sx=sx, sy=sy, dx=dx, dy=dy))
                print(f"f{f} {name:10s} ref bbox {sa['bbox']} c({sa['centroid'][0]:.0f},{sa['centroid'][1]:.0f}) | {k} bbox {sb['bbox']} "
                      f"c({sb['centroid'][0]:.0f},{sb['centroid'][1]:.0f}) | {k}-ref centroid ({off[0]:+.0f},{off[1]:+.0f}) bbox-centre "
                      f"({offb[0]:+.0f},{offb[1]:+.0f}) size W x{sb['W'] / sa['W']:.3f} H x{sb['H'] / sa['H']:.3f} | IoU {iou(A, B):.3f} -> {v:.3f} "
                      f"at sx {sx:.2f} sy {sy:.2f} dx {dx:+.0f} dy {dy:+.0f}")
                if v < 0.75:
                    print(f"      note: best IoU {v:.2f} < 0.75: the two masks may not be the same thing; look at them before trusting the shift")
    M.dump_json(a.json, dict(frames=frames, parts=parts, results=res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
