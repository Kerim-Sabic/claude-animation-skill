"""animation-quality skill: WHAT KIND of difference is a region's diffmap error? Split it into colour, placement and blur with
per-block "oracle" fixes before touching any knob.

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/error_split.py REF=ref.mp4 OURS=ours.mp4 --frames 150-250:3
         [--box x0,y0,x1,y1] [--block 120] [--shift 40] [--shift-step 8] [--blur 1.6,3.2,4.8,6.4,9.6]
         [--sharpen 0.4,0.8] [--sharpen-sigma 4.8] [--mask FILE] [--mask-boxes out|in] [--mask-tiles 60]
         [--block-stats mask|all] [--per-frame] [--decoder ffmpeg|diffmap|cv2] [--json out.json]

  sources   LABEL=PATH, reference first; every other source is split against the reference (videos: exact frame n from 0;
            a still is used for every frame).
  --frames  e.g. 150-250:3 (every 3rd frame is plenty: the oracles are slow-ish, ~1 s per frame and source).
  --box     the region (px of the reference frame, or 0..1 fractions); default the whole frame.
  --block   block size in reference px (default 120): the colour and shift oracles fix each block on its own.
  --shift   the shift oracle searches every offset within +-shift px (reference px) in steps of --shift-step.
  --blur    Gaussian sigmas (reference px) tried on OURS; the best one is the blur oracle.
  --sharpen unsharp amounts tried on OURS (ours * (1 + k) - k * blur(ours, --sharpen-sigma)); the best is the sharpen oracle.
  --mask    restrict the score to some pixels:
              an image (png/jpg, any size, scaled to the frame): nonzero = scored;
              a JSON of boxes per frame {"<frame>": [[x0,y0,x1,y1], ...], ...} (or one box per frame, px or 0..1); with
              --mask-boxes out (default) the boxes are EXCLUDED (e.g. the characters' boxes, to score the set only), with
              in only the boxes are scored. A frame missing from the JSON uses the nearest listed frame: list every frame
              (an empty list [] = nothing excluded on that frame).
  --mask-tiles PX  decide in / out per PX x PX tile by its centre pixel (diffmap's labelling rule). Tiles start at the
            --box corner: put the box on a 60 px grid with --mask-tiles 60 to get exactly diffmap's tiles.
  --block-stats  mask (default): the colour oracle matches each block's mean over the SCORED pixels only (a character
            inside the block does not tint the set's correction); all: over every pixel of the block (older notes).
  --decoder ffmpeg (default) = the stream's colour matrix, then area-resize to the analysis size; diffmap = one ffmpeg
            area-scale pass exactly as diffmap.py decodes (use it to match a diffmap.py region mean to the decimal);
            cv2 = OpenCV's reader (bt601; only to reproduce old notes).

The metric is diffmap.py's, unchanged: both frames scaled to 270x480 (480x270 landscape), CIELAB of a sigma 1.5 blur, per
pixel dE76 + 0.5 x |local L contrast difference| (sigma 2); the region's error = the mean over its scored pixels, averaged
over frames. Blocks, shifts and sigmas given in reference px are converted to that analysis grid (1/4 for 1080x1920),
so the numbers are directly comparable with diffmap.py's region means.

The oracles (each is a what-if on OURS alone, scored against the untouched reference):
  colour   every block of ours gets the reference block's mean RGB (a per-block offset): what is left is NOT flat colour.
  shift    every block of ours is moved by the integer offset (within +-shift) that scores best: what it removes is
           placement (a block-wise misplacement, or structure that a shift can fake: treat small gains as noise).
  shift+colour  both per block: what is left is shape, detail and texture that neither colour nor placement explains.
  blur     the whole frame of ours blurred (best sigma): a big gain = ours is too SHARP / has too much detail there.
  sharpen  ours sharpened (best amount): a big gain = ours is too SOFT (more blur will not help).
Gains (base - oracle) overlap, so they do not add up; compare their sizes. The split printed is each gain as a share of
the base error. Read it like this: the biggest gain names the first fix (colour -> a grade / material / light knob on
that region; placement -> register.py and camera / layout; blur -> depth of field / texture detail; sharpen -> detail, not
blur). A small shift+colour residual means colour and placement explain the region; a big one means the shapes differ.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep the skill folder free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import measurelib as M  # noqa: E402


# ---------------------------------------------------------------- diffmap's metric (identical to diffmap.py)
def lab(rgb: np.ndarray) -> np.ndarray:
    x = cv2.GaussianBlur(rgb.astype(np.float32) / 255.0, (0, 0), 1.5)
    return cv2.cvtColor(x, cv2.COLOR_RGB2Lab)


def local_std(L: np.ndarray) -> np.ndarray:
    m = cv2.GaussianBlur(L, (0, 0), 2.0)
    return np.sqrt(np.maximum(cv2.GaussianBlur(L * L, (0, 0), 2.0) - m * m, 0))


def error_map(r: np.ndarray, o: np.ndarray) -> np.ndarray:
    a, b = lab(r), lab(o)
    return np.sqrt(((a - b) ** 2).sum(axis=2)) + 0.5 * np.abs(local_std(a[..., 0]) - local_std(b[..., 0]))


# ---------------------------------------------------------------- frames at the analysis size (RGB)
def analysis_size(w: int, h: int) -> tuple[int, int]:
    return (270, 480) if h >= w else (480, 270)


def src_size(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in M.IMG:
        im = cv2.imread(str(path))
        if im is None:
            sys.exit(f"cannot read image {path}")
        return im.shape[1], im.shape[0]
    info = M.probe(path)
    return int(info["width"]), int(info["height"])


def read_diffmap_way(path: Path, frames: list[int], W: int, H: int) -> dict[int, np.ndarray]:
    """exactly diffmap.py's decode: one ffmpeg pass, scale=W:H:flags=area, rgb24."""
    if path.suffix.lower() in M.IMG:
        im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        return {f: im for f in frames}
    want, got, last = set(frames), {}, max(frames)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", f"scale={W}:{H}:flags=area", "-frames:v", str(last + 1),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n, i = W * H * 3, 0
    while i <= last:
        b = pr.stdout.read(n)
        if len(b) < n:
            break
        if i in want:
            got[i] = np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()
        i += 1
    pr.stdout.close()
    pr.wait()
    missing = sorted(want - set(got))
    if missing:
        print(f"warning: {path.name} has no frame(s) {missing[:8]}{'...' if len(missing) > 8 else ''} (decoded {i})", file=sys.stderr)
    return got


def load(path: Path, frames: list[int], W: int, H: int, decoder: str) -> dict[int, np.ndarray]:
    if decoder == "diffmap":
        return read_diffmap_way(path, frames, W, H)
    out = {}
    for f, im in M.read_frames(path, frames, decoder).items():
        out[f] = cv2.cvtColor(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return out


# ---------------------------------------------------------------- mask
class Mask:
    def __init__(self, path: str | None, mode: str, tiles: int, sw: int, sh: int, W: int, H: int):
        self.img = None
        self.per: dict[int, list] = {}
        self.mode, self.sw, self.sh, self.W, self.H = mode, sw, sh, W, H
        self.tiles = tiles
        if not path:
            return
        p = Path(path)
        if not p.exists():
            sys.exit(f"mask not found: {path}")
        if p.suffix.lower() in M.IMG:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                sys.exit(f"cannot read mask image {p}")
            self.img = cv2.resize(im, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
        else:
            raw = json.loads(p.read_text(encoding="utf-8"))
            for k, v in raw.items():
                if v and not isinstance(v[0], (list, tuple)):
                    v = [v]
                self.per[int(k)] = [[float(x) for x in b] for b in v]

    def full(self, f: int) -> np.ndarray | None:
        """the scored pixels at reference size (None = every pixel)."""
        if self.img is not None:
            return self.img
        if not self.per:
            return None
        k = f if f in self.per else min(self.per, key=lambda q: abs(q - f))
        inb = np.zeros((self.sh, self.sw), bool)
        for b in self.per[k]:
            x0, y0, x1, y1 = M.box_px(b, self.sw, self.sh)
            inb[y0:y1, x0:x1] = True
        return inb if self.mode == "in" else ~inb

    def at(self, f: int, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
        """the scored pixels on the analysis grid (tiles start at origin = the region's top-left, reference px)."""
        m = self.full(f)
        if m is None:
            return np.ones((self.H, self.W), bool)
        sx, sy = self.sw / self.W, self.sh / self.H
        if self.tiles:
            out = np.zeros((self.H, self.W), bool)
            T = self.tiles
            for ty in range(origin[1], self.sh, T):
                for tx in range(origin[0], self.sw, T):
                    cy, cx = min(self.sh - 1, int(ty + T / 2)), min(self.sw - 1, int(tx + T / 2))
                    if m[cy, cx]:
                        out[int(round(ty / sy)):int(round((ty + T) / sy)), int(round(tx / sx)):int(round((tx + T) / sx))] = True
            return out
        return cv2.resize(m.astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_AREA) >= 0.5


# ---------------------------------------------------------------- the oracles
def blocks(x0, y0, x1, y1, B):
    for by in range(y0, y1, B):
        for bx in range(x0, x1, B):
            yield by, bx, min(by + B, y1), min(bx + B, x1)


def oracles(r, o, m, region, B, shifts, step, sigmas, amounts, s_sigma, block_stats):
    """one frame: dict of oracle -> mean error over the scored pixels m (analysis grid, region applied already)."""
    H, W = r.shape[:2]
    x0, y0, x1, y1 = region
    res = {"base": float(error_map(r, o)[m].mean())}
    rf, of = r.astype(np.float32), o.astype(np.float32)

    def block_offset(rb, ob, mb):
        if block_stats == "mask" and mb.any():
            return rb[mb].mean(0) - ob[mb].mean(0)
        if block_stats == "mask":
            return np.zeros(3, np.float32)
        return rb.reshape(-1, 3).mean(0) - ob.reshape(-1, 3).mean(0)

    oc = of.copy()
    for by, bx, ey, ex in blocks(x0, y0, x1, y1, B):
        sl = (slice(by, ey), slice(bx, ex))
        oc[sl] += block_offset(rf[sl], of[sl], m[sl])
    res["colour"] = float(error_map(r, np.clip(oc, 0, 255).astype(np.uint8))[m].mean())

    # shift (and shift+colour): the whole frame is moved once per offset; each block keeps its best offset
    pad = 12  # context around a block so the metric's blurs see real neighbours (~3x its sigma)
    blist = [b for b in blocks(x0, y0, x1, y1, B) if m[b[0]:b[2], b[1]:b[3]].any()]
    best = {b: (1e9, 1e9) for b in blist}
    rng = range(-shifts, shifts + 1, step)
    for dy in rng:
        for dx in rng:
            sh = cv2.warpAffine(o, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
            shf = sh.astype(np.float32)
            for b in blist:
                by, bx, ey, ex = b
                cy0, cx0, cy1, cx1 = max(0, by - pad), max(0, bx - pad), min(H, ey + pad), min(W, ex + pad)
                rc, sc = r[cy0:cy1, cx0:cx1], sh[cy0:cy1, cx0:cx1]
                inner = (slice(by - cy0, ey - cy0), slice(bx - cx0, ex - cx0))
                mb = m[by:ey, bx:ex]
                v = float(error_map(rc, sc)[inner][mb].mean())
                if v < best[b][0]:
                    scf = shf[cy0:cy1, cx0:cx1].copy()
                    scf[inner] += block_offset(rf[by:ey, bx:ex], shf[by:ey, bx:ex], mb)
                    vc = float(error_map(rc, np.clip(scf, 0, 255).astype(np.uint8))[inner][mb].mean())
                    best[b] = (v, vc)
    n = [int(m[b[0]:b[2], b[1]:b[3]].sum()) for b in blist]
    res["shift"] = sum(best[b][0] * k for b, k in zip(blist, n)) / max(1, sum(n))
    res["shift+colour"] = sum(best[b][1] * k for b, k in zip(blist, n)) / max(1, sum(n))

    for px, s in sigmas:  # named by the sigma in reference px, applied in analysis px
        res[f"blur {px:g}px"] = float(error_map(r, cv2.GaussianBlur(o, (0, 0), s))[m].mean())
    soft = cv2.GaussianBlur(o, (0, 0), s_sigma).astype(np.float32)
    for k in amounts:
        sharp = np.clip(of * (1 + k) - k * soft, 0, 255).astype(np.uint8)
        res[f"sharpen {k:g}"] = float(error_map(r, sharp)[m].mean())
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--box", help="the region, x0,y0,x1,y1 px or 0..1 (default: the whole frame)")
    ap.add_argument("--block", type=float, default=120)
    ap.add_argument("--shift", type=float, default=40)
    ap.add_argument("--shift-step", type=float, default=8)
    ap.add_argument("--blur", default="1.6,3.2,4.8,6.4,9.6")
    ap.add_argument("--sharpen", default="0.4,0.8")
    ap.add_argument("--sharpen-sigma", type=float, default=4.8)
    ap.add_argument("--mask")
    ap.add_argument("--mask-boxes", choices=["out", "in"], default="out")
    ap.add_argument("--mask-tiles", type=int, default=0)
    ap.add_argument("--block-stats", choices=["mask", "all"], default="mask")
    ap.add_argument("--per-frame", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--decoder", choices=["ffmpeg", "diffmap", "cv2"], default="ffmpeg",
                    help="ffmpeg = the stream's colour matrix (default); diffmap = exactly diffmap.py's decode; cv2 = OpenCV (bt601)")
    a = ap.parse_args()
    srcs = M.parse_sources(a.sources)
    if len(srcs) < 2:
        sys.exit("need the reference and at least one source to split")
    frames = M.parse_frames(a.frames)
    sw, sh = src_size(srcs[0][1])
    W, H = analysis_size(sw, sh)
    sx, sy = sw / W, sh / H
    for k, p in srcs[1:]:
        w1, h1 = src_size(p)
        if abs(w1 / h1 - sw / sh) > 0.01:
            print(f"warning: {k} is {w1}x{h1}, the reference {sw}x{sh}: aspect differs, both are squeezed to {W}x{H}", file=sys.stderr)
    bx0, by0, bx1, by1 = M.box_px(M.parse_box(a.box), sw, sh) if a.box else (0, 0, sw, sh)
    region = (int(round(bx0 / sx)), int(round(by0 / sy)), int(round(bx1 / sx)), int(round(by1 / sy)))
    B = max(2, int(round(a.block / sx)))
    S = int(round(a.shift / sx))
    step = max(1, int(round(a.shift_step / sx)))
    sigmas = [(float(v), float(v) / sx) for v in a.blur.split(",") if v.strip()]
    amounts = [float(v) for v in a.sharpen.split(",") if v.strip()]
    s_sigma = a.sharpen_sigma / sx
    mask = Mask(a.mask, a.mask_boxes, a.mask_tiles, sw, sh, W, H)
    rmask = np.zeros((H, W), bool)
    rmask[region[1]:region[3], region[0]:region[2]] = True
    print(f"analysis {W}x{H} (1 analysis px = {sx:g} x {sy:g} ref px); region {bx0},{by0},{bx1},{by1} px = analysis {region}; "
          f"block {B} ({a.block:g} px); shift +-{S} step {step} (+-{S * sx:g} px step {step * sx:g}); blur sigmas (analysis px) "
          f"{', '.join(f'{s:g}' for _, s in sigmas)}; sharpen {a.sharpen} at sigma {s_sigma:g}; decoder {a.decoder}")
    data = {k: load(p, frames, W, H, a.decoder) for k, p in srcs}
    ref = srcs[0][0]
    out: dict = {"frames": frames, "region_px": [bx0, by0, bx1, by1], "analysis": [W, H], "block_px": a.block, "shift_px": a.shift,
                 "decoder": a.decoder, "block_stats": a.block_stats, "sources": {}}
    for k, _ in srcs[1:]:
        rows = {}
        for f in frames:
            if f not in data[ref] or f not in data[k]:
                continue
            m = mask.at(f, (bx0, by0)) & rmask
            if not m.any():
                if a.per_frame:
                    print(f"{k} f{f}: no scored pixels in the region")
                continue
            res = oracles(data[ref][f], data[k][f], m, region, B, S, step, sigmas, amounts, s_sigma, a.block_stats)
            rows[f] = res
            if a.per_frame:
                print(f"{k} f{f}: " + "  ".join(f"{n} {v:.2f}" for n, v in res.items()) + f"  (scored {int(m.sum())} px)")
        if not rows:
            print(f"{k}: no frame with scored pixels")
            continue
        keys = list(next(iter(rows.values())).keys())
        mean = {n: float(np.mean([r[n] for r in rows.values()])) for n in keys}
        bl = min((n for n in keys if n.startswith("blur ")), key=lambda n: mean[n])
        sp = min((n for n in keys if n.startswith("sharpen ")), key=lambda n: mean[n])
        base = mean["base"]
        gain = {"colour": base - mean["colour"], "placement": base - mean["shift"], "blur": base - mean[bl], "sharpen": base - mean[sp]}
        resid = mean["shift+colour"]
        print(f"\n{k} vs {ref}: {len(rows)} frame(s) with scored pixels, mean over frames (diffmap units)")
        print(f"  base {base:.2f} | colour-matched {mean['colour']:.2f} | best shift +-{S * sx:g} px {mean['shift']:.2f} | "
              f"shift+colour {resid:.2f} | best blur ({bl.split()[1]}) {mean[bl]:.2f} | "
              f"best sharpen ({sp.split()[1]}) {mean[sp]:.2f}")
        print("  every blur:    " + "  ".join(f"{n.split()[1]} {mean[n]:.2f}" for n in keys if n.startswith("blur ")))
        print("  every sharpen: " + "  ".join(f"{n.split()[1]} {mean[n]:.2f}" for n in keys if n.startswith("sharpen ")))
        print("  SPLIT (gain of each oracle, share of base; overlapping, not additive):")
        for n, v in gain.items():
            note = "" if v > 0.05 * base else ("  -> makes it WORSE: not the problem" if v < -0.05 * base else "  -> negligible")
            print(f"    {n:10s} {v:+6.2f}  {100 * v / base:5.1f} %{note}")
        print(f"    left after shift+colour: {resid:.2f} ({100 * resid / base:.0f} % of base): shape / detail / texture")
        for nm, lst in ((bl, [n for n in keys if n.startswith("blur ")]), (sp, [n for n in keys if n.startswith("sharpen ")])):
            if len(lst) > 1 and nm == lst[-1]:
                print(f"  note: the best {nm.split()[0]} is the LAST value tried: extend --{nm.split()[0]} to find its real best")
        top = max(("colour", "placement", "blur", "sharpen"), key=lambda n: gain[n])
        print(f"  biggest single fix: {top}" if gain[top] > 0.05 * base else
              "  biggest single fix: none (no oracle gains 5 % of the base: the difference is shape / detail, or noise)")
        out["sources"][k] = {"per_frame": {str(f): r for f, r in rows.items()}, "mean": mean, "best_blur": bl, "best_sharpen": sp,
                             "gain": gain, "share": {n: v / base for n, v in gain.items()}, "residual_shift_colour": resid}
    M.dump_json(a.json, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
