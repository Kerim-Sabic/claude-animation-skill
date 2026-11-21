"""animation-quality skill: WHERE on screen does our render differ most from a reference, and did a change help?

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/diffmap.py REF OURS --out DIR [--tag NAME] [--regions regions.json]
         [--baseline DIR/diff.json] [--tol 0.10] [--heat 4] [--frames 0-149]

  REF, OURS   two videos (any size/fps; compared frame by frame from frame 0 over the shorter one) or two images.
              REF is whatever the user supplied to match against: a reference clip, an approved earlier render, a still.
  --regions   optional JSON file, or inline JSON, {"name": [x0, y0, x1, y1], ...} in 0..1 frame fractions (e.g. {"hero head": [0.4,0.3,0.6,0.42]}).
              Tiles inside a named box get that label; every other tile is labelled by height: upper / middle / lower.
  --baseline  diff.json of an earlier run against the SAME reference (e.g. before your change). Prints the per-region deltas
              and the keep verdict: KEEP only if the mean improved AND no region got worse by more than --tol.
  --frames    only this inclusive frame range (videos).
  --heat N    write heat maps (reference | ours | ours with the difference overlaid) for the N worst frames.

Method: both inputs scaled to 270x480 (or 480x270 for landscape), lightly blurred (sigma 1.5) so a 1-2 px misplacement does not
dominate; per pixel CIELAB dE76 plus half the difference of local L contrast (so a blurred wall vs a crisp one counts even when the
colours match); averaged in 15x15 analysis tiles. A region's SHARE = its summed tile error / the total, so the top rows are where
the eye sees the most difference (large and wrong beats small and very wrong). Numbers rank work; they never replace looking.

Writes DIR/diff.json, DIR/diff.md and DIR/heat_fNNNN.png. Exit 0; exit 1 when --baseline says REVERT.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
T = 15


def size_of(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in IMG:
        im = cv2.imread(str(path))
        return im.shape[1], im.shape[0]
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json",
                        str(path)], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return s["width"], s["height"]


def load(path: Path, W: int, H: int, rng: tuple[int, int] | None) -> np.ndarray:
    if path.suffix.lower() in IMG:
        im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)[None]
    vf = f"scale={W}:{H}:flags=area"
    if rng:
        vf = f"select='between(n\\,{rng[0]}\\,{rng[1]})'," + vf
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def lab(rgb: np.ndarray) -> np.ndarray:
    x = cv2.GaussianBlur(rgb.astype(np.float32) / 255.0, (0, 0), 1.5)
    return cv2.cvtColor(x, cv2.COLOR_RGB2Lab)


def local_std(L: np.ndarray) -> np.ndarray:
    m = cv2.GaussianBlur(L, (0, 0), 2.0)
    return np.sqrt(np.maximum(cv2.GaussianBlur(L * L, (0, 0), 2.0) - m * m, 0))


def error_map(r: np.ndarray, o: np.ndarray) -> np.ndarray:
    a, b = lab(r), lab(o)
    return np.sqrt(((a - b) ** 2).sum(axis=2)) + 0.5 * np.abs(local_std(a[..., 0]) - local_std(b[..., 0]))


def tile_labels(W: int, H: int, regions: dict) -> list[list[str]]:
    rows = []
    for ty in range(H // T):
        row = []
        for tx in range(W // T):
            cx, cy = (tx + 0.5) * T / W, (ty + 0.5) * T / H
            name = next((k for k, b in regions.items() if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]), None)
            row.append(name or ("upper" if cy < 1 / 3 else "middle" if cy < 2 / 3 else "lower"))
        rows.append(row)
    return rows


def heat_png(r, o, e, path: Path, title: str):
    h = np.clip(e / 40.0, 0, 1)
    cm = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    over = (0.45 * o + 0.55 * cm).astype(np.uint8)
    big = [cv2.resize(x, (x.shape[1] * 2, x.shape[0] * 2), interpolation=cv2.INTER_NEAREST) for x in (r, o, over)]
    sheet = np.concatenate(big, axis=1)
    bar = np.full((40, sheet.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    for i, t in enumerate(["REFERENCE", "OURS", "DIFFERENCE"]):
        cv2.putText(bar, t, (i * big[0].shape[1] + big[0].shape[1] - 150, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate([bar, sheet], axis=0), cv2.COLOR_RGB2BGR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("ours")
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default="run")
    ap.add_argument("--regions"); ap.add_argument("--baseline"); ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--heat", type=int, default=4); ap.add_argument("--frames")
    a = ap.parse_args()
    ref, ours, out = Path(a.ref), Path(a.ours), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w0, h0 = size_of(ref)
    w1, h1 = size_of(ours)
    if abs(w0 / h0 - w1 / h1) > 0.01:
        print(f"WARNING: aspect ratios differ (ref {w0}x{h0}, ours {w1}x{h1}); both are squeezed to the same size, so the numbers are not meaningful")
    W, H = (270, 480) if h0 >= w0 else (480, 270)
    rng = tuple(int(x) for x in a.frames.split("-")) if a.frames else None
    R, O = load(ref, W, H, rng), load(ours, W, H, rng)
    n = min(len(R), len(O))
    if n == 0:
        print("nothing decoded", file=sys.stderr); return 2
    if len(R) != len(O):
        print(f"note: frame counts differ (ref {len(R)}, ours {len(O)}); comparing the first {n}")
    regions = {}
    if a.regions:  # a JSON file, or the JSON itself inline (bash <(...) does not work with Windows Python)
        regions = json.loads(a.regions if a.regions.lstrip().startswith("{") else Path(a.regions).read_text(encoding="utf-8"))
    labels = tile_labels(W, H, regions)
    f0 = rng[0] if rng else 0
    reg_sum, reg_n = defaultdict(float), defaultdict(int)
    per_frame, maps = [], {}
    for i in range(n):
        e = error_map(R[i], O[i])
        per_frame.append(round(float(e.mean()), 3))
        tiles = e[: (H // T) * T, : (W // T) * T].reshape(H // T, T, W // T, T).mean(axis=(1, 3))
        for ty, row in enumerate(labels):
            for tx, lb in enumerate(row):
                reg_sum[lb] += float(tiles[ty, tx]); reg_n[lb] += 1
        maps[i] = e
    total = sum(reg_sum.values()) or 1.0
    regions_out = {k: {"mean": round(reg_sum[k] / reg_n[k], 3), "share": round(reg_sum[k] / total, 4)} for k in reg_sum}
    mean = round(float(np.mean(per_frame)), 3)
    worst = sorted(range(n), key=lambda i: -per_frame[i])[: a.heat]
    for i in worst:
        heat_png(R[i], O[i], maps[i], out / f"heat_f{f0 + i:04d}.png", f"{a.tag}  frame {f0 + i}  mean dE {per_frame[i]:.2f}")
    res = {"tag": a.tag, "ref": str(ref), "ours": str(ours), "frames": n, "first_frame": f0, "analysis_size": [W, H],
           "mean": mean, "regions": regions_out, "per_frame": per_frame, "worst_frames": [f0 + i for i in worst]}
    md = [f"# diffmap {a.tag}", "", f"ref `{ref}` vs ours `{ours}`, {n} frame(s) from {f0}, mean difference **{mean}** (lower is better)", "",
          "| region | share of difference | mean |", "|---|---|---|"]
    for k, v in sorted(regions_out.items(), key=lambda kv: -kv[1]["share"]):
        md.append(f"| {k} | {100 * v['share']:.1f} % | {v['mean']:.2f} |")
    md += ["", f"worst frames: {res['worst_frames']} (heat maps heat_fNNNN.png)"]
    rc = 0
    if a.baseline:
        b = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
        md += ["", f"## vs baseline `{b['tag']}` (mean {b['mean']})", "", "| region | before | now | delta |", "|---|---|---|---|"]
        worse = []
        for k, v in regions_out.items():
            bv = b["regions"].get(k, {}).get("mean")
            if bv is None:
                continue
            d = v["mean"] - bv
            md.append(f"| {k} | {bv:.2f} | {v['mean']:.2f} | {d:+.2f} |")
            if d > a.tol:
                worse.append(f"{k} {d:+.2f}")
        keep = mean < b["mean"] and not worse
        verdict = "KEEP" if keep else "REVERT"
        why = f"mean {b['mean']} -> {mean}" + (f"; worse regions: {', '.join(worse)}" if worse else "")
        md += ["", f"**{verdict}** ({why}; tolerance {a.tol}). Look at the heat maps and crops before you believe it."]
        res["baseline"] = {"tag": b["tag"], "mean": b["mean"], "verdict": verdict, "worse": worse}
        rc = 0 if keep else 1
    (out / "diff.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "diff.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return rc


if __name__ == "__main__":
    sys.exit(main())
