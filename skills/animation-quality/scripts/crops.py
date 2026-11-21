"""animation-quality skill: a labelled still sheet for judging by eye. One row per frame, one column per source.

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/crops.py --out SHEET.png --frames 0,60,120 [--box x0,y0,x1,y1] [--col-w 540]
         LABEL=PATH [LABEL=PATH ...]

  LABEL=PATH  a video (frame n is taken exactly, counted from 0) or a still (used for every row), e.g.
              REFERENCE=ref.mp4 "BEFORE=renders/a.mp4" NOW=renders/b.mp4
  --box       crop box in 0..1 frame fractions (e.g. a face: 0.35,0.25,0.65,0.45); default the whole frame
  --col-w     width of each column in px (the crop is scaled to it)

Look at the result at full size (Read the PNG). Crops at 1:1 or larger show line weight, shading and texture that a
full-frame thumbnail hides.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def grab(path: Path, f: int) -> np.ndarray | None:
    if path.suffix.lower() in IMG:
        return cv2.imread(str(path))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{f})", "-vsync", "0", "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--frames", required=True)
    ap.add_argument("--box"); ap.add_argument("--col-w", type=int, default=540)
    a = ap.parse_args()
    srcs = []
    for s in a.sources:
        if "=" not in s:
            print(f"source must be LABEL=PATH: {s}", file=sys.stderr); return 2
        k, p = s.split("=", 1); srcs.append((k, Path(p)))
    frames = [int(x) for x in a.frames.split(",")]
    box = [float(x) for x in a.box.split(",")] if a.box else [0, 0, 1, 1]
    rows = []
    for f in frames:
        cells = []
        for label, p in srcs:
            im = grab(p, f)
            if im is None:
                im = np.zeros((960, 540, 3), np.uint8); label += " (no frame)"
            h, w = im.shape[:2]
            c = im[int(box[1] * h): int(box[3] * h), int(box[0] * w): int(box[2] * w)]
            ch = max(1, round(c.shape[0] * a.col_w / c.shape[1]))
            c = cv2.resize(c, (a.col_w, ch), interpolation=cv2.INTER_AREA if c.shape[1] > a.col_w else cv2.INTER_NEAREST)
            bar = np.full((34, a.col_w, 3), 24, np.uint8)
            cv2.putText(bar, f"{label}  f{f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
            cells.append(np.concatenate([bar, c], axis=0))
        hmax = max(x.shape[0] for x in cells)
        cells = [np.pad(x, ((0, hmax - x.shape[0]), (0, 4), (0, 0))) for x in cells]
        rows.append(np.concatenate(cells, axis=1))
    wmax = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 4), (0, wmax - r.shape[1]), (0, 0))) for r in rows], axis=0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print(f"{a.out} {sheet.shape[1]}x{sheet.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
