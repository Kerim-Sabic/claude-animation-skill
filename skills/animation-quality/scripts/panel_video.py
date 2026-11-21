"""animation-quality skill: a side-by-side comparison video. Two or three videos side by side, each with a label bar on top.

usage (paths relative to the skill folder, or give the full path to the script):
  python scripts/panel_video.py --out OUT.mp4 [--panel-w 720] [--audio-from 1,3]
         LABEL=PATH LABEL=PATH [LABEL=PATH]

  typical: REFERENCE=ref.mp4 "BEFORE=renders/prev.mp4" NOW=renders/now.mp4   (3 x 720 wide = 2160)
           REFERENCE=ref.mp4 NOW=renders/now.mp4 --panel-w 1080                 (full-size side by side)
  --audio-from  1-based panel numbers whose audio become audio tracks 1, 2, ... (default: every panel that has audio).
                Players switch tracks (VLC: Audio > Audio Track).

Every panel is scaled to the same width keeping its aspect ratio, padded to the tallest, cut to the shortest panel's length
(-t), 30 fps, H.264 CRF 18, AAC 192k. Labels are drawn with OpenCV into a PNG bar (no font paths needed).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def probe(p: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json",
                        str(p)], capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    st = j["streams"]
    v = next(s for s in st if s["codec_type"] == "video")
    return {"w": v["width"], "h": v["height"], "audio": any(s["codec_type"] == "audio" for s in st),
            "dur": float(j["format"]["duration"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--panel-w", type=int, default=720); ap.add_argument("--audio-from")
    a = ap.parse_args()
    srcs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.sources]
    if len(srcs) < 2 or any(not p.exists() for _, p in srcs):
        print("need 2-3 existing LABEL=PATH videos", file=sys.stderr); return 2
    info = [probe(p) for _, p in srcs]
    PW = a.panel_w - a.panel_w % 2
    ph = max(round(i["h"] * PW / i["w"] / 2) * 2 for i in info)
    BAR = 56
    tmp = Path(tempfile.mkdtemp(prefix="panel_"))
    bar = np.full((BAR, PW * len(srcs), 3), 20, np.uint8)
    for k, (label, _) in enumerate(srcs):
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cv2.putText(bar, label, (k * PW + (PW - tw) // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (245, 245, 245), 3, cv2.LINE_AA)
    cv2.imwrite(str(tmp / "bar.png"), bar)
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for _, p in srcs:
        cmd += ["-i", str(p)]
    dur = min(i["dur"] for i in info)   # the looped label image is endless: cut everything to the shortest panel
    cmd += ["-loop", "1", "-framerate", "30", "-i", str(tmp / "bar.png")]
    n = len(srcs)
    fl = [f"[{i}:v]fps=30,scale={PW}:-2,pad={PW}:{ph}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]" for i in range(n)]
    fl.append("".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[row]")
    fl.append(f"[{n}:v]fps=30[bar];[bar][row]vstack=inputs=2:shortest=1,format=yuv420p[out]")
    cmd += ["-filter_complex", ";".join(fl), "-map", "[out]"]
    picks = [int(x) - 1 for x in a.audio_from.split(",")] if a.audio_from else [i for i in range(n) if info[i]["audio"]]
    for i in picks:
        if info[i]["audio"]:
            cmd += ["-map", f"{i}:a:0"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}", a.out]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:], file=sys.stderr); return 1
    o = probe(Path(a.out))
    print(f"{a.out} {o['w']}x{o['h']} audio={o['audio']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
