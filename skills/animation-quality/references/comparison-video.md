# The review package

People judge animation by watching it next to what it should look like. Give the reviewer one folder per round with the
videos, one still sheet and a short README in plain words. They should be able to answer in a line or two.

If any panel contains reference pixels, the folder is private study material: never upload or publish it.

## Files
1. **Three-panel video per clip**: REFERENCE | BEFORE (the round they last watched) | NOW, 3 × 720 wide:
   `python scripts/panel_video.py --out round07/CLIP_ref_before_now.mp4 REFERENCE=ref.mp4 BEFORE=prev.mp4 NOW=now.mp4`
2. **Full-size side by side per clip**: REFERENCE | NOW at 1080 wide each:
   `python scripts/panel_video.py --panel-w 1080 --out round07/CLIP_side_by_side.mp4 REFERENCE=ref.mp4 NOW=now.mp4`
3. **Close-ups**: one row per chosen frame, cropped to where the change is (faces, shading, a set detail):
   `python scripts/crops.py --out round07/closeups.png --frames 40,120 --box 0.3,0.2,0.7,0.5 --col-w 720 REFERENCE=ref.mp4 BEFORE=prev.mp4 NOW=now.mp4`
4. **Audio**: each panel's audio is its own track; say in the README which track is which.
   With no reference, a single BEFORE | NOW pair.

## README template
```
# Round NN (YYYY-MM-DD)

**Files:** what each file is, panel order, size. Audio: track 1 = ..., track 2 = ...

**Your last note:** "<the reviewer's last words, verbatim>"

## What changed
- 3-6 bullets in plain words, about what they will SEE (not code names).

## Measured difference from the reference (lower is better)
| Clip | Last round | Now |
|---|---|---|

## Honest read
What is clearly closer, what is not yet, and the one question to ask.

Frames checked: <which frames, which files>.

Nothing is approved until you say so.
```

## Rules
- Before writing the README, open the videos at a few frames and the close-up sheet yourself, and list exactly what you
  checked. Do not describe a change you did not see in the output.
- Sweep every ~5th frame, and every frame of dissolves and fast moves, for new flaws before you send it.
- Record the reviewer's reply word for word in the next round's README.
