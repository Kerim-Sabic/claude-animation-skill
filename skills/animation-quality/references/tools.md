# The tools

All tools live in `scripts/`. Run them from anywhere: `python path/to/scripts/<tool>.py -h` prints the full usage.
Requirements: Python 3.10+, `numpy`, `opencv-python`, `ffmpeg` and `ffprobe` on the PATH.

## Shared conventions

- **Sources** are `LABEL=PATH`. The first source is the REFERENCE; every other source is compared with it. A video gives
  frame n exactly (counted from 0, decoded in order, no seeking); a still is used for every frame. A source of another
  size is resized to the reference's size (with a warning).
- **Frames**: `40,50,60`, `40-60`, `40-60:5` (every 5th) or a mix: `10,40-60:10`.
- **Boxes**: `x0,y0,x1,y1` in pixels of the reference frame, or 0..1 fractions when every value is <= 1. A per-frame box
  file is JSON `{"<frame>": [x0,y0,x1,y1], ...}`; a missing frame uses the nearest listed frame.
- **Colour**: L\*, a\*, b\* are CIELAB (L\* 0..100). Video is decoded with the stream's own colour matrix (untagged HD =
  bt709). `--decoder cv2` uses OpenCV's reader (bt601), only to reproduce numbers measured that way. Never mix decoders
  inside one comparison: they differ by ~3 levels on average.
- **Angles**: screen degrees counter-clockwise from screen-right (90 = up, 180 = left, 270 = down), except
  `edge_profile.py`'s `normal_deg`, which uses the image convention (y down).
- **The metric** (`diffmap.py`, shared by `error_split.py`, `colour_solve.py`, `set_register.py`): both frames scaled
  to 270x480 (480x270 for landscape), a light blur (sigma 1.5), then per pixel CIELAB dE76 plus half the difference of
  local L contrast (so a blurred wall vs a crisp one counts even when the colours match), averaged in 15x15 tiles.
  It sees colour, brightness, placement and blur / texture. It does not understand shape or acting: it ranks work, it
  does not judge quality.

---

## diffmap.py: where is the difference, and did a change help?

```
python scripts/diffmap.py REF OURS --out DIR [--tag NAME] [--regions regions.json] [--baseline DIR/diff.json]
       [--tol 0.10] [--heat 4] [--frames 0-149]
```
- REF / OURS: two videos (compared frame by frame from frame 0 over the shorter one) or two images.
- `--regions`: a JSON file or inline JSON of named boxes in 0..1 fractions, e.g.
  `{"hero head": [0.4,0.3,0.6,0.42], "hero body": [0.35,0.44,0.65,0.80]}`. Every other tile is labelled
  upper / middle / lower. Name the boxes after what is in the REFERENCE there.
- `--baseline`: the `diff.json` of an earlier run against the same reference. Prints per-region deltas and the verdict:
  **KEEP** only if the mean improved AND no region got worse by more than `--tol`. Exit code 1 on REVERT.
- `--heat N`: heat maps (reference | ours | ours with the difference overlaid) for the N worst frames.
Writes `diff.json`, `diff.md` and `heat_fNNNN.png`. A region's SHARE is its summed tile error over the total, so the
top rows are where the eye sees the most difference.

## error_split.py: what kind of difference is it?

```
python scripts/error_split.py REF=ref.mp4 OURS=ours.mp4 --frames 150-250:3 [--box x0,y0,x1,y1] [--block 120]
       [--shift 40] [--shift-step 8] [--blur 1.6,3.2,4.8,6.4,9.6] [--sharpen 0.4,0.8] [--mask FILE]
       [--mask-boxes out|in] [--mask-tiles 60] [--per-frame] [--decoder ffmpeg|diffmap|cv2] [--json out.json]
```
Runs "oracle" fixes on OURS alone and scores each against the untouched reference:
- **colour**: every block gets the reference block's mean colour. What is left is not flat colour.
- **shift**: every block moved by its best offset (within ±shift). What it removes is placement.
- **shift+colour**: both. What is left is shape, detail and texture.
- **blur**: the whole frame blurred (best sigma). A big gain = ours is too sharp or too detailed there.
- **sharpen**: ours sharpened. A big gain = ours is too soft (more blur will not help).
Gains overlap, so they do not add up; compare their sizes. The biggest gain names the first fix. `--mask` takes a JSON
of boxes per frame (excluded by default, e.g. the characters' boxes to score the set only; `--mask-boxes in` scores only
the boxes) or a black / white image. `--mask-tiles 60` decides in / out per 60 px tile like diffmap's labelling.

## register.py: where does our character sit, and how big is it?

```
python scripts/register.py REF=ref.mp4 OURS=ours.mp4 --frames 45-90:5 --parts parts.json
       [--scale 0.88,1.20,0.02] [--shift 40] [--json out.json]
```
`parts.json` names each part with a search box and a colour test, e.g. a pale face, a dark hat above it and a red
sweater below:
```json
{"face":    {"box": [150,650,560,960], "where": {"L": [62,null], "absa": 14, "b": [-35,12]}, "close": 15},
 "hat":     {"rel": "face", "box": [-0.8,-1.3,1.8,0.1], "where": {"L": [null,40]}, "open": 1},
 "sweater": {"rel": "face", "box": [-0.3,1.0,1.3,2.6], "pick": "all", "fill": false, "close": 9,
             "where": [{"R": [150,null], "G": [null,110], "B": [null,120]}]}}
```
- `where`: one condition or a list OR'ed together; each ANDs exclusive bounds on `L a b R G B R-B R-G G-B` (`[lo, hi]`,
  null = open) and `absa` / `absb` (an upper bound on |a\*| / |b\*|).
- `rel`: the box is in units of an earlier part's bounding box in the SAME source, so each source gets its own box
  around its own character.
- `pick`: `largest` (default), `all`, or `seed` (the component with most pixels inside `seed`).
Prints per frame and part: both bounding boxes and centroids, ours minus ref (px), the size ratio, the mask IoU, and the
best scale + shift that lays ours onto the reference with the IoU it reaches. A consistent offset over a hold is a
placement fix; scale > 1 means ours is smaller; sy != sx means a different proportion. A best IoU under 0.75 means the
masks do not describe the same thing: look at them before trusting the shift.

## set_register.py: camera or layout?

```
python scripts/set_register.py --ref ref.mp4 --normal "stills/f{frame}_C.png" --hidden "stills/f{frame}_H.png"
       [--wide "stills/f{frame}_HW.png" --wide-scale 0.74] [--frames 124,132,140] [--char-mask chars.png]
       [--box ...] [--mask M] [--shift 200] [--shift-step 40] [--scale 0.9:1.35] [--strips 2]
       [--stretch 1.0:1.4 --baseline-y 1400:1600:50] [--pool] [--noise 0.3] [--fov-v DEG] [--save DIR] [--json out.json]
```
Needs our still with the character (C), the same camera with the character hidden (H), and optionally the hidden still
zoomed out (HW, with `--wide-scale` = how big the set looks in it, e.g. exp(-0.3) = 0.741). It searches shifts, zoom and
an optional height stretch as an image warp of the set (no render per candidate), puts our character back on top, and
scores on diffmap's metric, for the whole region and for left / right strips:
- **AGREE**: both strips want the same move → a camera aim error.
- **OPPOSITE**: the strips want opposite shifts → layout or proportion (move set pieces, not the camera).
- **STRETCH wins**: the set stands taller (or shorter) above the same ground line.
Signs: shift +x = the set moves right on screen, +y = down; to get it with the camera, pan / tilt the other way.
Read gains against `sim(identity)`, not the real still: resampling the wide still is softer. An image warp has no
parallax, so always bake the result, render stills and score them for real. On a repeating pattern (windows, laps,
bricks) keep `--shift` under half the repeat: in a test with windows every 120 px and a true shift of +30 px, a ±80 px
search let one strip lock onto the next window (-90 px) and the verdict read OPPOSITE; with ±50 px both strips found +30
and agreed.

## head_profile.py: how is a face shaded?

```
python scripts/head_profile.py REF=ref.mp4 OURS=ours.mp4 --frames 40-160:20 (--box x0,y0,x1,y1 | --boxes boxes.json)
       [--skin warm|pale|"L=66:,a=-6:30,b=2:45"] [--use-min-l 55] [--step 15] [--sectors 165-375] [--core-d 0.2]
       [--core-min-l L] [--polar] [--vprof] [--json out.json]
```
Mask = the largest skin component in the search box, holes filled. W = its width; d = distance from the outline / W.
Prints: the lit core L\*; per 15° sector the shade plateau just inside the outline, its depth, width and the
terminator's 10-90 % rise; a crescent fit (where the shade is widest = the light comes from the other side; how
one-sided it is); the shade colour as a multiplier of the lit colour. A flat plateau to ~0.09 W with a crisp terminator
wants a two-tone toon edge; L\* rising steadily from the outline wants a soft gradient; a sector with no depth on the lit
side means no edge ring there. Register the head first.

## outline_profile.py: how wide and dark is the outline, side by side?

```
python scripts/outline_profile.py REF=ref.mp4 OURS=ours.mp4 --frames 40-160:20 (--box ... | --boxes ...)
       [--fill warm|pale|"L=62:,absa=14,b=-22:17"|JSON] [--step 30] [--out-max 0.12] [--min-step 12] [--close 15]
       [--key-deg D] [--per-frame] [--fit] [--skip 60-120] [--w0 PX] [--json out.json]
```
The fill test finds the shape (not its ink); every second contour point samples L\* along the outward normal and
measures the ink run's width and core colour. Prints per sector: points read, median width [IQR], width / W, ink L\* and
RGB; a cosine fit (the thick side, thick / thin ratio); `--key-deg` the lit vs far side; `--fit` a two-width model
(thin + (thick − thin) · smoothstep(lo, hi, cos(θ − pole))). Hand-inked references usually have one thick side away from
the light (ratio 2-4). Sectors under a hat brim read the brim: skip them in the fit. Shapes under ~80 px wide are noisy.

## edge_profile.py: how soft is an edge?

```
python scripts/edge_profile.py REF=ref.mp4 OURS=ours.mp4 (--points points.json | --point F,X,Y[,NORMAL_DEG[,LABEL]] ...)
       [--tol 0.20] [--search 10] [--profile] [--json out.json]
```
`points.json`: `[{"f": 70, "pt": [x, y], "normal_deg": 0, "label": "wall edge"}, ...]` (px of the reference frame;
`normal_deg` across the edge, 0 = a vertical edge measured left to right, 90 = a horizontal edge top to bottom; omit it
to use the local gradient). Spread = contrast / peak slope (≈ the 10-90 % rise; ≈ 2.5 sigma of a Gaussian blur).
PASS when |ours / ref − 1| <= tol. It measures soft steps, not thin lines (`--profile` prints both profiles so you can
tell). Use the rendered VIDEO for ours: a still and the encoded video of the same frame disagreed by ~0.1 in ratio.

## limb_track.py: where are the legs and arms?

```
python scripts/limb_track.py REF=ref.mp4 OURS=ours.mp4 --frames 150-250:5 (--box ... | --boxes ...) [--row-step 8]
       [--dark 20] [--width 6,60] [--at-y Y] [--band 12] [--json out.json]
```
Scans the box row by row for dark runs (darker than the row's median by `--dark` L\*) of a width inside `--width`.
Prints per frame: run count, width median [IQR], core colour, darkest L\*; with `--at-y` the limb centres on that row and
ours − ref offsets (the number for a per-frame shift). Keep the box to limbs and plain background only.

## colour_solve.py: which input colours render closest to the reference?

```
python scripts/colour_solve.py --ref ref.mp4 --manifest toggles.json [--frames 4,20] [--box ...] [--mask chars.json]
       [--fix wall=#506868] [--bounds trim=#000000:#ffffff] [--current today.json] [--init start.json]
       [--step 8] [--iters 3] [--refine 6] [--colours solved.json] [--json out.json]
```
Through a grade, haze or a coloured light, a colour measured on the reference is not the input that renders as it. This
tool measures what your renderer does to each surface from one batch of stills and solves the inputs on diffmap's metric:
```json
{"levels": {"low": "#202020", "base": "#808080", "high": "#e0e0e0"},
 "frames": [4, 20],
 "surfaces": ["wall", "trim", "roof"],
 "base":   "stills/f{frame}_G.png",
 "toggle": "stills/f{frame}_T{surface}{level}.png",
 "level_tags": {"low": "d", "high": "l"}}
```
`G` = every surface at base; each toggle = that ONE surface at low / high, all others at base, with geometry, light,
grade and camera exactly as the render you compare with. Model: one per-channel piecewise-linear response per surface
(assumes surfaces add and the grade does not mix channels). The prediction ranks candidates and usually runs a little
optimistic: always render the solved colours and score the real stills. A surface solved to 0 or 255 means the render
cannot reach the reference there (light, grade or placement): check 1:1 crops before using it.

## crops.py: a labelled still sheet

```
python scripts/crops.py --out SHEET.png --frames 0,60,120 [--box x0,y0,x1,y1] [--col-w 540] LABEL=PATH [LABEL=PATH ...]
```
One row per frame, one column per source, any crop box. Open it at full size: crops at 1:1 or larger show line weight,
shading and texture that a full-frame thumbnail hides.

## panel_video.py: a side-by-side video

```
python scripts/panel_video.py --out OUT.mp4 [--panel-w 720] [--audio-from 1,3] LABEL=PATH LABEL=PATH [LABEL=PATH]
```
Two or three videos side by side with a label bar on top, each scaled to the same width, padded to the tallest, cut to
the shortest, 30 fps, H.264 CRF 18. Every panel's audio becomes its own audio track (players switch tracks; in VLC:
Audio > Audio Track). Typical: `REFERENCE=ref.mp4 BEFORE=prev.mp4 NOW=now.mp4` (3 × 720), or
`REFERENCE=ref.mp4 NOW=now.mp4 --panel-w 1080` (full size).
