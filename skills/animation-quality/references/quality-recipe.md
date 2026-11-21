# Quality recipe: from first render to finished quality

The default path for any shot that must match a look: a frame-aligned reference, or a new original shot built to a look
target. It is the order that a hundred rounds of reference matching settled on. Each stage names the tool and the
condition that ends it. Commands are relative to the skill folder; `S = scripts`.

Two kinds of target:
- **Frame-aligned reference** (same length, frame 0 aligned, same aspect; cut it with ffmpeg if needed): every stage
  applies.
- **Look target only** (a new original shot, nothing frame-aligned): skip the stages marked *aligned*. For the property
  tools (head / outline / edge / limb), run each tool once on the target with its own box and once on ours with ours, and
  compare the printed numbers. Widths and distances are printed in shape widths (W), so sizes need not match. Copy the
  target's craft (line weight, shading, light, colour, softness, timing), never its designs.

Rules that hold in every stage: back up each file you touch; put each change behind a named knob; render variants as
stills; rank them with the clip metric (`diffmap.py` / `error_split.py`, never plain RGB difference); confirm in a full
render (stage 13). Reference pixels and everything cut from them stay in a private folder.

## 0. Build the shot and look at it
Render 3-5 stills and open each one. Stop when nothing is broken: no camera inside a wall, no floating feet, no prop
through a face, no blank or white areas in a clip strip.

## 1. Baseline: first full render and diffmap  *aligned*
`python S/diffmap.py REF.mp4 OURS.mp4 --out D/base --tag base --regions regions.json --heat 6`.
Name the regions after what is in the REFERENCE at that place (`{"hero head": [0.38,0.30,0.62,0.44], ...}`, 0..1
fractions). Open the heat maps. Stop when `diff.json` is saved: every later keep decision is made against it.

## 2. Visible-difference pass at normal size (the reviewer's bar)
Watch the whole clip beside the reference at normal size (`panel_video.py --panel-w 1080`) and list the 3-6 differences
a viewer sees first, in plain words. This list, not the metric, decides what "done" means. Redo this pass at the start
of every round. Stop when the list is written down.

## 3. Split the error of each top row  *aligned*
For the top 1-3 diffmap rows:
`python S/error_split.py REF=ref.mp4 OURS=ours.mp4 --frames A-B:3 --box ROW --mask chars.json --mask-tiles 60`.
Mask every character (and animals, props, effects, captions) for a set row. For an exact character-vs-set split, render
one still with the characters hidden and score both. Read: colour share → stage 7; placement → stages 4-6; blur gain →
stage 11 (ours too crisp); sharpen gain → missing detail (stage 11, not more blur); a big residual after shift + colour →
shapes or layout differ. Stop when each top row has a named cause.

## 4. Register the characters, per frame  *aligned*
`python S/register.py REF=... OURS=... --frames A-B:2 --parts parts.json --json reg.json`.
- Overlay the masks on the frames before trusting any number (a strict colour test can drop a shaded half).
- Under a strong colour grade, anchor on the most saturated garment and place the head relative to it.
- Size before placement: fix the scale (about the head, or a prop about its root), then the offset.
- Every frame of a moving character (a per-frame shift table, median of 3 neighbours); legs (`limb_track.py`), shadows
  and props too. Check feature rows (eyes, mouth) against the chin: a head pitch can fix the face alone.
Stop when the offset is within 2-3 px and the best IoU is high (> 0.75) on every frame of every hold.

## 5. Register the set  *aligned*
Add two probe knobs to the shot: one hides the characters, one zooms the camera out. Render one still batch: C (normal),
H (hidden), HW (hidden + zoomed out about 0.3 in log zoom, so the set looks ~0.74 as big). Check H against C by pixel
difference (only the character may change).
`python S/set_register.py --ref ref.mp4 --normal "stills/f{frame}_C.png" --hidden "stills/f{frame}_H.png"
--wide "stills/f{frame}_HW.png" --wide-scale 0.741 --frames ... --mask refchars.json --strips 2
[--stretch 1.0:1.4 --baseline-y lo:hi:step]`.
Strips AGREE → a camera aim error (move the camera, or only the set if the character is pinned to the screen);
OPPOSITE → layout or proportion (move set pieces); STRETCH wins → the set stands taller above the same ground line.
On a repetitive facade a shift search is not a registration: compare runs of two distinctive colours down a few columns.
Stop when neither strip gains more than the noise (0.3) after baking the winner and re-rendering stills.

## 6. Place the surfaces (toggle masks)  *aligned*
Render one still with each surface toggled (or an ID still: each surface a pure colour) and outline each toggle's
difference mask over the reference frame. Fix where a surface lands before its colour: x first, then heights, then the
depth of parts that stick out. A drift high or low over a camera move can be a set scale error about the frame centre,
not a camera tilt. Split surfaces whose reference colour differs part to part into their own knobs. Stop when every
solvable surface's mask sits on the reference's matching colour at 1:1.

## 7. Solve the colours
Render a toggle batch with geometry, light, grade and camera exactly as the render you compare with: `G` = every surface
#808080; `T<s>d` / `T<s>l` = one surface at #202020 / #e0e0e0, the rest at #808080. Then
`python S/colour_solve.py --ref ref.mp4 --manifest toggles.json --mask chars.json --mask-tiles 60 --current today.json
--colours solved.json`.
- `--fix` surfaces under the mask (reference median printed as "-") or solved to 0 / 255; use `--bounds` (about ±56
  around today's colours) under a strong warm light; hold light thin trims that land on dark reference pixels.
- Render the solved colours and score the real stills. If they miss the prediction by more than 0.2, re-centre the
  toggles on the solve (`surface_levels`, ±48) and solve again (`--init`).
- Compare two geometries only after each has its own solve. Solve frames must cover every camera phase.
Look target only: solve against a painted mock-up still, or set colours from `crops.py` patch medians and judge at 1:1.
Stop when the real stills score within ~0.2 of the prediction and the solved surfaces look right at 1:1.

## 8. Skies and light
Measure the sky as a column profile (colour by screen height) on both. A flat sky does not render flat (glow, haze,
grade add a height-dependent veil): render flat test skies, fit `out = a·in + b` per row, invert it, then fit the sky
colours; verify the height mapping once with a hard two-colour step sky. Shadows: measure the reference's shadow
direction before adding shadow casters (it tells you which side the light is on); toggle each shade source before
touching the light. Ramp shot-wide grades and shades by frame range. Stop when the column profile matches and no
colour check elsewhere flipped.

## 9. Line and ink
`python S/outline_profile.py REF=... OURS=... --frames ... --boxes heads.json --fit --w0 PX` on a registered head.
Hand-inked references usually have one thick side away from the light and a thin lit side (ratio 2-4); a ratio near 1
is a uniform ring. Match both the width by side and the ink colour (a thinner line in a lighter ink is much less ink).
Set lines: missing ink on flush trim, or on thin parts, is often the whole difference between a "soft" and a "crisp"
set. A set line drawn through a character is a post-pass leak (an ink or edge pass that does not respect the character's
coverage), not a draw-order bug. Ragged boxes or a seam on a flat board can be two outline ids fighting on overlapping
coplanar pieces. Stop when width ratio and ink colour match by side and the line reads the same at normal size.

## 10. Heads and shading
`python S/head_profile.py REF=... OURS=... --frames ... --boxes heads.json --polar` on registered heads only.
A flat plateau plus a crisp step → a two-tone toon shade; a steady rise from the outline → a soft gradient; no depth on
the lit side → no edge ring there. One key light shared by every character reads as one world. A separate grade for the
characters (less haze, more saturation) stops a hazy set from washing them out. Stop when plateau, depth, terminator
width and light direction match, and the faces read better at normal size.

## 11. Sharpness last
Only after placement and colours are solved.
`python S/edge_profile.py REF=ref.mp4 OURS=ours.mp4 --points edges.json` on the VIDEO, plus a count of strong reference
edges that have a matching edge in ours (a spread ratio of 1.0 with missing edges still looks soft: that is layout or
detail, not blur).
- Background blur flat across depth in the reference (a "layer blur") → cap the maximum blur; ground near the focus
  crisp → keep a sharp band in front of the focus; sharpen only the frames and depths the reference has crisp.
- Re-solve colours (or keep the soft-look solve; test both) before judging a blur change. Check 1:1 for flaws the old
  blur hid (prop textures, default colours, letters of the wrong light / dark polarity).
Stop when edge checks PASS on the video and the matched-edge count stops rising.

## 12. Props and hinged parts, per frame
Measure each prop's edges and colour on every reference frame before animating it. A cartoon door "swing" may be a
darkening plus a narrowing from the free edge (a screen-space width key plus a colour key), not a 3D rotation. Props a
character holds or uses belong to the character: when the set moves, they take only the part of the move that keeps
them on their surface. Moving a wall changes what hides the props in front of it. Stop when every prop is placed on
every frame it is visible, checked on a clip strip.

## 13. Keep rule, checks, noise proof
Full render → `diffmap.py ... --baseline D/base/diff.json` must print KEEP (mean down, no region worse than `--tol`
0.10). Re-run the stage's own tool on the VIDEO. Run the checks you wrote down at the start (framing IoU, region colours,
luma percentiles, line widths, edge spreads, timing) on video frames with one decoder. A check that flips in an
untouched part of the clip is probably noise: prove it with a second identical render, or with 0 px changed inside its
box on stills. Then the visible-difference list from stage 2: a gain not visible at normal size is reverted.
Tag each kept change GENERAL (a reusable rule that helps any video) or THIS-CLIP (per-frame keys, hand values that only
fit this reference). Start the next video from the GENERAL list only. Stop when KEEP, checks PASS (or are marked
UNSCORED with a reason), and the list item is visibly closer.

## 14. Comparison video for the reviewer
`panel_video.py` 3-panel (REFERENCE | BEFORE | NOW) and 2-panel full size, `crops.py` close-ups, and a short README
(`comparison-video.md`). Open the outputs yourself first. Nothing is approved until the reviewer says so.

Loop: back to stage 2 with the new render as the baseline, until the reviewer closes the clip.

## Final sweep before any review video
Look at every ~5th frame of the whole clip, and every frame of dissolves, whip pans and fast moves, at reduced size.
A faint flaw noted once can grow into a visible one and go unchecked for many rounds. Pay special attention to
characters during dissolves: a character drawn outside the dissolve stays solid while the rest of the shot fades, then
pops out in one frame.
