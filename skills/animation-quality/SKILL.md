---
name: animation-quality
description: Bring an animated video or shot up to the quality of a target look by measuring instead of guessing. Ranks where on screen our render differs most from a reference (or an approved earlier render), splits each difference into colour, placement, blur or shape, registers characters and sets on the reference's own pixels, measures shading, outline width, edge softness and limbs, solves flat surface colours through the renderer, keeps a change only when a full render proves it, and builds side-by-side review videos. Use it whenever someone asks to make an animation, cartoon, Short, shot or scene "look like the reference", "look better", "more 3D", "less cheap", or "perfect"; to compare two renders; to find out why a shot looks wrong; or to build a before/after or reference/ours comparison video. Works with any renderer (Remotion, Three.js, Blender, After Effects, game engines) because every tool reads plain video files and stills.
---

# animation-quality: measure, fix, prove

This skill is a working method plus twelve measuring tools. It came out of a hundred rounds of rebuilding short
cartoon clips frame for frame, where tuning by eye kept producing frames that "looked a bit better" while the real gap
barely moved. What moved it was always the same loop: find where the difference is biggest, name its cause with a
measurement, change one thing behind a knob, and keep it only if a full render proves it is closer.

All tools are Python scripts in `scripts/` (paths below are relative to this skill's folder). They need Python 3.10+,
`numpy`, `opencv-python`, and `ffmpeg` / `ffprobe` on the PATH. They read any video or still, so they work with any
renderer. Every script prints its full usage with `-h`.

## 0. Rules

- **Original work only.** A reference clip is something you *measure against* for craft: line weight, shading, light,
  colour, softness, camera timing. Never copy its characters, designs, frames, text or audio into anything you publish.
  Keep the reference and every crop, diff or frame cut from it in a private folder that is never uploaded.
- **Honest reporting.** Say exactly what you rendered, opened, measured or ran. A still you did not open is not checked.
  Claude cannot hear: for sound, name the proxy (loudness numbers, onsets, speech recognition) and say nobody listened.
- **The numbers rank the work; the person's eye decides.** Never call a shot "matched" or "approved". Only the person
  who asked closes it.
- **Ask before paid steps** (voice, music or image generation, paid render farms) and before publishing anything.

## 1. Pick the job

| The person asks for | Do |
|---|---|
| "make it look like this reference" / "perfect" | Follow `references/quality-recipe.md` stage by stage |
| "why does this shot look wrong / cheap?" | Stages 2-3 of the recipe: the visible-difference list, then `diffmap.py` + `error_split.py` |
| a look for a new original shot (nothing frame-aligned) | The recipe's look-target path: property tools on both, compare numbers, never copy designs |
| "did my change help?" | `diffmap.py ... --baseline` (section 3) |
| a comparison video | `references/comparison-video.md` |

## 2. Get the inputs right first

- **Align the reference**: same frame count, frame 0 at the same moment, same aspect ratio. Cut it with ffmpeg by frame
  (`-vf "select=between(n\,A\,B)"`), never by guessing seconds.
- **A reference can be**: a clip the person supplies, an earlier render they approved (a regression check), or the
  previous round of the same video (did this round help?).
- **Knobs, not edits.** Put every change behind a named parameter so variants are cheap and reverting is one line. Back
  up each file you touch first.
- **One renderer at a time** if several agents share a machine: use a lock file, so two renders never fight.

## 3. The loop

1. **Baseline.** Full render of the current state, then
   `python scripts/diffmap.py REF.mp4 OURS.mp4 --out D/base --tag base --regions regions.json --heat 6`.
   Read `diff.md` (regions ranked by their share of the total difference) and open the heat maps.
2. **Watch both at normal size first** and write down the 3-6 differences a viewer sees first. That list decides what
   "done" means; the metric then ranks and proves the work.
3. **Work the top row.** Large and wrong beats small and very wrong.
   - Split its error: `error_split.py` says how much is colour, placement, blur or missing detail.
   - Register any character in it first (`register.py`), then measure the property itself on the reference's pixels:
     `head_profile.py` (face shading), `outline_profile.py` (ink width and colour by side), `limb_track.py` (legs, arms),
     `edge_profile.py` (edge softness, on the video). Write the reference's numbers and ours down: that pair is the target.
   - Set out of place? `set_register.py` tells a camera aim error from a layout error.
   - Flat colours wrong? Solve them with `colour_solve.py`; do not pick them off the reference.
4. **Variants as stills.** 2-4 values of the knob, rendered at the 3-8 frames that show the problem. Score each still
   with `diffmap.py` (images work) and look at `crops.py` sheets at 1:1. Drop anything worse in numbers *or* to the eye.
5. **Full render to decide.** `diffmap.py ... --baseline D/base/diff.json` prints **KEEP** only if the mean went down
   and no region got worse by more than `--tol` (0.10). Then look at a crop strip at 1:1 and the frame at normal size:
   a gain nobody can see at normal size is reverted anyway. Stills can mislead; the full render decides.
6. **Log the round**: what you tried, the numbers, which frames and crops you opened, kept or dropped, how to roll back.

Full renders are not perfectly deterministic: two renders of the same code can differ by about 0.5-1 level of 8-bit
value. A small change in a region you did not touch is noise, not a gain or a loss.

## 4. Tools

| Tool | Answers |
|---|---|
| `diffmap.py` | Where on screen is the difference, region by region, and did a change help (KEEP / REVERT)? |
| `error_split.py` | Is a region's difference colour, placement, blur or missing detail? |
| `register.py` | Where does our character (each part: face, hat, body) sit and how big is it, vs the reference? |
| `set_register.py` | Is the background off because of the camera (pan / tilt / zoom) or the layout? |
| `head_profile.py` | How is a face or head shaded: lit core, shade plateau, terminator softness, light direction? |
| `outline_profile.py` | How wide and how dark is the outline, side by side around a shape? Thick on which side? |
| `edge_profile.py` | How soft is an edge (depth of field, blur), ours vs the reference, PASS / FAIL? |
| `limb_track.py` | Where are the thin dark shapes (legs, arms, poles), how wide, what colour? |
| `colour_solve.py` | Which input colour per surface renders closest to the reference, through our own light and grade? |
| `crops.py` | A labelled still sheet: one row per frame, one column per source, any crop box |
| `panel_video.py` | A side-by-side video with label bars and every panel's audio as its own track |
| `measurelib.py` | Shared helpers (not a command) |

Every command, its options and how to read its output: `references/tools.md`.

## 5. Where to read more

- `references/quality-recipe.md`: the ordered 15-stage path from first render to finished quality, with the tool and
  the stop condition at each stage.
- `references/tools.md`: every tool in detail.
- `references/lessons.md`: the traps a hundred rounds ran into, grouped by topic.
- `references/comparison-video.md`: the review package (videos, close-up sheet, short README).

## 6. Report

End with: what exists now (paths), what you looked at (which frames, which crops), what you measured (numbers), what you
did not do or could not check, and the next step. Short sentences, plain words.
