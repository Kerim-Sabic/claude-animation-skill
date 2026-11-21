# Lessons: the traps a hundred rounds ran into

Grouped by topic, in the order of `quality-recipe.md`. The numbers are from real rounds and only indicate size.

## Process: where to look and how to decide
- **Start every round with the whole clip at normal size, beside the reference**, and list what a viewer sees first.
  Keep only what a full render shows is better, AND what is visible at normal size: several measurable gains nobody could
  see were reverted; one that read clearly as "warmer, brighter faces" was kept.
- **Measure on the reference's own pixels**, not on annotation boxes or by eye from a crop sheet (annotated feet sat
  15-25 px below the real ones). Read directions (left / right, up / down) from a tool's output, never from a crop.
- **Split the error before touching a knob.** A set band that "looked blurry" split into colour and placement in about
  equal parts, with blur negligible and sharpening worse: that ruled out the depth-of-field knob before any render.
- **Mask the characters before blaming them.** A top "character" row in a close-up was mostly set pixels. For an exact
  character-vs-set split, render one still with the character hidden.
- **Rank variants with the clip metric**, never plain RGB difference: the two ranked every variant of one railing in
  opposite directions. Score every region, not only the whole frame.
- **Test hiding something with a toggle still before deleting it**: the gap may be the layout behind it. Check that a
  knob actually changes pixels before tuning it.
- **Stills can mislead; the full render decides.** A sky that won in stills lost in the video (the video's sky came
  out brighter). A camera change that improved the mean broke a transition.
- **Tag each kept change GENERAL or THIS-CLIP.** General = a reusable rule that helps any video. This-clip = per-frame
  keys and hand values that only fit this reference. The next video starts from the general list only.

## Registration: characters
- **Register a character before testing its shading.** A correct shading model drawn on a head that sat ~20 px off and
  ~10 % short scored WORSE (the shade fell on the reference's lit chin); after registering, the same region dropped by
  about a quarter.
- **Look at the masks before trusting the numbers.** A strict colour test dropped the shaded half of a face and the
  offsets came out 20-40 px wrong.
- **Under a strong grade, anchor on the most saturated garment.** Face and hat colour tests picked up sky under a sunset
  grade; a saturated shirt held.
- **Size before placement.** A figure "in the right place" was 1.2-1.3x too big. Scale about the head so placement stays.
- **Register every frame of a moving character, legs and shadows too.** A walk bob out of phase swung the head offset
  from 0 to ~50 px within 4 frames; a per-frame shift table fixed it. Legs can be far off while the head is fine.
- **Track a character to the last frame it is visible**, through dissolves. "He's slow" turned out to mean "he stops":
  match step timing and stride, not only position.
- **A head can be in the right place while its face is not.** Measure eyes and mouth against the chin; a head pitch can
  move the features without moving the outline.
- **Scaling a head changes more than the head**: the face centre moves and the outline thickens.

## Registration: set and surfaces
- **Register the background per frame too.** A far layer drifted up to ~20 px against the ground over a hold. Tilting
  the whole camera fixed the set and broke the near layers: move the far layer only.
- **A shift search on a repetitive pattern (clapboard, bricks) is not a registration**: offsets disagreed from -60 to
  +80 px. Compare runs of two distinctive colours down a few columns instead.
- **Check where each surface lands before solving its colour.** One building row sat ~130 px off, and the colour solve
  had been averaging colours in the wrong places.
- **A geometry fix can score worse at the old colours and still win after a re-solve.** Fix geometry, re-render the
  toggles, re-solve, then compare.
- **Moving one set piece moves its neighbours' problems**: moving a far row away uncovered sky; moving a wall changed
  what hid the props in front of it.

## Colour solving
- **Solve colours, don't pick them**: through a grade or haze, a light-grey input can render as a dark blue-grey.
- **A muted solve can mean misplaced, not mis-coloured.** The solver pays for misplaced edges by greying thin bright
  surfaces (white trim came out as mud). Fix placement first.
- **Give each distinct surface its own knob** (window casings vs corner boards, sign face vs frame). Hidden default
  colours in your renderer (a default door colour, curtains that mix in the wall colour) are colour sources too: expose
  them.
- **Hold light thin surfaces that land on dark reference pixels**, and bound inputs near today's hue under a strong warm
  light: a bounded solve (about ±56) rendered better than the free one.
- **Predictions run optimistic**: ~0.2 on a sharp set, ~0.6 on a heavy depth-of-field background. Always render and score.
- **Solve frames must cover every camera phase**, and the mask must cover animals, props and effects as well as
  characters.
- **A two-zone flat colour needs no new feature**: copy the geometry in a second colour, fade it in with a soft patch,
  and solve both.

## Sky, light and shadow
- **Measure the sky as a column profile** (colour by screen height). Where the gradient turns pale above the roofs was
  worth about a third of that band's error.
- **A flat sky does not render flat**: glow, haze and grade add a height-dependent veil (+30 red measured on a sunset).
  Fit and invert it per row.
- **A sky change reaches checks far from the sky**: any colour box or edge point with sky behind it.
- **Toggle each shade source before touching the light.** Leftover decoration (shadow strips painted where nothing casts
  them) made a wall far too dark.
- **Check the light's direction before adding shadow casters**: casters lit from the wrong side throw stripes across the
  characters. A soft shadow map turns thin casters into one blurred patch.
- **Line a shadow up before changing its tone**; a corrected shadow can score worse until the ground it uncovers has the
  right colour.

## Sharpness and blur
- **Sharpness comes last**: a sharper set loses until placement and colours are right.
- **Measure background blur against depth.** One cartoon reference's background blur was flat across depths (~8 px edge
  spread, a "layer blur"); ours grew with distance. Capping the maximum blur reproduced it.
- **Sharpen only what the reference has sharp**, over the frames where it is sharp.
- **A blur cap fixes blur, not missing detail.** One set reached an edge-spread ratio of ~1.0 and still looked soft: the
  reference carried more DETAIL (ink on trim edges, regular laps, grain). Count matched edges, not only spreads.
- **Sharpening exposes every wrong small detail** (default colours, frames, letters of the wrong polarity, prop
  textures the blur hid). Fix those first. When the eye and the metric disagree on sharpness (a sign becomes readable),
  the reviewer's eye wins if the cost is small; say so.

## Ink and outline
- **A set line drawn through a character is a post-pass leak** (set ink or edge light), not a draw-order bug. Let the
  character's coverage decide, not a depth tolerance: a character standing 15 cm in front of a door was inside the
  tolerance and the door's outline showed through him.
- **Ragged dark patches plus a seam on a flat board can be two outline ids fighting** on overlapping coplanar pieces.
- **Missing heavy ink on thin parts** (railings) can be a flat dark card behind each part.
- **A part that "reads grey" can be its outline, not its fill.** A thinner line in a lighter ink is much less ink.

## Characters, garments and props
- **"The jacket looks wrong" can mean a whole panel is missing.** Measure the order of panels and lines across a garment
  on reference rows (panel, ink, shirt strip, ink, panel...), not mean box colours. Rules like "the far panel hides past
  a given turn" are assumptions: check them on the reference's turned frames.
- **"Arms inside the silhouette"** in a cartoon reference = arm roots shifted into the torso, angled out, drawn in front.
  Widening the torso lost badly.
- **A part drawn on top can hide another** ("stitch too short" was "arm drawn over the stitch").
- **Check small face extras per shot** (a straw, a cigarette): one was missing in a close-up where it was the boldest
  line in the reference.
- **Measure a hinged prop on every frame before animating it**: a cartoon door "swing" was mostly a darkening plus a
  narrowing from the free edge.
- **Props a character uses belong to the character**: a plate under a pinned hand must not slide with the table.

## Checks and noise
- **Full renders are not perfectly deterministic** (about 0.5-1 level of 8-bit value). Stills nearly are. Never claim a
  gain smaller than the noise; prove a flipped check is noise with a second identical render or 0 px changed in its box.
- **Read a colour check's distribution before chasing it**: a median sitting on a dark / bright balance flips from encode
  noise anywhere in the clip.
- **Check where a check's sample point actually sits**: a "background edge" point sat on a character's jacket edge.
- **Edge checks belong on video frames**: stills and the encoded video of the same frame disagreed by ~0.1 in ratio.
- **One decoder for every number in a comparison.**
- **Write the checks and their tolerances down before you start**, for example: subject box IoU >= 0.85 every frame,
  horizon within 2 % of frame height, region colours within dE 6, luma percentiles (5 / 50 / 95) within ±8, line
  widths within ±15 %, blur widths within ±20 %, cuts, camera moves, captions and sound onsets within ±1 frame. Report a
  check you could not measure as UNSCORED with the reason, never as passed.

## Code and render hygiene
- **If a flaw shows in the video but not in a still of the same frame, look for state carried between frames** (a
  cache keyed on first use, a mask made once). A character resized every frame in a depth-of-field shot smeared because
  its blur mask was stale; stills looked clean because each still starts fresh.
- **Visibility before update**: hide or show an element before drawing it for the frame, or a clip shows an empty quad
  while stills look fine.
- **Code after an inline `//` comment is silently dropped.** One round's probe put a statement there and every still lacked
  that element; the round's conclusions had to be redone. Check edited lines for code to the right of `//`.
- **Never edit source while a batch render runs** if your runner snapshots and restores files.
- **One broken file breaks every render**: typecheck or lint before a long render.
- **Repaint every colour setting in loud tag colours on one still** to find a colour you cannot trace: whatever keeps
  its colour comes from a hard-coded path.
- **When a part moves per frame, list what it covered at rest**: the move can uncover a stripe or a gap.
- **Clean up**: render bundles and probe stills are large; delete only the ones your own run created.

## Judging by eye
- Look at crops at 1:1 or larger, not thumbnails.
- Three columns when you can: REFERENCE | BEFORE | NOW. "Better" means closer to the target, not prettier.
- Say what you saw per frame ("f40: shade now visible on the left cheek; reference still darker under the chin").
- Name what you did NOT look at.
