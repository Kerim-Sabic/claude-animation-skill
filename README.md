# claude-animation-skill

A [Claude Code](https://docs.claude.com/en/docs/claude-code) skill that makes animated videos better by **measuring
instead of guessing**.

You give Claude a target: a reference clip, an approved earlier render, or the last round of your own video. The skill
finds where on screen your render differs most, works out why (colour, placement, blur, outline, shading, layout), and
fixes one thing at a time. It keeps a change only when a full render proves it is closer. Then it builds a side-by-side
video so you can judge the result yourself.

It came out of about 100 rounds of rebuilding short cartoon clips frame for frame. Tuning by eye kept producing frames
that "looked a bit better" while the real gap barely moved. This loop is what actually closed it.

**Works with any renderer:** Remotion, Three.js, Blender, After Effects, Unity, Unreal, hand-drawn. Every tool reads
plain video files and stills.

## What's inside

```
skills/animation-quality/
  SKILL.md                      the method Claude follows
  references/
    quality-recipe.md           15 ordered stages from first render to finished quality, with a stop condition each
    tools.md                    every tool: options, output, how to read it
    lessons.md                  the traps those rounds ran into, by topic
    comparison-video.md         the review package (videos, close-ups, short README)
  scripts/
    diffmap.py                  where is the difference, region by region; KEEP / REVERT against a baseline
    error_split.py              is a region's difference colour, placement, blur or missing detail?
    register.py                 where does our character sit and how big is it, part by part, vs the reference?
    set_register.py             is the background off because of the camera or the layout?
    head_profile.py             how is a face shaded: lit core, shade plateau, terminator, light direction
    outline_profile.py          outline width and ink colour by side; thick / thin ratio
    edge_profile.py             edge softness (blur, depth of field), PASS / FAIL against the reference
    limb_track.py               legs, arms and poles: position, width, colour per frame
    colour_solve.py             input colours that render closest to the reference, through your own light and grade
    crops.py                    labelled close-up sheets
    panel_video.py              side-by-side videos with labels and one audio track per panel
    measurelib.py               shared helpers
```

## Install

1. Requirements: Python 3.10+, and `ffmpeg` / `ffprobe` on your PATH.
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the skill folder where Claude Code looks for skills:
   - for all your projects: `~/.claude/skills/animation-quality/`
   - for one project: `<your-project>/.claude/skills/animation-quality/`
   ```bash
   git clone https://github.com/Kerim-Sabic/claude-animation-skill.git
   cp -r claude-animation-skill/skills/animation-quality ~/.claude/skills/
   ```
3. Start Claude Code in your animation project and ask in plain words. For example:
   - "Make shot 3 look like ref/shot3.mp4."
   - "Why does this render look cheap next to the reference?"
   - "Did my lighting change help? Compare renders/v6.mp4 with renders/v5.mp4."
   - "Build me a before / after video of the last two renders."

The tools also work on their own, without Claude:
```bash
python skills/animation-quality/scripts/diffmap.py ref.mp4 ours.mp4 --out diff/base --heat 4
python skills/animation-quality/scripts/panel_video.py --out compare.mp4 REFERENCE=ref.mp4 NOW=ours.mp4
```
Every script prints its full usage with `-h`.

## The loop in one screen

1. **Baseline**: render, then `diffmap.py` ranks the screen regions by their share of the difference.
2. **Watch both at normal size** and write down the 3-6 differences a viewer sees first. That list decides "done".
3. **Name the cause** of the top region with a measurement: `error_split.py` first, then the right property tool.
4. **Change one thing behind a knob**, try 2-4 values as stills, score them, look at 1:1 crops.
5. **Full render**: `diffmap.py --baseline` prints KEEP only if the mean went down and no region got worse. A gain you
   can't see at normal size gets reverted anyway.
6. **Show it** with `panel_video.py` and `crops.py`, and repeat.

## Use it responsibly

- **Reference clips are for study, not copying.** Measure their craft (line weight, shading, light, colour, softness,
  timing) and keep your own characters, designs, text and sound. Keep reference footage, and every frame or crop made
  from it, private. Respect the copyright of whoever made it.
- The numbers rank the work. Your eye decides what is good.

## Licence

MIT. See [LICENSE](LICENSE).
