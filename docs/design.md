# Design

MujocoReplay draws recorded positions of many worlds of one MuJoCo model in a
single scene. It never simulates: it composes a model with one copy of the
moving bodies per world, writes each world's recorded positions into its copy's
joints frame by frame, computes the resulting body poses with MuJoCo's
kinematics, and draws. Everything it knows about a run comes from the recording
file described in [recording-format.md](recording-format.md).

## Goals and limits

- Any MuJoCo model, any number of worlds in the file; the tool chooses how many
  to draw.
- The best world in its natural colours, the others as faint grey ghosts, so
  that the spread of behaviours is visible without hiding the best.
- Slow by default: a person must be able to see each step. Playback pauses,
  steps frame by frame, and runs from a few seconds per frame to faster than
  real time.
- A clean overlay with the facts of the run that never covers the bodies, and
  can be hidden.
- A free camera, driven with the mouse as in MuJoCo's own viewer.
- A video of the same scene, from the same code, for presentations.

Not goals: physics, editing, side-by-side scenes, or a graphical user
interface beyond keys and the mouse. Drawing is bounded by legibility and the
graphics card: with 64 shapes per copy, 128 copies are 8,192 shapes per frame,
which a laptop draws smoothly; a thousand overlapping ghosts would be a grey
blob anyway.

## Modules

| Module | Does | Imports at module level |
| --- | --- | --- |
| `recording.py` | The file format: `Recording`, `write_recording`, `read_recording` | NumPy |
| `selection.py` | `selected_ranks` and `choose_worlds(recording, levels, per_level, world_ids=None, all_worlds=False)`: which of a file's worlds to draw, and in what rank order | NumPy |
| `scene.py` | `ComposedScene`: the composite model, the joint mapping, colours, markers, `set_frame(frame_index)`, and `fits`/`show` to reuse the composite for another file of the same model | MuJoCo, NumPy |
| `playback.py` | `Playback`: the playlist, current frame, play/pause, speed presets, stepping, the events just passed, and the playback line of the overlay; pure logic, no graphics | none |
| `render.py` | `SceneRenderer`: the MuJoCo render context, camera, drawing of the scene, markers, overlay, timeline, and reading pixels back; works in a window or offscreen | MuJoCo, NumPy |
| `viewer.py` | The window: GLFW setup, mouse and key handling, the main loop | GLFW, MuJoCo, NumPy; imported only by the `view` command |
| `video.py` | Offscreen rendering of a playlist to an MP4 | `imageio` inside the function |
| `cli.py`, `__main__.py` | The `mujoco-replay` command and `python -m mujoco_replay` | argparse, NumPy; the window and video modules inside the commands |

`recording` and `selection` are the producer-facing half: Centipede imports
them to write files. They must stay free of MuJoCo and graphics imports, and a
test checks it in a fresh process. The rest may import MuJoCo freely; note that
`import mujoco` itself loads the `glfw` package (unless `MUJOCO_GL` selects
EGL or OSMesa), which is harmless without a display, since nothing opens until
a window or context is asked for.

## Composing the scene

An `MjSpec` is MuJoCo's editable form of a model: parsed from XML, changed
element by element, then compiled into the `MjModel` that simulation and
drawing use. The steps below were first tried with MuJoCo 3.12.0 on
2026-10-07 against Centipede's model (69 position coordinates, one root body
`segment_00` holding the whole centipede, 64 shapes, 55 actuators, a floor and
one light in the world body). Stage R2 built them and verified them against
the test model of `tests/conftest.py` and a centipede-like chain written for
the purpose (65 position coordinates, 52 shapes, a mesh, a skybox, and a
textured floor): every body of 32 copies matched the original model posed with
its world's `qpos` exactly.

1. Parse the recording's `model_xml` with `mujoco.MjSpec.from_string(xml,
   assets=...)`; the assets are the `asset/<name>` arrays as bytes. Every
   unnamed body and joint is given a name (`body7`, `joint3`), the same one on
   every parse, because joints are matched by name below and attaching leaves
   unnamed parts unnamed. Compiling this spec gives the original model, whose
   `qpos` layout the recording uses; its `nq` must match the file's.
2. The same spec becomes the static scene: delete every actuator, sensor,
   tendon, equality constraint, contact pair and exclusion, keyframe, tuple,
   skin, and flex (`spec.delete(item)`), then every replicated body
   (`spec.delete(spec.body(name))`). These parts are not needed for drawing
   and would otherwise name bodies and joints that no longer exist.
3. For each drawn world `k`, parse the XML again, reduce that spec to the
   replicated bodies and the materials, textures, meshes, and height fields
   they use, and attach it whole under a prefix: `frame =
   spec.worldbody.add_frame(); spec.attach(source, frame=frame,
   prefix=f"w{k}_")`. Names of bodies, joints, geoms, sites, and the copied
   materials and meshes carry the prefix. Attaching a spec copies everything
   left in it, hence the reduction: the floor, the world's lights, or a skybox
   texture would otherwise be copied once per world. Attaching each
   replicated body with `frame.attach_body` instead fails as soon as two root
   bodies share a material ("incompatible id in material array"). Attaching
   moves the contents out of the source spec, so the XML is parsed again for
   every copy.
4. Two fixes before compiling. A camera or light of the static scene aimed at
   a replicated body (`mode="targetbody"`) would aim at a body that no longer
   exists, which does not compile; it is aimed at copy 0's body, the best
   world's, instead. Lights inside the replicated bodies are kept in copy 0
   only, so that 32 copies of a tracking light do not light the scene 32
   times over.
5. Compile, and build the mapping by name. Copy `k`'s joints happen to sit in
   a contiguous block of the composite's `qpos`, but nothing relies on that:
   for each joint of the original model and each copy `k`, the composite's
   `joint(f"w{k}_{name}")` gives its `qposadr`, and the joint type its width
   (7 for free, 4 for ball, 1 for slide and hinge). The result is an index
   array `(K, columns)` saying where each replicated `qpos` entry goes. Joints
   of the static scene map to their own addresses and take values from the
   highlighted world's row.
6. `set_frame(t)` scatters `qpos[t, drawn_worlds]` through the index array
   into `data.qpos`, then runs `mujoco.mj_kinematics`, `mujoco.mj_comPos` (the
   centres of mass, which the camera looks at), and `mujoco.mj_camlight`
   (lights attached to bodies need it, or the scene is lit only by the
   headlight). No `mj_forward`: no contacts, no dynamics. A frame takes 0.2 ms
   for 32 copies of the chain.
7. The scene (`mujoco.MjvScene`) is allocated with `maxgeom` of the composite
   model's shapes plus the markers plus a margin.

Composing takes about 0.7 s for 32 copies of the chain, 2 s for 64, and 8 s
for 128 in the development container. MuJoCo's compiler grows with the square
of the number of bodies (the same copies written out as plain XML compile 6
to 8 times slower still), so `--all` on a file with hundreds of worlds is slow
to open. Spatial tendons are not drawn, since they go with the other unused
parts.

Colours: every geom of a ghost copy gets the ghost grey `(0.55, 0.55, 0.55)`
and `geom_matid = -1`, so that a material or texture cannot override the grey.
Its alpha is 0.15 times the shape's own drawn alpha, so that a shape the model
hides stays hidden; the drawn alpha is the material's when the geom's colour
is MuJoCo's default grey `(0.5, 0.5, 0.5, 1)`, and the geom's otherwise, which
is the rule MuJoCo draws by. The highlighted copy keeps the colours the
composite was compiled with, which are the original model's; they are read
from the composite itself, because every copy has its own prefixed materials,
whose ids differ from the original's. Both arrays are plain fields of
`MjModel` and can be changed at any time, so the highlight can move from one
world to another while playing. Hiding the ghosts sets their alpha to 0, and
MuJoCo leaves shapes with alpha 0 out of the scene altogether. MuJoCo draws
transparent shapes after opaque ones, so the highlighted world shows through
the ghosts. The alpha value is a starting point to tune by eye; it may need to
drop as more worlds are drawn.

Markers: after `mjv_updateScene`, one sphere per drawn world and marker is
added with `mjv_initGeom` into the scene's spare slots: the highlighted
world's in a saturated magenta, the ghosts' in the ghost grey. They are
decoration, so they cast no shadow; they hide with the ghosts, and on their
own with `M`.

## Choosing the worlds

The file may hold more worlds than are drawn. The drawn set is chosen with
`selected_ranks(K, levels, per_level)` over the file's `score`, descending and
stable (`numpy.argsort(-score, kind="stable")`), with defaults of 4 levels and 8
per level. Options widen or narrow this: `--levels`, `--per-level`, `--all`
(draw every world), `--worlds 3,7,9` (explicit `world_ids`). The highlighted
world starts as rank 0; a key moves the highlight through the drawn worlds in
rank order, so each ghost can be inspected in colour.

## Playback and keys

`Playback` owns a playlist of recordings and a position: file index, frame
index, playing or paused, and the speed as seconds of wall time per recorded
frame. The default is 0.3 s per frame, so that a 20 ms step is visible as a
step. The presets, in seconds per frame, are 3, 2, 1, 0.5, 0.3, 0.2, 0.1,
0.05, then real time (`frame_seconds`), 2× and 4× real time; the overlay shows
both the seconds per frame and the multiple of real time. Advancing uses wall
time, so a slow renderer skips frames at fast speeds rather than slowing down.
When a file ends, playback continues with the next file in the playlist and
stops at the end of the last one; Space at the very end replays that file.
Stepping crosses into the neighbouring file. `Playback.advance(now)` returns
the events whose frame was passed since the last call, so the overlay can
flash them: an event at frame `T` of a file is passed on moving into the next
file, or, in the last file, on stopping there while playing.

Changing file needs the scene of the new file. When its model and number of
drawn worlds match the current scene's (`ComposedScene.fits`), as for
consecutive windows of one run, the same composite shows it (`show`), at once
and with the camera kept; otherwise a new scene is composed, after which the
viewer calls `Playback.sync(now)` so that the time spent composing is not
played. The highlight stays on the same world, by `world_ids`, when the new
file draws it, and returns to rank 0 otherwise.

| Key | Action |
| --- | --- |
| Space | Play or pause |
| Right, Left | One frame forward or back; pauses |
| Up, Down | Faster or slower, through the presets |
| 0 | Back to the default speed |
| Home, End | First or last frame of the file |
| R | Restart the file |
| N, P | Next or previous file in the playlist |
| B, Shift+B | Highlight the next or previous drawn world, by rank |
| G | Show or hide the ghosts |
| M | Show or hide the markers |
| H | Show or hide the overlay |
| I | Show or hide the setup panel |
| C | Centre the camera on the highlighted world |
| F | Follow the highlighted world (tracking camera) on or off |
| Esc, Q | Quit |

Right and Left repeat while held. Mouse, as in MuJoCo's `simulate`: left drag
rotates, right drag pans, middle drag or the wheel zooms, and Shift turns
rotating and panning to the horizontal plane, through `mujoco.mjv_moveCamera`
with the standard action mapping (in MuJoCo 3.12 it takes no scene argument).

## Overlay

Drawn with `mujoco.mjr_overlay` in the corners, and with
`mujoco.mjr_rectangle` and `mujoco.mjr_label` for the timeline, in the normal
font, so that the middle of the window stays clear. MuJoCo's fonts hold ASCII
only, so the separators are ` | ` and `x`, and other characters in a
recording's text are replaced: `·` by `|`, accents dropped, the rest by `?`.

- Top left: the title; then the `frame_info` values at the current frame on
  one line (`updates 15 | steps per world 3,840`); then the highlighted world
  (`world 512 | rank 1 of 1,024 | summed reward +1.23 | level 1 of 4`); then
  the drawn set (`32 of 1,024 worlds drawn | 31 ghosts`, with `(hidden)` while
  `G` hides them).
- Bottom left: the playback line (`file 2 of 3 | frame 124 / 256 | 2.46 s |
  0.3 s per frame (0.067x real time) | paused`); the file shows only for a
  playlist, frames count from 1, and the time is the frame's since the start.
- Bottom: a thin timeline bar across the window, filled to the current frame,
  with an orange tick per event and its label in a box above it, and small
  blue ticks where the highlighted world's episodes start.
- Event flash: when playback passes an event, its label is drawn large at the
  bottom centre, above the playback line, for one second.
- Setup panel (`I`): the `setup_json` object flattened to `key = value` lines,
  nested keys joined by dots, drawn top right, as many as fit.

The overlay is drawn in the same way into the window and into the offscreen
buffer, so the video carries it unless `--no-hud` is given.

## Camera

A free camera (`mjCAMERA_FREE`) starts looking at the centre of the drawn
worlds and their markers at frame 0, from a raised angle, at a distance of 0.8
times their spread plus 1.2 times the composite model's `stat.extent`. `C`
recentres on the highlighted world's centre of mass; `F` switches to a
tracking camera (`mjCAMERA_TRACKING`) on the highlighted world's root body,
which moves with the highlight, and back to a free camera where it stands.
MuJoCo's tracking camera looks at the body's `subtree_com`, which
`mj_kinematics` leaves at zero, so `set_frame` also runs `mj_comPos`, which
computes it. The camera is independent of playback; it is kept when the next
file reuses the composite, and reframed for a file of another model.

## Rendering in a window and offscreen

`SceneRenderer` wraps one `mujoco.MjrContext`, one `MjvScene`, `MjvCamera`,
and `MjvOption`, and needs a current OpenGL context before it is made. The
viewer creates a visible GLFW window and renders to its framebuffer at the
framebuffer size, not the window size, so high-DPI screens render at full
resolution, with a font scale taken from the window's content scale.
Offscreen drawing, for the video and the tests, uses `mujoco.GLContext`,
MuJoCo's own helper, rather than a hand-made hidden GLFW window: by default it
is exactly that hidden window, and on a Linux machine without a display it
uses EGL or OSMesa when `MUJOCO_GL` says so, with no code of ours. The
renderer sets the model's `vis.global_.offwidth/offheight` before making the
context, selects the offscreen buffer (`mjr_setBuffer(mjFB_OFFSCREEN)`), and
reads pixels with `mjr_readPixels`, flipped vertically, as MuJoCo returns them
bottom-up. Without OpenGL, making the context raises `mujoco.FatalError`,
which the renderer reports as a clear error. This is the same structure as
MuJoCo's `basic` sample and the Python `mujoco.Renderer`; the tool keeps its
own loop because the passive viewer offers no text overlay.

Each frame, `mjv_updateScene` lists the shapes to draw; the ghosts' shapes are
then marked as decoration (`mjCAT_DECOR`), because MuJoCo draws translucent
shapes with full, dark shadows, which would cover the floor in grey, and casts
none from decoration; the markers are added; `mjr_render` draws, and the
overlay goes on top. The highlighted world keeps its shadow. For 32 copies of
the chain (1,633 shapes), listing the shapes takes 0.05 ms, marking the ghosts
1 ms in Python, and posing the frame 0.2 ms.

## Video

`mujoco-replay render FILES --out replay.mp4 [--width 1280 --height 720
--fps 30 --speed 0.3 --no-hud]` plays the playlist at `--speed` seconds per
recorded frame and writes `fps` video frames per second, so each recorded
frame repeats for `speed × fps` video frames (9 at the defaults). Frames go to
`imageio.get_writer` with the `ffmpeg` plugin from `imageio-ffmpeg`, which
bundles its own encoder, so no system installation is needed. The `video`
extra installs both; the viewer does not need them.

## Command line

```text
mujoco-replay [view] FILE [FILE ...] [--levels 4] [--per-level 8] [--all]
              [--worlds IDS] [--speed SECONDS] [--no-hud] [--width W] [--height H]
mujoco-replay render FILE [FILE ...] --out PATH [--fps 30] [--speed SECONDS]
              [--width 1280] [--height 720] [--levels 4] [--per-level 8]
              [--all] [--worlds IDS] [--no-hud]
```

`view` is the default subcommand. `python -m mujoco_replay` is the same
command.

## Dependencies

`mujoco==3.12.0` (the version the composition was verified with; the `glfw`
package comes with it), `numpy`. Optional `video`: `imageio`,
`imageio-ffmpeg`. Development: `pytest`, `ruff`. Python 3.11 or newer.

## Verification without a display

An assistant cannot use the window as a person does. Rendering is verified by
writing a few offscreen frames of a recording to PNG files in the scratchpad
and inspecting them: ghosts grey and translucent, the best world coloured and
on top, markers present, overlay text in the corners, the timeline filled to
the right frame. The user verifies the window, the mouse, and the keys. Tests
that need an OpenGL context skip with a clear reason where none can be
created.

In stage R3 the development container had no GPU and no display; MuJoCo drew
through Mesa's software renderer under a virtual X display (`xvfb-run`), at
about 1.4 s per 1280 × 720 frame with 31 ghosts, so smooth playback needs a
real graphics card. The window itself was also started there and driven with
synthetic key presses (the X test extension) and screenshots, which showed
every key acting as listed; that is a check before the user's, not instead of
it.
