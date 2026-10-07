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
- An application, not only a command: it opens on an empty world, takes
  files from a file picker or dropped onto it, and offers a small panel of
  its own settings, never MuJoCo's raw options, with a Quality and a
  Performance mode so that it plays on a weak graphics card (stage R6).

Not goals: physics, editing, side-by-side scenes, or a graphical toolkit:
the panel is drawn with MuJoCo's own overlay functions. Drawing is bounded by
legibility and the
graphics card: with 64 shapes per copy, 128 copies are 8,192 shapes per frame,
which a laptop draws smoothly; a thousand overlapping ghosts would be a grey
blob anyway.

## Modules

| Module | Does | Imports at module level |
| --- | --- | --- |
| `recording.py` | The file format: `Recording`, `write_recording`, `read_recording` | NumPy |
| `selection.py` | `selected_ranks` and `level_of_ranks`, shared with producers; `world_counts` and `choose_worlds(recording, count, world_ids=None)`: which of a file's worlds to draw, and in what rank order | NumPy |
| `scene.py` | `ComposedScene`: the composite model, the joint mapping, colours, markers, `set_frame(frame_index)`, and `fits`/`show` to reuse the composite for another file of the same model | MuJoCo, NumPy |
| `playback.py` | `Playback`: the playlist, current frame, play/pause, speed presets, stepping and seeking, the events just passed, and the playback line of the overlay; pure logic, no graphics | NumPy, through `recording` |
| `render.py` | `SceneRenderer`: the MuJoCo render context, camera, graphics settings, drawing of the scene, markers, overlay, timeline, and reading pixels back; works in a window or offscreen | MuJoCo, NumPy |
| `settings.py` | `Settings` and `Graphics`: the Quality and Performance presets, the ghosts' strengths, and keeping the settings in the user's settings folder | NumPy, through `selection` |
| `ui.py` | `Panel`: the side panel's rows, their layout, drawing, and clicks | MuJoCo |
| `viewer.py` | The application: the window, the empty world, opening files, the panel's actions, the keys and the mouse, the main loop that draws only on change | GLFW, MuJoCo, NumPy; imported only by the `view` command |
| `video.py` | Offscreen rendering of a playlist to an MP4 | `imageio`, `imageio-ffmpeg`, MuJoCo, NumPy; imported only by the `render` command, so a missing `video` extra fails before any work |
| `cli.py`, `__main__.py` | The `mujoco-replay` command and `python -m mujoco_replay` | argparse, NumPy; the window and video modules inside the commands |
| `launcher.py` | The `MujocoReplay` program: the command without a console window, its failures shown in a message box | The standard library; the command inside `main` |

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
   unnamed parts unnamed. The compiler's `fusestatic` is turned off: it would
   merge bodies without joints into their parents inside the spec while
   compiling, so that bodies the model names would vanish from the copies
   (the third review found that crash). Compiling a parse of its own gives
   the original model, whose `qpos` layout the recording uses; its `nq` must
   match the file's.
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
   moves the contents out of the source spec, so each copy attaches a copy
   (`MjSpec.copy`) of a spec reduced once, which is quicker than parsing and
   reducing again; copy 0 has a reduced spec of its own, with the lights.
4. Fixes before compiling. Cameras and lights aimed at a body
   (`mode="targetbody"`) keep aiming at it across the split: one of the
   static scene aimed at a replicated body would aim at a body that no longer
   exists, which does not compile, so it is aimed at copy 0's body, the best
   world's; one inside a copy aimed at a static body was given the copy's
   prefix by attaching (`w3_post`), and loses it. Lights inside the
   replicated bodies are kept in copy 0 only, so that 32 copies of a tracking
   light do not light the scene 32 times over.
5. Set the composite's memory for MuJoCo's working arrays to 4 MiB
   (`spec.memory`): nothing is simulated, and MuJoCo's own sizing reserved
   118 MB for 16 worlds and 3.5 GB for 128, which Linux leaves unused but
   Windows may count against its memory. Compile, and build the mapping by
   name. Copy `k`'s joints happen to sit in
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

Composing Centipede's model takes 0.17 s for 16 copies, 0.45 s for 32, and
6.2 s for 128 in the development container. MuJoCo's compiler and attaching
grow with the square of the number of bodies (the same copies written out as
plain XML compile 6 to 8 times slower still), so 128 worlds take several
seconds to open; the composed-scene cache of stage R6 removes the wait the
second time. Spatial
tendons, skins, and flexible bodies (cloth) are not drawn, since they go with
the other unused parts; a model compiled with `discardvisual` shows only its
collision shapes, as MuJoCo itself would.

Colours: every geom of a ghost copy gets the ghost grey `(0.55, 0.55, 0.55)`
and `geom_matid = -1`, so that a material or texture cannot override the grey.
Its alpha is the ghosts' alpha times the shape's own drawn alpha, so that a
shape the model hides stays hidden; the drawn alpha is the material's when the geom's colour
is MuJoCo's default grey `(0.5, 0.5, 0.5, 1)`, and the geom's otherwise, which
is the rule MuJoCo draws by. The highlighted copy keeps the colours the
composite was compiled with, which are the original model's; they are read
from the composite itself, because every copy has its own prefixed materials,
whose ids differ from the original's. Both arrays are plain fields of
`MjModel` and can be changed at any time, so the highlight can move from one
world to another while playing. The ghosts' alpha comes from their strength,
a setting of the panel: faint 0.07, normal 0.15, strong 0.35 with up to 8
worlds drawn, falling with the square root of the number of worlds beyond
(normal is 0.075 with 32). MuJoCo draws transparent shapes after opaque ones,
blended over them, and a run's worlds start on top of one another, so with a
fixed alpha 15 ghosts in front of the highlighted world washed it out; the
interaction review found that, and the fading keeps it readable. Hidden
ghosts have alpha 0, and MuJoCo leaves such shapes out of the scene
altogether.

Markers: after `mjv_updateScene`, one sphere per drawn world and marker is
added with `mjv_initGeom` into the scene's spare slots, at the marker's
radius: the highlighted world's in a saturated magenta, the ghosts' in the
ghost grey. Over each of the highlighted world's markers stands a beacon
sized to the scene: a pole half the composite's `stat.extent` high and a
head that carries the marker's name, drawn as the shape's label. Centipede's
target is a point 10 to 20 mm ahead of a 34 mm centipede, with an arrival
radius of 1 mm, too small to find by its sphere alone; the beacon shows it
from any distance, and the sphere still shows the arrival zone. Markers are
decoration, so they cast no shadow; they hide with the ghosts, and on their
own with `M`.

## Choosing the worlds

The file may hold more worlds than are drawn. The viewer draws `N` of them,
1, 2, 4, … up to 128 and up to the file's worlds (`world_counts`): the file's
worlds are ordered by `score`, descending and stable
(`numpy.argsort(-score, kind="stable")`), split into `N` bands of as equal a
size as possible, and the best world of each band is drawn, which is
`selected_ranks(K, N, 1)`. So 1 draws the best world, 2 the best of the upper
and of the lower half, and a count at or above the file's worlds draws them
all; the rule asked for by the user on 2026-10-07 replaced the first design's
8 evenly spaced worlds from each of 4 bands. The bands follow rank, not score
values, so that every band holds a world to draw. `--worlds N` sets the count
(16 by default) and `--ids 3,7,9` draws exactly those `world_ids` instead; the
two exclude each other, and an `--ids` list that matches no world of a file is
an error naming the file; while `--ids` holds, the panel names the worlds as
chosen by id instead of offering the count. The highlighted world starts as
rank 0; `B` moves the highlight through the drawn worlds in rank order, and a
double-click on a world highlights it, so each ghost can be inspected in
colour. The rank shown is the producer's, among all its worlds, when the file
gives one (`rank` and `ranked_worlds`), and the rank within the file
otherwise.

## Playback and keys

`Playback` owns a playlist of recordings and a position: file index, frame
index, playing or paused, and the speed as seconds of wall time per recorded
frame. The default is 0.3 s per frame, so that a 20 ms step is visible as a
step. The presets, in seconds per frame, are 3, 2, 1, 0.5, 0.3, 0.2, 0.1,
0.05, then real time (`frame_seconds`), 2× and 4× real time, kept in order of
speed, so that a file with frames longer than 0.05 s places real time among
the fixed presets; the overlay shows both the seconds per frame and the
multiple of real time. Advancing uses wall
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
played. A world the user highlighted stays highlighted, by `world_ids`, when
the new file draws it; otherwise, and whenever the user has not picked one,
the new file's best world is. Clicking or dragging on the timeline pauses at
that frame (`Playback.seek`).

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
| G | Hide the ghosts, or show them again at their strength |
| M | Show or hide the markers |
| V | Reset the view: frame every drawn world from a raised angle |
| T | Look straight down, from above |
| C | Centre the camera on the highlighted world |
| F | Follow the highlighted world on or off |
| H | Show or hide the overlay (remembered) |
| I | Show or hide the setup information |
| O | Open recordings with the system's file picker |
| F1, ? | Show or hide this list of keys, at the top right |
| Tab | Show or hide the side panel (remembered) |
| Esc, Q | Ask whether to quit: Enter, Y, or Q again quits; Esc or N stays. The window's close button asks too |

Right and Left repeat while held; every other key acts once per press. Letter
keys are read as the characters they type (GLFW's character callback), so
that they follow the keyboard's layout: on a French keyboard Q is the key
marked Q, not the one where an American Q sits. Shift is read from the key
itself, so that Caps Lock does not turn B into Shift+B. Mouse: left drag rotates, right drag pans, middle drag zooms, through
`mujoco.mjv_moveCamera` (in MuJoCo 3.12 it takes no scene argument); Shift
with the left drag turns around the vertical only, and Shift with the right
drag pans in the horizontal plane. The wheel zooms in when turned away from
the user, as in most programs (MuJoCo's own viewer does the opposite). The
camera stays at least 2° above the horizon, so it cannot slip under the
floor, and panning switches following off, since following would undo it.
A double-click on a world highlights it (`mjv_select` names the shape under
the cursor, hence its copy); the timeline takes clicks and drags.

## Overlay

Drawn with `mujoco.mjr_overlay` in the corners, and with
`mujoco.mjr_rectangle` and `mujoco.mjr_label` for the timeline, in the normal
font, so that the middle of the window stays clear. In the window the corners
and the timeline keep to the right of the side panel. A message (a file that
could not be opened, or "Composing 32 worlds ...") is shown at the top, and the
empty world shows its hint large in the middle. MuJoCo's fonts hold ASCII
only, so the separators are ` | ` and `x`, and other characters in a
recording's text are replaced: `·` by `|`, accents dropped, the rest by `?`.

- Top left: the title; then the `frame_info` values at the current frame on
  one line (`updates 15 | steps per world 3,840`); then the highlighted world
  (`world 512 | rank 37 of 1,024 | summed reward +1.23 | level 1 of 4`, the
  level's "of 4" only when the file gives `level_count`); then the drawn set
  (`32 of 1,024 worlds drawn | 31 ghosts`, with `(hidden)` while `G` hides
  them). A line too long for the window is cut with `...`.
- Bottom left: the playback line (`file 2 of 3 | frame 124 / 256 | 2.46 s |
  0.3 s per frame (0.067x real time) | paused`); the file shows only for a
  playlist, frames count from 1, and the time is the frame's since the start.
- Bottom: a thin timeline bar across the window, filled to the current frame,
  with an orange tick per event and its label in a box above it (a label that
  would cover the one before it is left out), and small blue ticks where the
  highlighted world's episodes start. A click or drag on it, or just above it,
  goes to that frame.
- Event flash: when playback passes an event, its label is drawn large at the
  bottom centre, above the playback line, for one second.
- Setup (`I`): the `setup_json` object as `key = value` lines, a nested
  object's values under a heading of its dotted path, indented, at the top
  right, right of the top-left lines and above the playback line. MuJoCo's
  `mjr_overlay` keeps only 500 characters, which cut Centipede's 2,250 off
  mid-word, so each line is drawn on its own, in as many columns as it takes;
  the columns share the free width, a line too long for its column is cut,
  and lines beyond the columns that fit are counted on the last line. The
  lines go right of the top-left lines, or below them across the whole
  width, whichever leaves out fewer lines and then cuts fewer: right of a
  long title there may be no room at all. The list of keys (`F1`) is drawn
  the same way, in short lines, so that two columns fit beside the panel at
  150 % scaling.
- Bottom right, above the playback line: the frame-rate readout, when it is
  on.

The setup, the list of keys, the readout, and messages show when the overlay
is hidden too.

The overlay is drawn in the same way into the window and into the offscreen
buffer, so the video carries it unless `--no-hud` is given.

## Camera

A free camera (`mjCAMERA_FREE`) starts looking at the centre of the drawn
worlds and their markers at frame 0, from a raised angle, at a distance of 0.8
times their spread plus 1.2 times the composite model's `stat.extent`; `V`,
or the panel's Reset button, frames them so again, and `T` looks straight
down. Framing leaves out a world whose positions are not finite or lie more
than 1,000 extents from the others' median (from the best world's centre,
when half the worlds diverged and the median lies between them), as a
diverged simulation leaves them, and neither centring nor following moves the camera to a point that is
not finite: one such world could otherwise take the camera with it and blank
the picture (the third review found that). A
world's centre is the centre of mass of all its replicated root bodies,
weighted by their masses, from the `subtree_com` that `mj_comPos` computes
(`mj_kinematics` leaves it at zero, so `set_frame` runs both). `C` recentres
on the highlighted world's centre; `F` keeps the camera's look-at point on it
every frame, following the highlight as it moves, and leaves the camera where
it stands when switched off. This is what MuJoCo's tracking camera
(`mjCAMERA_TRACKING`) does for one body, but a world may have several root
bodies. The camera is independent of playback; it is kept when the next file
reuses the composite, and reframed for a file of another model, in the window
and in the video alike.

## Rendering in a window and offscreen

`SceneRenderer` wraps one `mujoco.MjrContext`, one `MjvScene`, `MjvCamera`,
and `MjvOption`, and needs a current OpenGL context before it is made. The
viewer creates a visible GLFW window and renders to its framebuffer at the
framebuffer size, not the window size, so high-DPI screens render at full
resolution, with a font scale taken from the window's content scale (100 at
normal density, 150 at 125 % and 150 %, 200 at double). Without a size from
the command line, the window takes 85 % of the screen's free area, in its
middle, so that the panel has room at the larger font scales. The scene is
drawn right of the panel, so that the camera's centre is the visible
centre; the panel is opaque.
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

Each frame, `mjv_updateScene` lists the shapes to draw; while shadows are on,
the ghosts' and the floors' shapes are then marked as decoration
(`mjCAT_DECOR`), which casts no shadow but still receives them: MuJoCo draws
translucent shapes with full, dark shadows, which would cover the floor in
grey, and a plane can only shade itself, which speckled the floor with black
dashes; the markers are added; `mjr_render` draws, and the overlay goes on
top. The highlighted world keeps its shadow. Sites are not
drawn at all: they mark points for sensors and attachments, and every ghost
would carry opaque copies of them. For 32 copies of the chain (1,633 shapes),
listing the shapes takes 0.05 ms, marking the ghosts 1 ms in Python, and
posing the frame 0.2 ms.

## Graphics settings

Stage R7's performance review measured where a frame's time goes, with 16,
32, and 128 copies of Centipede's model: the Python and MuJoCo work on the
processor, which carries over to the user's laptop, and the drawing in Mesa's
software renderer, whose numbers only compare with each other. MuJoCo issues
one draw call per shape, has no instancing, draws a translucent ghost twice,
and draws everything again for a reflection: 32 worlds make 3,969 draws per
frame without reflections and 8,064 with them. With Mesa's no-op driver, which
issues the calls without drawing, issuing them took 9.8 ms at 32 worlds and
41 ms at 128, the largest cost on the processor; posing a frame took 0.2 and
0.8 ms. In the software renderer, against Performance, reflections added 93 %,
4× anti-aliasing 86 %, shadows 57 %, and MuJoCo's own tessellation of round
shapes (28 facets around, 16 along) 280 %; reflections changed the picture of
Centipede's dark floor by 0.3 of 255 on average, and 16 by 8 facets changed
it by less, even close up. So fine shapes are 16 by 8, which cut Quality's
triangles by about four. Quality keeps reflections all the same, as the user
decided on 2026-10-07: it is the mode for the look, and the Reflections
switch, or Performance, drops them when the frame rate matters. A shadow map
of 2048 or 1024 texels saved no time and showed jagged shadows, so MuJoCo's
4096 stays. With 128 worlds, issuing the draws alone may take 15 to 30 ms on
the laptop (an estimate: a third of Mesa's time), so 32 worlds is the
smooth maximum there, and 128 a view for pausing.

`Graphics` holds five switches, which two presets set at once:

| Switch | Quality | Performance | How it is done |
| --- | --- | --- | --- |
| Shadows | on | off | MuJoCo's `mjRND_SHADOW` flag; a scene whose lights cast no shadow gets one from its first light (Centipede's light casts none), over at least 4 model extents (`vis.map.shadowclip`) |
| Reflections | on | off | `mjRND_REFLECTION`; a floor (a plane with a material) that does not reflect reflects 0.08 |
| Anti-aliasing | on | off | 4 samples in MuJoCo's offscreen buffer (`vis.quality.offsamples`) |
| Fine shapes | on | off | `vis.quality.numslices` and `numstacks`: 16 and 8, or 12 and 6 |
| Resolution | 100 % | 75 % | the share of the window's width and height drawn, then scaled up |

Changing a switch after a preset makes the mode "custom". The anti-aliasing
and the shapes are part of the `MjrContext`, which is made again when they
change; the others apply at the next frame. The ghosts never cast shadows
(they are marked as decoration), so a shadow belongs to the highlighted world.

The visual review found Quality looking worse than Performance on Centipede's
small model: black dashes on the floor and speckles on the robot's back,
where shadows fell on the shapes casting them, and the floor's reflection
showing the legs as pale sticks, like extra ghosts, and the target's beacon
as a second pole under the floor. Floors no longer cast shadows; MuJoCo's
default shadow area of one model extent (`shadowclip = 1`) both left worlds
away from the first without shadows and gave too little depth margin at this
scale, and 3 extents and more cleared the robot's speckles, so the area is
widened to 4; the floor reflects 0.08 instead of 0.15.

In the window, the scene is always drawn into MuJoCo's offscreen buffer at
the chosen share of the window's size and copied into the window with
`mjr_blitBuffer`, which scales it up smoothly; the overlay and the panel are
then drawn into the window at its full resolution, so text stays sharp. The
window itself has no multisampling, since the offscreen buffer does it. The
buffer grows with `mjr_resizeOffscreen` when the window does. When only the
overlay or the panel changed, as when the mouse moves over a button, a
message goes, or the panel scrolls, the scene's picture is the same: the
renderer compares what the picture depends on (the scene and context shown,
the frame, the highlight, the ghosts, the markers, the graphics, the camera,
the size) with the last time, and copies the buffer into the window again
instead of drawing the scene. The review measured a redraw for a hover at 128
worlds falling from 51 ms to 2.3 ms with the no-op driver.

The frame-rate readout, a switch of the panel, shows how long a frame takes
to draw and the frame rate that allows (`draw 12.3 ms | up to 81 frames/s`):
while it is on, each frame waits for the graphics card (`mjr_finish`) so that
the time is the card's, not only the program's; frames that only copy the
picture again are not counted, and the first frame after the readout is
switched on, or after the graphics or the scene change, draws the scene even
when nothing moved, so that there is always a time to show. It shows with the overlay
hidden too, and starts measuring again when the graphics or the scene change,
drawing once more at once so that a paused window shows the new time. The
window draws only when
something changed: a frame passed, the camera or a setting moved, a message
came or went. Between such moments the main loop sleeps in
`glfw.wait_events_timeout` until the next frame is due
(`Playback.seconds_to_next_frame`), so a paused window, or one playing at
3 s per frame, keeps the graphics card idle.

## The application

The window opens on an empty world: a checkered floor under a sky, in
Centipede's colours, with the line "Open recordings (O), or drop .npz files
here". Recordings come from the command line, from the panel's Open button,
which shows the system's file picker, or from files dropped onto the window
(GLFW's drop callback). The picker is tkinter's, run in a process of its own,
which prints the chosen paths in UTF-8 into a temporary file: that keeps
tkinter's event loop apart from GLFW's, the window keeps drawing while the
picker is open, and no pipe can fill and hold the picker open, which a few
dozen paths would do on Windows; where tkinter is missing, a message says so
and points to dropping. Several files picked or dropped play in the order of
their names, numbers by value (`cycle_2` before `cycle_10`). A file that cannot be
read or composed is reported as a message at the top of the window, and the
window goes on showing what it showed; a later file of a playlist whose model
fails is left out of the playlist, and playback goes back, paused, to the
frame shown before.

The side panel, on the left, is drawn with MuJoCo's own overlay functions
(`mjr_rectangle`, `mjr_label`), so the viewer needs nothing beyond MuJoCo and
NumPy. Its sections are Playback (the transport buttons, the time per frame,
the file), Worlds (how many are shown, the highlighted rank, the ghosts'
strength), View (Reset, Top, Follow), Graphics (the two presets, the four
switches, the resolution), and Options (overlay, markers, setup, keys, frame
rate, the cache). Switches are buttons lit while on, two or three to a row,
so that the panel fits a laptop's window at 150 % scaling; when it still
does not fit, a scroll bar along its edge says so, and the wheel scrolls it.
The time per frame is the stepper's value, so that its minus makes playback
faster and its value smaller, as the eye expects. The viewer describes the
panel as plain rows each frame; `ui.Panel` lays them out, draws them, and
names the action under a click, which the viewer then carries out, the same
action a key would. A button acts when the mouse is released over it, so that
a drag that begins on the panel, or slips off a button, does nothing. Tab
hides the panel and leaves a small Panel button. A click's position is the
one the cursor callback recorded in event order: asking GLFW for the cursor
at the click would give where it is after the events still queued, so on a
slow card a quick second click would land the first one too; driving the
window found that. For the same reason the panel is laid out again right
after each action, so that a click queued behind Tab does not meet the
hidden panel.

The settings (the graphics, the number of worlds, the ghosts' strength, the
cache, the frame-rate readout, the panel, and the overlay) are saved as JSON
in the user's settings folder (`%APPDATA%\MujocoReplay` on Windows,
`~/.config/mujoco-replay` on Linux) whenever they change, and read at the
next start; a missing or
damaged file, or a wrong value, gives the default for that value, and a
byte-order mark, which some Windows editors write, is accepted. The first
start is in Performance mode with 16 worlds.

Composed scenes are cached in the user's cache folder
(`%LOCALAPPDATA%\MujocoReplay\cache` on Windows, `~/.cache/mujoco-replay` on
Linux) as compiled models in MuJoCo's binary format, named by a hash of the
model, its assets, the replicated bodies, the number of copies, the MuJoCo
version, and a version of the composition itself, each part hashed with its
length so that two different sets of parts never run together. MuJoCo
writes the model into memory (`mj_saveModel` with a buffer) and reads it from
memory (`MjModel.from_binary_path` with the bytes as an asset), and Python
does the file work, because MuJoCo's own file access may not open a path
with characters outside the system's code page. A run opened again, or a count changed back, then
skips composing: with Centipede's model, 16 or 32 copies load in about 30 ms
instead of 0.17 or 0.45 s, and 128 in 0.1 s instead of 6.2 s, which includes
about 15 ms to compile the single model, which the joint mapping still
needs. The files take 5 to 27 MB; the newest 16 are kept, and a damaged one
is composed again (MuJoCo then prints a warning and, as it does for every
warning, appends it to `MUJOCO_LOG.TXT` in the current folder). The panel's
switch turns the cache off, and says where the cache is.

## Video

`mujoco-replay render FILES --out replay.mp4 [--width 1280 --height 720
--fps 30 --speed 0.3 --mode quality --no-hud]` plays the playlist at `--speed` seconds per
recorded frame and writes `fps` video frames per second, so each recorded
frame repeats for `speed × fps` video frames (9 at the defaults). Frames go to
`imageio.get_writer` with the `ffmpeg` plugin from `imageio-ffmpeg`, which
bundles its own encoder, so no system installation is needed. The `video`
extra installs both; the viewer does not need them.

Playback runs on a clock of video frames instead of the wall clock: video
frame `i` shows the scene at `i / fps` seconds, so the result is the same
however long the drawing takes. A recorded frame is drawn once and its pixels
repeated, and drawn again only when the event flash appears or goes. When the
last file ends with an event (frame `T`), the last frame is held for the
second the flash lasts, so that the video shows it too. Each file shows its
own best world in colour, and consecutive files of one model reuse the
composite, as in the window. The size is rounded down to even numbers,
which the H.264 encoder needs, and the font scale follows the height (150 at
720 rows, 200 at 1,080). A progress line counts the video frames on the error
stream. With the software renderer of the development container, two 40-frame
files at 640 × 360 and 0.1 s per frame took 2 min 18 s to export.

## Command line

```text
mujoco-replay [view] [FILE ...] [--worlds N | --ids IDS] [--mode quality|performance]
              [--speed SECONDS] [--no-hud] [--width W] [--height H]
mujoco-replay render FILE [FILE ...] --out PATH [--fps 30] [--speed SECONDS]
              [--width 1280] [--height 720] [--worlds N | --ids IDS]
              [--mode quality|performance] [--no-hud]
```

`view` is the default subcommand, and the files are optional: without them
the window opens on the empty world. `python -m mujoco_replay` is the same
command. The saved settings supply what the command line leaves out; `--mode`
and `--worlds` given to `view` are remembered, as the panel's changes are, and
`--no-hud` hides the overlay for this run only: it is not saved, and a video
carries the overlay unless `render` is given `--no-hud`, whatever the
viewer's switch. `render` draws with the Quality graphics unless given
`--mode performance`, and at 1280 × 720 unless told otherwise; the window
takes most of the screen. Both subcommands expand wildcards in file names
themselves (`recordings\*.npz`), since Windows' shells pass them on as they
are, read and check every file before opening anything (the checks that need
a later file's model come when playback reaches it), and exit with status 1
and a one-line message for a bad file, a missing OpenGL, or a video file that
cannot be written; `render` names a missing `video` extra the same way,
refuses an output folder that does not exist before any work, and reports
its progress on the error stream. Options may come before or after the
files, and sizes are at least 16 pixels.

## The program

The `gui-scripts` entry of `pyproject.toml` makes pip install a program,
`MujocoReplay`, beside the command: on Windows, `MujocoReplay.exe` in the
environment's `Scripts` folder, which runs `pythonw`, Python without a
console window, so that the tool starts from the Start menu, the taskbar, or
a desktop shortcut like any application. The program runs the command
(`launcher.main` calls `cli.main`, which reads the program's arguments): it
opens the empty world, or the files dropped onto the program or its
shortcut, which Windows passes as arguments, with the same saved settings.
Having no code of its own beyond that, it follows every change to the
command and the viewer, and with the editable install a pull is enough; only
a change to the entry points or the dependencies needs `pip install` again.
`pythonw -m mujoco_replay.launcher` runs the same program without the
`.exe`.

Without a console, `sys.stderr` is `None`, and what the command prints when
it fails (a file it cannot read, a missing OpenGL, a wrong option, or a
traceback) would go nowhere. The program keeps the error stream in memory
instead, and when the command fails, it shows the last 40 lines in a message
box: Windows' own `MessageBoxW`, through `ctypes`, so no library is needed.
What the window reports itself, such as a file opened from the panel that
cannot be read, stays a message in the window. With a console, as on Linux
or when started from a terminal, the program is the command itself. A
failure before the program's own code runs, such as this folder moved away
from where the editable install points, shows nothing; `mujoco-replay`, run
in a terminal, prints it.

The program's current folder is the one its shortcut starts in, by default
the `Scripts` folder, where MuJoCo appends its warnings to `MUJOCO_LOG.TXT`.
The process that draws is the base Python's `pythonw.exe`, which the
environment's own `pythonw.exe` starts, so that is the one Windows' graphics
settings must send to the GeForce.

## Dependencies

`mujoco==3.12.0` (the version the composition was verified with), `numpy`,
and `glfw`, which MuJoCo also brings but the window imports directly.
Optional `video`: `imageio`, `imageio-ffmpeg`. Development: `pytest`, `ruff`.
Python 3.11 to 3.14: MuJoCo 3.12.0 has no wheels for newer versions.

## Verification without a display

An assistant cannot see the user's screen. Rendering is verified by
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

In stage R6 the application was driven the same way, through the panel this
time: started empty, started with a stand-in recording simulated from
Centipede's own model, paused, fewer and more worlds, Quality with the
frame-rate readout, the panel hidden and shown again, the highlight moved,
the resolution lowered, and three quick clicks while a frame was drawing,
which found the click-position fault described above. Dropping files cannot
be faked that way, so a test calls the drop callback directly; the file
picker could not be shown at all, since the container's Python has no
tkinter.

In stage R7 five reviewers, each working on its own, went through the tool by
running it: one composed some thirty unusual models and fed the reader
seventy damaged or hostile files; one wrote recordings with Centipede's own
recorder code and opened them; one drove the window through every control
at sizes from 480 × 320 to 1600 × 1000; one timed each step of a frame; and
one checked the documents, a fresh install, and the Windows code paths by
reading. Screenshots of the window and offscreen frames in both modes made
the visual review. Their findings and the fixes are listed in `plan.md`
under R7, and the reasons for each change are in the sections above.
