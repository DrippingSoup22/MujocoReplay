# Development plan

This plan builds MujocoReplay in fourteen stages; the user asked for the sixth
on 2026-10-07, once the first five were built, for the seventh, a full review,
once the sixth was, for the eighth, a program to start it from, once the
seventh was, and for the ninth, any world on its own and new defaults, on
2026-10-08; the tenth, rings and radii that change, Centipede asked for on
the user's behalf the same day, and the user asked for the eleventh, tabs,
the twelfth, an icon with Loop and Play next, and the thirteenth, an
executable, on 2026-10-09, and the fourteenth, a camera that follows, on
2026-10-10. It is paired with Stage 7 of the
sibling `Centipede` project (`../Centipede/plan.md`), which records the files
this tool replays. The format is fixed in
[docs/recording-format.md](docs/recording-format.md) and the design in
[docs/design.md](docs/design.md); this plan says what to build in which order
and when each stage is done.

## Rules

- One scope per stage; later files are not created early.
- Each stage ends with its tests passing under `tests/`, one file per module.
- The assistant writes everything in this project (decided 2026-10-07).
- The tool stays independent of Centipede: nothing centipede-specific in the
  code, and no import from it. Centipede depends on this package, never the
  other way round.
- Rendering is verified with offscreen frames written to PNG files; the window
  and its controls are verified by the user.
- Commits carry Claude's credit lines, as the user chose on 2026-10-08;
  `AGENTS.md` holds the project's rules.

## Order of work across the two projects

1. **R1** here: the package, the format, and the selection rule. Centipede
   needs these to write files.
2. **Centipede Stage 7, parts A to D**: the simulation diagnostics, the
   recorder, the settings, and the files. Ends with a user-launched smoke run
   that writes `recordings/cycle_0001.npz` to `cycle_0003.npz`, and a
   user-launched evaluation that writes its recording.
3. **R2 and R3** here: the scene, the renderer, the viewer, verified on those
   files.
4. **R4** here: the video.
5. **R5** here and Centipede Stage 7 part E: documents, status, and the
   end-of-stage commits in both projects.

## Stages

### R1 — Package and recording format

**Builds:** `pyproject.toml` (package `mujoco_replay` under `src/`, the
`mujoco-replay` console script pointing at `cli:main`, dependencies and extras
as in the design, pytest and ruff settings as in Centipede's `pyproject.toml`);
`src/mujoco_replay/__init__.py`, `recording.py`, `selection.py`; a `cli.py`
and `__main__.py` that only parse arguments and report that viewing is not
built yet; `tests/test_recording.py`, `tests/test_selection.py`.

**Done when** tests show that:

- a `Recording` with every optional group written and read back is equal,
  array by array, and one with only the required keys reads with the defaults
  filled in;
- a file with a wrong shape, an event frame beyond `T`, a level below 1, a
  marker group with a missing key, or `setup_json` that is not an object is
  rejected with an error naming the key;
- `selected_ranks` selects every rank when the worlds fit, always includes
  rank 0 and the last rank otherwise, returns `levels × per_level` distinct
  sorted ranks with levels from 1 to `levels` when the worlds are plentiful,
  and handles `per_level = 1` and uneven bands;
- importing `mujoco_replay`, `mujoco_replay.recording`, and
  `mujoco_replay.selection` loads neither `glfw` nor `OpenGL` nor `mujoco`
  (checked through `sys.modules` in a fresh subprocess).

**Result (2026-10-07):** done; 15 tests pass. The bands of the rank rule
start at ceilings, `⌈i·W/L⌉`, so that a rank's level is `⌊rank·L/W⌋ + 1`
and the best world is always level 1; the format document was corrected to
say so. `level_of_ranks` is public next to `selected_ranks`, because
Centipede needs the level of arbitrary ranks for its `"first"` selection.
Centipede's experiment tests write and read back real recordings of model v2.

### R2 — Scene composition

**Builds:** `src/mujoco_replay/scene.py` and `tests/test_scene.py`. The tests
use a small model written in the test itself (a floor, a light, and one root
body with a free joint, a hinge, and two shapes), and Centipede's
`../Centipede/models/assembly_v2.xml` through its path when that file exists,
skipping otherwise.

**Done when** tests show that:

- the composite model has `K` copies of the replicated bodies and one copy of
  the static scene, no actuators, and `nq` equal to the static joints plus
  `K` times the replicated joints;
- the default `replicated_bodies` is every root body whose subtree has a
  joint, and a root body without joints stays static;
- after `set_frame`, copy `k`'s body positions equal those of the original
  model posed with world `k`'s `qpos` alone (`mj_kinematics` on the original
  model, compared by body name), and no other copy moved;
- the ghost copies' shapes are grey with the ghost alpha and no material, the
  highlighted copy keeps the original colours, and moving the highlight swaps
  them;
- a recording whose `nq` differs from the model's is rejected with a clear
  error.

**Result (2026-10-07):** done; 7 tests pass, and the case with Centipede's
model skips in the development container, which has no `../Centipede`; it
runs wherever that folder exists. The small model lives in `tests/conftest.py`
for the later stages' tests too, and adds what real models contain: a static
door on a hinge, a hidden shape, a material, an unnamed joint, and a light
aimed at the robot. Three steps of the composition changed, with the reasons
in [docs/design.md](docs/design.md#composing-the-scene): each copy is a
reduced spec attached whole, because attaching two root bodies that share a
material fails; unnamed bodies and joints are named before joints are matched;
and a static camera or light aimed at a moving body is aimed at copy 0.
Composing 32 copies of a 52-shape chain takes 0.7 s.

### R3 — Renderer, playback, viewer

**Builds:** `src/mujoco_replay/playback.py`, `render.py`, `viewer.py`, the
`view` command in `cli.py`; `tests/test_playback.py`, `tests/test_render.py`.

**Done when:**

- playback tests show that advancing by wall time moves the expected number of
  frames at each preset, pauses, steps one frame either way, runs on into the
  next file and stops at the end of the last, reports the events passed, and
  cycles the speed presets;
- an offscreen render test draws one frame of a two-world recording of the
  small test model and checks the image is not blank and differs between the
  ghosts-on and ghosts-off settings (skipped with a reason where no OpenGL
  context can be made);
- frames of Centipede's smoke recording rendered offscreen to PNG files show
  grey translucent ghosts, the best world coloured, the target markers, the
  overlay in the corners, and the timeline, inspected by the assistant;
- the user opens `mujoco-replay runs/<smoke run>/recordings/cycle_0003.npz`
  and confirms the camera, play and pause, stepping, the speed presets, the
  highlight key, and the overlay toggles.

**Result (2026-10-07):** built; the user's check of the window is pending.
The six playback tests and the render test pass (30 tests in all, with the
render test skipped where no OpenGL context exists), plus a scene test for
reusing the composite across files of one model. Centipede's smoke recording
was not available in the development container, so the frames were rendered
from recordings of the centipede-like chain of stage R2, simulated with
MuJoCo for the purpose (64 worlds, a target marker, two events, `frame_info`,
`setup_json`): they showed the grey translucent ghosts, the best world
coloured, the markers, the corners, the setup panel, the flash, and the
timeline. The window was also driven there under a virtual display with
synthetic key presses: every key acted as listed, a three-file playlist ran
on from file to file, and Q closed it cleanly. What changed from the design,
with the reasons in [docs/design.md](docs/design.md): ghosts cast no shadows
(marked as decoration), offscreen drawing uses `mujoco.GLContext`, the
tracking camera needs `mj_comPos`, and the overlay keeps to ASCII.

### R4 — Video

**Builds:** `src/mujoco_replay/video.py`, the `render` command, the `video`
extra; `tests/test_video.py`.

**Done when** a test exports a two-frame recording of the test model at a
small size and the written file has the expected number of video frames
(skipped where OpenGL or `imageio-ffmpeg` is missing), and the user-launched
export of the smoke recording plays in an ordinary video player.

**Result (2026-10-07):** built; the user's check in a video player is pending.
The test writes two recorded frames at 0.3 s per frame and 10 frames per
second, reads the file back, and finds six frames of 96 × 64 pixels. Through
the command, two 40-frame recordings of the chain became an 8-second H.264
file of 240 frames, whose frames showed the overlay, the file change, and the
end-of-window event flashing as the second file began. A second failed
attempt to make an OpenGL context aborts the process in MuJoCo 3.12, so the
tests ask once per session (`opengl` in `tests/conftest.py`).

### R5 — Documents and status

**Builds:** the README's status and use sections checked against the built
commands; `docs/design.md` corrected where the implementation had to differ,
with the reason; a link from Centipede's `docs/architecture.md` and
`README.md` to this project as the way to watch recordings.

**Done when** the documents describe the tool as built, and the user has
accepted the end-of-stage commit in both projects.

**Result (2026-10-07):** this project's half is done. The README's status,
setup, and use sections match the built commands; `docs/design.md` records
every change from the first design with its reason, stage by stage; and
`docs/recording-format.md` now names the checks made at composition and the
ASCII-only overlay. The links from Centipede's documents remain, because the
development container had no access to that project, and with them the
user's acceptance and the window and video checks of stages R3 and R4.

### R6 — The application: panel, settings, and performance

Asked by the user on 2026-10-07, after R5: the window becomes an application
that starts on an empty flat world and is driven by a side panel, with a
Quality and a Performance mode for weak graphics cards such as the user's
GeForce MX330, a setting for how many worlds are drawn, and a cache of
composed scenes. The panel offers the tool's own settings only, never
MuJoCo's raw visual options. One file holds the model and its environment, as
a recording already does, so nothing is loaded separately.

**Builds:**

- the world count: 1, 2, 4, … up to 128, and up to the file's worlds. With
  `N` worlds, the file's worlds are split by score rank into `N` bands of as
  equal a size as possible and the best world of each band is drawn
  (`selected_ranks(K, N, 1)`): 1 shows the best, 2 the best of each half. It
  replaces `--levels`, `--per-level`, and `--all`; `selected_ranks` and
  `level_of_ranks`, which Centipede imports, stay as they are;
- `src/mujoco_replay/settings.py`: the settings, the two presets, and keeping
  them in the user's settings folder between runs; `tests/test_settings.py`;
- graphics settings applied by the renderer: shadows (added to the model's
  lights), reflections (a slight one added to a floor that has none), 4×
  anti-aliasing, shape detail (MuJoCo's `numslices` and `numstacks`), and a
  resolution scale, drawing offscreen at a fraction of the window and scaling
  up, with the overlay at full resolution; a frame-rate readout; no redrawing
  while nothing changes;
- the composed-scene cache: the compiled composite saved in MuJoCo's binary
  format in the user's cache folder, keyed by the model, the number of copies,
  and the MuJoCo version;
- `src/mujoco_replay/ui.py`: the side panel drawn with MuJoCo's overlay
  functions (buttons, switches, steppers), its layout, and its hit testing;
  `tests/test_ui.py`;
- the empty flat world at start; an Open button that shows the system's file
  picker (tkinter, in a separate process, so the window keeps drawing);
  dropping files onto the window; the target drawn at its true size with a
  beacon sized to the scene, so that Centipede's 1 mm target can be found.

**Done when:**

- tests show the count rule, the presets and the saving of settings, the
  panel's hit testing, a cached composite posing exactly like a freshly
  composed one, and both presets drawing;
- offscreen frames of a stand-in recording made with Centipede's own model,
  in both modes, inspected; the window driven under a virtual display through
  the panel: opening by dropping files, switching modes, changing the count,
  and a reopened run served from the cache;
- the user runs it on the MX330 with Centipede's recordings and finds
  settings that play smoothly.

**Result (2026-10-07):** built; the user's check on the MX330 is pending. The
count rule, the settings and presets, the cache, the panel, both presets
drawing, the scaled window drawing, and dropped files each have tests (54 in
all; the 12 that draw skip without OpenGL). Centipede's repository was cloned
read-only as the sibling folder, so the scene test with its model now runs,
and a stand-in recording was simulated from that model with targets placed as
Centipede places them. Measured in the container's software renderer, fine
shapes cost 3.5 times the coarse ones, anti-aliasing 1.6 times, and shadows
with reflections twice; the cache turned 0.76 s of composing 32 copies into
13 ms. Driving the window through the panel found one fault, fixed: a click
made while a frame was drawing landed where the next click did. The file
picker could not be tried, as the container's Python has no tkinter.
A second independent review then found eleven defects, each reproduced and
fixed: stepping back into a file that needed composing crashed; a later file
whose model failed closed the window; the end of a file without an event left
the last frame undrawn; changing the count while playing skipped the
composing time; with `--ids` the count stepper composed again for nothing and
saved a count; `--no-hud` was saved with the next change, and `render`
followed the viewer's overlay switch instead of `--no-hud`; the frame-rate
readout lagged a graphics change and vanished with the overlay; "fewer"
skipped a step from a count between steps; a settings file starting with a
byte-order mark was ignored; and a click queued behind Tab acted on the
hidden panel. `docs/design.md` describes the graphics, the application, and
the cache.

### R7 — Full review

Asked by the user on 2026-10-07, after R6: review the whole application from
several sides, its soundness and its look, fix what is found, and add a few
settings that let the user shape the view and interact better, each with its
reason, keeping the application simple.

**Builds:** five independent reviews, each by running code: varied MuJoCo
models and damaged or hostile files; the fit with what Centipede writes,
using Centipede's own recorder code; the window driven like a user through
every control, at several sizes; where the time of a frame goes; and the
documents, the packaging, and Windows. A visual review of screenshots of the
window and of offscreen frames in both modes. Then the fixes, and these
additions, each answering a finding:

- the ghosts' strength (faint, normal, strong, hidden), fading as more worlds
  are drawn, because with a fixed alpha the ghosts buried the highlighted
  world;
- Reset view (`V`) and a view from above (`T`), because a lost camera had no
  way back, and a top view shows how far each world got;
- a double-click on a world highlights it, because inspecting an outlier took
  up to 127 presses of `B`;
- a click or drag on the timeline goes to that frame, the natural way to
  reach an event in a replay;
- the list of keys on `F1`, because the panel never mentioned the keys;
- `render --mode`, so that a video can be drawn as plainly as the window;
- leaner graphics, from the performance review: 16 by 8 facets instead of 28
  by 16; the scene not drawn again when only the overlay or the panel
  changed; 4 MiB instead of up to 3.5 GB reserved for MuJoCo's working
  arrays; composing 40 % quicker;
- the optional format keys `level_count`, `rank`, and `ranked_worlds`, so
  that a file holding a selection of a producer's worlds can show "rank 37 of
  1,024" and "level 1 of 4" truthfully.

**Done when:** every finding is fixed or answered, with a test where a test
can hold it, the documents describe the result, and the user has the list of
what remains for them.

**Result (2026-10-07):** done; `docs/design.md` gives each change with its
reason. The reviews found, among others: a half-written file (Centipede
writes its files in place) closing the window; models compiled with
`fusestatic` crashing the composition; the reader taking text for numbers;
one diverged world blanking the picture; a cache key that two different sets
of assets could share; the file picker able to hang on a full pipe; the
setup cut at MuJoCo's 500 characters, mid-word; Quality drawing black dashes
and speckles with its shadows and pale sticks with its reflection; the scene
centred behind the panel; the panel overflowing a laptop's window at 150 %
scaling; buttons acting on press, so that a drag switched to Quality; the
highlight wandering off the best world between files; and a README that
did not say how to set up on Windows or which Python versions work. A last
review of the changes themselves then found eight more defects, all fixed:
a recording with one damaged byte (an unknown zip compression, a broken
array header, a bad zlib stream) still closed the window; the key list and
the setup could be cut short or vanish right of a long title at 150 %
scaling, and now go below the top-left lines when that shows more; the
frame-rate readout could stay on "measuring" forever, since an unchanged
picture is not drawn again; cutting a long text took quadratic time; held
letter keys repeated, and Caps Lock turned B into Shift+B; a `Recording`
took ranks or a level count its own reader refused; framing failed when
half the worlds diverged; and one file name could not be sorted. Last, the
user kept reflections in Quality, which the performance review had taken
out, and asked for a question before quitting, which Q, Esc, and the
window's close button now ask, in a small box drawn in the window.

### R8 — The program

Asked by the user on 2026-10-07, after R7: start the tool like an
application, without a terminal, on the empty world, and add the files from
there. A standalone executable, which would not need Python installed, is
left for later.

**Builds:** a `gui-scripts` entry that makes pip install `MujocoReplay` (on
Windows `MujocoReplay.exe`, which runs Python without a console window), and
`launcher.py` behind it, which runs the `mujoco-replay` command with the
program's arguments and, without a console, shows the command's failure in
a message box. The command and the viewer stay as they are: the user tries
them first.

**Done when:** the program opens the same window as the command, empty or
with the files given; a failure without a console reaches the message box;
tests hold both; and the README says how to pin the program.

**Result (2026-10-07):** built. In the container, with the error stream
taken away as `pythonw` has it, the program opened the empty world and a
recording given to it, and an unreadable file and a missing display each
ended in the text the message box would show. The message box and
`MujocoReplay.exe` exist only on Windows, so the user checks them.

### R9 — Any world on its own, and new defaults

Asked by the user on 2026-10-08, after R8: the highlight becomes a switch over
every world of the run, so that each world's run can be watched on its own
even when fewer worlds are drawn than the file holds; faint ghosts, 0.1 s per
frame, and full resolution become the defaults; and why a run of more than
32 worlds shows 32, and how many worlds the tool could load.

**Builds:** `choose_worlds(..., keep=index)`, which draws a world in place
of the best world of its band; the Highlight stepper and `B` stepping through
every world of the file, drawn or not, the stepper shown whenever the file
holds more than one world; the new defaults, and a version in the settings
file, so that a file saved under the first defaults takes the new ghosts and
resolution once; measurements of composing and drawing beyond 128 worlds.

**Done when:** tests show the kept world drawn in its band, the highlight
reaching every world on the same composite with the count unchanged, the
panel offering the highlight with one world drawn, the picked world drawn in
the next file, and an old settings file taking the new defaults; the window,
driven under a virtual display, starts with the new defaults and shows the
worlds of a stand-in recording one at a time.

**Result (2026-10-08):** built; the user's check pending. 96 tests pass (26
draw and skip without OpenGL); one of them holds a fault found in review: a
file whose worlds share an id, which the reader accepts, crashed the window
when the highlight stepped onto the second, so `keep` is an index into the
file, not an id. The window was driven under a virtual display with a
32-world stand-in recording simulated from Centipede's model and a settings
file in the first defaults' format (normal ghosts, 75 %, 32 worlds): it
opened at 0.1 s per frame, with faint ghosts, at 100 %, still with 32 worlds;
with one world drawn, the Highlight stepper, `B`, and Shift+B showed ranks 2,
4, and 3 alone; back at 4 worlds, rank 3 stayed highlighted among 3 ghosts,
and a double-click on a ghost highlighted rank 25. A 1,024-world stand-in
of 256 frames (76 MB) read in 0.08 s and stepped the same way.

The 32 worlds come from Centipede, which keeps `record_levels ×
record_per_level` = 4 × 8 worlds in each recording by default, out of the 64
worlds of the user's current runs (1,024 in its full training); this tool
draws up to 128 of a file's worlds, and a file may hold any number. For
drawing more at once, measured with Centipede's model: composing
took 0.4, 5.3, 22, and 89 s for 32, 128, 256, and 512 copies, four times as
long for each doubling, and 1,024 copies overflowed the composite's 4 MiB of
working memory and took 6.7 min with 64 MiB; issuing one frame's draws took
12, 38, 106, 164, and 392 ms for 32 to 1,024 worlds with Mesa's no-op driver,
work on the processor that a faster graphics card does not remove. Posing one
model per world and adding its shapes to the scene (`mjv_addGeoms`) instead
of composing took 11 ms a frame for 1,024 worlds, with no wait to open. The
user then chose to record every world of a run, with a setting for how many
worlds a recording keeps: a change to Centipede, handed to its sessions in a
note outside this repository and made there the same day, after its commit
`1b42442`. Its `[run] record_worlds` is `"all"` by default, or a number N
of worlds evenly spaced over all ranks, best and worst included
(`selected_ranks(W, 1, N)`, which gives exactly N, checked here for 64 to
1,024 worlds). Drawing stays at up to 128 worlds at once.

### R10 — Rings and radii that change

Asked on 2026-10-08 by a Centipede session, for the user, after R9:
Centipede's new task ends an episode when the head's tip leaves a range
circle around the target, 2.5 times the head's distance at the episode's
start, and the user wants to see that circle in replays. Its radius differs
between worlds and changes at each new episode, which the format could not
say: a marker had one radius and was always a sphere.

**Builds:** in the format, `marker_radius` of shape `(T, K, M)` besides
`(M,)`, and an optional `marker_shapes`, each `"sphere"` or `"ring"`,
joining the marker group; format version 2 for a file that uses either,
version 1 otherwise, both read; the viewer drawing a ring flat around its
marker, for the highlighted world, at the current frame's radius, and
framing it.

**Done when:** tests show a file with a ring and a changing radius written
as version 2 and read back, other files still written as version 1,
inconsistent radii and shapes refused by key when read and when made, a ring
drawn flat at the radius of the frame for the highlighted world alone, and
framing taking it in; offscreen frames and the window show the ring.

**Result (2026-10-08):** built; the user's check pending. 106 tests pass (28
draw and skip without OpenGL). Frames of a 32-world stand-in recording with
a `target` sphere and a `range` ring, whose radius grew from 38 to 92 mm
where an episode restarted, showed the ring flat around the target in both
modes, smooth in Quality; in the window, 2 pixels read thin, so rings are 3
pixels wide at font scale 100. Copies of MujocoReplay from before this stage
refuse version-2 files ("format_version must be 1"), so every copy that
opens Centipede's new files, and the one on the GPU desktop that writes
them, needs a pull first.

### R11 — Tabs

Asked by the user on 2026-10-09, after R10: open several recordings at once,
each in a tab along the top of the window as an editor shows its files, and
switch between them with a click, so that a run's start and end of training
can be compared without opening the files again; no limit on their number
while none is needed.

**Builds:** in `ui.py`, `TabBar`, the tabs' layout, scrolling, and buttons,
and `tab_names`; in `playback.py`, `show_file`, `remove_file`, and
`run_on`, so that each tab of the window plays on its own while a video
still runs on from file to file; in `viewer.py`, opening files into tabs,
switching at the same frame, closing one or all, Ctrl+Tab and Ctrl+W, and
the scene drawn below the tabs; `ComposedScene.same_model`, so that the
camera stays for the files of one model.

**Done when:** tests show the tabs' layout, scrolling, and names, a switch
keeping the frame, the pause, and the camera, a file opened twice showing
its tab, closing tabs down to the empty world, a click on a tab and the tab
keys, a file whose model fails closed again, and each tab playing on its
own; the window, driven under a virtual display with stand-in recordings,
opens, switches, scrolls, and closes tabs.

**Result (2026-10-09):** built; the user's check pending. 121 tests pass (36
draw and skip without OpenGL). The window was driven under a virtual display
with stand-in recordings posed from Centipede's model: an early and a late
file of one run, the same cycle of a second run, and 24 files for many tabs.
The tabs read `cycle_0016 (runA)`, `cycle_0304`, and `cycle_0016 (runB)`; a
click and Ctrl+Tab switched files at the same frame, 43 of 128, paused, with
the camera kept; the second run's shorter file stopped at its end, and
Space replayed it; Ctrl+W, a tab's `x`, and Close all closed tabs down to
the empty world, the remaining `cycle_0016` dropping its folder once its
name was unique; and 24 tabs in a 1,000-pixel window scrolled with the
arrows and the wheel, P bringing the shown tab into view. Two choices were
made along the way, with the reasons in
[docs/design.md](docs/design.md#tabs): switching keeps the frame instead of
each tab keeping a place of its own, so that two files compare at the same
moment; and each tab plays on its own, since running on into the next tab,
as the playlist did, switched tabs by itself after a switch from a longer
file. Opening files no longer replaces what is open, and the panel's file
stepper gave way to the tabs and to Close all.

An independent review then drove the change with 300 random sequences of
actions, 4,000 random tab layouts, and real key presses, and found one crash
and six lesser defects, each reproduced and fixed: after `--ids`, a file
opened from the window whose model failed brought the last tab back still
drawing the worlds chosen by id, and the next B or double-click crashed; one
failure's message hid another's; Close all left the tabs scrolled; the same
file named twice on the command line opened two tabs; in a window under
about 400 pixels wide the arrows and `+` went past the edge, and a tab cut
to one pixel closed when clicked; three files of one name, such as the same
cycle under `e1/runA`, `e2/runA`, and `e1/runB`, were all named `c`; and a
command whose only file's model failed opened the empty world instead of
ending with status 1, as it did before the tabs. A narrowed window now also
brings the shown tab back into view. The review's random sequences then ran
clean on the fixed code, but for a 300-pixel window, whose 30-pixel strip
beside the panel holds no tab at all.

### R12 — An icon, Loop, and Play next

Asked by the user on 2026-10-09, after R11: an icon for the application,
chosen from what the tool is for; whether several files can be opened at
once; a loop that starts an episode again when it ends; and, since the user
watches a run's recordings one after the other to see the centipede
progress, a setting that goes on with the next tab by itself.

**Builds:** `icon.py`, the icon as pixel art, scaled by whole pixels and
written as a Windows icon file; the window's icon and, on Windows, its own name on the
taskbar; `Playback.loop`, and the window's `run_on` following Play next;
the settings `loop` and `play_next`, saved; the panel's Loop and Play next
and the `L` key; the file picker's title saying that it takes several
files; `tests/test_icon.py`.

**Done when:** tests show the icon drawn at every size and written as an
icon file of the same pixels, looping and running on in playback, the
panel's switches remembered and driving the tabs, and a shorter file pausing
at its end; the window, driven under a virtual display, carries the icon,
plays on into the next tab, and comes round from the last.

**Result (2026-10-09):** built; the user's check pending, on Windows in
particular for the taskbar and the shortcut's icon. 126 tests pass (37 draw
and skip without OpenGL). The first icon, a centipede drawn smooth with a
play button, gave way at the user's word to 8-bit pixel art about the tool
rather than one model: one figure in three worlds, two in ghost grey and
the best in orange, on a checkered floor, with a play sign; 32 by 32 pixels,
and a simpler 16 by 16 picture for the smallest sizes, each scaled by whole
pixels. Read back from the window on the virtual display (`_NET_WM_ICON`),
the icon was there at its sizes up to 64 pixels. Driven there with three stand-in files,
Play next took Space at the end of the second tab on into the third, and
with Loop, the third's end came round to the first, playing throughout. The
file picker already took several files at once (tkinter's
`askopenfilenames`), with Ctrl+click or Shift+click, which its title now
says. A switch that lands past a shorter file's end now pauses there, so
that neither switch leaves that file at once.

### R13 — The executable

Asked by the user on 2026-10-09, after R12, on a branch of its own
(`claude/executable`): an actual executable instead of only the program pip
makes, so that the tool runs without Python installed. R8 had left it for
later.

**Builds:** `packaging/MujocoReplay.spec`, PyInstaller's recipe for one
folder with the window program, without a console, carrying the icon (a
console program beside it was dropped at the user's word, below); the file
picker run as a hidden `--pick-files` mode of
the command, since a frozen program's `sys.executable` is the program
itself; `packaging/check.py`, which checks a built folder as a person would
use it; `.github/workflows/executable.yml`, which builds the folder on
GitHub's Windows machines, runs the tests, checks the build, and keeps the
zip; `tests/test_cli.py`.

**Done when:** the workflow passes on Windows: the tests, and the built
programs printing their help, writing a video whose frames show the test
recording's box, opening the window on it, showing the box, and quitting
when asked, and opening and cancelling the file picker; the zip holds no
Mesa; the README says where to get the zip and how to build it.

**Result (2026-10-09):** built; the user's check of the zip on a real
display pending. The first run of the workflow passed: 128 tests on Windows
(the one that needs Centipede's model skipped), drawing with Mesa's
software OpenGL, since GitHub's machines have no graphics card; the build
in 30 seconds, a folder of 1,093 files and 186 MB, zipped to 72 MB; and
every check: the help, a 90-frame video with the box in each frame, the
window titled `MujocoReplay - check` with about 29,000 pixels of the box in
its screenshot, closed by its close button and Y with status 0, and the
picker, cancelled with nothing chosen. In the container the same recipe
built a Linux folder first: its window, driven under the virtual display,
showed the tabs and the scene, the check's help and video passed on it, and
its picker mode, without tkinter there, ended with the one line of reason
that the window shows. The zip and the screenshots are the run's artifacts.

The user's first start of the zip met an empty terminal opening with the
window, and did not see the icon on the taskbar. The window program is
windowed (PyInstaller's `runw` bootloader, by the build log), so Windows
opens no console for it; but the console program beside it, then
`mujoco-replay.exe`, carried the same icon under a name that reads the same
with Windows' extensions hidden, and a double-click on it opens a console
window. It is now `MujocoReplay-console.exe`, without an icon. The check now
starts the programs from the extracted zip as a double-click does: the
window program opened no console window, the icon it carries and its
window's big and small icons matched the drawing pixel for pixel, and the
taskbar button showed the icon, in a screenshot read back from the run's
log; the console program, started the same way, opened the console window
the user saw, which shows that the check sees one. The executable also no
longer names itself on the taskbar (`MujocoReplay.Viewer`), which would
have kept its window apart from the program pinned there.

The user's second start found everything working, and asked what the
second program was for, since both opened the window. It was for a
terminal and for videos; the user chose one program, with the command line,
`render` included, coming from the Python package, a dependency the user
accepts for it. The zip now holds `MujocoReplay.exe` alone; the check runs
its help and its video with its output taken, and starts Python itself as
the console program that shows the check sees console windows. The README
was rewritten at the same time, shorter and with pictures, at the user's
request: the window and a replay of Centipede's model, from stand-in
recordings simulated from the model with an open-loop gait, since no real
run's recordings are in the container.

The user then found that the camera stayed where it was when another world
was highlighted, so that the centipede just picked could be out of the
picture. Highlighting a world, with B, Shift+B, the panel's stepper, or a
double-click, now brings the camera to it, keeping its distance and angle,
and keeps it following if it followed. A test holds it, and fails without
the change; in the window, driven under the virtual display with the early
stand-in zoomed in at its last frame, three presses of B centred the camera
on the worlds of ranks 2, 3, and 4 in turn.

### R14 — A camera that follows

Asked by the user on 2026-10-10: the camera stayed where the centipede
started, while Centipede's training branch now lets a world walk on after
reaching its target, to a new target placed from where its head is
(`[environment.target] after_arrival = "new_target"`), so that a centipede
travels many body lengths in one recording.

**Builds:** following on from the start, as the setting `follow`,
remembered; F and the panel's Follow switch it and save the choice, while
panning stops it for the moment only; framing, at a file's opening or with
V, and centring with C or a new highlight, leave it as it is; a video
follows each file's best world.

**Done when:** tests show following on by default and saved when switched,
panning not saved, framing and centring keeping it, and a video keeping in
view a world that leaves the first picture; in the window, a recording of
centipedes walking on from target to target keeps the best one in the
middle of the picture to its end.

**Result (2026-10-10):** built; the user's check pending. Centipede's
recordings keep no run in the repository, so a stand-in was posed from
Centipede's model as the training branch records such a run: 16 worlds of
600 frames walking from target to target, each world's target and range
ring moving on at every arrival, with `episode_start` marking the arrivals;
the worlds reached 2 to 8 targets and ended 12 to 42 cm from where they
started, the centipede being 3 cm long. Driven under the virtual display,
the window lost the best world before the change and kept it in the middle
of the picture, at the start, halfway, and at the end, after it. 134 tests
pass (41 draw and skip without OpenGL); the new ones fail without the
change. A world's centre is the centre of mass of all its replicated root
bodies, which for Centipede's model is the centipede alone; a test model
whose world is a robot and a door on a hinge keeps the camera between them.

## Status

| Stage | Status |
| --- | --- |
| R1 — Package and recording format | Complete (2026-10-07) |
| R2 — Scene composition | Complete (2026-10-07) |
| R3 — Renderer, playback, viewer | Built (2026-10-07); the user's window check pending |
| R4 — Video | Built (2026-10-07); the user's check in a video player pending |
| R5 — Documents and status | Complete (2026-10-07); Centipede's README and architecture document now point here |
| R6 — The application: panel, settings, and performance | Built (2026-10-07); the user's check on the MX330 pending |
| R7 — Full review | Done (2026-10-07); the user's check on the MX330, with Centipede's own recordings, pending |
| R8 — The program | Built (2026-10-07); the user's check on Windows pending |
| R9 — Any world on its own, and new defaults | Built (2026-10-08); the user's check pending |
| R10 — Rings and radii that change | Built (2026-10-08); the user's check pending |
| R11 — Tabs | Built (2026-10-09); the user's check pending |
| R12 — An icon, Loop, and Play next | Built (2026-10-09); the user's check pending |
| R13 — The executable | Built (2026-10-09); checked by the user on Windows, then reduced to one program; merged into `main` |
| R14 — A camera that follows | Built (2026-10-10); the user's check pending |
