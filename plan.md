# Development plan

This plan builds MujocoReplay in seven stages; the user asked for the sixth on
2026-10-07, once the first five were built, and for the seventh, a full
review, once the sixth was. It is paired with Stage 7 of the
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
- The assistant commits and pushes on its own, under the user's identity,
  with no AI attribution, and keeps branches short; `AGENTS.md` holds the
  rules the user set on 2026-10-07.

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
did not say how to set up on Windows or which Python versions work.

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
