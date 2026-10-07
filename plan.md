# Development plan

This plan builds MujocoReplay in five stages. It is paired with Stage 7 of the
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
- Commits and pushes only when the user asks, under the user's identity, with
  no AI attribution. The user creates the remote repository; the assistant
  initialises Git locally when asked.

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

## Status

| Stage | Status |
| --- | --- |
| R1 — Package and recording format | Complete (2026-10-07) |
| R2 — Scene composition | Complete (2026-10-07) |
| R3 — Renderer, playback, viewer | Built (2026-10-07); the user's window check pending |
| R4 — Video | Built (2026-10-07); the user's check in a video player pending |
| R5 — Documents and status | This project's documents done (2026-10-07); Centipede's links and the user's acceptance pending |
