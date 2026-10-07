# MujocoReplay

A small tool that replays recorded poses of any MuJoCo model: many worlds of
the same model drawn on top of each other in one scene, with the best world in
its natural colours and the others as faint grey ghosts. It is made for
watching what reinforcement-learning agents do during training without slowing
the training down: the training only records positions, and this tool draws
them later, as slowly as a person needs.

It does not simulate anything. It reads a self-contained recording file, builds
one scene with one copy of the moving bodies per recorded world, sets every
copy's joints from the recorded positions frame by frame, and draws. The camera
is free, playback can be paused and slowed to a few seconds per step, and an
overlay shows the facts of the run. The same scene can be rendered to a video.

## Status

**Stage R7, a full review, done (2026-10-07):** five independent reviews and
a visual one went through the whole application, and their findings are
fixed: damaged and half-written files, unusual models, a diverged world, the
cache, the file picker, the shadows and reflections of Quality mode, the
panel at a laptop's display scaling, and many smaller points of interaction.
A few settings were added where a finding asked for them: the ghosts'
strength, Reset and Top views, double-click to highlight a world, clicks on
the timeline, and the list of keys on F1. The tool is an application: it
opens on an empty world, takes recordings from a file picker, from files
dropped onto it, or from the command line, and offers a side panel with its
own settings, remembered between runs; `mujoco-replay render` writes a video.
Everything was verified in a container without a GPU, on offscreen frames and
with the window driven by synthetic input; it awaits the user's check on a
real display, on a weak graphics card (a GeForce MX330), with Centipede's own
recordings.

## Documentation

| Document | Describes |
| --- | --- |
| [`plan.md`](plan.md) | The stages that build the tool, each with its checks |
| [`docs/recording-format.md`](docs/recording-format.md) | The recording file: every key, its shape and meaning, and the writer and reader |
| [`docs/design.md`](docs/design.md) | How the tool is built: modules, scene composition, world selection, playback, overlay, camera, graphics settings, the application and its panel, the cache, video |
| [`AGENTS.md`](AGENTS.md) | Working rules for coding assistants |

## Project folders

```text
MujocoReplay/
├─ docs/                Format and design documents
├─ src/mujoco_replay/   The package: recording format, selection, scene, playback, renderer, settings, panel, viewer, video
├─ tests/               Automated tests, one file per module; shared helpers in conftest.py
└─ archive/             Superseded material; local only
```

## Setup

Python 3.11 to 3.14 (MuJoCo 3.12.0, which this tool and Centipede pin, has no
wheels for newer versions). The tool can share the Python environment of the
sibling projects; from this folder, in PowerShell:

```powershell
& "$HOME\.venvs\Centipede\Scripts\Activate.ps1"
python -m pip install -e ".[video,dev]"
python -m pytest
```

or have an environment of its own (`py -3.12 -m venv .venv`, then
`.venv\Scripts\Activate.ps1` and the same `pip install`). `video` adds the
packages that write MP4 files; `dev` adds the test tools. The viewer itself
needs only MuJoCo, NumPy, and GLFW. Use the native Windows environment for the
window: OpenGL through WSL is unreliable on this machine.

On a laptop with two graphics chips, Windows may run Python on the
processor's own chip instead of the GeForce. To use the GeForce, open
Settings, System, Display, Graphics, add the base Python's `python.exe` (an
environment's `python.exe` starts it), and choose High performance; the
frame-rate readout shows the difference.

The tests that draw need OpenGL; without it they skip and say why. On a Linux
machine without a display, `xvfb-run -a python -m pytest` runs them on a
virtual display. MuJoCo can also draw there through EGL or OSMesa
(`MUJOCO_GL=egl`), which the video's offscreen drawing supports through
MuJoCo's own context helper; that path has not been tried yet.

## Use

With the environment active:

```powershell
mujoco-replay
mujoco-replay RECORDING.npz [RECORDING2.npz ...]
mujoco-replay runs\RUN\recordings\*.npz --mode performance --worlds 16
mujoco-replay render RECORDING.npz --out replay.mp4
```

(`python -m mujoco_replay` is the same command.) Without files the window
opens empty: open recordings with the panel's Open button (or `O`), or drop
`.npz` files onto it; several files play one after the other, in the order of
their names. The panel on the left sets everything else: playback, how many
worlds are drawn and how strongly the ghosts show, the highlighted world, the
view, the graphics (Quality, Performance, or single switches), and the
options; Tab hides it. On a weak graphics card, start in Performance mode
with 16 worlds, turn on the frame-rate readout, and change one switch at a
time: the readout says how long a frame takes to draw. Space plays and
pauses, the arrows step and change speed, a double-click highlights a world,
a click on the timeline jumps there, V resets the view, and F1 lists every
key; [`docs/design.md`](docs/design.md) describes every key, option, and
switch.
