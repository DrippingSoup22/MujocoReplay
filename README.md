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

**Stage R6 built (2026-10-07):** the tool is an application. It opens on an
empty world, takes recordings from a file picker, from files dropped onto the
window, or from the command line, and offers a side panel with its own
settings: how many worlds to draw (1, 2, 4, … up to 128, the best of as many
score bands), a Quality and a Performance mode with five switches, a
frame-rate readout, and a cache of composed scenes. Settings are remembered
between runs. `mujoco-replay render` writes a video. Everything was verified
in a container without a GPU, on offscreen frames, with the window driven by
synthetic clicks and keys, and with a stand-in recording made from
Centipede's own model; it awaits the user's check on a real display, on a
weak graphics card (a GeForce MX330), with Centipede's own recordings.

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

The tool shares the Python environment of the sibling projects. From this
folder, in PowerShell:

```powershell
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pip install -e ".[video,dev]"
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pytest
```

`video` adds the packages that write MP4 files; `dev` adds the test tools. The
viewer itself needs only MuJoCo and NumPy. Use the native Windows environment
for the window: OpenGL through WSL is unreliable on this machine.

The tests that draw need OpenGL; without it they skip and say why. On a Linux
machine without a display, `xvfb-run -a python -m pytest` runs them on a
virtual display. MuJoCo can also draw there through EGL or OSMesa
(`MUJOCO_GL=egl`), which the video's offscreen drawing supports through
MuJoCo's own context helper; that path has not been tried yet.

## Use

```powershell
mujoco-replay
mujoco-replay RECORDING.npz [RECORDING2.npz ...]
mujoco-replay RECORDING.npz --mode performance --worlds 16
mujoco-replay render RECORDING.npz --out replay.mp4
```

Without files the window opens empty: open recordings with the panel's Open
button (or `O`), or drop `.npz` files onto it. Several files play one after
the other. The panel on the left sets everything else: playback, how many
worlds are drawn, the highlighted world, ghosts and markers, the graphics
(Quality, Performance, or single switches), and the display; Tab hides it.
On a weak graphics card, start in Performance mode with 16 worlds, turn on the
frame-rate readout, and change one switch at a time: the readout says how
long a frame takes to draw. Space plays and pauses, the arrows step and
change speed, B moves the highlight, and the mouse moves the camera; every
key, option, and switch is described in [`docs/design.md`](docs/design.md).
