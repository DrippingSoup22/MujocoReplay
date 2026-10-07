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

**Stage R1 complete (2026-10-07):** the package, the recording file format,
and the rank rule exist and are tested; Centipede writes recordings with them.
The scene, the viewer, and the video (stages R2 to R5 of the plan) are not
built yet: the command only parses its arguments.

## Documentation

| Document | Describes |
| --- | --- |
| [`plan.md`](plan.md) | The stages that build the tool, each with its checks |
| [`docs/recording-format.md`](docs/recording-format.md) | The recording file: every key, its shape and meaning, and the writer and reader |
| [`docs/design.md`](docs/design.md) | How the tool is built: modules, scene composition, world selection, playback, overlay, camera, video |
| [`AGENTS.md`](AGENTS.md) | Working rules for coding assistants |

## Project folders

```text
MujocoReplay/
├─ docs/                Format and design documents
├─ src/mujoco_replay/   The package: recording format, selection, scene, renderer, viewer, video
├─ tests/               Automated tests, one file per module
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

## Use

```powershell
mujoco-replay RECORDING.npz [RECORDING2.npz ...]
mujoco-replay render RECORDING.npz --out replay.mp4
```

Several files play one after the other. The keys and options are listed in
[`docs/design.md`](docs/design.md#playback-and-keys).
