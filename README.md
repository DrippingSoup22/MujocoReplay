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

**Stage R11, tabs, built (2026-10-09):** recordings open in tabs along the
top of the window, one per file, as an editor shows its files, so that the
start and the end of a run's training can be compared without opening them
again. Opening more files adds tabs instead of replacing what is open; a
click on a tab, or N, P, and Ctrl+Tab, shows its file at the same frame, with
the highlighted world kept, and the camera too for files of one model, so
that two files compare moment by moment; a tab's `x`, Ctrl+W, or the panel's
Close all closes tabs. Each
tab plays on its own: the end of a file no longer runs on into the next,
which only a video's playlist still does. There is no limit on the number of
tabs; each open file stays in memory at about its size on disk.

**Stage R10, rings and radii that change, built (2026-10-08):** a marker
can be drawn as a ring, a circle lying flat around it, and its radius can
change from frame to frame and differ between worlds. Centipede asked for
it so that the user sees each episode's range around the target: the
highlighted world's ring shows when its head leaves the range. Files that
use either are format version 2, which copies of this tool from before
refuse; pull every copy, including the GPU desktop's, before Centipede
writes them.

**Stage R9, any world on its own and new defaults, built (2026-10-08):**
the panel's Highlight stepper and `B` step through every world of the file,
not only the drawn ones; a world the count does not draw takes the place of
the best world of its band, so that with one world drawn each world's run
can be watched alone. The ghosts are faint, playback runs at 0.1 s per frame,
and both graphics presets draw at full resolution by default; settings saved
before keep their other values. Centipede's recordings held 32 worlds, 4
levels of 8; at the user's request it now records every world by default
(`record_worlds` in its `[run]` section), so a run's files hold all its
worlds, while earlier runs keep 32 per file. This tool draws up to 128 of
the worlds a file holds.

**Stage R8, a program to start it from, built (2026-10-07):** the install
also makes `MujocoReplay`, which opens the window without a console window,
to start from the Start menu, the taskbar, or the desktop (see Use). It runs
the `mujoco-replay` command, unchanged, and says in a message box why it
cannot start, if it cannot. `MujocoReplay.exe` awaits the user's check on
Windows.

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
| [`docs/design.md`](docs/design.md) | How the tool is built: modules, scene composition, world selection, playback, the tabs, overlay, camera, graphics settings, the application and its panel, the cache, video |
| [`AGENTS.md`](AGENTS.md) | Working rules for coding assistants |

## Project folders

```text
MujocoReplay/
├─ docs/                Format and design documents
├─ src/mujoco_replay/   The package: recording format, selection, scene, playback, renderer, settings, panel, viewer, video, command, program
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
Settings, System, Display, Graphics, add the base Python's `python.exe`, and
for the program (see Use) its `pythonw.exe` from the same folder, and choose
High performance for each: an environment's own `python.exe` and
`pythonw.exe` only start these. This prints where the base `python.exe` is:

```powershell
python -c "import sys; print(sys._base_executable)"
```

The frame-rate readout shows the difference.

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
opens empty: open recordings with the panel's Open button (or `O`, or the `+`
after the tabs), or drop `.npz` files onto it. Each file opens in a tab of
its own along the top, in the order of their names, and opening more adds
tabs: a click on a tab, or N, P, and Ctrl+Tab, shows its file at the same
frame, so that the start and the end of a run compare moment by moment, and
a tab's `x`, Ctrl+W, or Close all closes them. The panel on the left sets
everything else: playback, how many
worlds are drawn and how strongly the ghosts show, the highlighted world, the
view, the graphics (Quality, Performance, or single switches), and the
options; Tab hides it. On a weak graphics card, start in Performance mode
with 16 worlds, turn on the frame-rate readout, and change one switch at a
time: the readout says how long a frame takes to draw. Space plays and
pauses, the arrows step and change speed, a double-click highlights a world,
a click on the timeline jumps there, V resets the view, F1 lists every key,
and Q or Esc asks before quitting; [`docs/design.md`](docs/design.md) describes every key, option, and
switch.

The install also makes `MujocoReplay`, a program that opens the same window
without a console window. pip puts it in the environment, not in this
folder: on Windows, `MujocoReplay.exe` in the environment's `Scripts` folder
(`$HOME\.venvs\Centipede\Scripts` for the shared environment above). An
environment installed before the program existed needs the `pip install`
above once more, with the window closed. Then, with the environment active,
`MujocoReplay` starts it, and this puts a shortcut to it on the desktop:

```powershell
$desktop = [Environment]::GetFolderPath("Desktop")
$link = (New-Object -ComObject WScript.Shell).CreateShortcut("$desktop\MujocoReplay.lnk")
$link.TargetPath = (Get-Command MujocoReplay).Path
$link.WorkingDirectory = Split-Path $link.TargetPath
$link.Save()
```

Right-click the shortcut to pin it to Start or the taskbar. The program
opens the empty world, or the files dropped onto it or onto its shortcut,
with the same remembered settings. It runs this folder's code, so a pull
needs no new install; only a change to `pyproject.toml` needs the
`pip install` again. When it cannot start, a message box says why; if
nothing appears at all, run `mujoco-replay` in the terminal, which prints
the reason.
