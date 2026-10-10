"""The ``mujoco-replay`` command: ``view`` (the default) and ``render``.

Both read the recordings given, or MuJoCo model files read as recordings, and
check them before opening anything; ``view`` also starts without any, on the
empty world. The window and the video modules, and MuJoCo, are imported only
when they run, so that the command starts quickly. The saved settings supply
what the command line leaves out; ``--mode`` and ``--worlds`` given to
``view`` are remembered, as the panel's changes are.
"""

import argparse
import glob
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from mujoco_replay.playback import DEFAULT_SECONDS_PER_FRAME
from mujoco_replay.recording import (
    Recording,
    RecordingError,
    in_name_order,
)
from mujoco_replay.selection import MAX_WORLDS, choose_worlds
from mujoco_replay.settings import PRESETS, load_settings, save_settings, user_folder

# The window shows the system's file picker by running the command again, in a
# process of its own, with this and a file to write the chosen paths into.
PICK_FILES = "--pick-files"


def main(arguments: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand."""
    raw = list(sys.argv[1:] if arguments is None else arguments)
    if len(raw) == 2 and raw[0] == PICK_FILES:
        return pick_files(raw[1])
    parser = build_parser()
    if not raw or raw[0] not in ("view", "render", "-h", "--help"):
        raw.insert(0, "view")
    options = parser.parse_args(raw)
    options.files = _expand(options.files)
    if options.command == "render" and not Path(options.out).parent.is_dir():
        parser.error(f"--out: the folder {Path(options.out).parent} does not exist")
    settings = load_settings()
    if options.command == "view" and (options.mode or options.worlds):
        if options.mode:
            settings = settings.with_mode(options.mode)
        if options.worlds:
            settings = replace(settings, worlds=options.worlds)
        save_settings(settings)  # remembered, as changes in the panel are
    count = options.worlds or settings.worlds
    size = (options.width or 1280, options.height or 720)
    if options.command == "view" and not (options.width or options.height):
        size = None  # the window takes most of the screen
    try:
        from mujoco_replay.scene import read_file  # MuJoCo reads model files

        recordings = [read_file(path) for path in options.files]
        worlds = [
            drawn_worlds(recording, path, count, options.ids, parser)
            for recording, path in zip(recordings, options.files, strict=True)
        ]
        if options.command == "view":
            from mujoco_replay import viewer

            viewer.run(
                recordings,
                settings,
                options.speed,
                options.ids,
                size,
                hud=not options.no_hud,
                paths=options.files,
            )
            return 0
        try:
            from mujoco_replay import video
        except ImportError as error:
            print(
                f"mujoco-replay render needs the video extra ({error}); "
                'install it with: pip install -e ".[video]"',
                file=sys.stderr,
            )
            return 1
        frames = video.export(
            recordings,
            worlds,
            options.out,
            options.speed,
            options.fps,
            size,
            hud=not options.no_hud,
            progress=_report_progress,
            cache=user_folder("cache") if settings.cache else None,
            graphics=PRESETS[options.mode or "quality"],
        )
        print(f"wrote {options.out}: {frames} frames, {frames / options.fps:.1f} s")
        return 0
    except (RecordingError, RuntimeError, OSError) as error:  # OSError: --out
        print(f"mujoco-replay: {error}", file=sys.stderr)
        return 1


def pick_files(output: str) -> int:
    """Show the system's file picker, tkinter's, and write the chosen paths
    into ``output``, one per line in UTF-8.

    The window runs this in a process of its own, which keeps tkinter's event
    loop apart from GLFW's. A picker that cannot open, as without tkinter,
    puts one line on the error stream, which the window reads as UTF-8 and
    shows.
    """
    if sys.stderr is not None:
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        paths = filedialog.askopenfilenames(
            parent=root,
            title="Open recordings or models (Ctrl+click or Shift+click picks several)",
            filetypes=[
                ("Recordings and models", "*.npz *.xml"),
                ("Recordings", "*.npz"),
                ("MuJoCo models", "*.xml"),
                ("All files", "*.*"),
            ],
        )
    except Exception as error:  # the window says why, from the error stream
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    Path(output).write_text("\n".join(paths), encoding="utf-8")
    return 0


def drawn_worlds(
    recording: Recording,
    path: str,
    count: int,
    ids: list[int] | None,
    parser: argparse.ArgumentParser,
) -> np.ndarray:
    """The indices of the worlds to draw from one file, best first."""
    worlds = choose_worlds(recording, count, ids)
    if not len(worlds):
        listed = ",".join(str(world_id) for world_id in ids)
        parser.error(f"--ids {listed}: none of these worlds is in {path}")
    return worlds


def build_parser() -> argparse.ArgumentParser:
    """The command line of both subcommands, as docs/design.md lists it."""
    parser = argparse.ArgumentParser(
        prog="mujoco-replay",
        description="Replay recorded poses of many worlds of one MuJoCo model.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    view = subcommands.add_parser("view", help="open the interactive window")
    render = subcommands.add_parser("render", help="write a video")
    files = "recordings (.npz) or MuJoCo model files (.xml), in order"
    view.add_argument("files", nargs="*", help=files)
    render.add_argument("files", nargs="+", help=files)
    for subcommand in (view, render):
        which = subcommand.add_mutually_exclusive_group()
        which.add_argument(
            "--worlds",
            type=_between(1, MAX_WORLDS),
            help="how many worlds to draw: the best of as many rank bands",
        )
        which.add_argument(
            "--ids", type=_world_ids, help="draw these producer world ids instead"
        )
        subcommand.add_argument(
            "--speed",
            type=_positive(float),
            default=DEFAULT_SECONDS_PER_FRAME,
            help="seconds per recorded frame",
        )
        subcommand.add_argument("--no-hud", action="store_true")
        subcommand.add_argument("--width", type=_between(16), help="in pixels")
        subcommand.add_argument("--height", type=_between(16), help="in pixels")
    view.add_argument(
        "--mode",
        choices=("quality", "performance"),
        help="the graphics preset, remembered for the next runs",
    )
    render.add_argument(
        "--mode",
        choices=("quality", "performance"),
        help="the graphics preset; quality unless given",
    )
    render.add_argument("--out", required=True, help="the MP4 file to write")
    render.add_argument("--fps", type=_positive(int), default=30)
    return parser


def _expand(paths: list[str]) -> list[str]:
    """The files with wildcards expanded, as Windows' shells leave them.

    A pattern's files come in the order of their names; a pattern that
    matches nothing stays as it is, to be reported as a missing file.
    """
    expanded = []
    for path in paths:
        matches = glob.glob(path) if not Path(path).exists() else []
        expanded += in_name_order(matches) if matches else [path]
    return expanded


def _report_progress(done: int, total: int) -> None:
    """One line on the error stream, rewritten in place, ended when done."""
    if done == total or done % 30 == 0:
        end = "\n" if done == total else ""
        print(f"\rvideo frames: {done} / {total}", end=end, file=sys.stderr, flush=True)


def _positive(kind: type):
    """An argument type: a number of ``kind`` above zero."""

    def parse(text: str):
        value = kind(text)
        if not value > 0:
            raise argparse.ArgumentTypeError(f"{text} is not above zero")
        return value

    parse.__name__ = kind.__name__  # argparse names the type in its messages
    return parse


def _between(low: int, high: int | None = None):
    """An argument type: a whole number from ``low`` to ``high``, both included."""

    def parse(text: str) -> int:
        value = int(text)
        if value < low or (high is not None and value > high):
            allowed = f"from {low} to {high}" if high else f"{low} or more"
            raise argparse.ArgumentTypeError(f"{text} is not {allowed}")
        return value

    parse.__name__ = "int"
    return parse


def _world_ids(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of world ids"
        ) from None
