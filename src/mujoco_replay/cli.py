"""The ``mujoco-replay`` command: ``view`` (the default) and ``render``.

Both read the recordings, choose the worlds to draw in each, and hand them on;
the window and the video modules are imported only when they run, so that the
command starts quickly and reports a bad file before opening anything.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from mujoco_replay.recording import Recording, RecordingError, read_recording
from mujoco_replay.selection import choose_worlds


def main(arguments: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand."""
    parser = build_parser()
    raw = list(sys.argv[1:] if arguments is None else arguments)
    if not raw or raw[0] not in ("view", "render", "-h", "--help"):
        raw.insert(0, "view")
    options = parser.parse_args(raw)
    if options.command == "render" and not Path(options.out).parent.is_dir():
        parser.error(f"--out: the folder {Path(options.out).parent} does not exist")
    try:
        recordings = [read_recording(path) for path in options.files]
        worlds = [
            drawn_worlds(recording, path, options, parser)
            for recording, path in zip(recordings, options.files, strict=True)
        ]
        size = (options.width or 1280, options.height or 720)
        if options.command == "view":
            from mujoco_replay import viewer

            viewer.run(
                recordings, worlds, options.speed, hud=not options.no_hud, size=size
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
        )
        print(f"wrote {options.out}: {frames} frames, {frames / options.fps:.1f} s")
        return 0
    except (RecordingError, RuntimeError) as error:
        print(f"mujoco-replay: {error}", file=sys.stderr)
        return 1


def drawn_worlds(
    recording: Recording,
    path: str,
    options: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> np.ndarray:
    """The indices of the worlds to draw from one file, best first."""
    worlds = choose_worlds(
        recording, options.levels, options.per_level, options.worlds, options.all
    )
    if not len(worlds):
        ids = ",".join(str(world_id) for world_id in options.worlds)
        parser.error(f"--worlds {ids}: none of these worlds is in {path}")
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
    for subcommand in (view, render):
        subcommand.add_argument("files", nargs="+", help="recording files, in order")
        subcommand.add_argument("--levels", type=_positive(int), default=4)
        subcommand.add_argument("--per-level", type=_positive(int), default=8)
        which = subcommand.add_mutually_exclusive_group()
        which.add_argument("--all", action="store_true", help="draw every world")
        which.add_argument(
            "--worlds", type=_world_ids, help="producer world ids, comma-separated"
        )
        subcommand.add_argument(
            "--speed",
            type=_positive(float),
            default=0.3,
            help="seconds per recorded frame",
        )
        subcommand.add_argument("--no-hud", action="store_true")
        subcommand.add_argument("--width", type=_at_least(16), help="in pixels")
        subcommand.add_argument("--height", type=_at_least(16), help="in pixels")
    render.add_argument("--out", required=True, help="the MP4 file to write")
    render.add_argument("--fps", type=_positive(int), default=30)
    return parser


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


def _at_least(minimum: int):
    """An argument type: a whole number of at least ``minimum``."""

    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{text} is below {minimum}")
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
