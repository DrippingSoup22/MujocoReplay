"""The ``mujoco-replay`` command: ``view`` (the default) and ``render``.

Both read the recordings, choose the worlds to draw in each, and hand them on;
the window and the video modules are imported only when they run, so that the
command starts quickly and reports a bad file before opening anything.
"""

import argparse
import sys

import numpy as np

from mujoco_replay.recording import Recording, RecordingError, read_recording
from mujoco_replay.selection import choose_worlds


def main(arguments: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand."""
    parser = build_parser()
    raw = list(sys.argv[1:] if arguments is None else arguments)
    if raw and raw[0] not in ("view", "render") and not raw[0].startswith("-"):
        raw.insert(0, "view")
    options = parser.parse_args(raw)
    try:
        recordings = [read_recording(path) for path in options.files]
        worlds = [
            drawn_worlds(recording, path, options, parser)
            for recording, path in zip(recordings, options.files, strict=True)
        ]
        if options.command == "view":
            from mujoco_replay import viewer

            viewer.run(
                recordings,
                worlds,
                options.speed,
                hud=not options.no_hud,
                size=(options.width or 1280, options.height or 720),
            )
            return 0
        print("mujoco-replay render: not built yet; see plan.md (stage R4)")
        return 2
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
        subcommand.add_argument("--width", type=_positive(int))
        subcommand.add_argument("--height", type=_positive(int))
    render.add_argument("--out", required=True, help="the MP4 file to write")
    render.add_argument("--fps", type=_positive(int), default=30)
    return parser


def _positive(kind: type):
    """An argument type: a number of ``kind`` above zero."""

    def parse(text: str):
        value = kind(text)
        if not value > 0:
            raise argparse.ArgumentTypeError(f"{text} is not above zero")
        return value

    parse.__name__ = kind.__name__  # argparse names the type in its messages
    return parse


def _world_ids(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of world ids"
        ) from None
