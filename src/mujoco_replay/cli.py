"""The ``mujoco-replay`` command: ``view`` (the default) and ``render``.

Only the arguments exist so far; viewing and rendering are built in the later
stages of plan.md.
"""

import argparse
import sys


def main(arguments: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand."""
    parser = build_parser()
    raw = list(sys.argv[1:] if arguments is None else arguments)
    if raw and raw[0] not in ("view", "render") and not raw[0].startswith("-"):
        raw.insert(0, "view")
    options = parser.parse_args(raw)
    print(
        f"mujoco-replay {options.command}: not built yet; see plan.md. "
        f"Files: {', '.join(options.files)}"
    )
    return 2


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
        subcommand.add_argument("--levels", type=int, default=4)
        subcommand.add_argument("--per-level", type=int, default=8)
        subcommand.add_argument("--all", action="store_true", help="draw every world")
        subcommand.add_argument("--worlds", help="producer world ids, comma-separated")
        subcommand.add_argument(
            "--speed", type=float, default=0.3, help="seconds per recorded frame"
        )
        subcommand.add_argument("--no-hud", action="store_true")
        subcommand.add_argument("--width", type=int)
        subcommand.add_argument("--height", type=int)
    render.add_argument("--out", required=True, help="the MP4 file to write")
    render.add_argument("--fps", type=int, default=30)
    return parser
