"""The ``MujocoReplay`` program: the viewer, started without a console window.

pip makes the program from the ``gui-scripts`` entry of ``pyproject.toml``; on
Windows it is ``MujocoReplay.exe`` in the environment's ``Scripts`` folder,
which runs ``pythonw``, Python without a console window. The program runs the
``mujoco-replay`` command with what it is given (nothing, or the files dropped
onto it), so that it opens the same window and follows every change to the
command and the viewer with no code of its own. Without a console, what the
command prints when it fails would go nowhere, so the program keeps it and
shows it in a message box. With a console, as on Linux or when started from a
terminal, it is the command itself.
"""

import io
import sys
import traceback

TITLE = "MujocoReplay"
SHOWN_LINES = 40  # the end of a longer report, so that the message box fits
# MessageBoxW's flags: the error icon, and in front of the other windows.
MB_ICONERROR, MB_SETFOREGROUND, MB_TOPMOST = 0x10, 0x10000, 0x40000


def main() -> int:
    """Run the command; without a console, show why it failed, if it did.

    The file picker's process, which the window starts by running the
    program again, reports to the window instead of in a message box.
    """
    picker = sys.argv[1:2] == ["--pick-files"]  # cli.PICK_FILES, before cli loads
    if sys.stderr is not None or picker:
        from mujoco_replay import cli

        return cli.main()
    sys.stderr = report = io.StringIO()
    try:
        status = _run()
    finally:
        sys.stderr = None
    if status:
        lines = report.getvalue().strip().splitlines()[-SHOWN_LINES:]
        _show("\n".join(lines) or f"MujocoReplay stopped with status {status}.")
    return status


def _run() -> int:
    """The command's exit status, with what went wrong on the error stream."""
    try:
        from mujoco_replay import cli

        return cli.main()
    except SystemExit as stop:  # argparse's errors, or an exit with a message
        if stop.code is None or isinstance(stop.code, int):
            return stop.code or 0
        print(stop.code, file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1


def _show(text: str) -> None:
    """Show ``text`` in a message box, which Windows offers without a library."""
    if sys.platform == "win32":
        import ctypes

        flags = MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST
        ctypes.windll.user32.MessageBoxW(None, text, TITLE, flags)


if __name__ == "__main__":
    sys.exit(main())
