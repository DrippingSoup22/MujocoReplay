"""Check a built folder of the programs the way a person would use them.

    python packaging/check.py FOLDER [--screenshots FOLDER]

``FOLDER`` is the folder PyInstaller built, ``dist/MujocoReplay``. The checks
run the programs on a recording of their own, an orange box in three worlds:

- ``mujoco-replay --help`` prints the command's help;
- ``mujoco-replay render`` writes a video, whose every frame shows the box;
- on Windows, ``MujocoReplay``, the program without a console, opens its
  window on the recording, shows the box, and quits when asked, as its close
  button and Y do; and its file picker opens and closes.

The screenshots of the window and the picker go into ``--screenshots``. Each
check prints what it saw, and the first that fails stops the run with why.
The windows need a desktop with OpenGL: GitHub's Windows machines have no
graphics card, so the workflow checks a copy of the folder with Mesa's
software OpenGL next to the programs, never the zip. Run this with a Python
that has the package and its video extra, which write the recording and read
the video.
"""

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from mujoco_replay.recording import Recording, write_recording

# The recording's model: an orange box over a grey floor.
MODEL = """
<mujoco model="check">
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.6 0.6 0.6 1"/>
    <body name="box">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.2" rgba="1 0.5 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
TITLE = "check"
EXE = ".exe" if sys.platform == "win32" else ""
WAIT_SECONDS = 180  # a first start on a new machine is slow
# The fewest of the box's pixels a video frame, and the window, must show.
FRAME_PIXELS, WINDOW_PIXELS = 100, 1000
# Window messages: the close button's, and a typed character's.
WM_CLOSE, WM_CHAR = 0x0010, 0x0102


class CheckFailed(Exception):
    """A check that failed, with why."""


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="the folder PyInstaller built")
    parser.add_argument("--screenshots", type=Path, default=Path("screenshots"))
    options = parser.parse_args(arguments)
    built = options.folder.resolve()  # the programs run in a folder of their own
    command = built / f"mujoco-replay{EXE}"
    window_program = built / f"MujocoReplay{EXE}"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        folder = Path(temporary)
        recording = folder / f"{TITLE}.npz"
        write_check_recording(recording)
        try:
            for program in (command, window_program):
                if not program.is_file():
                    raise CheckFailed(f"{program} is missing")
            check_help(command, folder)
            check_render(command, recording, folder)
            if sys.platform == "win32":
                shots = options.screenshots
                shots.mkdir(parents=True, exist_ok=True)
                desktop = Desktop()
                check_window(window_program, recording, folder, desktop, shots)
                check_picker(window_program, folder, desktop, shots)
            else:
                say("window and picker: not checked, as only Windows checks them")
        except CheckFailed as failure:
            print(f"check failed: {failure}", file=sys.stderr, flush=True)
            return 1
    say("every check passed")
    return 0


def write_check_recording(path: Path) -> None:
    """Three worlds of the box, side by side, turning; the middle one scores best."""
    frames, worlds = 30, 3
    frame = np.arange(frames)[:, None]
    world = np.arange(worlds)[None, :]
    angle = 0.1 * frame + world
    qpos = np.zeros((frames, worlds, 7))
    qpos[..., 0] = 0.7 * (world - 1)
    qpos[..., 2] = 0.2
    qpos[..., 3], qpos[..., 6] = np.cos(angle / 2), np.sin(angle / 2)
    score = np.array([1.0, 3.0, 2.0])
    write_recording(path, Recording(MODEL, 0.02, qpos, score=score, title=TITLE))


def check_help(command: Path, folder: Path) -> None:
    status, printed = run([command, "--help"], folder)
    if status or "render" not in printed:
        raise CheckFailed(f"mujoco-replay --help: status {status}\n{printed}")
    say("help: printed")


def check_render(command: Path, recording: Path, folder: Path) -> None:
    video = folder / "check.mp4"
    size = ["--width", "320", "--height", "240"]
    status, printed = run([command, "render", recording, "--out", video, *size], folder)
    written = re.search(r"(\d+) frames", printed)
    if status or not written:
        raise CheckFailed(f"mujoco-replay render: status {status}\n{printed}")
    reader = imageio.get_reader(video)
    try:
        shown = [box_pixels(frame) for frame in reader]
    finally:
        reader.close()
    if len(shown) != int(written[1]):
        raise CheckFailed(f"render: {len(shown)} frames in the video, not {written[1]}")
    if min(shown) < FRAME_PIXELS:
        frame = shown.index(min(shown))
        raise CheckFailed(f"render: frame {frame} shows {min(shown)} pixels of the box")
    say(f"render: {len(shown)} frames, each with {min(shown)}+ pixels of the box")


def check_window(
    program: Path, recording: Path, folder: Path, desktop, shots: Path
) -> None:
    title = f"MujocoReplay - {TITLE}"
    with Started([program, recording], folder, "window") as started:
        window = started.wait_for_window(desktop, lambda text: text == title, shots)
        time.sleep(5)  # its first frames drawn
        left, top, right, bottom = desktop.inside(window)
        picture = screenshot(shots / "window.png")
        shown = box_pixels(picture[top:bottom, left:right])
        if shown < WINDOW_PIXELS:
            raise CheckFailed(f"window: {shown} pixels of the box; see window.png")
        desktop.post(window, WM_CLOSE)  # the close button, which asks first
        desktop.post(window, WM_CHAR, ord("y"))
        status = started.wait()
    if status:
        raise CheckFailed(f"window: quit with status {status}\n{started.printed()}")
    say(f"window: opened as {title!r}, showed {shown} pixels of the box, and quit")


def check_picker(program: Path, folder: Path, desktop, shots: Path) -> None:
    chosen = folder / "chosen.txt"
    with Started([program, "--pick-files", chosen], folder, "picker") as started:
        window = started.wait_for_window(
            desktop, lambda text: text.startswith("Open recordings"), shots
        )
        time.sleep(2)  # its folder listed
        screenshot(shots / "picker.png")
        desktop.post(window, WM_CLOSE)  # as Cancel does
        status = started.wait()
    if status or not chosen.is_file() or chosen.read_text(encoding="utf-8"):
        raise CheckFailed(f"picker: closed with status {status}\n{started.printed()}")
    say("picker: opened, and closed with nothing chosen")


def box_pixels(image: np.ndarray) -> int:
    """How many pixels of an image show the box's orange."""
    red, green, blue = (image[..., channel].astype(int) for channel in range(3))
    orange = (red > 100) & (red - blue > 70) & (green > red / 4) & (green < red * 0.8)
    return int(orange.sum())


def run(arguments: list, folder: Path) -> tuple[int, str]:
    """Run a program to its end; its status, and what it printed."""
    try:
        done = subprocess.run(
            arguments,
            cwd=folder,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=WAIT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        name = f"{Path(arguments[0]).name} {arguments[1]}"
        raise CheckFailed(f"{name} did not end in {WAIT_SECONDS} s") from None
    return done.returncode, done.stdout + done.stderr


class Started:
    """A program running beside the checks, its output kept in files; it is
    stopped when the checks leave it, if it still runs."""

    def __init__(self, arguments: list, folder: Path, name: str) -> None:
        self.output = folder / f"{name}.txt"
        with self.output.open("wb") as output:
            self.process = subprocess.Popen(
                arguments, cwd=folder, stdout=output, stderr=subprocess.STDOUT
            )
        self.name = name

    def __enter__(self) -> "Started":
        return self

    def __exit__(self, *failure) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()

    def wait_for_window(self, desktop, wanted, shots: Path) -> int:
        """The first visible window of the program whose title ``wanted``
        accepts; without one in time, a screenshot of what there is instead."""
        deadline, seen = time.monotonic() + WAIT_SECONDS, []
        while time.monotonic() < deadline:
            seen = desktop.windows_of(self.process.pid)
            for window, title in seen:
                if wanted(title):
                    return window
            if self.process.poll() is not None:
                raise CheckFailed(
                    f"{self.name}: stopped with status {self.process.returncode}"
                    f"\n{self.printed()}"
                )
            time.sleep(0.5)
        screenshot(shots / f"{self.name}-timeout.png")
        titles = [title for _, title in seen]
        raise CheckFailed(
            f"{self.name}: no such window in {WAIT_SECONDS} s; its windows: {titles}"
            f"\n{self.printed()}"
        )

    def wait(self) -> int:
        """The program's status once it has ended, as it must soon."""
        try:
            return self.process.wait(60)
        except subprocess.TimeoutExpired:
            raise CheckFailed(f"{self.name}: still running after 60 s") from None

    def printed(self) -> str:
        return self.output.read_text(encoding="utf-8", errors="replace")


class Desktop:
    """The few calls of Windows' user32 that find, place, and close windows."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes, self._types = ctypes, wintypes
        user32 = ctypes.WinDLL("user32")
        self._visit = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        handle, pointer = wintypes.HWND, ctypes.POINTER
        user32.EnumWindows.argtypes = [self._visit, wintypes.LPARAM]
        user32.GetWindowThreadProcessId.argtypes = [handle, pointer(wintypes.DWORD)]
        user32.IsWindowVisible.argtypes = [handle]
        user32.GetWindowTextW.argtypes = [handle, wintypes.LPWSTR, ctypes.c_int]
        user32.GetClientRect.argtypes = [handle, pointer(wintypes.RECT)]
        user32.ClientToScreen.argtypes = [handle, pointer(wintypes.POINT)]
        user32.PostMessageW.argtypes = [
            handle,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.SetProcessDPIAware()  # window places in the screenshot's pixels
        self._user32 = user32

    def windows_of(self, pid: int) -> list[tuple[int, str]]:
        """The visible top-level windows of a process, with their titles."""
        ctypes, user32, found = self._ctypes, self._user32, []

        def visit(window, _):
            owner = self._types.DWORD()
            user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
            if owner.value == pid and user32.IsWindowVisible(window):
                title = ctypes.create_unicode_buffer(512)
                user32.GetWindowTextW(window, title, len(title))
                found.append((window, title.value))
            return True

        user32.EnumWindows(self._visit(visit), 0)
        return found

    def inside(self, window: int) -> tuple[int, int, int, int]:
        """Where a window's inside is on the screen: left, top, right, bottom."""
        ctypes, user32 = self._ctypes, self._user32
        size, corner = self._types.RECT(), self._types.POINT(0, 0)
        user32.GetClientRect(window, ctypes.byref(size))
        user32.ClientToScreen(window, ctypes.byref(corner))
        return corner.x, corner.y, corner.x + size.right, corner.y + size.bottom

    def post(self, window: int, message: int, value: int = 0) -> None:
        self._user32.PostMessageW(window, message, value, 0)


def screenshot(path: Path) -> np.ndarray:
    """The screen, saved to ``path``, as RGB pixels."""
    from PIL import ImageGrab  # Pillow comes with imageio

    picture = ImageGrab.grab().convert("RGB")
    picture.save(path)
    return np.asarray(picture)


def say(text: str) -> None:
    print(text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
