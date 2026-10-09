"""Check a built folder of the programs the way a person would use them.

    python packaging/check.py FOLDER [--screenshots FOLDER] [--print-screenshots]

``FOLDER`` is the folder PyInstaller built, ``dist/MujocoReplay``. The checks
run the programs on a recording of their own, an orange box in three worlds:

- ``MujocoReplay-console --help`` prints the command's help;
- ``MujocoReplay-console render`` writes a video, whose every frame shows the
  box;
- on Windows, ``MujocoReplay`` is a windowed program and carries the icon,
  while ``MujocoReplay-console`` is a console program; ``MujocoReplay``,
  started as a double-click starts it, opens its window on the recording with
  no console window beside it, carries the icon, shows the box, and quits
  when asked, as its close button and Y do; ``MujocoReplay-console``, started
  the same way, does open a console window, which shows that the check for
  one sees it; and the file picker opens and closes.

The screenshots of the windows, the taskbar, and the picker go into
``--screenshots``; ``--print-screenshots`` also prints the windows' and the
taskbar's in the output, as base64 text, for a reader who cannot download
them. Each check prints what it saw, and the first that fails stops the run
with why. The windows need a desktop with OpenGL: GitHub's Windows machines
have no graphics card, so the workflow checks a copy of the folder with
Mesa's software OpenGL next to the programs, never the zip. Run this with a
Python that has the package and its video extra, which write the recording
and read the video.
"""

import argparse
import base64
import io
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from mujoco_replay import icon
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
WINDOW_TITLE = f"MujocoReplay - {TITLE}"
EXE = ".exe" if sys.platform == "win32" else ""
WAIT_SECONDS = 180  # a first start on a new machine is slow
# The fewest of the box's pixels a video frame, and the window, must show.
FRAME_PIXELS, WINDOW_PIXELS = 100, 1000
# Windows' subsystems: a program that opens windows, and one that needs a console.
WINDOWED, CONSOLE = 2, 3
# The window classes of a console window: the console host's, Windows Terminal's.
CONSOLES = ("ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS")
# Window messages: the close button's, and a typed character's.
WM_CLOSE, WM_CHAR = 0x0010, 0x0102


class CheckFailed(Exception):
    """A check that failed, with why."""


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="the folder PyInstaller built")
    parser.add_argument("--screenshots", type=Path, default=Path("screenshots"))
    parser.add_argument("--print-screenshots", action="store_true")
    options = parser.parse_args(arguments)
    built = options.folder.resolve()  # the programs run in a folder of their own
    command = built / f"MujocoReplay-console{EXE}"
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
                shots = Shots(options.screenshots, options.print_screenshots)
                desktop = Desktop()
                check_programs(window_program, command, desktop)
                check_window(window_program, recording, folder, desktop, shots)
                check_console_program(command, recording, folder, desktop, shots)
                check_picker(window_program, folder, desktop, shots)
            else:
                say("windows and picker: not checked, as only Windows checks them")
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
        raise CheckFailed(f"{command.name} --help: status {status}\n{printed}")
    say("help: printed")


def check_render(command: Path, recording: Path, folder: Path) -> None:
    video = folder / "check.mp4"
    size = ["--width", "320", "--height", "240"]
    status, printed = run([command, "render", recording, "--out", video, *size], folder)
    written = re.search(r"(\d+) frames", printed)
    if status or not written:
        raise CheckFailed(f"{command.name} render: status {status}\n{printed}")
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


def check_programs(window_program: Path, command: Path, desktop) -> None:
    """The window program is windowed, so that Windows opens no console for
    it, and carries the icon, which Explorer shows; the other needs a console."""
    for program, wanted in ((window_program, WINDOWED), (command, CONSOLE)):
        kind = subsystem(program)
        if kind != wanted:
            raise CheckFailed(f"{program.name} has subsystem {kind}, not {wanted}")
    difference = icon_difference(desktop.file_icon(window_program))
    if difference:
        raise CheckFailed(f"{window_program.name}'s own icon: {difference}")
    say(
        f"programs: {window_program.name} windowed, with the icon; "
        f"{command.name} a console program"
    )


def check_window(program: Path, recording: Path, folder: Path, desktop, shots) -> None:
    with Started([program, recording], folder, "window", desktop) as started:
        window = started.wait_for_window(lambda text: text == WINDOW_TITLE, shots)
        time.sleep(5)  # its first frames drawn
        picture = shots.take("window.png")
        shots.take_taskbar("taskbar.png", picture, desktop)
        consoles = started.consoles()
        if consoles:
            raise CheckFailed(f"window: a console window opened with it: {consoles}")
        icons = zip(("big", "small"), desktop.window_icons(window), strict=True)
        for which, handle in icons:
            if not handle:
                raise CheckFailed(f"window: it has no {which} icon")
            difference = icon_difference(desktop.icon_pixels(handle))
            if difference:
                raise CheckFailed(f"window: its {which} icon: {difference}")
        left, top, right, bottom = desktop.inside(window)
        shown = box_pixels(picture[top:bottom, left:right])
        if shown < WINDOW_PIXELS:
            raise CheckFailed(f"window: {shown} pixels of the box; see window.png")
        started.quit(window)
    say(
        f"window: opened as {WINDOW_TITLE!r} with no console window, carried the "
        f"icon, showed {shown} pixels of the box, and quit"
    )


def check_console_program(
    command: Path, recording: Path, folder: Path, desktop, shots
) -> None:
    with Started([command, recording], folder, "console program", desktop) as started:
        window = started.wait_for_window(lambda text: text == WINDOW_TITLE, shots)
        time.sleep(2)  # the console window shown too
        shots.take_taskbar("taskbar-console.png", shots.take("console.png"), desktop)
        consoles = started.consoles()
        if not consoles:
            raise CheckFailed(
                "console program: no console window seen, so the check that "
                "the window program opens none cannot be trusted"
            )
        started.quit(window)
    say(f"console program: opened the window and a console window, {consoles}")


def check_picker(program: Path, folder: Path, desktop, shots) -> None:
    chosen = folder / "chosen.txt"
    arguments = [program, "--pick-files", chosen]
    with Started(arguments, folder, "picker", desktop, output=True) as started:
        window = started.wait_for_window(
            lambda text: text.startswith("Open recordings"), shots
        )
        time.sleep(2)  # its folder listed
        shots.take("picker.png")
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


def icon_difference(pixels: np.ndarray) -> str:
    """How RGBA pixels differ from the application's icon at their size, or
    nothing when they are it: their size, their opaque pixels' colours, and
    which pixels are see-through, unless Windows dropped the opacity."""
    size = pixels.shape[0]
    drawn = icon.draw(size)
    if pixels.shape != drawn.shape:
        return f"{pixels.shape[1]} by {size} pixels, which no drawing of it has"
    opaque = drawn[..., 3] == 255
    wrong = (pixels[..., :3] != drawn[..., :3]).any(axis=2) & opaque
    if pixels[..., 3].any():
        wrong |= (pixels[..., 3] == 255) != opaque
    if wrong.any():
        return f"{int(wrong.sum())} of its {size * size} pixels are not the icon's"
    return ""


def subsystem(program: Path) -> int:
    """Windows' subsystem of a program, read from its header: ``WINDOWED`` or
    ``CONSOLE``."""
    with program.open("rb") as file:
        head = file.read(4096)
    # The optional header follows the signature and the file header, 24 bytes.
    start = int.from_bytes(head[0x3C:0x40], "little") + 24
    return int.from_bytes(head[start + 68 : start + 70], "little")


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
    """A program running beside the checks; it is stopped when the checks
    leave it, if it still runs.

    By default it starts as a double-click starts it: a console program gets a
    console window of its own, and neither kind gets output handles, so that
    the window program shows its errors in a message box. With ``output``,
    what it prints goes into a file instead.
    """

    def __init__(
        self, arguments: list, folder: Path, name: str, desktop, output: bool = False
    ) -> None:
        self.name, self.desktop = name, desktop
        self.output = folder / f"{name}.txt" if output else None
        self.before = set(desktop.top_windows())
        if self.output is None:
            flags = subprocess.CREATE_NEW_CONSOLE
            self.process = subprocess.Popen(arguments, cwd=folder, creationflags=flags)
        else:
            with self.output.open("wb") as file:
                self.process = subprocess.Popen(
                    arguments, cwd=folder, stdout=file, stderr=subprocess.STDOUT
                )

    def __enter__(self) -> "Started":
        return self

    def __exit__(self, *failure) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()

    def wait_for_window(self, wanted, shots) -> int:
        """The first visible window of the program whose title ``wanted``
        accepts; a message box in its place, or no window in time, fails."""
        deadline, seen = time.monotonic() + WAIT_SECONDS, []
        while time.monotonic() < deadline:
            seen = self.desktop.windows_of(self.process.pid)
            for window, _, title in seen:
                if wanted(title):
                    return window
            for window, kind, title in seen:
                if kind == "#32770" and title == "MujocoReplay":  # its message box
                    text = self.desktop.dialog_text(window)
                    raise CheckFailed(f"{self.name}: a message box says\n{text}")
            if self.process.poll() is not None:
                raise CheckFailed(
                    f"{self.name}: stopped with status {self.process.returncode}"
                    f"\n{self.printed()}"
                )
            time.sleep(0.5)
        shots.take(f"{self.name}-timeout.png")
        found = [(kind, title) for _, kind, title in seen]
        raise CheckFailed(
            f"{self.name}: no such window in {WAIT_SECONDS} s; its windows: {found}"
            f"\n{self.printed()}"
        )

    def consoles(self) -> list[str]:
        """The titles of the console windows opened since the program started."""
        return [
            title
            for window, (kind, title) in self.desktop.top_windows().items()
            if kind in CONSOLES and window not in self.before
        ]

    def quit(self, window: int) -> None:
        """Quit as a person does: the close button, which asks first, then Y."""
        self.desktop.post(window, WM_CLOSE)
        self.desktop.post(window, WM_CHAR, ord("y"))
        status = self.wait()
        if status:
            raise CheckFailed(f"{self.name}: quit with status {status}")

    def wait(self) -> int:
        """The program's status once it has ended, as it must soon."""
        try:
            return self.process.wait(60)
        except subprocess.TimeoutExpired:
            raise CheckFailed(f"{self.name}: still running after 60 s") from None

    def printed(self) -> str:
        if self.output is None:
            return ""
        return self.output.read_text(encoding="utf-8", errors="replace")


class Shots:
    """The screenshots: saved into a folder, and some printed when asked."""

    PRINTED = ("window.png", "console.png")  # at half size; taskbars in full

    def __init__(self, folder: Path, printed: bool) -> None:
        self.folder, self.printed = folder, printed
        folder.mkdir(parents=True, exist_ok=True)

    def take(self, name: str) -> np.ndarray:
        """The screen, saved as ``name``, as RGB pixels."""
        from PIL import ImageGrab  # Pillow comes with imageio

        picture = ImageGrab.grab().convert("RGB")
        picture.save(self.folder / name)
        if self.printed and name in self.PRINTED:
            self._print(name, picture.resize((picture.width // 2, picture.height // 2)))
        return np.asarray(picture)

    def take_taskbar(self, name: str, picture: np.ndarray, desktop) -> None:
        """The taskbar's part of a screenshot, saved as ``name``."""
        from PIL import Image

        box = desktop.taskbar()
        if box is None:
            say("taskbar: none found")
            return
        left, top, right, bottom = (max(edge, 0) for edge in box)
        strip = Image.fromarray(picture[top:bottom, left:right])
        strip.save(self.folder / name)
        if self.printed:
            self._print(name, strip)

    def _print(self, name: str, picture) -> None:
        data = io.BytesIO()
        picture.save(data, "PNG", optimize=True)
        text = base64.b64encode(data.getvalue()).decode()
        lines = [text[at : at + 100] for at in range(0, len(text), 100)]
        group = [f"::group::{name} as base64", *lines, "::endgroup::"]
        print(*group, sep="\n", flush=True)


class Desktop:
    """The few calls of Windows' user32, gdi32, and shell32 that find, read,
    and close windows, and read icons."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes as w

        class IconInfo(ctypes.Structure):
            _fields_ = [
                ("fIcon", w.BOOL),
                ("xHotspot", w.DWORD),
                ("yHotspot", w.DWORD),
                ("hbmMask", w.HBITMAP),
                ("hbmColor", w.HBITMAP),
            ]

        class Bitmap(ctypes.Structure):
            _fields_ = [
                ("bmType", w.LONG),
                ("bmWidth", w.LONG),
                ("bmHeight", w.LONG),
                ("bmWidthBytes", w.LONG),
                ("bmPlanes", w.WORD),
                ("bmBitsPixel", w.WORD),
                ("bmBits", ctypes.c_void_p),
            ]

        class BitmapInfo(ctypes.Structure):  # BITMAPINFOHEADER, and room for masks
            _fields_ = [
                ("biSize", w.DWORD),
                ("biWidth", w.LONG),
                ("biHeight", w.LONG),
                ("biPlanes", w.WORD),
                ("biBitCount", w.WORD),
                ("biCompression", w.DWORD),
                ("biSizeImage", w.DWORD),
                ("biXPelsPerMeter", w.LONG),
                ("biYPelsPerMeter", w.LONG),
                ("biClrUsed", w.DWORD),
                ("biClrImportant", w.DWORD),
                ("bmiColors", w.DWORD * 3),
            ]

        self._c, self._w = ctypes, w
        self._IconInfo, self._Bitmap, self._BitmapInfo = IconInfo, Bitmap, BitmapInfo
        user32, gdi32, shell32 = map(ctypes.WinDLL, ("user32", "gdi32", "shell32"))
        self._visit = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        window, pointer, text = w.HWND, ctypes.POINTER, w.LPWSTR
        user32.EnumWindows.argtypes = [self._visit, w.LPARAM]
        user32.GetWindowThreadProcessId.argtypes = [window, pointer(w.DWORD)]
        user32.IsWindowVisible.argtypes = [window]
        user32.GetWindowTextW.argtypes = [window, text, ctypes.c_int]
        user32.GetClassNameW.argtypes = [window, text, ctypes.c_int]
        user32.GetDlgItemTextW.argtypes = [window, ctypes.c_int, text, ctypes.c_int]
        user32.GetClientRect.argtypes = [window, pointer(w.RECT)]
        user32.GetWindowRect.argtypes = [window, pointer(w.RECT)]
        user32.ClientToScreen.argtypes = [window, pointer(w.POINT)]
        user32.PostMessageW.argtypes = [window, w.UINT, w.WPARAM, w.LPARAM]
        user32.SendMessageTimeoutW.argtypes = [
            window, w.UINT, w.WPARAM, w.LPARAM, w.UINT, w.UINT,
            pointer(ctypes.c_size_t),
        ]  # fmt: skip
        user32.SendMessageTimeoutW.restype = w.LPARAM
        user32.FindWindowW.argtypes = [w.LPCWSTR, w.LPCWSTR]
        user32.FindWindowW.restype = window
        user32.GetIconInfo.argtypes = [w.HICON, pointer(IconInfo)]
        user32.DestroyIcon.argtypes = [w.HICON]
        user32.GetDC.argtypes = [window]
        user32.GetDC.restype = w.HDC
        user32.ReleaseDC.argtypes = [window, w.HDC]
        gdi32.GetObjectW.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p]
        gdi32.GetDIBits.argtypes = [
            w.HDC, w.HBITMAP, w.UINT, w.UINT, ctypes.c_void_p, ctypes.c_void_p, w.UINT,
        ]  # fmt: skip
        gdi32.DeleteObject.argtypes = [w.HGDIOBJ]
        shell32.ExtractIconExW.argtypes = [
            w.LPCWSTR, ctypes.c_int, pointer(w.HICON), pointer(w.HICON), w.UINT,
        ]  # fmt: skip
        shell32.ExtractIconExW.restype = w.UINT
        user32.SetProcessDPIAware()  # window places in the screenshot's pixels
        self._user32, self._gdi32, self._shell32 = user32, gdi32, shell32

    def top_windows(self) -> dict[int, tuple[str, str]]:
        """Every visible top-level window: its class and its title."""
        return {window: (kind, title) for window, kind, title in self._windows()}

    def windows_of(self, pid: int) -> list[tuple[int, str, str]]:
        """A process's visible top-level windows, with classes and titles."""
        return self._windows(pid)

    def _windows(self, pid: int | None = None) -> list[tuple[int, str, str]]:
        c, user32, found = self._c, self._user32, []

        def visit(window, _):
            owner = self._w.DWORD()
            user32.GetWindowThreadProcessId(window, c.byref(owner))
            if user32.IsWindowVisible(window) and pid in (None, owner.value):
                kind, title = (c.create_unicode_buffer(512) for _ in range(2))
                user32.GetClassNameW(window, kind, len(kind))
                user32.GetWindowTextW(window, title, len(title))
                found.append((window, kind.value, title.value))
            return True

        user32.EnumWindows(self._visit(visit), 0)
        return found

    def dialog_text(self, window: int) -> str:
        """The message of a message box."""
        text = self._c.create_unicode_buffer(8192)
        self._user32.GetDlgItemTextW(window, 0xFFFF, text, len(text))
        return text.value

    def inside(self, window: int) -> tuple[int, int, int, int]:
        """Where a window's inside is on the screen: left, top, right, bottom."""
        c, user32 = self._c, self._user32
        size, corner = self._w.RECT(), self._w.POINT(0, 0)
        user32.GetClientRect(window, c.byref(size))
        user32.ClientToScreen(window, c.byref(corner))
        return corner.x, corner.y, corner.x + size.right, corner.y + size.bottom

    def taskbar(self) -> tuple[int, int, int, int] | None:
        """Where the taskbar is on the screen, if there is one."""
        bar = self._user32.FindWindowW("Shell_TrayWnd", None)
        if not bar:
            return None
        box = self._w.RECT()
        self._user32.GetWindowRect(bar, self._c.byref(box))
        return box.left, box.top, box.right, box.bottom

    def window_icons(self, window: int) -> tuple[int, int]:
        """A window's big and small icons, which its taskbar button and its
        title bar show; 0 for one it was not given."""
        icons = []
        for which in (1, 0):  # ICON_BIG, ICON_SMALL
            result = self._c.c_size_t()
            self._user32.SendMessageTimeoutW(  # WM_GETICON, unless it hangs
                window, 0x007F, which, 0, 0x0002, 2000, self._c.byref(result)
            )
            icons.append(result.value)
        return icons[0], icons[1]

    def file_icon(self, program: Path) -> np.ndarray:
        """The icon a program carries, at the big size Explorer shows."""
        handle = self._w.HICON()
        count = self._shell32.ExtractIconExW(
            str(program), 0, self._c.byref(handle), None, 1
        )
        if not count or not handle:
            raise CheckFailed(f"{program.name} carries no icon")
        try:
            return self.icon_pixels(handle)
        finally:
            self._user32.DestroyIcon(handle)

    def icon_pixels(self, handle) -> np.ndarray:
        """An icon's colour pixels as RGBA, top row first."""
        c, user32, gdi32 = self._c, self._user32, self._gdi32
        info = self._IconInfo()
        if not user32.GetIconInfo(handle, c.byref(info)):
            raise CheckFailed("an icon cannot be read")
        try:
            if not info.hbmColor:
                raise CheckFailed("an icon has no colours")
            bitmap = self._Bitmap()
            gdi32.GetObjectW(info.hbmColor, c.sizeof(bitmap), c.byref(bitmap))
            width, height = bitmap.bmWidth, bitmap.bmHeight
            header = self._BitmapInfo()
            header.biSize = 40  # the header alone
            header.biWidth, header.biHeight = width, -height  # top row first
            header.biPlanes, header.biBitCount = 1, 32
            pixels = (c.c_ubyte * (width * height * 4))()
            screen = user32.GetDC(None)
            try:
                gdi32.GetDIBits(
                    screen, info.hbmColor, 0, height, pixels, c.byref(header), 0
                )
            finally:
                user32.ReleaseDC(None, screen)
        finally:
            for bitmap_handle in (info.hbmColor, info.hbmMask):
                if bitmap_handle:
                    gdi32.DeleteObject(bitmap_handle)
        blue_first = np.frombuffer(pixels, np.uint8).reshape(height, width, 4)
        return blue_first[..., [2, 1, 0, 3]].copy()

    def post(self, window: int, message: int, value: int = 0) -> None:
        self._user32.PostMessageW(window, message, value, 0)


def say(text: str) -> None:
    print(text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
