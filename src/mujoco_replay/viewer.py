"""The application window: the scene, the panel, the tabs, the keys, and the mouse.

The window opens on an empty flat world and plays recordings once they are
opened: from the command line, with the panel's Open button (the system's file
picker), or by dropping files onto the window. Each file opens in a tab of its
own along the top, and switching tabs shows another file at the same frame,
so that two files compare at the same moment. The panel holds the settings,
which are saved as they change. Each turn of the main loop advances playback
by wall time and draws only when something changed, so a paused window costs
next to nothing. GLFW opens the window and delivers the input; only the
``view`` command imports this module. docs/design.md lists the keys.
"""

import contextlib
import math
import os
import subprocess
import sys
import tempfile
import time
import warnings
from collections import deque
from dataclasses import fields, replace
from pathlib import Path

import glfw
import mujoco
import numpy as np

from mujoco_replay import icon, ui
from mujoco_replay.cli import PICK_FILES
from mujoco_replay.playback import DEFAULT_SECONDS_PER_FRAME, Playback
from mujoco_replay.recording import (
    Recording,
    RecordingError,
    in_name_order,
)
from mujoco_replay.render import SceneRenderer, setup_lines, world_rank
from mujoco_replay.scene import ComposedScene, read_file
from mujoco_replay.selection import MAX_WORLDS, choose_worlds, world_counts
from mujoco_replay.settings import (
    GHOST_STRENGTHS,
    RESOLUTIONS,
    Settings,
    save_settings,
    user_folder,
)

FLASH_SECONDS = 1.0
MESSAGE_SECONDS = 6.0
IDLE_WAIT = 0.5  # the longest the loop sleeps between looks at the clock
SCREEN_SHARE = 0.85  # of the screen's free area the window takes, unless sized
DOUBLE_CLICK = 0.4  # seconds between the presses of a double-click, at most
# The icon's sizes for the window, whole multiples of its pixel art: Windows
# shows 16 and 32 pixels at normal density, up to 64 at double; GLFW picks the
# nearest.
ICON_SIZES = (16, 32, 48, 64)
# Windows groups a window by this name on the taskbar, with the window's icon,
# instead of with every other Python program under Python's; see
# _name_for_the_taskbar.
APP_ID = "MujocoReplay.Viewer"
# The world shown before any recording: a floor under a sky, as Centipede's.
EMPTY_WORLD = """
<mujoco model="empty world">
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" width="256" height="1536"
             rgb1="0.10 0.15 0.20" rgb2="0.35 0.40 0.44"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.16 0.20 0.23" rgb2="0.20 0.24 0.27"/>
    <material name="grid" texture="grid" texrepeat="16 16"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="4 4 0.05" material="grid"/>
  </worldbody>
</mujoco>
"""
# Keys that type no character.
KEYS = {
    glfw.KEY_SPACE: "play",
    glfw.KEY_RIGHT: "step",
    glfw.KEY_LEFT: "back",
    glfw.KEY_UP: "faster",
    glfw.KEY_DOWN: "slower",
    glfw.KEY_HOME: "first",
    glfw.KEY_END: "last",
    glfw.KEY_TAB: "panel",
    glfw.KEY_ESCAPE: "quit",
    glfw.KEY_F1: "keys",
}
# The question before quitting, and what its keys and buttons do.
QUIT_QUESTION = "Quit MujocoReplay?"
QUIT_CHOICES = (("Quit (Enter)", "quit now"), ("Cancel (Esc)", "stay"))
QUIT_KEYS = {
    glfw.KEY_ENTER: "quit now",
    glfw.KEY_KP_ENTER: "quit now",
    glfw.KEY_ESCAPE: "stay",
}
QUIT_CHARACTERS = {"y": "quit now", "q": "quit now", "n": "stay"}
# Keys by the character they type, so that they follow the keyboard's layout;
# a capital letter does what its small letter does, and Shift+B goes back.
CHARACTERS = {
    "0": "default speed",
    "r": "restart",
    "l": "loop",
    "n": "next file",
    "p": "previous file",
    "b": "next world",
    "g": "ghosts",
    "m": "markers",
    "v": "reset view",
    "t": "top view",
    "c": "centre",
    "f": "follow",
    "h": "overlay",
    "i": "setup",
    "o": "open",
    "q": "quit",
    "?": "keys",
}
# What F1 shows, in short lines, so that two columns fit beside the panel.
HELP = [
    "Space: play, pause",
    "Right, Left: step, step back",
    "Up, Down: faster, slower",
    "0: default speed",
    "Home, End: first, last frame",
    "R: restart the file",
    "L: loop",
    "N, Ctrl+Tab: next tab",
    "P, Ctrl+Shift+Tab: previous tab",
    "Ctrl+W: close the tab",
    "B: next world",
    "Shift+B: previous world",
    "Double-click: highlight a world",
    "G: ghosts   M: markers",
    "V: reset the view",
    "T: look from above",
    "C: centre on the highlight",
    "F: follow the highlight",
    "Left drag: rotate",
    "Shift and left drag: turn",
    "Right drag: pan",
    "Wheel, middle drag: zoom",
    "Click the timeline: go there",
    "H: overlay   I: setup",
    "O: open recordings or models",
    "Tab: panel   F1, ?: these keys",
    "Q, Esc: quit (asks first)",
]


def run(
    recordings: list[Recording],
    settings: Settings,
    seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME,
    ids: list[int] | None = None,
    size: tuple[int, int] | None = None,
    hud: bool = True,
    paths: list[str] | None = None,
) -> None:
    """Open the window, with ``recordings`` if any, until it is closed.

    ``ids``, when given, are the producer world ids to draw instead of the
    best of each rank band; the caller has checked that each file has some.
    Without ``size``, the window takes most of the screen, in its middle.
    ``hud`` off hides the overlay for this run without saving that.
    ``paths`` are the recordings' files, which name their tabs. When none of
    them can be shown, the first file's problem is raised as a
    ``RecordingError``, as the command does for a file it cannot read.
    """
    if not glfw.init():
        raise RuntimeError("GLFW cannot start: the window needs a display")
    try:
        _name_for_the_taskbar()
        glfw.window_hint(glfw.SAMPLES, 0)  # anti-aliasing is drawn offscreen
        glfw.window_hint(glfw.VISIBLE, False)  # shown once in its place
        size, position = _placement(size)
        window = glfw.create_window(*size, "MujocoReplay", None, None)
        if not window:
            raise RuntimeError("GLFW cannot open a window with OpenGL")
        _give_icon(window)
        if position is not None:
            glfw.set_window_pos(window, *position)
        glfw.show_window(window)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        viewer = Viewer(window, settings, seconds_per_frame, hud)
        if recordings:
            problems = viewer.load(recordings, ids, paths)
            if viewer.playback is None:  # none could be shown
                raise RecordingError(problems[0])
        viewer.loop()
    finally:
        glfw.terminate()


class FilePicker:
    """The system's file picker, shown in a process of its own.

    The window runs the command again with ``--pick-files`` and a file to
    write the chosen paths into: an executable runs itself, and otherwise
    Python runs the package, since an executable's ``sys.executable`` is the
    executable. Its own process keeps tkinter's event loop apart from GLFW's,
    and the window goes on drawing while the picker is open. Files, not pipes,
    carry the paths and the errors: a pipe that the process filled before
    exiting would hold it open for good, and a file takes any path as UTF-8.
    """

    def __init__(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        handle, self._chosen = tempfile.mkstemp(prefix="mujoco-replay-", suffix=".txt")
        os.close(handle)
        self._errors = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
        program = [sys.executable]
        if not getattr(sys, "frozen", False):
            program += ["-m", "mujoco_replay"]
        self._process = subprocess.Popen(
            [*program, PICK_FILES, self._chosen],
            stdout=subprocess.DEVNULL,
            stderr=self._errors,
            creationflags=flags,
        )

    def poll(self) -> list[str] | None:
        """The chosen paths once the picker has closed; ``None`` while it is open."""
        if self._process.poll() is None:
            return None
        self._errors.seek(0)
        errors = self._errors.read()
        try:
            chosen = Path(self._chosen).read_text(encoding="utf-8", errors="replace")
        except OSError:
            chosen = ""
        self.close()
        if self._process.returncode:
            reason = (errors.strip().splitlines() or ["no reason given"])[-1]
            raise RuntimeError(
                f"The file picker could not open ({reason}); "
                "drop the files onto the window instead."
            )
        return [line for line in chosen.splitlines() if line.strip()]

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._errors.close()
        with contextlib.suppress(OSError):
            os.unlink(self._chosen)


class Viewer:
    """The state of the window: files, playback, scene, renderer, panel, tabs."""

    def __init__(
        self,
        window,
        settings: Settings,
        seconds_per_frame: float,
        hud: bool = True,
    ) -> None:
        self.window = window
        self.settings = settings if hud else replace(settings, overlay=False)
        self._saved = settings  # what is saved: the run's own choices left out
        self.seconds_per_frame = seconds_per_frame
        self.paths: list[str | None] = []  # each tab's file; None without one
        self.ids: list[int] | None = None
        self.playback: Playback | None = None  # its files are the tabs'
        self._empty = _empty_world()
        self.scene = ComposedScene(self._empty, np.arange(1))
        self.renderer = SceneRenderer(
            self.scene, font_scale=_font_scale(window), graphics=settings.graphics
        )
        _look_at_empty_world(self.renderer)
        self.panel = ui.Panel()
        self.tabs = ui.TabBar()
        self._names: list[str] = []  # the tabs' names
        self.boxes: list[ui.Box] = []
        self.setup = False
        self.help = False
        self.asking = False  # whether the question before quitting is shown
        self._picker: FilePicker | None = None
        self._dropped: list[str] = []
        self._passed: list[str] = []  # events passed, not yet flashed
        self._flash, self._flash_until = "", 0.0
        self._message, self._message_until = "", 0.0
        self._picked: int | None = None  # the world id the user highlighted
        self._shown_ghosts = (
            Settings().ghosts if settings.ghosts == "hidden" else settings.ghosts
        )
        self._dirty = True
        self._cursor = glfw.get_cursor_pos(window)
        self._held = self._shift = False  # of the key whose character comes next
        self._hovered: str | None = None
        self._pressed: str | None = None  # the panel action a press began on
        self._dragging = False  # a drag that began on the scene, not the panel
        self._scrubbing = False  # a drag that began on the timeline
        self._last_press = (-math.inf, 0.0, 0.0)  # its time and place
        # Where the scene and the overlay begin on the left, and the pixels
        # the tabs take at the top.
        self._left = self._inset = self._top = 0
        self._draw_seconds: deque[float] = deque(maxlen=30)
        self._actions = self._make_actions()
        glfw.set_key_callback(window, self._on_key)
        glfw.set_char_callback(window, self._on_char)
        glfw.set_mouse_button_callback(window, self._on_button)
        glfw.set_cursor_pos_callback(window, self._on_cursor)
        glfw.set_scroll_callback(window, self._on_scroll)
        glfw.set_drop_callback(window, self._on_drop)
        glfw.set_framebuffer_size_callback(window, self._on_change)
        glfw.set_window_refresh_callback(window, self._on_change)
        glfw.set_window_close_callback(window, self._on_close)

    def loop(self) -> None:
        while not glfw.window_should_close(self.window):
            wait = self._wait()
            if wait > 0:
                glfw.wait_events_timeout(wait)
            else:
                glfw.poll_events()
            self._take_files()
            now = time.monotonic()
            if self.playback is not None:
                before = self._position()
                self._passed += self.playback.advance(now)
                self._dirty |= self._position() != before
            self._report(self._show_file())  # when another tab's file is due
            if self._passed:  # timed from now, so that composing cannot eat it
                self._flash = "  ".join(self._passed)
                self._flash_until = time.monotonic() + FLASH_SECONDS
                self._passed = []
                self._dirty = True
            if self._flash and now >= self._flash_until:
                self._flash, self._dirty = "", True
            if self._message and now >= self._message_until:
                self._message, self._dirty = "", True
            if self._dirty:
                self._draw()
        self.renderer.close()
        if self._picker is not None:
            self._picker.close()

    @property
    def recordings(self) -> list[Recording]:
        """The open files' recordings, in the order of their tabs."""
        return self.playback.recordings if self.playback is not None else []

    def load(
        self,
        recordings: list[Recording],
        ids: list[int] | None = None,
        paths: list[str | None] | None = None,
    ) -> list[str]:
        """Open ``recordings`` in tabs after those open, and show the first.

        ``paths`` are their files, which name the tabs; a file open already
        is not opened again. ``ids``, the world ids to draw instead of the
        best of each rank band, hold until files are opened from the window.
        As when switching tabs, the frame, the pause, and the highlighted
        world stay; a file whose model fails is closed again. Returns the
        problems found, which are also told.
        """
        problems = self._add(recordings, paths, ids)
        self._report(problems)
        return problems

    def open_files(self, paths: list[str]) -> None:
        """Open the files at ``paths`` in tabs, and show the first.

        A file already open is not opened again: when every file is, the
        first one's tab is shown. A file that cannot be read is reported,
        and the others open.
        """
        recordings, opened, problems = [], [], []
        shown = None  # the tab of the first of the files open already
        for path in paths:
            if any(_where(path) == _where(other) for other in opened):
                continue
            tab = self._tab_of(path)
            if tab is not None:
                shown = tab if shown is None else shown
                continue
            try:
                recordings.append(read_file(path))
                opened.append(path)
            except RecordingError as error:
                problems.append(str(error))
        if recordings:
            problems += self._add(recordings, opened)
        elif shown is not None:
            self._show_tab(shown)
        self._report(problems)

    def _add(
        self,
        recordings: list[Recording],
        paths: list[str | None] | None,
        ids: list[int] | None = None,
    ) -> list[str]:
        """Open tabs after those open and show the first; return the problems
        of the files closed again."""
        paths = paths or [None] * len(recordings)
        fresh = []
        for recording, path in zip(recordings, paths, strict=True):
            places = [_where(other) for _, other in fresh if other is not None]
            if path is not None and (
                self._tab_of(path) is not None or _where(path) in places
            ):
                continue  # open already
            fresh.append((recording, path))
        if not fresh:
            return []
        if self.playback is None:
            self.playback = Playback(
                [],
                self.seconds_per_frame,
                time.monotonic(),
                run_on=self.settings.play_next,
                loop=self.settings.loop,
            )
        first = len(self.recordings)
        self.recordings.extend(recording for recording, _ in fresh)
        self.paths += [path for _, path in fresh]
        self.ids = ids
        self._name_tabs()
        self.playback.show_file(first)
        return self._show_file()

    def _draw(self) -> None:
        self._dirty = False
        width, height = glfw.get_framebuffer_size(self.window)
        if not width or not height:  # minimised; redrawn when restored
            return
        context = self.renderer.context
        line = context.charHeight
        panel = ui.Panel.width(line) if self.settings.panel else 0
        self._left, self._inset = panel, panel or 6 * line
        tabs = self.settings.panel and self.playback is not None
        self._top = ui.TabBar.height(line) if tabs else 0
        loaded = self.playback is not None and self.scene.recording is not self._empty
        # While another tab's file is composed, the scene shows the one before.
        if loaded and self.scene.recording is self.playback.recording:
            if self.scene.frame_index != self.playback.frame_index:
                self.scene.set_frame(self.playback.frame_index)
        side = None
        if self.help:
            side = HELP
        elif loaded and self.setup:
            side = setup_lines(self.scene.recording.setup) or ["(no setup)"]
        rate, first = self._rate(), self.settings.frame_rate and not self._draw_seconds
        start = time.perf_counter()
        self.renderer.render(
            width,
            height,
            status=self.playback.status() if loaded else "",
            flash=self._flash,
            hud=loaded and self.settings.overlay,
            side=side,
            corner=rate,
            left=self._left,
            inset=self._inset,
            message=self._message,
            hint="" if loaded else "Open recordings or models (O), or drop them here",
            fresh=first,  # a frame to time, even when only the overlay changed
            top=self._top,
        )
        context = self.renderer.context  # made again if the floor grew
        hover = self._pixels(*self._cursor)
        self.panel.draw(self._layout(), height, context, hover, self.settings.panel)
        if self.asking:  # over everything else, which it dims
            mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), 0, 0, 0, 0.5)
            self.panel.draw(self._question(), height, context, hover, False)
        self.boxes = self._targets()
        if self.settings.frame_rate and self.renderer.drew_scene:
            mujoco.mjr_finish()  # wait for the graphics card, to time it
            self._draw_seconds.append(time.perf_counter() - start)
            self._dirty |= first  # show the first time at once, even paused
        glfw.swap_buffers(self.window)

    def _layout(self) -> list[ui.Box]:
        """The panel's boxes and the tabs as things stand, or the button that
        shows them."""
        width, height = glfw.get_framebuffer_size(self.window)
        context = self.renderer.context
        line = context.charHeight
        if self.settings.panel:

            def measure(text: str) -> int:
                return ui.text_width(context, text)

            boxes = self.panel.layout(self._rows(), height, line, measure)
            if self.playback is not None:
                left, shown = ui.Panel.width(line), self.playback.file_index
                beside = max(0, width - left)
                boxes += self.tabs.layout(
                    self._names, shown, left, beside, height, line, measure
                )
            return boxes
        row = line + line // 2
        top = height - line // 2 - row
        return [ui.Box(line // 2, top, 5 * line, row, "button", "Panel", "panel")]

    def _question(self) -> list[ui.Box]:
        """The question before quitting, in the middle of the window."""
        width, height = glfw.get_framebuffer_size(self.window)
        context = self.renderer.context

        def measure(text: str) -> int:
            return ui.text_width(context, text)

        line = context.charHeight
        return ui.dialog(width, height, line, measure, QUIT_QUESTION, QUIT_CHOICES)

    def _targets(self) -> list[ui.Box]:
        """What a click can hit: the question's buttons while it is shown."""
        return self._question() if self.asking else self._layout()

    def _rows(self) -> list[ui.Row]:
        """The panel as it stands, top to bottom."""
        settings, graphics = self.settings, self.settings.graphics
        rows: list[ui.Row] = [ui.Title("MujocoReplay")]
        playback = self.playback
        if playback is None:
            rows += [
                ui.Buttons((("Open files ...", "open"),)),
                ui.Note("or drop .npz or .xml files here"),
            ]
        else:
            rows.append(ui.Buttons((("Open ...", "open"), ("Close all", "close all"))))
            play = "Pause" if playback.playing else "Play"
            controls = (("|<", "first"), ("<", "back"), (play, "play"))
            controls += ((">", "step"), (">|", "last"))
            per_frame = f"{playback.seconds_per_frame:.3g} s"
            rows += [
                ui.Section("Playback"),
                ui.Buttons(controls),
                ui.Stepper("Per frame", per_frame, "faster", "slower"),
                ui.Toggles(
                    (
                        ("Loop", "loop", settings.loop),
                        ("Play next", "play next", settings.play_next),
                    )
                ),
            ]
        if playback is None:
            shown = str(settings.worlds)
        else:
            shown = f"{len(self.scene.worlds)} of {self.scene.recording.world_count}"
        rows.append(ui.Section("Worlds"))
        if self.ids is None:
            rows.append(ui.Stepper("Shown", shown, "fewer", "more"))
        else:  # the worlds named on the command line, whatever the count
            rows.append(ui.Note(f"Shown: {shown}, chosen by id"))
        if playback is not None and len(self._highlightable()) > 1:
            world = self.scene.worlds[self.scene.highlight]
            rank = f"rank {world_rank(self.scene.recording, world)[0]:,}"
            rows.append(ui.Stepper("Highlight", rank, "previous world", "next world"))
        if playback is not None and len(self.scene.worlds) > 1:
            rows.append(ui.Stepper("Ghosts", settings.ghosts, "fainter", "stronger"))
        if playback is not None:
            rows += [
                ui.Section("View"),
                ui.Toggles(
                    (
                        ("Reset", "reset view", False),
                        ("Top", "top view", False),
                        ("Follow", "follow", self.renderer.follow),
                    )
                ),
            ]
        mode = settings.mode
        resolution = f"{graphics.resolution}%"
        rows += [
            ui.Section("Graphics" + (" (custom)" if mode == "custom" else "")),
            ui.Toggles(
                (
                    ("Quality", "quality", mode == "quality"),
                    ("Performance", "performance", mode == "performance"),
                )
            ),
            ui.Toggles(
                (
                    ("Shadows", "shadows", graphics.shadows),
                    ("Reflections", "reflections", graphics.reflections),
                )
            ),
            ui.Toggles(
                (
                    ("Anti-aliasing", "antialiasing", graphics.antialiasing),
                    ("Fine shapes", "fine shapes", graphics.fine_shapes),
                )
            ),
            ui.Stepper(
                "Resolution", resolution, "lower resolution", "higher resolution"
            ),
            ui.Section("Options"),
            ui.Toggles(
                (
                    ("Overlay", "overlay", settings.overlay),
                    ("Markers", "markers", self.renderer.markers_visible),
                    ("Setup", "setup", self.setup),
                )
            ),
            ui.Toggles(
                (
                    ("Keys", "keys", self.help),
                    ("Frame rate", "frame rate", settings.frame_rate),
                    ("Cache", "cache", settings.cache),
                )
            ),
            ui.Note("Tab hides this panel"),
        ]
        return rows

    def _make_actions(self) -> dict:
        """What every panel action and key does."""

        def playing(change):
            def act():
                if self.playback is not None:
                    change(self.playback)

            return act

        def graphics(name: str):
            def act():
                current = getattr(self.settings.graphics, name)
                self._change(self.settings.with_graphics(**{name: not current}))

            return act

        def setting(name: str):
            def act():
                current = getattr(self.settings, name)
                self._change(replace(self.settings, **{name: not current}))

            return act

        return {
            "open": self._open_picker,
            "first": playing(lambda playback: playback.first_frame()),
            "back": playing(lambda playback: playback.step(-1)),
            "play": playing(lambda playback: playback.toggle()),
            "step": playing(lambda playback: self._passed.extend(playback.step(1))),
            "last": playing(lambda playback: playback.last_frame()),
            "restart": playing(lambda playback: playback.restart()),
            "slower": playing(lambda playback: playback.slower()),
            "faster": playing(lambda playback: playback.faster()),
            "default speed": playing(lambda playback: playback.default_speed()),
            "previous file": lambda: self._step_tab(-1),
            "next file": lambda: self._step_tab(1),
            "close file": lambda: self._close_tab(self._shown_tab()),
            "close all": self._unload,
            "tabs left": lambda: self.tabs.scroll_by(-1),
            "tabs right": lambda: self.tabs.scroll_by(1),
            "fewer": lambda: self._step_worlds(-1),
            "more": lambda: self._step_worlds(1),
            "previous world": lambda: self._step_highlight(-1),
            "next world": lambda: self._step_highlight(1),
            "ghosts": self._toggle_ghosts,
            "fainter": lambda: self._step_ghosts(-1),
            "stronger": lambda: self._step_ghosts(1),
            "markers": self._toggle_markers,
            "reset view": self._reset_view,
            "top view": self.renderer.look_from_above,
            "follow": self._toggle_follow,
            "centre": self._centre,
            "quality": lambda: self._change(self.settings.with_mode("quality")),
            "performance": lambda: self._change(self.settings.with_mode("performance")),
            "shadows": graphics("shadows"),
            "reflections": graphics("reflections"),
            "antialiasing": graphics("antialiasing"),
            "fine shapes": graphics("fine_shapes"),
            "lower resolution": lambda: self._step_resolution(-1),
            "higher resolution": lambda: self._step_resolution(1),
            "overlay": setting("overlay"),
            "loop": setting("loop"),
            "play next": setting("play_next"),
            "frame rate": setting("frame_rate"),
            "cache": setting("cache"),
            "panel": setting("panel"),
            "setup": self._toggle_setup,
            "keys": self._toggle_help,
            "quit": self._ask_to_quit,
            "quit now": lambda: glfw.set_window_should_close(self.window, True),
            "stay": self._stay,
        }

    def _act(self, action: str) -> None:
        kind, _, tab = action.rpartition(" ")  # a tab's actions end in its number
        if kind == "tab":
            self._show_tab(int(tab))
        elif kind == "close tab":
            self._close_tab(int(tab))
        else:
            self._actions[action]()
        # Clicks still queued must meet the panel as it now is, not as drawn.
        self.boxes = self._targets()
        self._dirty = True

    def _change(self, settings: Settings) -> None:
        """Take new settings: save them, and redraw or recompose as they need."""
        previous, self.settings = self.settings, settings
        changed = {
            item.name: getattr(settings, item.name)
            for item in fields(Settings)
            if getattr(settings, item.name) != getattr(previous, item.name)
        }
        self._saved = replace(self._saved, **changed)
        save_settings(self._saved)
        if settings.graphics != previous.graphics:
            self.renderer.set_graphics(settings.graphics)
            self._draw_seconds.clear()
        if settings.ghosts != previous.ghosts:
            self.scene.set_ghosts(settings.ghosts)
        if settings.worlds != previous.worlds and self.playback is not None:
            self._recompose()
        if settings.frame_rate != previous.frame_rate:
            self._draw_seconds.clear()
        if self.playback is not None:  # what playing does at a file's end
            self.playback.loop, self.playback.run_on = settings.loop, settings.play_next
        if settings.cache != previous.cache:
            done = "from now on" if settings.cache else "no longer"
            self._say(f"Composed scenes are {done} kept in {user_folder('cache')}")

    def _step_resolution(self, direction: int) -> None:
        index = RESOLUTIONS.index(self.settings.graphics.resolution) + direction
        if 0 <= index < len(RESOLUTIONS):
            self._change(self.settings.with_graphics(resolution=RESOLUTIONS[index]))

    def _step_worlds(self, direction: int) -> None:
        """Draw the next count up or down, as far as the file's worlds allow."""
        total = self.scene.recording.world_count if self.playback else MAX_WORLDS
        counts = world_counts(total)
        current = min(self.settings.worlds, counts[-1])
        if direction < 0:
            chosen = [count for count in counts if count < current][-1:]
        else:
            chosen = [count for count in counts if count > current][:1]
        if chosen:
            self._change(replace(self.settings, worlds=chosen[0]))

    def _step_ghosts(self, direction: int) -> None:
        index = GHOST_STRENGTHS.index(self.settings.ghosts) + direction
        if 0 <= index < len(GHOST_STRENGTHS):
            self._set_ghosts(GHOST_STRENGTHS[index])

    def _toggle_ghosts(self) -> None:
        hidden = self.settings.ghosts == "hidden"
        self._set_ghosts(self._shown_ghosts if hidden else "hidden")

    def _set_ghosts(self, strength: str) -> None:
        if strength != "hidden":
            self._shown_ghosts = strength
        self._change(replace(self.settings, ghosts=strength))

    def _recompose(self) -> None:
        """Choose and compose the worlds again after the count changed."""
        recording = self.playback.recording
        worlds = self._choose(recording)
        self._say_now(f"Composing {len(worlds)} worlds ...")
        scene = ComposedScene(recording, worlds, self._cache())
        self._use(scene, reframe=False)
        self._message = ""
        self.playback.sync(time.monotonic())  # composing may have taken a while

    def _show_file(self) -> list[str]:
        """Show the shown tab's file, when the scene does not, at the frame of
        playback and with the world the user highlighted; return the problems
        of the files closed on the way.

        The camera stays for a file of the same model. A file whose model
        fails is closed, and the file shown before comes back, paused at the
        frame it showed; when that file is closed too, the one that took the
        failed file's place is shown, and the empty world after the last.
        """
        problems = []
        while self.playback and self.playback.recording is not self.scene.recording:
            recording = self.playback.recording
            worlds = self._choose(recording)
            previous = self.scene
            if previous.fits(recording, worlds):
                previous.show(recording, worlds, self.playback.frame_index)
                self._use(previous, reframe=False)
            else:
                self._say_now(f"Composing {len(worlds)} worlds ...")
                try:
                    scene = ComposedScene(recording, worlds, self._cache())
                except RecordingError as error:
                    problems.append(str(error))
                    self._close_tab(self.playback.file_index)
                    back = self._tab_of_recording(previous.recording)
                    if back is not None:
                        self.playback.show_file(back)
                        self.playback.frame_index = previous.frame_index
                        self.playback.playing = False
                        worlds = self._choose(previous.recording)
                        if not np.array_equal(worlds, previous.worlds):
                            self._recompose()  # opening ended the worlds by id
                    continue
                scene.set_frame(self.playback.frame_index)  # framed as it is now
                self._use(scene, reframe=not previous.same_model(recording))
                self._message = ""
            self.playback.sync(time.monotonic())  # composing may have taken a while
        return problems

    def _show_tab(self, index: int) -> None:
        """Show another tab's file at the same frame; the main loop shows it."""
        if self.playback is not None and 0 <= index < len(self.recordings):
            self.playback.show_file(index)

    def _step_tab(self, direction: int) -> None:
        """Show the next or previous tab's file, round from the end."""
        if self.playback is not None:
            count = len(self.recordings)
            self._show_tab((self.playback.file_index + direction) % count)

    def _shown_tab(self) -> int:
        return self.playback.file_index if self.playback is not None else -1

    def _close_tab(self, index: int) -> None:
        """Close a tab; when its file was shown, the next tab's file is shown,
        or the previous one's after the last tab, and the empty world after
        the only one."""
        if self.playback is None or not 0 <= index < len(self.recordings):
            return
        self.playback.remove_file(index)
        del self.paths[index]
        if self.recordings:
            self._name_tabs()
        else:
            self._unload()

    def _unload(self) -> None:
        """Close every tab: back to the empty world, as at the start."""
        if self.playback is not None:
            self.seconds_per_frame = self.playback.seconds_per_frame
        self.playback, self.paths, self._names = None, [], []
        self.ids = self._picked = None
        self.tabs = ui.TabBar()
        self.scene = ComposedScene(self._empty, np.arange(1))
        self.renderer.show(self.scene)
        _look_at_empty_world(self.renderer)
        glfw.set_window_title(self.window, "MujocoReplay")
        self._draw_seconds.clear()
        self._dirty = True

    def _tab_of(self, path: str) -> int | None:
        """The tab of the file at ``path``, when it is open."""
        place = _where(path)
        for index, other in enumerate(self.paths):
            if other is not None and _where(other) == place:
                return index
        return None

    def _tab_of_recording(self, recording: Recording) -> int | None:
        for index, other in enumerate(self.recordings):
            if other is recording:
                return index
        return None

    def _name_tabs(self) -> None:
        titles = [recording.title for recording in self.recordings]
        self._names = ui.tab_names(self.paths, titles)

    def _use(self, scene: ComposedScene, reframe: bool) -> None:
        """Show ``scene`` with the ghosts as set and the world the user picked.

        Unless the user picked a world that ``scene`` draws, the best is
        highlighted.
        """
        scene.set_ghosts(self.settings.ghosts)
        self.scene = scene
        self.renderer.show(scene)
        if reframe:
            self.renderer.frame_all()
            self.renderer.set_follow(self.settings.follow)
        drawn = scene.recording.world_ids[scene.worlds]
        same = np.flatnonzero(drawn == self._picked) if self._picked is not None else []
        scene.set_highlight(int(same[0]) if len(same) else 0)
        title = f"MujocoReplay - {scene.recording.title}"
        # A file name's undecodable bytes become lone surrogates, which GLFW refuses.
        glfw.set_window_title(self.window, title.encode(errors="replace").decode())
        self._draw_seconds.clear()  # another scene costs another time
        self._dirty = True

    def _position(self) -> tuple[int, int, bool]:
        """What the drawing shows of playback: the file, the frame, any pause."""
        playback = self.playback
        return playback.file_index, playback.frame_index, playback.playing

    def _choose(self, recording: Recording, keep: int | None = None) -> np.ndarray:
        """A file's worlds to draw: as the count or the ids choose them, with
        ``keep`` in place of the best world of its band; by default the world
        the user picked, when the file holds it."""
        if keep is None and self._picked is not None:
            picked = np.flatnonzero(recording.world_ids == self._picked)
            keep = int(picked[0]) if len(picked) else None
        return choose_worlds(recording, self.settings.worlds, self.ids, keep)

    def _highlightable(self) -> np.ndarray:
        """The worlds the highlight steps through, best first: every world of
        the file, drawn or not, or only the worlds chosen by id."""
        if self.ids is not None:
            return self.scene.worlds
        return np.argsort(-self.scene.recording.score, kind="stable")

    def _step_highlight(self, direction: int) -> None:
        """Highlight the next or previous world by rank, round from the end."""
        if self.playback is None:
            return
        worlds = self._highlightable()
        shown = self.scene.worlds[self.scene.highlight]
        place = int(np.flatnonzero(worlds == shown)[0]) + direction
        self._highlight(int(worlds[place % len(worlds)]))

    def _highlight(self, world: int) -> None:
        """Highlight a world of the file, keep it highlighted in other files,
        and bring the camera to it.

        A world that is not drawn takes the place of the best world of its
        band, on the same composite, so the count drawn stays as set. The
        camera keeps its distance and angle, and goes on following the
        highlight if it did; the world just picked may lie far from where it
        looked.
        """
        scene = self.scene
        self._picked = int(scene.recording.world_ids[world])
        worlds = self._choose(scene.recording, keep=world)
        if not np.array_equal(worlds, scene.worlds):
            scene.show(scene.recording, worlds, scene.frame_index)
            self.renderer.show(scene)  # another picture, the same model
        scene.set_highlight(int(np.flatnonzero(worlds == world)[0]))
        self.renderer.centre_on_highlight()

    def _reset_view(self) -> None:
        """Frame every drawn world again, following the highlight if the
        settings say so, as when a file opens; the empty world's own view
        without a file."""
        if self.playback is None:
            _look_at_empty_world(self.renderer)
            return
        self.renderer.frame_all()
        self.renderer.set_follow(self.settings.follow)

    def _toggle_follow(self) -> None:
        """Follow the highlighted world, or stop, and remember that for the
        next runs. Panning stops following for the moment only; the next file
        that opens, or the view reset, follows again."""
        if self.playback is None:  # nothing to follow
            return
        follow = not self.renderer.follow
        self.renderer.set_follow(follow)
        if follow != self.settings.follow:
            self._change(replace(self.settings, follow=follow))

    def _centre(self) -> None:
        if self.playback is not None:
            self.renderer.centre_on_highlight()

    def _toggle_markers(self) -> None:
        self.renderer.markers_visible = not self.renderer.markers_visible

    def _toggle_setup(self) -> None:
        self.setup = not self.setup

    def _toggle_help(self) -> None:
        self.help = not self.help

    def _ask_to_quit(self) -> None:
        self.asking = True

    def _stay(self) -> None:
        self.asking = False

    def _cache(self):
        return user_folder("cache") if self.settings.cache else None

    def _open_picker(self) -> None:
        if self._picker is None:
            self._picker = FilePicker()

    def _take_files(self) -> None:
        """Open what the picker chose or what was dropped, once there is any.

        Several files open in tabs in the order of their names, which is what
        a run's numbered files need, whatever order the system hands them
        over in.
        """
        if self._picker is not None:
            try:
                paths = self._picker.poll()
            except RuntimeError as error:
                self._picker = None
                self._say(str(error))
                return
            if paths is not None:
                self._picker = None
                self.open_files(in_name_order(paths))
        if self._dropped:
            paths, self._dropped = self._dropped, []
            self.open_files(in_name_order(paths))

    def _report(self, problems: list[str]) -> None:
        """Tell the first problem, and how many more files it left out."""
        if problems:
            more = len(problems) - 1
            left = f" (and {more} more file{'s' * (more > 1)} left out)" if more else ""
            self._say(problems[0] + left)

    def _say(self, message: str) -> None:
        self._message = message
        self._message_until = time.monotonic() + MESSAGE_SECONDS
        self._dirty = True

    def _say_now(self, message: str) -> None:
        """Show a message at once, before a step that keeps the window busy."""
        self._say(message)
        self._draw()

    def _rate(self) -> str:
        """The frame-rate line: how long a frame takes to draw, and what it allows."""
        if not self.settings.frame_rate:
            return ""
        if not self._draw_seconds:
            return "measuring ..."
        seconds = sum(self._draw_seconds) / len(self._draw_seconds)
        rate = f"{1 / seconds:.0f}" if seconds <= 0.1 else f"{1 / seconds:.1f}"
        return f"draw {1e3 * seconds:.1f} ms | up to {rate} frames/s"

    def _wait(self) -> float:
        """How long the loop may sleep before something has to be looked at."""
        if self._dirty:
            return 0.0
        now = time.monotonic()
        waits = [IDLE_WAIT]
        if self.playback is not None:
            waits.append(self.playback.seconds_to_next_frame(now))
        if self._flash:
            waits.append(self._flash_until - now)
        if self._message:
            waits.append(self._message_until - now)
        if self._picker is not None:
            waits.append(0.1)
        return max(0.0, min(waits))

    def _pixels(self, x: float, y: float) -> tuple[float, float]:
        """A cursor position as framebuffer pixels from the bottom left."""
        width, height = glfw.get_window_size(self.window)
        frame_width, frame_height = glfw.get_framebuffer_size(self.window)
        scale_x = frame_width / width if width else 1.0
        scale_y = frame_height / height if height else 1.0
        return x * scale_x, frame_height - y * scale_y

    def _over_panel(self, x: float) -> bool:
        return self.settings.panel and x < self._left

    def _over_tabs(self, y: float) -> bool:
        height = glfw.get_framebuffer_size(self.window)[1]
        return self._top > 0 and y >= height - self._top

    def _on_timeline(self, x: float, y: float) -> bool:
        """Whether a point is on the timeline, or just above it."""
        line = self.renderer.context.charHeight
        timeline = self.renderer.timeline_height
        loaded = self.playback is not None and self.settings.overlay
        return loaded and timeline > 0 and x >= self._inset and y < timeline + line // 2

    def _seek(self, x: float) -> None:
        width = glfw.get_framebuffer_size(self.window)[0]
        area = width - self._inset
        self.playback.seek(self.renderer.frame_at(x, self._inset, area))
        self._dirty = True

    def _pick(self, x: float, y: float) -> None:
        """Highlight the world under a point of the scene, if any."""
        width, height = glfw.get_framebuffer_size(self.window)
        copy = self.renderer.copy_at(x, y, width, height, self._left, self._top)
        if copy is not None:
            self._highlight(int(self.scene.worlds[copy]))
            self._dirty = True

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        # GLFW sends a key's character right after the key itself.
        self._held, self._shift = action == glfw.REPEAT, bool(mods & glfw.MOD_SHIFT)
        stepping = key in (glfw.KEY_RIGHT, glfw.KEY_LEFT)
        if action == glfw.RELEASE or (action == glfw.REPEAT and not stepping):
            return
        control = mods & glfw.MOD_CONTROL and not mods & glfw.MOD_ALT  # not AltGr
        if self.asking:
            name = QUIT_KEYS.get(key)
        elif control and key == glfw.KEY_TAB:  # the tabs, as in an editor
            name = "previous file" if self._shift else "next file"
        elif control and glfw.get_key_name(key, scancode) in ("w", "W"):
            name = "close file"
        else:
            name = KEYS.get(key)
        if name is not None:
            self._act(name)

    def _on_char(self, window, codepoint: int) -> None:
        """A typed character's action, once per press however long it is held."""
        if self._held:
            return
        characters = QUIT_CHARACTERS if self.asking else CHARACTERS
        name = characters.get(chr(codepoint).lower())
        if name == "next world" and self._shift:
            name = "previous world"
        if name is not None:
            self._act(name)

    def _on_button(self, window, button: int, action: int, mods: int) -> None:
        """A panel button acts on release over it; the scene and timeline on press.

        The position is the cursor as of this event, kept by the cursor
        callback: asking GLFW now would give where it is after events still
        queued, so a click made while a frame was drawing would land where a
        later one did.
        """
        x, y = self._pixels(*self._cursor)
        if action == glfw.RELEASE:
            pressed, self._pressed = self._pressed, None
            self._dragging = self._scrubbing = False
            if pressed is not None and ui.action_at(self.boxes, x, y) == pressed:
                self._act(pressed)
            return
        name = ui.action_at(self.boxes, x, y)
        if name is not None:
            self._pressed = name
            return
        if self.asking or self._over_panel(x) or self._over_tabs(y):
            return  # the question, the panel, and the tabs take the mouse
        left = button == glfw.MOUSE_BUTTON_LEFT
        if left and self._on_timeline(x, y):
            self._scrubbing = True
            self._seek(x)
            return
        now, (then, last_x, last_y) = time.monotonic(), self._last_press
        self._last_press = (now, x, y)
        if left and now - then < DOUBLE_CLICK and abs(x - last_x) + abs(y - last_y) < 8:
            self._pick(x, y)
            return
        self._dragging = True

    def _on_cursor(self, window, x: float, y: float) -> None:
        """Left drag rotates, right drag pans, middle drag zooms; Shift varies."""
        dx, dy = x - self._cursor[0], y - self._cursor[1]
        self._cursor = (x, y)
        pixels = self._pixels(x, y)
        hovered = ui.action_at(self.boxes, *pixels)
        if hovered != self._hovered:
            self._hovered, self._dirty = hovered, True
        if self._scrubbing:
            self._seek(pixels[0])
        if not self._dragging:
            return

        def pressed(button: int) -> bool:
            return glfw.get_mouse_button(window, button) == glfw.PRESS

        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        mouse = mujoco.mjtMouse
        if pressed(glfw.MOUSE_BUTTON_RIGHT):
            action = mouse.mjMOUSE_MOVE_H if shift else mouse.mjMOUSE_MOVE_V
        elif pressed(glfw.MOUSE_BUTTON_LEFT):
            action = mouse.mjMOUSE_ROTATE_H if shift else mouse.mjMOUSE_ROTATE_V
            dy = 0.0 if shift else dy  # Shift turns around the vertical only
        elif pressed(glfw.MOUSE_BUTTON_MIDDLE):
            action = mouse.mjMOUSE_ZOOM
        else:
            return
        height = glfw.get_window_size(window)[1]
        if height:
            self.renderer.move_camera(action, dx / height, dy / height)
            self._dirty = True

    def _on_scroll(self, window, x: float, y: float) -> None:
        """The wheel scrolls the panel under the cursor, and zooms elsewhere."""
        if self.asking:
            return
        pixels = self._pixels(*self._cursor)
        if self._over_panel(pixels[0]):
            self.panel.scroll_by(round(-2 * self.renderer.context.charHeight * y))
            self.boxes = self._layout()
        elif self._over_tabs(pixels[1]):  # forward, or a swipe left, goes left
            self.tabs.scroll_by(-1 if (y or -x) > 0 else 1)
            self.boxes = self._layout()
        else:  # forward, away from the user, zooms in
            self.renderer.move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, 0.05 * y)
        self._dirty = True

    def _on_drop(self, window, paths: list[str]) -> None:
        self._dropped.extend(paths)

    def _on_change(self, window, *ignored) -> None:
        self._dirty = True

    def _on_close(self, window) -> None:
        """The window's close button asks first, as Q and Esc do."""
        glfw.set_window_should_close(window, False)
        self._act("quit")


def _name_for_the_taskbar() -> None:
    """On Windows, give the process the tool's own name on the taskbar.

    Python runs the tool, and the taskbar would group the window with every
    other Python program, under Python's icon. An executable is a program of
    its own, which the taskbar groups by itself, with the icon it carries; a
    name of its own would only keep its window apart from the program pinned
    to the taskbar, which has no such name.
    """
    if sys.platform != "win32" or getattr(sys, "frozen", False):
        return
    import ctypes

    with contextlib.suppress(AttributeError, OSError):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)


def _give_icon(window) -> None:
    """The application's icon on the window, its taskbar button, and Alt+Tab.

    Windows on macOS and Wayland have no icon of their own, and GLFW warns
    there, which is of no use.
    """
    images = [(size, size, icon.draw(size).tolist()) for size in ICON_SIZES]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glfw.set_window_icon(window, len(images), images)


def _placement(
    size: tuple[int, int] | None,
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """The window's size, and its position when it is free to choose one."""
    monitor = glfw.get_primary_monitor()
    if size is None and monitor:
        left, top, width, height = glfw.get_monitor_workarea(monitor)
        if width > 0 and height > 0:
            size = (int(width * SCREEN_SHARE), int(height * SCREEN_SHARE))
            return size, (left + (width - size[0]) // 2, top + (height - size[1]) // 2)
    return size or (1280, 720), None


def _empty_world() -> Recording:
    return Recording(EMPTY_WORLD, 0.02, np.zeros((1, 1, 0)), title="no recording")


def _where(path: str) -> str:
    """A file's place, the same for every way of writing its path."""
    return os.path.normcase(os.path.realpath(path))


def _look_at_empty_world(renderer: SceneRenderer) -> None:
    renderer.set_follow(False)
    camera = renderer.camera
    camera.lookat[:] = 0.0
    camera.distance, camera.elevation, camera.azimuth = 3.0, -20.0, 120.0


def _font_scale(window) -> int:
    """MuJoCo's font scale for the screen: 100 at normal density, up to 300."""
    density = glfw.get_window_content_scale(window)[0]
    return int(np.clip(50 * math.floor(2 * density + 0.5), 100, 300))
