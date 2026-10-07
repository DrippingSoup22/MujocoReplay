"""The application window: the scene, the side panel, the keys, and the mouse.

The window opens on an empty flat world and plays recordings once they are
opened: from the command line, with the panel's Open button (the system's file
picker), or by dropping files onto the window. The panel holds the settings,
which are saved as they change. Each turn of the main loop advances playback
by wall time and draws only when something changed, so a paused window costs
next to nothing. GLFW opens the window and delivers the input; only the
``view`` command imports this module. docs/design.md lists the keys.
"""

import math
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import fields, replace

import glfw
import mujoco
import numpy as np

from mujoco_replay import ui
from mujoco_replay.playback import DEFAULT_SECONDS_PER_FRAME, Playback
from mujoco_replay.recording import (
    Recording,
    RecordingError,
    in_name_order,
    read_recording,
)
from mujoco_replay.render import SceneRenderer, setup_lines, world_rank
from mujoco_replay.scene import ComposedScene
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
# The file picker, run in a process of its own; it prints the chosen paths.
PICKER = """
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
import tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
paths = filedialog.askopenfilenames(
    parent=root,
    title="Open recordings",
    filetypes=[("Recordings", "*.npz"), ("All files", "*.*")],
)
print("\\n".join(paths))
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
# Keys by the character they type, so that they follow the keyboard's layout;
# a capital letter does what its small letter does, unless listed.
CHARACTERS = {
    "0": "default speed",
    "r": "restart",
    "n": "next file",
    "p": "previous file",
    "b": "next world",
    "B": "previous world",
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
# What F1 shows.
HELP = [
    "Space: play or pause",
    "Left, Right: one frame back, forward",
    "Up, Down: faster, slower; 0: default speed",
    "Home, End: first, last frame; R: restart",
    "N, P: next, previous file",
    "B, Shift+B: next, previous world",
    "Double-click a world: highlight it",
    "G: ghosts; M: markers",
    "V: reset the view; T: look from above",
    "C: centre on the highlight; F: follow it",
    "Left drag: rotate; with Shift: turn only",
    "Right drag: pan; wheel or middle drag: zoom",
    "Click the timeline: go to that frame",
    "H: overlay; I: setup; O: open recordings",
    "Tab: panel; F1 or ?: these keys; Q, Esc: quit",
]


def run(
    recordings: list[Recording],
    settings: Settings,
    seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME,
    ids: list[int] | None = None,
    size: tuple[int, int] | None = None,
    hud: bool = True,
) -> None:
    """Open the window, with ``recordings`` if any, until it is closed.

    ``ids``, when given, are the producer world ids to draw instead of the
    best of each rank band; the caller has checked that each file has some.
    Without ``size``, the window takes most of the screen, in its middle.
    ``hud`` off hides the overlay for this run without saving that.
    """
    if not glfw.init():
        raise RuntimeError("GLFW cannot start: the window needs a display")
    try:
        glfw.window_hint(glfw.SAMPLES, 0)  # anti-aliasing is drawn offscreen
        glfw.window_hint(glfw.VISIBLE, False)  # shown once in its place
        size, position = _placement(size)
        window = glfw.create_window(*size, "MujocoReplay", None, None)
        if not window:
            raise RuntimeError("GLFW cannot open a window with OpenGL")
        if position is not None:
            glfw.set_window_pos(window, *position)
        glfw.show_window(window)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        viewer = Viewer(window, settings, seconds_per_frame, hud)
        if recordings:
            viewer.load(recordings, ids)
        viewer.loop()
    finally:
        glfw.terminate()


class FilePicker:
    """The system's file picker, shown by tkinter in a process of its own.

    Its own process keeps tkinter's event loop apart from GLFW's, and the
    window goes on drawing while the picker is open. The process writes into
    files rather than pipes: a pipe it filled before exiting would hold it
    open for good, and the files take any path as UTF-8.
    """

    def __init__(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._output, self._errors = (
            tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
            for _ in range(2)
        )
        self._process = subprocess.Popen(
            [sys.executable, "-c", PICKER],
            stdout=self._output,
            stderr=self._errors,
            creationflags=flags,
        )

    def poll(self) -> list[str] | None:
        """The chosen paths once the picker has closed; ``None`` while it is open."""
        if self._process.poll() is None:
            return None
        output, errors = (self._read(file) for file in (self._output, self._errors))
        if self._process.returncode:
            reason = (errors.strip().splitlines() or ["no reason given"])[-1]
            raise RuntimeError(
                f"The file picker could not open ({reason}); "
                "drop recordings onto the window instead."
            )
        return [line for line in output.splitlines() if line.strip()]

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._output.close()
        self._errors.close()

    @staticmethod
    def _read(file) -> str:
        file.seek(0)
        text = file.read()
        file.close()
        return text


class Viewer:
    """The state of the window: files, playback, scene, renderer, and panel."""

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
        self.recordings: list[Recording] = []
        self.worlds: list[np.ndarray] = []
        self.ids: list[int] | None = None
        self.playback: Playback | None = None
        self.scene = ComposedScene(_empty_world(), np.arange(1))
        self.renderer = SceneRenderer(
            self.scene, font_scale=_font_scale(window), graphics=settings.graphics
        )
        _look_at_empty_world(self.renderer)
        self.panel = ui.Panel()
        self.boxes: list[ui.Box] = []
        self.setup = False
        self.help = False
        self._picker: FilePicker | None = None
        self._dropped: list[str] = []
        self._passed: list[str] = []  # events passed, not yet flashed
        self._flash, self._flash_until = "", 0.0
        self._message, self._message_until = "", 0.0
        self._shown_file = 0
        self._picked: int | None = None  # the world id the user highlighted
        self._shown_ghosts = (
            "normal" if settings.ghosts == "hidden" else settings.ghosts
        )
        self._dirty = True
        self._cursor = glfw.get_cursor_pos(window)
        self._hovered: str | None = None
        self._pressed: str | None = None  # the panel action a press began on
        self._dragging = False  # a drag that began on the scene, not the panel
        self._scrubbing = False  # a drag that began on the timeline
        self._last_press = (-math.inf, 0.0, 0.0)  # its time and place
        self._left = self._inset = 0  # where the scene and the overlay begin
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
                if self.playback.file_index != self._shown_file:
                    self._show_file()
                self._dirty |= self._position() != before
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

    def load(self, recordings: list[Recording], ids: list[int] | None = None) -> None:
        """Play ``recordings`` from the first; a model that fails leaves all as was."""
        count = self.settings.worlds
        worlds = [choose_worlds(recording, count, ids) for recording in recordings]
        self._say_now(f"Composing {len(worlds[0])} worlds ...")
        scene = ComposedScene(recordings[0], worlds[0], self._cache())
        self.recordings, self.worlds, self.ids = recordings, worlds, ids
        speed = self.seconds_per_frame
        if self.playback is not None:
            speed = self.playback.seconds_per_frame
        self.playback = Playback(recordings, speed, time.monotonic())
        self._picked = None
        self._use(scene, reframe=True)
        self._message = ""

    def open_files(self, paths: list[str]) -> None:
        """Read and play the files at ``paths``; a bad file is reported, not played."""
        if not paths:
            return
        try:
            self.load([read_recording(path) for path in paths])
        except RecordingError as error:
            self._say(str(error))

    def _draw(self) -> None:
        self._dirty = False
        width, height = glfw.get_framebuffer_size(self.window)
        if not width or not height:  # minimised; redrawn when restored
            return
        context = self.renderer.context
        line = context.charHeight
        panel = ui.Panel.width(line) if self.settings.panel else 0
        self._left, self._inset = panel, panel or 6 * line
        loaded = self.playback is not None
        # While the next file is composed, the scene still shows the last one.
        if loaded and self._shown_file == self.playback.file_index:
            self.scene.set_frame(self.playback.frame_index)
        side = None
        if self.help:
            side = HELP
        elif loaded and self.setup:
            side = setup_lines(self.scene.recording.setup) or ["(no setup)"]
        rate, first = self._rate(), not self._draw_seconds
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
            hint="" if loaded else "Open recordings (O), or drop .npz files here",
        )
        hover = self._pixels(*self._cursor)
        self.boxes = self._layout()
        self.panel.draw(self.boxes, height, context, hover, self.settings.panel)
        if self.settings.frame_rate:  # wait for the graphics card, to time it
            mujoco.mjr_finish()
            self._draw_seconds.append(time.perf_counter() - start)
            self._dirty |= first  # show the first time at once, even paused
        glfw.swap_buffers(self.window)

    def _layout(self) -> list[ui.Box]:
        """The panel's boxes as things stand, or the button that shows it."""
        height = glfw.get_framebuffer_size(self.window)[1]
        context = self.renderer.context
        line = context.charHeight
        if self.settings.panel:

            def measure(text: str) -> int:
                return ui.text_width(context, text)

            return self.panel.layout(self._rows(), height, line, measure)
        row = line + line // 2
        top = height - line // 2 - row
        return [ui.Box(line // 2, top, 5 * line, row, "button", "Panel", "panel")]

    def _rows(self) -> list[ui.Row]:
        """The panel as it stands, top to bottom."""
        settings, graphics = self.settings, self.settings.graphics
        rows: list[ui.Row] = [
            ui.Title("MujocoReplay"),
            ui.Buttons((("Open recordings ...", "open"),)),
        ]
        playback = self.playback
        if playback is None:
            rows.append(ui.Note("or drop .npz files on the window"))
        else:
            play = "Pause" if playback.playing else "Play"
            controls = (("|<", "first"), ("<", "back"), (play, "play"))
            controls += ((">", "step"), (">|", "last"))
            per_frame = f"{playback.seconds_per_frame:.3g} s"
            rows += [
                ui.Section("Playback"),
                ui.Buttons(controls),
                ui.Stepper("Per frame", per_frame, "faster", "slower"),
            ]
            if len(self.recordings) > 1:
                where = f"{playback.file_index + 1} of {len(self.recordings)}"
                rows.append(ui.Stepper("File", where, "previous file", "next file"))
        if playback is None:
            shown = str(settings.worlds)
        else:
            shown = f"{len(self.scene.worlds)} of {self.scene.recording.world_count}"
        rows.append(ui.Section("Worlds"))
        if self.ids is None:
            rows.append(ui.Stepper("Shown", shown, "fewer", "more"))
        else:  # the worlds named on the command line, whatever the count
            rows.append(ui.Note(f"Shown: {shown}, chosen by id"))
        if playback is not None and len(self.scene.worlds) > 1:
            world = self.scene.worlds[self.scene.highlight]
            rank = f"rank {world_rank(self.scene.recording, world)[0]:,}"
            rows += [
                ui.Stepper("Highlight", rank, "previous world", "next world"),
                ui.Stepper("Ghosts", settings.ghosts, "fainter", "stronger"),
            ]
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
            "previous file": playing(lambda playback: playback.previous_file()),
            "next file": playing(lambda playback: playback.next_file()),
            "fewer": lambda: self._step_worlds(-1),
            "more": lambda: self._step_worlds(1),
            "previous world": lambda: self._highlight(self.scene.highlight - 1),
            "next world": lambda: self._highlight(self.scene.highlight + 1),
            "ghosts": self._toggle_ghosts,
            "fainter": lambda: self._step_ghosts(-1),
            "stronger": lambda: self._step_ghosts(1),
            "markers": self._toggle_markers,
            "reset view": self.renderer.frame_all,
            "top view": self.renderer.look_from_above,
            "follow": lambda: self.renderer.set_follow(not self.renderer.follow),
            "centre": self.renderer.centre_on_highlight,
            "quality": lambda: self._change(self.settings.with_mode("quality")),
            "performance": lambda: self._change(self.settings.with_mode("performance")),
            "shadows": graphics("shadows"),
            "reflections": graphics("reflections"),
            "antialiasing": graphics("antialiasing"),
            "fine shapes": graphics("fine_shapes"),
            "lower resolution": lambda: self._step_resolution(-1),
            "higher resolution": lambda: self._step_resolution(1),
            "overlay": setting("overlay"),
            "frame rate": setting("frame_rate"),
            "cache": setting("cache"),
            "panel": setting("panel"),
            "setup": self._toggle_setup,
            "keys": self._toggle_help,
            "quit": lambda: glfw.set_window_should_close(self.window, True),
        }

    def _act(self, action: str) -> None:
        self._actions[action]()
        # Clicks still queued must meet the panel as it now is, not as drawn.
        self.boxes = self._layout()
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
        count = self.settings.worlds
        self.worlds = [
            choose_worlds(recording, count, self.ids) for recording in self.recordings
        ]
        index = self.playback.file_index
        self._say_now(f"Composing {len(self.worlds[index])} worlds ...")
        scene = ComposedScene(self.recordings[index], self.worlds[index], self._cache())
        self._use(scene, reframe=False)
        self._message = ""
        self.playback.sync(time.monotonic())  # composing may have taken a while

    def _show_file(self) -> None:
        """Show the playlist's current file, with the world the user highlighted.

        A file whose model fails is left out of the playlist, and playback
        goes back, paused, to the frame shown before.
        """
        index = self.playback.file_index
        recording, worlds = self.recordings[index], self.worlds[index]
        previous = self.scene
        if previous.fits(recording, worlds):
            previous.show(recording, worlds)
            self._use(previous, reframe=False)
        else:
            self._say_now(f"Composing {len(worlds)} worlds ...")
            try:
                scene = ComposedScene(recording, worlds, self._cache())
            except RecordingError as error:
                del self.recordings[index]  # the playback's playlist too
                del self.worlds[index]
                if index < self._shown_file:
                    self._shown_file -= 1
                self.playback.file_index = self._shown_file
                self.playback.frame_index = previous.frame_index
                self.playback.playing = False
                self._say(f"{error} (left out of the playlist)")
                return
            self._use(scene, reframe=True)
            self._message = ""
        self.playback.sync(time.monotonic())  # composing may have taken a while

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
        drawn = scene.recording.world_ids[scene.worlds]
        same = np.flatnonzero(drawn == self._picked) if self._picked is not None else []
        scene.set_highlight(int(same[0]) if len(same) else 0)
        self._shown_file = self.playback.file_index
        title = f"MujocoReplay - {scene.recording.title}"
        # A file name's undecodable bytes become lone surrogates, which GLFW refuses.
        glfw.set_window_title(self.window, title.encode(errors="replace").decode())
        self._draw_seconds.clear()  # another scene costs another time
        self._dirty = True

    def _position(self) -> tuple[int, int, bool]:
        """What the drawing shows of playback: the file, the frame, any pause."""
        playback = self.playback
        return playback.file_index, playback.frame_index, playback.playing

    def _highlight(self, copy: int) -> None:
        """Highlight a drawn world, and keep it highlighted in the next files."""
        if self.playback is not None:
            self.scene.set_highlight(copy)
            scene = self.scene
            self._picked = int(scene.recording.world_ids[scene.worlds[scene.highlight]])

    def _toggle_markers(self) -> None:
        self.renderer.markers_visible = not self.renderer.markers_visible

    def _toggle_setup(self) -> None:
        self.setup = not self.setup

    def _toggle_help(self) -> None:
        self.help = not self.help

    def _cache(self):
        return user_folder("cache") if self.settings.cache else None

    def _open_picker(self) -> None:
        if self._picker is None:
            self._picker = FilePicker()

    def _take_files(self) -> None:
        """Open what the picker chose or what was dropped, once there is any.

        Several files play in the order of their names, which is what a run's
        numbered files need, whatever order the system hands them over in.
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
        copy = self.renderer.copy_at(x, y, width, height, self._left)
        if copy is not None:
            self._highlight(copy)
            self._dirty = True

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        stepping = key in (glfw.KEY_RIGHT, glfw.KEY_LEFT)
        if action == glfw.RELEASE or (action == glfw.REPEAT and not stepping):
            return
        name = KEYS.get(key)
        if name is not None:
            self._act(name)

    def _on_char(self, window, codepoint: int) -> None:
        character = chr(codepoint)
        name = CHARACTERS.get(character) or CHARACTERS.get(character.lower())
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
        if self._over_panel(x):
            return
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
        if self._over_panel(self._pixels(*self._cursor)[0]):
            self.panel.scroll_by(round(-2 * self.renderer.context.charHeight * y))
            self.boxes = self._layout()
        else:  # forward, away from the user, zooms in
            self.renderer.move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, 0.05 * y)
        self._dirty = True

    def _on_drop(self, window, paths: list[str]) -> None:
        self._dropped.extend(paths)

    def _on_change(self, window, *ignored) -> None:
        self._dirty = True


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


def _look_at_empty_world(renderer: SceneRenderer) -> None:
    camera = renderer.camera
    camera.lookat[:] = 0.0
    camera.distance, camera.elevation, camera.azimuth = 3.0, -20.0, 120.0


def _font_scale(window) -> int:
    """MuJoCo's font scale for the screen: 100 at normal density, up to 300."""
    density = glfw.get_window_content_scale(window)[0]
    return int(np.clip(50 * math.floor(2 * density + 0.5), 100, 300))
