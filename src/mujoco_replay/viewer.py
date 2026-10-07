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
import time
from collections import deque
from dataclasses import fields, replace

import glfw
import mujoco
import numpy as np

from mujoco_replay import ui
from mujoco_replay.playback import DEFAULT_SECONDS_PER_FRAME, Playback
from mujoco_replay.recording import Recording, RecordingError, read_recording
from mujoco_replay.render import SceneRenderer
from mujoco_replay.scene import ComposedScene
from mujoco_replay.selection import MAX_WORLDS, choose_worlds, world_counts
from mujoco_replay.settings import RESOLUTIONS, Settings, save_settings, user_folder

FLASH_SECONDS = 1.0
MESSAGE_SECONDS = 6.0
IDLE_WAIT = 0.5  # the longest the loop sleeps between looks at the clock
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
KEYS = {
    glfw.KEY_SPACE: "play",
    glfw.KEY_RIGHT: "step",
    glfw.KEY_LEFT: "back",
    glfw.KEY_UP: "faster",
    glfw.KEY_DOWN: "slower",
    glfw.KEY_0: "default speed",
    glfw.KEY_KP_0: "default speed",
    glfw.KEY_HOME: "first",
    glfw.KEY_END: "last",
    glfw.KEY_R: "restart",
    glfw.KEY_N: "next file",
    glfw.KEY_P: "previous file",
    glfw.KEY_B: "next world",
    glfw.KEY_G: "ghosts",
    glfw.KEY_M: "markers",
    glfw.KEY_H: "overlay",
    glfw.KEY_I: "setup",
    glfw.KEY_C: "centre",
    glfw.KEY_F: "follow",
    glfw.KEY_O: "open",
    glfw.KEY_TAB: "panel",
    glfw.KEY_ESCAPE: "quit",
    glfw.KEY_Q: "quit",
}


def run(
    recordings: list[Recording],
    settings: Settings,
    seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME,
    ids: list[int] | None = None,
    size: tuple[int, int] = (1280, 720),
    hud: bool = True,
) -> None:
    """Open the window, with ``recordings`` if any, until it is closed.

    ``ids``, when given, are the producer world ids to draw instead of the
    best of each rank band; the caller has checked that each file has some.
    ``hud`` off hides the overlay for this run without saving that.
    """
    if not glfw.init():
        raise RuntimeError("GLFW cannot start: the window needs a display")
    try:
        glfw.window_hint(glfw.SAMPLES, 0)  # anti-aliasing is drawn offscreen
        window = glfw.create_window(*size, "MujocoReplay", None, None)
        if not window:
            raise RuntimeError("GLFW cannot open a window with OpenGL")
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
    window goes on drawing while the picker is open.
    """

    def __init__(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._process = subprocess.Popen(
            [sys.executable, "-c", PICKER],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=flags,
        )

    def poll(self) -> list[str] | None:
        """The chosen paths once the picker has closed; ``None`` while it is open."""
        if self._process.poll() is None:
            return None
        output, errors = self._process.communicate()
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
        self._picker: FilePicker | None = None
        self._dropped: list[str] = []
        self._passed: list[str] = []  # events passed, not yet flashed
        self._flash, self._flash_until = "", 0.0
        self._message, self._message_until = "", 0.0
        self._shown_file = 0
        self._dirty = True
        self._cursor = glfw.get_cursor_pos(window)
        self._hovered: str | None = None
        self._dragging = False  # a drag that began on the scene, not the panel
        self._draw_seconds: deque[float] = deque(maxlen=30)
        self._actions = self._make_actions()
        glfw.set_key_callback(window, self._on_key)
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
        inset = ui.Panel.width(line) if self.settings.panel else 6 * line
        loaded = self.playback is not None
        # While the next file is composed, the scene still shows the last one.
        if loaded and self._shown_file == self.playback.file_index:
            self.scene.set_frame(self.playback.frame_index)
        rate, first = self._rate(), not self._draw_seconds
        start = time.perf_counter()
        self.renderer.render(
            width,
            height,
            status=self.playback.status() if loaded else "",
            flash=self._flash,
            hud=loaded and self.settings.overlay,
            setup=loaded and self.setup,
            corner=rate,
            inset=inset,
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
            speed = f"{playback.seconds_per_frame:.3g} s/frame"
            rows += [
                ui.Section("Playback"),
                ui.Buttons(controls),
                ui.Stepper("Speed", speed, "slower", "faster"),
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
        if playback is not None:
            rank = f"rank {self._rank() + 1}"
            rows += [
                ui.Stepper("Highlight", rank, "previous world", "next world"),
                ui.Switch("Ghosts", self.scene.ghosts_visible, "ghosts"),
                ui.Switch("Markers", self.renderer.markers_visible, "markers"),
                ui.Switch("Follow highlight", self.renderer.follow, "follow"),
            ]
        mode = settings.mode
        choices = (
            ("Quality", "quality", mode == "quality"),
            ("Performance", "performance", mode == "performance"),
        )
        resolution = f"{graphics.resolution}%"
        rows += [
            ui.Section("Graphics" + (" (custom)" if mode == "custom" else "")),
            ui.Choice(choices),
            ui.Switch("Shadows", graphics.shadows, "shadows"),
            ui.Switch("Reflections", graphics.reflections, "reflections"),
            ui.Switch("Anti-aliasing", graphics.antialiasing, "antialiasing"),
            ui.Switch("Fine shapes", graphics.fine_shapes, "fine shapes"),
            ui.Stepper(
                "Resolution", resolution, "lower resolution", "higher resolution"
            ),
            ui.Section("Display"),
            ui.Switch("Overlay", settings.overlay, "overlay"),
            ui.Switch("Setup info", self.setup, "setup"),
            ui.Switch("Frame rate", settings.frame_rate, "frame rate"),
            ui.Switch("Cache scenes", settings.cache, "cache"),
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
            "previous world": lambda: self._highlight(-1),
            "next world": lambda: self._highlight(1),
            "ghosts": self._toggle_ghosts,
            "markers": self._toggle_markers,
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
        if settings.worlds != previous.worlds and self.playback is not None:
            self._recompose()
        if settings.frame_rate != previous.frame_rate:
            self._draw_seconds.clear()

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

    def _recompose(self) -> None:
        """Choose and compose the worlds again after the count changed."""
        previous = self.scene
        world_id = previous.recording.world_ids[previous.worlds[previous.highlight]]
        count = self.settings.worlds
        self.worlds = [
            choose_worlds(recording, count, self.ids) for recording in self.recordings
        ]
        index = self.playback.file_index
        self._say_now(f"Composing {len(self.worlds[index])} worlds ...")
        scene = ComposedScene(self.recordings[index], self.worlds[index], self._cache())
        self._use(scene, reframe=False, world_id=world_id)
        self._message = ""
        self.playback.sync(time.monotonic())  # composing may have taken a while

    def _show_file(self) -> None:
        """Show the playlist's current file, keeping the highlighted world.

        A file whose model fails is left out of the playlist, and playback
        goes back, paused, to the frame shown before.
        """
        index = self.playback.file_index
        recording, worlds = self.recordings[index], self.worlds[index]
        previous = self.scene
        world_id = previous.recording.world_ids[previous.worlds[previous.highlight]]
        if previous.fits(recording, worlds):
            previous.show(recording, worlds)
            self._use(previous, reframe=False, world_id=world_id)
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
            self._use(scene, reframe=True, world_id=world_id)
            self._message = ""
        self.playback.sync(time.monotonic())  # composing may have taken a while

    def _use(
        self, scene: ComposedScene, reframe: bool, world_id: int | None = None
    ) -> None:
        """Show ``scene``, carrying over the ghosts and the highlighted world."""
        scene.set_ghosts_visible(self.scene.ghosts_visible)
        self.scene = scene
        self.renderer.show(scene)
        if reframe:
            self.renderer.frame_all()
        drawn = scene.recording.world_ids[scene.worlds]
        same = np.flatnonzero(drawn == world_id) if world_id is not None else []
        scene.set_highlight(int(same[0]) if len(same) else 0)
        self._shown_file = self.playback.file_index
        title = f"MujocoReplay - {scene.recording.title}"
        glfw.set_window_title(self.window, title)
        self._draw_seconds.clear()  # another scene costs another time
        self._dirty = True

    def _position(self) -> tuple[int, int, bool]:
        """What the drawing shows of playback: the file, the frame, any pause."""
        playback = self.playback
        return playback.file_index, playback.frame_index, playback.playing

    def _highlight(self, direction: int) -> None:
        if self.playback is not None:
            self.scene.set_highlight(self.scene.highlight + direction)

    def _toggle_ghosts(self) -> None:
        self.scene.set_ghosts_visible(not self.scene.ghosts_visible)

    def _toggle_markers(self) -> None:
        self.renderer.markers_visible = not self.renderer.markers_visible

    def _toggle_setup(self) -> None:
        self.setup = not self.setup

    def _rank(self) -> int:
        """The highlighted world's rank among all the file's worlds, from 0."""
        recording, world = self.scene.recording, self.scene.worlds[self.scene.highlight]
        order = np.argsort(-recording.score, kind="stable")
        return int(np.flatnonzero(order == world)[0])

    def _cache(self):
        return user_folder("cache") if self.settings.cache else None

    def _open_picker(self) -> None:
        if self._picker is None:
            self._picker = FilePicker()

    def _take_files(self) -> None:
        """Open what the picker chose or what was dropped, once there is any."""
        if self._picker is not None:
            try:
                paths = self._picker.poll()
            except RuntimeError as error:
                self._picker = None
                self._say(str(error))
                return
            if paths is not None:
                self._picker = None
                self.open_files(paths)
        if self._dropped:
            paths, self._dropped = self._dropped, []
            self.open_files(paths)

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
        return f"draw {1e3 * seconds:.1f} ms | up to {1 / seconds:.0f} frames/s"

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

    def _panel_width(self) -> int:
        return ui.Panel.width(self.renderer.context.charHeight)

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        stepping = key in (glfw.KEY_RIGHT, glfw.KEY_LEFT)
        if action == glfw.RELEASE or (action == glfw.REPEAT and not stepping):
            return
        name = KEYS.get(key)
        if name == "next world" and mods & glfw.MOD_SHIFT:
            name = "previous world"
        if name is not None:
            self._act(name)

    def _on_button(self, window, button: int, action: int, mods: int) -> None:
        # The cursor as of this event, kept by the cursor callback: asking GLFW
        # now would give where it is after events still queued, so a click made
        # while a frame was drawing would land where a later one did.
        self._dragging = False
        if action != glfw.PRESS:
            return
        x, y = self._pixels(*self._cursor)
        name = ui.action_at(self.boxes, x, y)
        if name is not None:
            self._act(name)
            return
        self._dragging = not (self.settings.panel and x < self._panel_width())

    def _on_cursor(self, window, x: float, y: float) -> None:
        """Left drag rotates, right drag pans, middle drag zooms; Shift varies."""
        dx, dy = x - self._cursor[0], y - self._cursor[1]
        self._cursor = (x, y)
        hovered = ui.action_at(self.boxes, *self._pixels(x, y))
        if hovered != self._hovered:
            self._hovered, self._dirty = hovered, True
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
        over_panel = self._pixels(*self._cursor)[0] < self._panel_width()
        if self.settings.panel and over_panel:
            self.panel.scroll_by(round(-2 * self.renderer.context.charHeight * y))
            self.boxes = self._layout()
        else:
            self.renderer.move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * y)
        self._dirty = True

    def _on_drop(self, window, paths: list[str]) -> None:
        self._dropped.extend(paths)

    def _on_change(self, window, *ignored) -> None:
        self._dirty = True


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
