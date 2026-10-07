"""The interactive window: one GLFW window, its keys, and its mouse.

GLFW opens the window and delivers the input, ``SceneRenderer`` draws into it.
Each turn of the main loop advances playback by wall time, poses the scene,
draws it, and shows the result. Only the ``view`` command imports this module.
The keys and the mouse are listed in docs/design.md.
"""

import time

import glfw
import mujoco
import numpy as np

from mujoco_replay.playback import Playback
from mujoco_replay.recording import Recording
from mujoco_replay.render import SceneRenderer
from mujoco_replay.scene import ComposedScene

FLASH_SECONDS = 1.0


def run(
    recordings: list[Recording],
    worlds: list[np.ndarray],
    seconds_per_frame: float,
    hud: bool = True,
    size: tuple[int, int] = (1280, 720),
) -> None:
    """Open the window and play the recordings until it is closed.

    ``worlds`` holds, per recording, the indices of the worlds to draw, best
    first, as ``choose_worlds`` returns them.
    """
    if not glfw.init():
        raise RuntimeError("GLFW cannot start: the window needs a display")
    try:
        glfw.window_hint(glfw.SAMPLES, 4)
        window = glfw.create_window(*size, "MujocoReplay", None, None)
        if not window:
            raise RuntimeError("GLFW cannot open a window with OpenGL")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        Viewer(window, recordings, worlds, seconds_per_frame, hud).loop()
    finally:
        glfw.terminate()


class Viewer:
    """The state of the window: playback, scene, renderer, and toggles."""

    def __init__(
        self,
        window,
        recordings: list[Recording],
        worlds: list[np.ndarray],
        seconds_per_frame: float,
        hud: bool,
    ) -> None:
        self.window = window
        self.recordings = recordings
        self.worlds = worlds
        self.hud = hud
        self.setup = False
        self.playback = Playback(recordings, seconds_per_frame, time.monotonic())
        self.scene = ComposedScene(recordings[0], worlds[0])
        self.renderer = SceneRenderer(self.scene, font_scale=_font_scale(window))
        self._shown_file = 0
        self._flash = ""
        self._flash_until = 0.0
        self._cursor = glfw.get_cursor_pos(window)
        glfw.set_key_callback(window, self._on_key)
        glfw.set_mouse_button_callback(window, self._on_button)
        glfw.set_cursor_pos_callback(window, self._on_cursor)
        glfw.set_scroll_callback(window, self._on_scroll)
        glfw.set_window_title(window, f"MujocoReplay - {recordings[0].title}")

    def loop(self) -> None:
        self.playback.sync(time.monotonic())
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self._show_events(self.playback.advance(time.monotonic()))
            if self.playback.file_index != self._shown_file:
                self._show_file()
            self.scene.set_frame(self.playback.frame_index)
            width, height = glfw.get_framebuffer_size(self.window)
            if not width or not height:  # minimised
                glfw.wait_events_timeout(0.1)
                continue
            flash = self._flash if time.monotonic() < self._flash_until else ""
            self.renderer.render(
                width,
                height,
                status=self.playback.status(),
                flash=flash,
                hud=self.hud,
                setup=self.setup,
            )
            glfw.swap_buffers(self.window)
        self.renderer.close()

    def _show_file(self) -> None:
        """Show the playlist's current file, keeping the highlighted world."""
        index = self.playback.file_index
        recording, worlds = self.recordings[index], self.worlds[index]
        previous = self.scene
        world_id = previous.recording.world_ids[previous.worlds[previous.highlight]]
        if previous.fits(recording, worlds):
            previous.show(recording, worlds)
            self.renderer.show(previous)  # keeps the context; makes room for markers
        else:
            self.scene = ComposedScene(recording, worlds)
            self.scene.set_ghosts_visible(previous.ghosts_visible)
            self.renderer.show(self.scene)
            self.renderer.frame_all()
        same = np.flatnonzero(recording.world_ids[worlds] == world_id)
        self.scene.set_highlight(int(same[0]) if len(same) else 0)
        self._shown_file = index
        glfw.set_window_title(self.window, f"MujocoReplay - {recording.title}")
        self.playback.sync(time.monotonic())  # composing may have taken a while

    def _show_events(self, labels: list[str]) -> None:
        if labels:
            self._flash = "  ".join(labels)
            self._flash_until = time.monotonic() + FLASH_SECONDS

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        stepping = key in (glfw.KEY_RIGHT, glfw.KEY_LEFT)
        if action == glfw.RELEASE or (action == glfw.REPEAT and not stepping):
            return
        playback, scene, renderer = self.playback, self.scene, self.renderer
        if key == glfw.KEY_SPACE:
            playback.toggle()
        elif key == glfw.KEY_RIGHT:
            self._show_events(playback.step(1))
        elif key == glfw.KEY_LEFT:
            playback.step(-1)
        elif key == glfw.KEY_UP:
            playback.faster()
        elif key == glfw.KEY_DOWN:
            playback.slower()
        elif key in (glfw.KEY_0, glfw.KEY_KP_0):
            playback.default_speed()
        elif key == glfw.KEY_HOME:
            playback.first_frame()
        elif key == glfw.KEY_END:
            playback.last_frame()
        elif key == glfw.KEY_R:
            playback.restart()
        elif key == glfw.KEY_N:
            playback.next_file()
        elif key == glfw.KEY_P:
            playback.previous_file()
        elif key == glfw.KEY_B:
            scene.set_highlight(scene.highlight + (-1 if mods & glfw.MOD_SHIFT else 1))
        elif key == glfw.KEY_G:
            scene.set_ghosts_visible(not scene.ghosts_visible)
        elif key == glfw.KEY_M:
            renderer.markers_visible = not renderer.markers_visible
        elif key == glfw.KEY_H:
            self.hud = not self.hud
        elif key == glfw.KEY_I:
            self.setup = not self.setup
        elif key == glfw.KEY_C:
            renderer.centre_on_highlight()
        elif key == glfw.KEY_F:
            renderer.set_follow(not renderer.follow)
        elif key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(window, True)

    def _on_button(self, window, button: int, action: int, mods: int) -> None:
        self._cursor = glfw.get_cursor_pos(window)

    def _on_cursor(self, window, x: float, y: float) -> None:
        """Left drag rotates, right drag pans, middle drag zooms; Shift varies."""
        dx, dy = x - self._cursor[0], y - self._cursor[1]
        self._cursor = (x, y)

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
        self.renderer.move_camera(action, dx / height, dy / height)

    def _on_scroll(self, window, x: float, y: float) -> None:
        self.renderer.move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * y)


def _font_scale(window) -> int:
    """MuJoCo's font scale for the screen: 150 at normal density, up to 300."""
    density = glfw.get_window_content_scale(window)[0]
    return int(np.clip(50 * round(3 * density), 100, 300))
