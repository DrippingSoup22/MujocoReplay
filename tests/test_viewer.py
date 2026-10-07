"""Tests for the window: files dropped and played, the playlist, the settings."""

import glfw
import numpy as np
import pytest

from mujoco_replay.recording import Recording, write_recording
from mujoco_replay.settings import Settings, load_settings
from mujoco_replay.viewer import Viewer


@pytest.fixture
def window(opengl):
    """A hidden window with a current OpenGL context."""
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(640, 720, "test", None, None)
    glfw.default_window_hints()
    glfw.make_context_current(window)
    yield window
    glfw.destroy_window(window)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """The settings folder, in the test's own folder, on every system."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))


def test_dropped_recordings_play_with_the_settings_count_of_worlds(
    window, make_recording, tmp_path
):
    paths = []
    for index in range(2):
        path = tmp_path / f"cycle_{index}.npz"
        write_recording(path, make_recording(frames=2, worlds=6, seed=index))
        paths.append(str(path))
    viewer = Viewer(window, Settings(worlds=4, cache=False), 0.3)

    viewer._on_drop(window, paths)
    viewer._take_files()

    assert len(viewer.recordings) == 2 and viewer.playback.file_index == 0
    assert len(viewer.scene.worlds) == 4
    viewer.renderer.close()


def test_a_dropped_file_that_is_no_recording_is_told_and_changes_nothing(
    window, tmp_path
):
    notes = tmp_path / "notes.npz"
    notes.write_text("not a recording")
    viewer = Viewer(window, Settings(cache=False), 0.3)

    viewer._on_drop(window, [str(notes)])
    viewer._take_files()

    assert viewer.playback is None
    assert "cannot be read as a recording" in viewer._message
    viewer.renderer.close()


def test_stepping_back_into_a_file_that_needs_composing_shows_it(
    window, make_recording
):
    first = make_recording(frames=5, worlds=3, seed=1)
    second = make_recording(frames=2, worlds=1, seed=2)  # another composite
    viewer = Viewer(window, Settings(worlds=4, cache=False), 0.3)
    viewer.load([first, second])
    viewer.playback.next_file()
    viewer._show_file()

    viewer.playback.step(-1)  # to the first file's last frame
    viewer._show_file()

    assert viewer.scene.recording is first and viewer.scene.frame_index == 0
    viewer._draw()
    assert viewer.scene.frame_index == 4
    viewer.renderer.close()


def test_a_later_file_whose_model_fails_is_left_out_of_the_playlist(
    window, make_recording, small_model
):
    good = make_recording(frames=3, worlds=3)
    bad = Recording(small_model, 0.02, np.zeros((2, 3, 10)), title="bad")
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer.load([good, bad])
    viewer.playback.last_frame()
    viewer._draw()

    viewer.playback.step(1)  # into the bad file
    viewer._show_file()

    assert viewer.playback.recordings == [good] and viewer.scene.recording is good
    assert viewer.playback.frame_index == 2 and not viewer.playback.playing
    assert "bad: qpos has 10 positions" in viewer._message
    viewer.renderer.close()


def test_a_hidden_overlay_for_one_run_is_not_saved_with_other_changes(window, saved):
    viewer = Viewer(window, Settings(cache=False), 0.3, hud=False)

    viewer._act("shadows")

    assert not viewer.settings.overlay
    assert load_settings().overlay and load_settings().graphics.shadows
    viewer.renderer.close()


def test_fewer_worlds_from_a_count_between_steps_goes_to_the_step_below(window, saved):
    viewer = Viewer(window, Settings(worlds=20, cache=False), 0.3)

    viewer._act("fewer")

    assert viewer.settings.worlds == load_settings().worlds == 16
    viewer.renderer.close()


def press(viewer, window, x: float, y: float, action) -> None:
    """A left click's press or release at framebuffer pixels from the bottom left."""
    viewer._on_cursor(window, x, glfw.get_framebuffer_size(window)[1] - y)
    viewer._on_button(window, glfw.MOUSE_BUTTON_LEFT, action, 0)


def test_a_panel_button_acts_on_release_over_it_and_not_after_a_drag_away(
    window, saved
):
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer._draw()
    box = next(box for box in viewer.boxes if box.action == "quality")
    middle = (box.x + box.width / 2, box.y + box.height / 2)

    press(viewer, window, *middle, glfw.PRESS)
    press(viewer, window, 600, 600, glfw.RELEASE)  # dragged off the button
    assert viewer.settings.mode == "performance"
    press(viewer, window, *middle, glfw.PRESS)
    press(viewer, window, *middle, glfw.RELEASE)
    assert viewer.settings.mode == "quality"
    viewer.renderer.close()


def test_the_next_file_highlights_its_best_world_unless_one_was_picked(
    window, make_recording
):
    ids = [10, 11, 12]
    first = make_recording(worlds=3, score=[3, 2, 1], world_ids=ids, seed=1)
    second = make_recording(worlds=3, score=[1, 2, 3], world_ids=ids, seed=2)
    viewer = Viewer(window, Settings(worlds=3, cache=False), 0.3)
    viewer.load([first, second])

    def highlighted() -> int:
        scene = viewer.scene
        return int(scene.recording.world_ids[scene.worlds[scene.highlight]])

    viewer.playback.next_file()
    viewer._show_file()
    assert highlighted() == 12  # the second file's best
    viewer._act("next world")
    viewer.playback.previous_file()
    viewer._show_file()
    assert highlighted() == 11  # the picked world, in the first file too
    viewer.renderer.close()


def test_a_click_on_the_timeline_pauses_at_that_frame(window, make_recording):
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer.load([make_recording(frames=10)])
    viewer._draw()
    left = viewer._inset
    width = glfw.get_framebuffer_size(window)[0]

    press(viewer, window, left + 0.75 * (width - left), 2, glfw.PRESS)

    assert viewer.playback.frame_index == 7 and not viewer.playback.playing
    viewer.renderer.close()


def test_switching_the_frame_rate_on_times_a_frame_even_when_nothing_moved(
    window, saved
):
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer._draw()

    viewer._act("frame rate")
    viewer._draw()  # the picture is the same, but there is a frame to time

    assert len(viewer._draw_seconds) == 1 and viewer._dirty  # shown at once
    viewer.renderer.close()


def test_a_held_letter_acts_once_and_shift_b_goes_back_whatever_the_case(
    window, make_recording
):
    viewer = Viewer(window, Settings(worlds=3, cache=False), 0.3)
    viewer.load([make_recording(worlds=3)])

    def type_b(action, mods) -> int:
        viewer._on_key(window, glfw.KEY_B, 0, action, mods)
        viewer._on_char(window, ord("B"))  # a capital, as with Shift or Caps Lock
        return viewer.scene.highlight

    assert type_b(glfw.PRESS, glfw.MOD_SHIFT) == 2  # Shift+B: back, from 0 to 2
    assert type_b(glfw.REPEAT, glfw.MOD_SHIFT) == 2  # held: no more
    assert type_b(glfw.PRESS, 0) == 0  # Caps Lock alone: forward
    viewer.renderer.close()
