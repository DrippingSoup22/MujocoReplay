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
    window = glfw.create_window(320, 240, "test", None, None)
    glfw.default_window_hints()
    glfw.make_context_current(window)
    yield window
    glfw.destroy_window(window)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """The settings folder, in the test's own folder."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))


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
