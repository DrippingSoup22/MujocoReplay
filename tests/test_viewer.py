"""Tests for the window's files: dropped recordings play, a bad file is told."""

import glfw
import pytest

from mujoco_replay.recording import write_recording
from mujoco_replay.settings import Settings
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
