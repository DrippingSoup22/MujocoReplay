"""Tests for the window: files dropped and played, the tabs, the settings."""

import ctypes
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import glfw
import mujoco
import numpy as np
import pytest

from mujoco_replay import cli, ui
from mujoco_replay.recording import Recording, write_recording
from mujoco_replay.settings import Settings, load_settings
from mujoco_replay.viewer import APP_ID, FilePicker, Viewer, _name_for_the_taskbar


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


def test_a_dropped_file_that_is_no_recording_is_told_and_the_others_open(
    window, make_recording, tmp_path
):
    notes, good = tmp_path / "notes.npz", tmp_path / "run.npz"
    notes.write_text("not a recording")
    write_recording(good, make_recording())
    viewer = Viewer(window, Settings(cache=False), 0.3)

    viewer._on_drop(window, [str(notes)])
    viewer._take_files()
    assert viewer.playback is None
    viewer._on_drop(window, [str(notes), str(good)])
    viewer._take_files()

    assert len(viewer.recordings) == 1
    assert "notes.npz cannot be read as a recording" in viewer._message
    viewer.renderer.close()


def test_a_tab_that_needs_composing_shows_its_file_at_the_frame_shown(
    window, make_recording
):
    first = make_recording(frames=5, worlds=3, seed=1)
    second = make_recording(frames=2, worlds=1, seed=2)  # another composite
    viewer = Viewer(window, Settings(worlds=4, cache=False), 0.3)
    viewer.load([first, second])
    viewer.playback.last_frame()
    viewer._draw()

    viewer._act("next file")  # composed while the scene shows the first file
    viewer._show_file()
    viewer._draw()

    assert viewer.scene.recording is second and viewer.scene.frame_index == 1
    viewer.renderer.close()


def test_a_later_file_whose_model_fails_is_closed_and_the_last_comes_back(
    window, make_recording, small_model
):
    good = make_recording(frames=3, worlds=3)
    bad = Recording(small_model, 0.02, np.zeros((2, 3, 10)), title="bad")
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer.load([good, bad])
    viewer.playback.last_frame()
    viewer._draw()

    viewer._act("tab 1")  # the bad file
    problems = viewer._show_file()

    assert viewer.recordings == [good] and viewer.scene.recording is good
    assert viewer.playback.frame_index == 2 and not viewer.playback.playing
    assert len(viewer.paths) == 1  # its tab is closed
    assert len(problems) == 1 and "bad: qpos has 10 positions" in problems[0]
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
    viewer = Viewer(window, Settings(worlds=1, cache=False), 0.3)
    viewer.load([first, second])

    def highlighted() -> int:
        scene = viewer.scene
        return int(scene.recording.world_ids[scene.worlds[scene.highlight]])

    viewer._act("next file")
    viewer._show_file()
    assert highlighted() == 12  # the second file's best
    viewer._act("next world")
    viewer._act("previous file")
    viewer._show_file()
    assert highlighted() == 11  # the picked world, drawn in the first file too
    viewer.renderer.close()


def test_the_highlight_reaches_every_world_each_in_place_of_its_bands_best(
    window, make_recording
):
    ids = [10, 11, 12, 13]  # in rank order; two bands: 10 and 11, 12 and 13
    recording = make_recording(worlds=4, score=[4, 3, 2, 1], world_ids=ids)
    viewer = Viewer(window, Settings(worlds=2, cache=False), 0.3)
    viewer.load([recording])
    model = viewer.scene.model

    def shown() -> tuple[list[int], int]:
        scene = viewer.scene
        drawn = scene.recording.world_ids[scene.worlds]
        return drawn.tolist(), int(drawn[scene.highlight])

    steps = []
    for _ in range(4):
        viewer._act("next world")
        steps.append(shown())

    assert steps == [([11, 12], 11), ([10, 12], 12), ([10, 13], 13), ([10, 12], 10)]
    assert viewer.scene.model is model  # the same composite throughout
    viewer.renderer.close()


def test_the_camera_follows_the_highlight_unless_turned_off_which_is_remembered(
    window, make_recording, saved
):
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer._act("follow")  # nothing to follow in the empty world
    assert not viewer.renderer.follow and load_settings().follow
    viewer.load([make_recording(frames=4, worlds=2)])
    renderer, scene = viewer.renderer, viewer.scene
    viewer.playback.seek(3)
    viewer._draw()

    assert renderer.follow  # by default, from the first file on
    assert np.allclose(renderer.camera.lookat, scene.world_centre(scene.highlight))
    viewer._act("follow")  # F, or the panel's Follow
    viewer._act("reset view")
    assert not renderer.follow and not load_settings().follow
    viewer._act("follow")
    assert renderer.follow and load_settings().follow
    renderer.move_camera(mujoco.mjtMouse.mjMOUSE_MOVE_V, 0.1, 0.1)  # a pan
    assert not renderer.follow and load_settings().follow  # for the moment only
    viewer._act("reset view")
    assert renderer.follow
    renderer.close()


def test_highlighting_another_world_brings_the_camera_to_it(window, make_recording):
    recording = make_recording(worlds=3, score=[3, 2, 1])
    viewer = Viewer(window, Settings(worlds=3, cache=False), 0.3)
    viewer.load([recording])
    camera, scene = viewer.renderer.camera, viewer.scene
    camera.lookat[:] = (5.0, 5.0, 5.0)  # looking elsewhere, as after a pan
    angle = (camera.distance, camera.azimuth, camera.elevation)

    viewer._act("next world")  # B, or the panel's stepper
    second = scene.world_centre(scene.highlight)
    looked = camera.lookat.copy()
    viewer.renderer.set_follow(True)
    viewer._highlight(int(scene.worlds[0]))  # a double-click on the best world

    assert scene.highlight == 0 and np.allclose(looked, second)
    assert np.allclose(camera.lookat, scene.world_centre(0))
    assert (camera.distance, camera.azimuth, camera.elevation) == angle
    assert viewer.renderer.follow  # following still, as before the click
    viewer.renderer.close()


def test_a_world_whose_id_another_world_shares_can_be_highlighted_too(
    window, make_recording
):
    recording = make_recording(worlds=3, score=[3, 2, 1], world_ids=[7, 7, 8])
    viewer = Viewer(window, Settings(worlds=1, cache=False), 0.3)
    viewer.load([recording])

    viewer._act("next world")

    assert viewer.scene.worlds.tolist() == [1]  # the second world, not the first
    viewer.renderer.close()


def test_with_one_world_drawn_the_panel_offers_the_highlight_but_no_ghosts(
    window, make_recording
):
    viewer = Viewer(window, Settings(worlds=1, cache=False), 0.3)
    viewer.load([make_recording(worlds=3)])

    labels = [row.label for row in viewer._rows() if isinstance(row, ui.Stepper)]

    assert "Highlight" in labels and "Ghosts" not in labels
    viewer.renderer.close()


def tabs(viewer) -> list[str]:
    """The names on the tabs, as last laid out."""
    return [box.text for box in viewer.boxes if box.action.startswith("tab ")]


def test_opening_more_files_adds_tabs_and_shows_the_first_at_the_same_frame(
    window, make_recording, tmp_path
):
    early, late = tmp_path / "early.npz", tmp_path / "late.npz"
    write_recording(early, make_recording(frames=6, seed=1))
    write_recording(late, make_recording(frames=6, seed=2))
    viewer = Viewer(window, Settings(cache=False, follow=False), 0.3)
    viewer.open_files([str(early)])
    viewer.playback.seek(4)  # paused at the fifth frame
    viewer.renderer.camera.lookat[:] = (1.0, 2.0, 3.0)  # where the user put it

    viewer.open_files([str(late)])
    viewer._draw()

    assert tabs(viewer) == ["early", "late"] and viewer.playback.file_index == 1
    assert viewer.scene.recording is viewer.recordings[1]
    assert viewer.scene.frame_index == 4 and not viewer.playback.playing
    assert list(viewer.renderer.camera.lookat) == [1.0, 2.0, 3.0]  # the same model
    viewer.renderer.close()


def test_a_file_open_already_shows_its_tab_instead_of_a_second_one(
    window, make_recording, tmp_path
):
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.npz"
        write_recording(path, make_recording(seed=len(paths)))
        paths.append(str(path))
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer.open_files(paths)
    viewer._act("tab 1")
    viewer._show_file()

    viewer.open_files([str(tmp_path / "." / "a.npz")])  # the same file
    viewer._show_file()
    viewer.load([make_recording(seed=2)], paths=[paths[1]])  # as from the command

    assert len(viewer.recordings) == 2 and viewer.playback.file_index == 0
    viewer.renderer.close()


def test_closing_the_shown_tab_shows_its_neighbour_and_the_last_the_empty_world(
    window, make_recording
):
    recordings = [make_recording(seed=seed) for seed in range(3)]
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer.load(recordings)
    viewer._act("tab 1")
    viewer._show_file()

    viewer._act("close tab 1")  # the next tab takes its place
    viewer._show_file()
    assert viewer.scene.recording is recordings[2]
    viewer._act("close tab 1")  # the last tab: the one before
    viewer._show_file()
    assert viewer.scene.recording is recordings[0]
    viewer._act("close all")
    viewer._draw()

    assert viewer.playback is None and viewer.recordings == []
    assert viewer.scene.recording.title == "no recording" and not tabs(viewer)
    viewer.renderer.close()


def test_close_all_leaves_no_scrolled_tabs_behind(window, make_recording):
    recordings = [make_recording(seed=seed) for seed in range(12)]
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer.load(recordings)
    viewer.tabs.scroll_by(6)
    viewer._draw()
    assert "tab 0" not in {box.action for box in viewer.boxes}

    viewer._act("close all")
    viewer.load(recordings)
    viewer._draw()

    assert "tab 0" in {box.action for box in viewer.boxes}
    viewer.renderer.close()


def test_each_file_left_out_is_counted_in_the_message(
    window, make_recording, small_model, tmp_path
):
    junk, bad = tmp_path / "a_junk.npz", tmp_path / "b_bad.npz"
    junk.write_text("not a recording")
    write_recording(bad, Recording(small_model, 0.02, np.zeros((2, 3, 10))))
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer.load([make_recording()])

    viewer.open_files([str(junk), str(bad)])

    assert "a_junk.npz cannot be read" in viewer._message
    assert viewer._message.endswith("(and 1 more file left out)")
    viewer.renderer.close()


def test_a_failing_file_after_worlds_chosen_by_id_brings_the_last_back_whole(
    window, make_recording, small_model
):
    ids = [10, 11, 12, 13]
    good = make_recording(worlds=4, world_ids=ids)
    bad = Recording(small_model, 0.02, np.zeros((2, 3, 10)), title="bad")
    viewer = Viewer(window, Settings(worlds=4, cache=False), 0.3)
    viewer.load([good], ids=[10, 12])

    viewer.load([bad])  # opened from the window: the worlds by id end
    viewer._act("next world")

    assert viewer.scene.recording is good and len(viewer.scene.worlds) == 4
    viewer.renderer.close()


def test_the_command_ends_with_the_reason_when_none_of_its_files_shows(
    opengl, saved, small_model, tmp_path, capsys
):
    bad = tmp_path / "bad.npz"
    write_recording(bad, Recording(small_model, 0.02, np.zeros((2, 3, 10))))

    status = cli.main([str(bad), "--width", "160", "--height", "120"])

    assert status == 1
    assert "bad: qpos has 10 positions" in capsys.readouterr().err


def test_play_next_runs_on_into_the_next_tab_and_loop_back_to_the_first(
    window, make_recording, saved
):
    first, second = make_recording(frames=2, seed=1), make_recording(frames=2, seed=2)
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer.load([first, second])

    viewer._act("play next")
    viewer._act("loop")
    playback = viewer.playback
    playback.sync(0.0)

    playback.advance(0.6)  # to the first file's end, and on into the second
    assert playback.recording is second and playback.playing
    playback.advance(1.2)  # to the second's end, and round to the first

    assert playback.recording is first and playback.playing
    assert load_settings().play_next and load_settings().loop  # remembered
    viewer.renderer.close()


def test_a_click_on_a_tab_shows_its_file_and_ctrl_tab_and_ctrl_w_act_on_tabs(
    window, make_recording
):
    viewer = Viewer(window, Settings(cache=False), 0.3)
    viewer.load([make_recording(seed=1), make_recording(seed=2)])
    viewer._draw()
    box = next(box for box in viewer.boxes if box.action == "tab 1")
    middle = (box.x + box.width / 2, box.y + box.height / 2)

    press(viewer, window, *middle, glfw.PRESS)
    press(viewer, window, *middle, glfw.RELEASE)
    assert viewer.playback.file_index == 1
    viewer._on_key(window, glfw.KEY_TAB, 0, glfw.PRESS, glfw.MOD_CONTROL)
    assert viewer.playback.file_index == 0 and viewer.settings.panel  # round
    viewer._on_key(window, glfw.KEY_W, 0, glfw.PRESS, glfw.MOD_CONTROL)

    assert len(viewer.recordings) == 1 and viewer.settings.panel
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


def test_quitting_asks_first_and_only_the_quit_button_or_enter_quits(window):
    viewer = Viewer(window, Settings(cache=False), 0.3)

    viewer._act("quit")
    viewer._draw()
    assert viewer.asking and not glfw.window_should_close(window)
    viewer._on_key(window, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)  # Esc: stay
    assert not viewer.asking and not glfw.window_should_close(window)
    viewer._on_close(window)  # the window's close button asks too
    viewer._draw()
    box = next(box for box in viewer.boxes if box.action == "quit now")
    middle = (box.x + box.width / 2, box.y + box.height / 2)
    press(viewer, window, *middle, glfw.PRESS)
    press(viewer, window, *middle, glfw.RELEASE)

    assert glfw.window_should_close(window)
    viewer.renderer.close()


def test_the_file_picker_runs_the_command_again_and_reads_the_paths_it_wrote(
    monkeypatch,
):
    commands = []

    class Picker:  # in place of the process: two files chosen at once
        returncode = 0

        def __init__(self, command, **options):
            commands.append(command)
            Path(command[-1]).write_text("a.npz\nb \u00e9.npz\n", encoding="utf-8")

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Picker)
    assert FilePicker().poll() == ["a.npz", "b \u00e9.npz"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    FilePicker().close()

    assert commands[0][:4] == [sys.executable, "-m", "mujoco_replay", cli.PICK_FILES]
    assert commands[1][:2] == [sys.executable, cli.PICK_FILES]  # it runs itself
    assert not any(Path(command[-1]).exists() for command in commands)


def test_the_window_names_itself_on_the_taskbar_unless_it_is_an_executable(
    monkeypatch,
):
    named = []
    shell32 = SimpleNamespace(SetCurrentProcessExplicitAppUserModelID=named.append)
    monkeypatch.setattr(
        ctypes, "windll", SimpleNamespace(shell32=shell32), raising=False
    )
    monkeypatch.setattr(sys, "platform", "win32")
    _name_for_the_taskbar()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    _name_for_the_taskbar()

    assert named == [APP_ID]  # Python's process, and not the executable's
