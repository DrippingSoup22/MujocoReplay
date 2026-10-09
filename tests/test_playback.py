"""Tests for playback: wall time, pausing, stepping, files, events, and speeds."""

import numpy as np
import pytest

from mujoco_replay.playback import DEFAULT_SECONDS_PER_FRAME, Playback
from mujoco_replay.recording import Recording


def clip(frames: int, events: dict[int, str] | None = None) -> Recording:
    """A recording with 20 ms frames; playback needs only its timing and events."""
    events = events or {}
    return Recording(
        "<mujoco/>",
        0.02,
        np.zeros((frames, 1, 0)),
        event_frames=list(events) or None,
        event_labels=tuple(events.values()) or None,
    )


def position(playback: Playback) -> tuple[int, int, bool]:
    return playback.file_index, playback.frame_index, playback.playing


def test_wall_time_moves_one_frame_per_interval_at_every_preset():
    for seconds_per_frame in Playback([clip(1)]).presets:
        playback = Playback([clip(1000)], seconds_per_frame, now=100.0)

        playback.advance(100.0 + 0.6 * seconds_per_frame)
        playback.advance(100.0 + 10.5 * seconds_per_frame)

        assert playback.frame_index == 10, seconds_per_frame


def test_a_paused_playback_stays_and_resumes_without_a_jump():
    playback = Playback([clip(100)], 0.3, now=0.0)
    playback.advance(0.35)

    playback.toggle()
    playback.advance(60.0)
    assert (playback.frame_index, playback.playing) == (1, False)

    playback.toggle()
    playback.advance(60.35)
    assert playback.frame_index == 2


def test_stepping_moves_one_frame_either_way_across_files_and_pauses():
    playback = Playback([clip(2), clip(2)])

    playback.step(1)
    assert position(playback) == (0, 1, False)
    playback.step(1)
    assert position(playback) == (1, 0, False)
    playback.step(-1)
    assert position(playback) == (0, 1, False)
    playback.step(-1)
    playback.step(-1)
    assert position(playback) == (0, 0, False)


def test_playback_runs_on_into_the_next_file_and_stops_at_the_end_of_the_last():
    playback = Playback([clip(3), clip(2)], 1.0, now=0.0)

    playback.advance(3.0)
    assert position(playback) == (1, 0, True)
    playback.advance(100.0)
    assert position(playback) == (1, 1, False)

    playback.toggle()  # playing again at the very end replays the last file
    assert position(playback) == (1, 0, True)


def test_without_running_on_each_file_plays_and_steps_on_its_own():
    playback = Playback([clip(2, {2: "end"}), clip(2)], 1.0, now=0.0, run_on=False)

    assert playback.advance(5.0) == ["end"]
    assert position(playback) == (0, 1, False)  # stopped at the file's end
    playback.step(1)
    assert position(playback) == (0, 1, False)
    playback.toggle()  # playing again at the end replays the file
    assert position(playback) == (0, 0, True)
    playback.show_file(1)
    playback.step(-1)
    assert position(playback) == (1, 0, False)


def test_showing_another_file_keeps_the_frame_or_pauses_at_a_shorter_files_end():
    playback = Playback([clip(10), clip(8), clip(4)], 1.0, now=0.0)
    playback.advance(6.0)

    playback.show_file(1)
    assert position(playback) == (1, 6, True)  # the same frame, still playing
    playback.show_file(2)
    assert position(playback) == (2, 3, False)  # its last frame, paused


def test_looping_starts_the_file_again_and_running_on_the_first_file():
    playback = Playback([clip(2, {2: "end"}), clip(2)], 1.0, 0.0, False, loop=True)

    assert playback.advance(2.0) == ["end"]
    assert position(playback) == (0, 0, True)  # the file again
    playback.run_on = True
    playback.advance(5.0)
    assert position(playback) == (1, 1, True)  # on into the next file
    playback.advance(7.0)
    assert position(playback) == (0, 1, True)  # the first again, after the last


def test_closing_the_shown_file_shows_the_next_or_after_the_last_the_one_before():
    first, second, third, fourth = clip(5), clip(5), clip(2), clip(5)
    playback = Playback([first, second, third, fourth])
    playback.show_file(2)

    playback.remove_file(0)  # before the shown file
    assert playback.recording is third and playback.file_index == 1
    playback.seek(1)
    playback.remove_file(1)  # the shown file: the next takes its place
    assert playback.recording is fourth and playback.frame_index == 1
    playback.seek(4)
    playback.remove_file(1)  # the shown file, the last: the one before
    assert playback.recording is second and playback.frame_index == 4
    playback.remove_file(0)
    assert playback.recordings == []


def test_each_event_is_reported_once_when_playback_passes_it():
    playback = Playback(
        [clip(3, {1: "one", 3: "end of first"}), clip(2, {0: "second", 2: "end"})],
        1.0,
        now=0.0,
    )

    assert playback.advance(1.0) == ["one"]
    assert playback.advance(2.0) == []
    assert playback.advance(3.0) == ["end of first", "second"]
    assert playback.advance(10.0) == ["end"]
    assert playback.advance(20.0) == []


def test_the_speed_presets_run_both_ways_and_stop_at_the_ends():
    playback = Playback([clip(10)])
    faster, slower = [], []

    for _ in range(5):
        playback.faster()
        faster.append(playback.seconds_per_frame)
    multiple = playback.real_time_multiple
    for _ in range(11):
        playback.slower()
        slower.append(playback.seconds_per_frame)

    assert faster == pytest.approx([0.05, 0.02, 0.01, 0.005, 0.005])
    assert multiple == pytest.approx(4)
    assert slower == pytest.approx([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 3])
    playback.default_speed()
    assert playback.seconds_per_frame == DEFAULT_SECONDS_PER_FRAME
