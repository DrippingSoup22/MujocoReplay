"""Tests for GIFs: their delays, a video's frames written, and the window's
recording."""

import numpy as np
import pytest

from mujoco_replay.gif import RECORD_WIDTH, GifRecorder, GifWriter, delays

Image = pytest.importorskip("PIL.Image", reason="Pillow, of the video extra")


def frames_of(path) -> list[tuple[tuple[int, int], int]]:
    """Each frame of a GIF as its size and delay in milliseconds."""
    with Image.open(path) as gif:
        found = []
        for index in range(gif.n_frames):
            gif.seek(index)
            found.append((gif.size, gif.info["duration"]))
        return found


def picture(width: int, height: int, shade: int) -> np.ndarray:
    return np.full((height, width, 3), shade, dtype=np.uint8)


def test_delays_round_the_running_time_and_none_is_too_short_to_show():
    fifteen = delays([0.0, 1 / 15, 2 / 15, 3 / 15], 4 / 15)
    assert fifteen == [70, 60, 70, 70]  # 0.27 s in all, as shown
    assert delays([0.0, 0.001], 0.5) == [20, 500]  # browsers stretch under 20 ms


def test_a_videos_frames_make_a_gif_that_loops_at_their_rate(tmp_path):
    path = tmp_path / "video.gif"
    with GifWriter(path, fps=10) as writer:
        for shade in (0, 80, 160):
            writer.append_data(picture(32, 24, shade))

    assert frames_of(path) == [((32, 24), 100)] * 3
    with Image.open(path) as gif:
        assert gif.info["loop"] == 0


def test_a_recording_shrinks_to_its_width_keeps_its_size_and_its_time(tmp_path):
    path = tmp_path / "folder made" / "window.gif"
    recorder = GifRecorder(path, now=10.0)
    recorder.add(picture(1280, 720, 0), 10.0)
    recorder.add(picture(1280, 720, 0), 10.1)  # the same: merged into the first
    recorder.add(picture(900, 900, 200), 10.2)  # the window changed its size
    recorder.stop(10.5)
    recorder.wait()

    assert recorder.finished and not recorder.error and not recorder.recording
    size = (RECORD_WIDTH, round(720 * RECORD_WIDTH / 1280))
    assert frames_of(path) == [(size, 200), (size, 300)]


def test_a_recording_that_cannot_be_written_tells_why(tmp_path):
    blocked = tmp_path / "a file"
    blocked.write_text("in the way of a folder")
    recorder = GifRecorder(blocked / "window.gif", now=0.0)
    recorder.add(picture(16, 16, 0), 0.0)
    recorder.stop(1.0)
    recorder.wait()
    empty = GifRecorder(tmp_path / "empty.gif", now=0.0)
    empty.stop(1.0)
    empty.wait()

    assert recorder.finished and recorder.error
    assert empty.error == "no frame was recorded"
    assert not (tmp_path / "empty.gif").exists()
