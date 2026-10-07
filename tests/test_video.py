"""Tests for the video: short recordings written to MP4 files."""

import numpy as np
import pytest

pytest.importorskip("imageio_ffmpeg", reason="the video extra is not installed")

import imageio.v2 as imageio  # noqa: E402

from mujoco_replay.video import FLASH_SECONDS, export  # noqa: E402


def frames_in(path) -> list[np.ndarray]:
    with imageio.get_reader(path, format="FFMPEG") as reader:
        return [frame for frame in reader]


def test_each_recorded_frame_lasts_its_seconds_in_the_video(
    gl_context, make_recording, tmp_path
):
    path = tmp_path / "two_frames.mp4"
    written = export(
        [make_recording(frames=2, worlds=2)],
        [np.arange(2)],
        path,
        seconds_per_frame=0.3,
        fps=10,
        size=(96, 64),
    )

    frames = frames_in(path)
    assert written == len(frames) == 2 * 3  # two recorded frames, 0.3 s at 10 fps
    assert frames[0].shape == (64, 96, 3)


def test_an_event_at_the_end_holds_the_last_frame_while_it_flashes(
    gl_context, make_recording, tmp_path
):
    path = tmp_path / "ending.mp4"
    recording = make_recording(
        frames=2, worlds=2, event_frames=[2], event_labels=("end",)
    )

    written = export(
        [recording],
        [np.arange(2)],
        path,
        seconds_per_frame=0.3,
        fps=10,
        size=(320, 240),
    )

    frames = frames_in(path)
    assert written == len(frames) == 2 * 3 + round(FLASH_SECONDS * 10)
    change = np.abs(frames[-1].astype(int) - frames[5].astype(int)).mean()
    assert change > 1  # the flash is drawn over the held frame
