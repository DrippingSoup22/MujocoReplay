"""Tests for the video: a short recording written to an MP4 file."""

import numpy as np
import pytest

from mujoco_replay.video import export


def test_each_recorded_frame_lasts_its_seconds_in_the_video(
    gl_context, make_recording, tmp_path
):
    pytest.importorskip("imageio_ffmpeg", reason="the video extra is not installed")
    import imageio.v2 as imageio

    path = tmp_path / "two_frames.mp4"
    written = export(
        [make_recording(frames=2, worlds=2)],
        [np.arange(2)],
        path,
        seconds_per_frame=0.3,
        fps=10,
        size=(96, 64),
    )

    with imageio.get_reader(path, format="FFMPEG") as reader:
        frames = [frame for frame in reader]
    assert written == len(frames) == 2 * 3  # two recorded frames, 0.3 s at 10 fps
    assert frames[0].shape == (64, 96, 3)
