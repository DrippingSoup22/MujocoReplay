"""Tests for the video: short recordings written to MP4 files."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

pytest.importorskip("imageio_ffmpeg", reason="the video extra is not installed")

import imageio.v2 as imageio  # noqa: E402

from mujoco_replay import cli  # noqa: E402
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


def test_the_video_follows_the_best_world_when_it_walks_away(
    gl_context, make_recording, small_model, tmp_path
):
    model = mujoco.MjModel.from_xml_string(small_model)
    x = model.jnt_qposadr[model.body("robot").jntadr[0]]  # its free joint
    recording = make_recording(frames=2, worlds=1, replicated_bodies=("robot",))
    qpos = recording.qpos.copy()
    qpos[:, 0, x + 2] = 0.5  # above the floor, which hides what lies under it
    qpos[1, 0, x] += 8.0  # past the 3 m floor's edge, far out of the first picture
    path = tmp_path / "away.mp4"

    moved = replace(recording, qpos=qpos)
    export([moved], [np.arange(1)], path, 0.1, 10, (160, 120), hud=False)

    last = frames_in(path)[-1].astype(int)
    red, green, blue = last[..., 0], last[..., 1], last[..., 2]
    torso = (blue > 120) & (blue > red + 50) & (blue > green + 20)  # its blue paint
    assert torso.sum() > 5  # none when the camera stays where it framed the start


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


def test_the_command_writes_a_video_of_a_model_file(
    gl_context, saved, small_model, tmp_path
):
    model, out = tmp_path / "small.xml", tmp_path / "small.mp4"
    model.write_text(small_model)

    status = cli.main(["render", str(model), "--out", str(out), "--fps", "10"])

    assert status == 0
    assert frames_in(out)[0].shape[2] == 3  # one still frame, as long as it lasts


def test_the_command_writes_a_gif_small_and_at_15_frames_a_second(
    gl_context, saved, make_recording, tmp_path
):
    from mujoco_replay.recording import write_recording

    run, out = tmp_path / "run.npz", tmp_path / "run.gif"
    write_recording(run, make_recording(frames=3, worlds=2))

    status = cli.main(["render", str(run), "--out", str(out), "--speed", "0.2"])

    assert status == 0
    from PIL import Image

    with Image.open(out) as gif:
        assert gif.size == (640, 360) and gif.info["loop"] == 0
        shown = []
        for index in range(gif.n_frames):
            gif.seek(index)
            shown.append(gif.info["duration"])
    # 9 frames at 15 a second, each recorded frame's 3 alike merged into one
    assert shown == [200, 200, 200]
