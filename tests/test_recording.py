"""Tests for the recording file: round trips, defaults, and rejected files."""

import numpy as np
import pytest

from mujoco_replay.recording import (
    Recording,
    RecordingError,
    read_recording,
    write_recording,
)

MODEL = "<mujoco><worldbody><body name='b'><joint/><geom size='1'/></body></worldbody></mujoco>"


def full_recording() -> Recording:
    frames, worlds = 3, 2
    return Recording(
        model_xml=MODEL,
        frame_seconds=0.02,
        qpos=np.arange(frames * worlds * 4, dtype=np.float64).reshape(
            frames, worlds, 4
        ),
        assets={"mesh.obj": b"v 0 0 0\n"},
        replicated_bodies=("b",),
        score=[1.5, -0.5],
        score_name="summed reward",
        world_ids=[7, 3],
        level=[1, 2],
        episode_start=[[True, False], [False, False], [False, True]],
        marker_names=("target",),
        marker_positions=np.ones((frames, worlds, 1, 3)),
        marker_radius=[0.001],
        frame_info_names=("updates", "steps"),
        frame_info=[[1, 10], [1, 11], [1, 12]],
        event_frames=[3],
        event_labels=("update 2",),
        title="probe · cycle 2 of 3",
        setup={"run": {"seed": 1}, "device": "cpu"},
    )


def test_a_full_recording_survives_a_round_trip(tmp_path):
    original = full_recording()
    path = tmp_path / "full.npz"

    write_recording(path, original)
    loaded = read_recording(path)

    for name in (
        "qpos",
        "score",
        "world_ids",
        "level",
        "episode_start",
        "marker_positions",
        "marker_radius",
        "frame_info",
        "event_frames",
    ):
        assert np.array_equal(getattr(loaded, name), getattr(original, name)), name
        assert getattr(loaded, name).dtype == getattr(original, name).dtype, name
    for name in (
        "model_xml",
        "frame_seconds",
        "assets",
        "replicated_bodies",
        "score_name",
        "marker_names",
        "frame_info_names",
        "event_labels",
        "title",
        "setup",
    ):
        assert getattr(loaded, name) == getattr(original, name), name
    assert (loaded.frame_count, loaded.world_count, loaded.position_count) == (3, 2, 4)


def test_a_minimal_recording_reads_with_the_defaults(tmp_path):
    path = tmp_path / "minimal.npz"
    write_recording(path, Recording(MODEL, 0.5, np.zeros((2, 3, 4))))

    loaded = read_recording(path)

    assert (
        np.array_equal(loaded.score, np.zeros(3)) and loaded.score.dtype == np.float32
    )
    assert np.array_equal(loaded.world_ids, [0, 1, 2])
    assert loaded.episode_start.shape == (2, 3) and not loaded.episode_start.any()
    assert loaded.level is None and loaded.marker_names is None
    assert loaded.frame_info is None and loaded.event_frames is None
    assert loaded.replicated_bodies is None and loaded.assets == {}
    assert (loaded.title, loaded.setup, loaded.score_name) == ("minimal", {}, "score")


@pytest.mark.parametrize(
    "change, key",
    [
        ({"score": np.zeros(3)}, "score"),
        (
            {"event_frames": np.array([4]), "event_labels": np.array(["x"])},
            "event_frames",
        ),
        ({"level": np.array([0, 1])}, "level"),
        ({"marker_names": np.array(["t"])}, "marker_positions"),
        ({"setup_json": np.array("[1, 2]")}, "setup_json"),
        ({"format_version": np.int64(2)}, "format_version"),
        ({"frame_seconds": np.float64(0.0)}, "frame_seconds"),
    ],
)
def test_a_file_that_breaks_the_format_is_rejected_by_key(tmp_path, change, key):
    path = tmp_path / "broken.npz"
    arrays = {
        "format_version": np.int64(1),
        "model_xml": np.array(MODEL),
        "frame_seconds": np.float64(0.02),
        "qpos": np.zeros((3, 2, 4), dtype=np.float32),
    }
    arrays.update(change)
    np.savez(path, **arrays)

    with pytest.raises(RecordingError, match=f"broken.npz: {key}"):
        read_recording(path)


def test_a_missing_required_key_or_unreadable_file_is_rejected(tmp_path):
    path = tmp_path / "no_qpos.npz"
    np.savez(path, format_version=np.int64(1), model_xml=np.array(MODEL))
    with pytest.raises(RecordingError, match="frame_seconds is missing"):
        read_recording(path)

    not_a_zip = tmp_path / "text.npz"
    not_a_zip.write_text("hello")
    with pytest.raises(RecordingError, match="cannot be read"):
        read_recording(not_a_zip)
