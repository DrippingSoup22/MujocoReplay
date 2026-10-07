"""Tests for the composite scene: its copies, their poses, and their colours."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_replay.recording import Recording, RecordingError
from mujoco_replay.scene import GHOST_RGBA, ComposedScene

CENTIPEDE_MODEL = (
    Path(__file__).resolve().parents[2] / "Centipede" / "models" / "assembly_v2.xml"
)


def body_names(model: mujoco.MjModel) -> list[str]:
    return [model.body(index).name for index in range(model.nbody)]


def test_the_composite_holds_the_static_scene_once_and_a_copy_per_world(
    make_recording,
):
    recording = make_recording(worlds=3, replicated_bodies=("robot",))

    scene = ComposedScene(recording, np.arange(3))

    names = body_names(scene.model)
    assert names.count("post") == names.count("door") == 1
    assert {f"w{copy}_{body}" for copy in range(3) for body in ("robot", "arm")} <= set(
        names
    )
    assert scene.model.nbody == 1 + 2 + 3 * 2
    assert scene.model.nu == 0
    # The door's hinge once, then a free joint and a hinge per copy.
    assert scene.model.nq == 1 + 3 * (7 + 1)


def test_by_default_every_root_body_with_a_joint_is_replicated(make_recording):
    scene = ComposedScene(make_recording(worlds=2), np.arange(2))

    names = body_names(scene.model)
    assert {"w0_door", "w1_door", "w0_robot", "w1_robot"} <= set(names)
    assert "post" in names and "w0_post" not in names


@pytest.mark.parametrize("model", ["small", "centipede"])
def test_each_copy_takes_the_pose_of_its_own_world(model, make_recording, small_model):
    if model == "centipede" and not CENTIPEDE_MODEL.exists():
        pytest.skip(f"Centipede's model is not at {CENTIPEDE_MODEL}")
    model_xml = (
        small_model
        if model == "small"
        else mujoco.MjSpec.from_file(str(CENTIPEDE_MODEL)).to_xml()
    )
    recording = make_recording(frames=2, worlds=3, model_xml=model_xml)
    # Frame 1 repeats frame 0 except for world 1, which copy 2 shows.
    recording.qpos[1] = recording.qpos[0]
    recording.qpos[1, 1] = make_recording(1, 1, model_xml, seed=1).qpos[0, 0]
    worlds = np.array([2, 0, 1])
    scene = ComposedScene(recording, worlds)
    original = mujoco.MjModel.from_xml_string(model_xml)
    alone = mujoco.MjData(original)
    composite_names = set(body_names(scene.model))
    copied = [name for name in body_names(original) if f"w0_{name}" in composite_names]
    assert copied

    for frame in (0, 1):
        scene.set_frame(frame)
        for copy, world in enumerate(worlds):
            alone.qpos[:] = recording.qpos[frame, world]
            mujoco.mj_kinematics(original, alone)
            for name in copied:
                expected = original.body(name).id
                actual = scene.model.body(f"w{copy}_{name}").id
                assert np.allclose(scene.data.xpos[actual], alone.xpos[expected])
                assert np.allclose(scene.data.xquat[actual], alone.xquat[expected])


def test_ghosts_are_grey_and_the_highlight_keeps_the_models_colours(
    make_recording, small_model
):
    scene = ComposedScene(make_recording(worlds=3), np.arange(3))
    model = scene.model
    original = mujoco.MjModel.from_xml_string(small_model)

    def shapes(copy: int) -> list[int]:
        return [model.geom(f"w{copy}_{name}").id for name in ("torso", "forearm")]

    def ghost(copy: int) -> bool:
        ids = shapes(copy)
        return np.allclose(model.geom_rgba[ids], GHOST_RGBA) and bool(
            (model.geom_matid[ids] == -1).all()
        )

    def natural(copy: int) -> bool:
        torso, forearm = shapes(copy)
        paint = original.material("paint").rgba
        return np.allclose(model.mat_rgba[model.geom_matid[torso]], paint) and (
            np.allclose(model.geom_rgba[forearm], original.geom("forearm").rgba)
        )

    assert natural(0) and ghost(1) and ghost(2)
    assert model.geom_rgba[model.geom("w1_bumper").id, 3] == 0  # hidden stays hidden

    scene.set_highlight(2)

    assert ghost(0) and ghost(1) and natural(2)


def test_a_worlds_centre_is_the_centre_of_mass_of_all_its_root_bodies(
    make_recording, small_model
):
    recording = make_recording(frames=1, worlds=2)
    scene = ComposedScene(recording, np.array([1, 0]))
    original = mujoco.MjModel.from_xml_string(small_model)
    alone = mujoco.MjData(original)
    alone.qpos[:] = recording.qpos[0, 1]
    mujoco.mj_kinematics(original, alone)
    mujoco.mj_comPos(original, alone)
    roots = [original.body("door").id, original.body("robot").id]
    mass = original.body_subtreemass[roots]

    expected = mass @ alone.subtree_com[roots] / mass.sum()
    assert np.allclose(scene.world_centre(0), expected)


def test_a_cached_composite_poses_exactly_like_a_freshly_composed_one(
    make_recording, tmp_path
):
    recording = make_recording(frames=2, worlds=3)
    fresh = ComposedScene(recording, np.arange(3), cache=tmp_path)
    cached = ComposedScene(recording, np.arange(3), cache=tmp_path)
    for scene in (fresh, cached):
        scene.set_frame(1)
        scene.set_highlight(2)

    assert not fresh.cached and cached.cached
    assert np.array_equal(cached.data.xpos, fresh.data.xpos)
    assert np.array_equal(cached.model.geom_rgba, fresh.model.geom_rgba)
    (stored,) = tmp_path.glob("*.mjb")
    stored.write_bytes(b"not a model")  # a damaged entry is composed again
    assert not ComposedScene(recording, np.arange(3), cache=tmp_path).cached


def test_another_recording_of_the_same_model_is_shown_on_the_same_composite(
    make_recording,
):
    scene = ComposedScene(make_recording(seed=1), np.arange(3))
    model = scene.model
    second, worlds = make_recording(seed=2), np.array([2, 1, 0])

    assert scene.fits(second, worlds) and not scene.fits(second, np.arange(2))
    scene.show(second, worlds)

    assert scene.model is model
    assert np.allclose(scene.data.xpos, ComposedScene(second, worlds).data.xpos)


def test_joints_of_the_static_scene_follow_the_highlighted_world(
    make_recording, small_model
):
    recording = make_recording(worlds=2, replicated_bodies=("robot",))
    scene = ComposedScene(recording, np.array([1, 0]))
    column = mujoco.MjModel.from_xml_string(small_model).joint("door_hinge").qposadr[0]
    door = scene.model.joint("door_hinge").qposadr[0]

    assert scene.data.qpos[door] == pytest.approx(recording.qpos[0, 1, column])
    scene.set_highlight(1)
    assert scene.data.qpos[door] == pytest.approx(recording.qpos[0, 0, column])


@pytest.mark.parametrize(
    "positions, fields, problem",
    [
        (10, {}, "qpos has 10 positions per world, but the model in model_xml has 9"),
        (9, {"replicated_bodies": ("arm",)}, "replicated_bodies names 'arm'"),
    ],
)
def test_a_recording_that_does_not_fit_its_model_is_rejected(
    small_model, positions, fields, problem
):
    recording = Recording(
        small_model, 0.02, np.zeros((1, 2, positions)), title="probe", **fields
    )

    with pytest.raises(RecordingError, match=f"probe: {problem}"):
        ComposedScene(recording, np.arange(2))
