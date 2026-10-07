"""Tests for drawing: offscreen frames of a small recording."""

from dataclasses import replace

import mujoco
import numpy as np

from mujoco_replay.render import SceneRenderer, _cut
from mujoco_replay.scene import ComposedScene
from mujoco_replay.settings import PERFORMANCE, QUALITY
from mujoco_replay.ui import text_width


def test_a_frame_shows_the_scene_and_changes_when_the_ghosts_hide(
    gl_context, make_recording
):
    scene = ComposedScene(make_recording(frames=1, worlds=2), np.arange(2))
    renderer = SceneRenderer(scene, offscreen_size=(160, 120))

    renderer.render(160, 120, status="frame 1 / 1")
    with_ghosts = renderer.read_pixels(160, 120)
    scene.set_ghosts("hidden")
    renderer.render(160, 120, status="frame 1 / 1")
    without_ghosts = renderer.read_pixels(160, 120)
    renderer.close()

    assert with_ghosts.shape == (120, 160, 3)
    assert len(np.unique(with_ghosts.reshape(-1, 3), axis=0)) > 50
    assert (with_ghosts != without_ghosts).any()


def test_sites_are_not_drawn(gl_context, make_recording):
    scene = ComposedScene(make_recording(frames=1, worlds=2), np.arange(2))
    renderer = SceneRenderer(scene, offscreen_size=(64, 48))

    renderer.render(64, 48, hud=False)
    shapes = renderer._shapes
    kinds = {shapes.geoms[index].objtype for index in range(shapes.ngeom)}
    renderer.close()

    assert mujoco.mjtObj.mjOBJ_GEOM in kinds
    assert mujoco.mjtObj.mjOBJ_SITE not in kinds


def test_both_presets_draw_and_look_different(gl_context, make_recording):
    images = {}
    for name, graphics in (("quality", QUALITY), ("performance", PERFORMANCE)):
        scene = ComposedScene(make_recording(frames=1, worlds=2), np.arange(2))
        renderer = SceneRenderer(scene, (160, 120), graphics=graphics)
        renderer.render(160, 120, hud=False)
        images[name] = renderer.read_pixels(160, 120)
        renderer.close()

    for image in images.values():
        assert len(np.unique(image.reshape(-1, 3), axis=0)) > 50
    assert (images["quality"] != images["performance"]).any()


def test_the_window_draws_a_share_of_its_pixels_and_scales_it_to_fill(
    gl_context, make_recording
):
    recording = make_recording(frames=1, worlds=2)
    whole = SceneRenderer(ComposedScene(recording, np.arange(2)), (160, 120))
    whole.render(160, 120, hud=False)
    expected = whole.read_pixels(160, 120).astype(int)
    whole.close()
    half = replace(QUALITY, resolution=50)
    window = SceneRenderer(ComposedScene(recording, np.arange(2)), graphics=half)

    window.render(160, 120, hud=False)  # into the window, scaled up from 80 x 60
    drawn = window.read_pixels(160, 120).astype(int)
    window.close()

    assert np.abs(drawn - expected).mean() < 12  # the same picture, only softer


def test_a_point_on_a_world_picks_that_world_and_the_sky_none(
    gl_context, make_recording
):
    recording = make_recording(frames=1, worlds=2)
    recording.qpos[0, :, 1:4] = [(0, -2, 0.5), (0, 2, 0.5)]  # the robots apart
    scene = ComposedScene(recording, np.arange(2))
    renderer = SceneRenderer(scene, offscreen_size=(160, 120))
    renderer.camera.lookat[:] = scene.data.xpos[scene.model.body("w1_robot").id]
    renderer.camera.distance, renderer.camera.elevation = 1.0, -89.0

    renderer.render(160, 120, hud=False)

    assert renderer.copy_at(80, 60, 160, 120, 0) == 1
    renderer.camera.elevation = -2.0  # the middle of the picture is sky now
    renderer.render(160, 120, hud=False)
    assert renderer.copy_at(80, 110, 160, 120, 0) is None
    renderer.close()


def test_framing_leaves_out_a_world_whose_poses_diverged(gl_context, make_recording):
    recording = make_recording(frames=1, worlds=3)
    recording.qpos[0, 2] = np.nan
    scene = ComposedScene(recording, np.arange(3))
    renderer = SceneRenderer(scene, offscreen_size=(64, 48))

    scene.set_highlight(2)
    renderer.centre_on_highlight()

    assert np.isfinite(renderer.camera.lookat).all()
    assert np.isfinite(renderer.camera.distance)
    renderer.close()


def test_the_window_draws_the_scene_again_only_when_its_picture_changes(
    gl_context, make_recording
):
    scene = ComposedScene(make_recording(frames=2, worlds=2), np.arange(2))
    renderer = SceneRenderer(scene)  # into the window, as the viewer draws

    renderer.render(160, 120, hud=False)
    drawn = renderer.read_pixels(160, 120)
    renderer.render(160, 120, hud=False, message="only the overlay changed")
    renderer.render(160, 120, hud=False)
    shown_again, again = renderer.drew_scene, renderer.read_pixels(160, 120)
    scene.set_frame(1)
    renderer.render(160, 120, hud=False)
    renderer.close()

    assert not shown_again and np.array_equal(again, drawn)
    assert renderer.drew_scene  # a new frame is a new picture


def test_framing_keeps_the_best_world_when_half_the_worlds_diverged(
    gl_context, make_recording
):
    recording = make_recording(frames=1, worlds=2)
    recording.qpos[0, :, 1:4] = [(10, 0, 0.5), (1e6, 0, 0.5)]  # far, but finite
    scene = ComposedScene(recording, np.arange(2))

    renderer = SceneRenderer(scene, offscreen_size=(64, 48))  # frames all at once
    renderer.close()

    assert np.allclose(renderer.camera.lookat, scene.world_centre(0))


def test_side_lines_are_cut_in_a_narrow_area_and_whole_in_a_wide_one(
    gl_context, make_recording
):
    scene = ComposedScene(make_recording(frames=1, worlds=1), np.arange(1))
    renderer = SceneRenderer(scene, offscreen_size=(64, 48))
    lines = [f"setting number {index} = a value of some length" for index in range(12)]

    _, narrow_hidden, narrow_cut = renderer._columns(
        mujoco.MjrRect(0, 0, 90, 600), lines
    )
    _, wide_hidden, wide_cut = renderer._columns(mujoco.MjrRect(0, 0, 900, 600), lines)
    long = _cut(renderer.context, "x" * 10_000, 200)  # one pass, however long
    long_width = text_width(renderer.context, long)
    renderer.close()

    assert narrow_cut == 12 and narrow_hidden == 0
    assert wide_cut == wide_hidden == 0
    assert long.endswith("...") and long_width <= 200
