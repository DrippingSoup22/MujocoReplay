"""Tests for drawing: offscreen frames of a small recording."""

import mujoco
import numpy as np

from mujoco_replay.render import SceneRenderer
from mujoco_replay.scene import ComposedScene


def test_a_frame_shows_the_scene_and_changes_when_the_ghosts_hide(
    gl_context, make_recording
):
    scene = ComposedScene(make_recording(frames=1, worlds=2), np.arange(2))
    renderer = SceneRenderer(scene, offscreen_size=(160, 120))

    renderer.render(160, 120, status="frame 1 / 1")
    with_ghosts = renderer.read_pixels(160, 120)
    scene.set_ghosts_visible(False)
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
