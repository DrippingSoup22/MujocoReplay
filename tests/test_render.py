"""Tests for drawing: offscreen frames of a small recording."""

from dataclasses import replace

import mujoco
import numpy as np

from mujoco_replay.recording import Recording
from mujoco_replay.render import RING_SEGMENTS, SceneRenderer, _cut, _spread_floors
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


def test_framing_and_centring_leave_following_as_it_was(gl_context, make_recording):
    scene = ComposedScene(make_recording(frames=1, worlds=2), np.arange(2))
    renderer = SceneRenderer(scene, offscreen_size=(64, 48))
    states = []
    for follow in (True, False):
        renderer.set_follow(follow)
        renderer.frame_all()
        renderer.centre_on_highlight()
        renderer.look_from_above()
        states.append(renderer.follow)
    renderer.close()

    assert states == [True, False]


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


def ring_recording(make_recording, radius) -> Recording:
    """Two worlds 1 m apart, each with a target and a range ring around it."""
    centres = np.zeros((2, 2, 2, 3))
    centres[:, 1] = (1.0, 0.0, 0.0)
    return make_recording(
        frames=2,
        worlds=2,
        marker_names=("target", "range"),
        marker_positions=centres,
        marker_radius=radius,
        marker_shapes=("sphere", "ring"),
    )


def test_a_ring_lies_flat_around_its_marker_at_the_radius_of_the_frame(
    gl_context, make_recording
):
    radius = np.full((2, 2, 2), 0.01, dtype=np.float32)
    radius[:, :, 1] = [[0.3, 0.5], [0.4, 0.6]]  # per frame and world
    scene = ComposedScene(ring_recording(make_recording, radius), np.arange(2))
    renderer = SceneRenderer(scene, offscreen_size=(64, 48))

    def ring_starts() -> np.ndarray:
        renderer.render(64, 48, hud=False)
        shapes = renderer._shapes
        lines = [
            shapes.geoms[index]
            for index in range(shapes.ngeom)
            if shapes.geoms[index].type == mujoco.mjtGeom.mjGEOM_LINE
        ]
        return np.array([line.pos for line in lines])

    first = ring_starts()
    scene.set_frame(1)
    second = ring_starts()
    renderer.close()

    assert len(first) == RING_SEGMENTS  # the highlighted world's ring alone
    for starts, expected in ((first, 0.3), (second, 0.4)):
        assert np.allclose(np.linalg.norm(starts[:, :2], axis=1), expected, atol=1e-5)
        assert np.allclose(starts[:, 2], starts[0, 2]) and starts[0, 2] > 0  # flat


def test_framing_takes_in_the_highlighted_worlds_ring(gl_context, make_recording):
    def distance(ring_radius: float) -> float:
        recording = ring_recording(make_recording, [0.01, ring_radius])
        renderer = SceneRenderer(
            ComposedScene(recording, np.arange(2)), offscreen_size=(64, 48)
        )
        renderer.close()
        return renderer.camera.distance

    assert distance(5.0) > distance(0.01) + 5


# A floor of 1 m by 4 m whose checks are 1.25 cm by 10 cm, a raised pad, a
# wall, and a sky that darkens upward; and a robot.
FLOORS = """
<mujoco>
  <asset>
    <texture type="skybox" builtin="gradient" width="16" height="96"
             rgb1="0.1 0.2 0.3" rgb2="0.5 0.6 0.7"/>
    <texture name="tiles" type="2d" builtin="checker" width="8" height="8"/>
    <material name="ground" texture="tiles" texrepeat="80 40"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0.5 2 0.1" material="ground"/>
    <geom name="pad" type="plane" pos="0 0 0.01" size="0.1 0.1 0.1"/>
    <geom name="wall" type="plane" pos="1 0 0" zaxis="1 0 0" size="1 1 0.1"/>
    <body name="robot" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def spread(make_recording, model_xml: str) -> mujoco.MjModel:
    """The composite of a recording of ``model_xml``, its floor spread."""
    scene = ComposedScene(make_recording(1, 1, model_xml), np.arange(1))
    _spread_floors(scene.model, scene.data)
    return scene.model


def test_the_floor_is_drawn_everywhere_with_its_checks_and_a_haze_like_the_sky(
    make_recording,
):
    model = spread(make_recording, FLOORS)

    sizes = {name: model.geom(name).size[:2].tolist() for name in ("floor", "pad")}
    assert sizes == {"floor": [0, 0], "pad": [0.1, 0.1]}  # the lowest plane alone
    assert model.geom("wall").size[:2].tolist() == [1, 1]  # facing sideways
    # Drawn everywhere, a plane repeats its texture every 2 / repeats metres.
    assert np.allclose(2 / model.material("ground").texrepeat, [0.0125, 0.1])
    assert np.allclose(model.vis.rgba.haze[:3], [0.3, 0.4, 0.5], atol=0.03)


def test_a_floor_keeps_its_size_when_its_texture_would_change_other_shapes(
    make_recording,
):
    shared = FLOORS.replace(
        'size="0.1 0.1 0.1"/>', 'size="0.1 0.1 0.1" material="ground"/>'
    )
    sizes, repeats, hazes = [], [], []
    for model_xml in (
        shared,  # the pad's checks would shrink with the floor's
        shared.replace('texrepeat="80 40"', 'texrepeat="80 40" texuniform="true"'),
        FLOORS.replace('size="0.5 2 0.1"', 'size="0 0 0.1"'),  # everywhere already
    ):
        model = spread(make_recording, model_xml)
        sizes.append(model.geom("floor").size[:2].tolist())
        repeats.append(model.material("ground").texrepeat.tolist())
        hazes.append(model.vis.rgba.haze[:3].tolist())

    assert sizes == [[0.5, 2], [0, 0], [0, 0]]
    assert repeats == [[80, 40]] * 3  # in metres when uniform: as everywhere
    assert hazes[0] == hazes[2] == [1, 1, 1]  # MuJoCo's own, for a floor not spread


def bright(image: np.ndarray) -> float:
    """The share of an image's pixels that are not the black of no sky."""
    return float((image.astype(int).sum(axis=2) > 60).mean())


def test_a_world_far_past_the_floors_edge_stands_on_the_floor(
    gl_context, make_recording
):
    recording = make_recording(frames=1, worlds=1, replicated_bodies=("robot",))
    recording.qpos[0, 0, 1:4] = (40.0, 0.0, 0.5)  # the floor ends at 3 m
    scene = ComposedScene(recording, np.arange(1))
    renderer = SceneRenderer(scene, offscreen_size=(160, 120))
    renderer.set_follow(True)
    renderer.camera.distance, renderer.camera.elevation = 3.0, -89.0

    renderer.render(160, 120, hud=False)
    image = renderer.read_pixels(160, 120)
    renderer.close()

    assert bright(image) > 0.95  # floor all around it


def test_the_floor_is_under_a_world_the_camera_jumps_to(gl_context, make_recording):
    recording = make_recording(frames=1, worlds=2, replicated_bodies=("robot",))
    recording.qpos[0, :, 1:4] = [(0.0, 0.0, 0.5), (100.0, 0.0, 0.5)]
    scene = ComposedScene(recording, np.arange(2))
    scene.model.vis.map.zfar = 2.0  # MuJoCo's far plane, and floor, 8 m away
    renderer = SceneRenderer(scene, offscreen_size=(160, 120))
    renderer.set_follow(True)
    renderer.camera.distance, renderer.camera.elevation = 3.0, -89.0
    renderer.render(160, 120, hud=False)

    scene.set_highlight(1)  # the camera follows it 100 m away, in one drawing
    renderer.render(160, 120, hud=False)
    image = renderer.read_pixels(160, 120)
    renderer.close()

    assert bright(image) > 0.95


def test_zooming_out_past_the_models_far_plane_keeps_the_world_in_view(
    gl_context, make_recording
):
    recording = make_recording(frames=1, worlds=1, replicated_bodies=("robot",))
    recording.qpos[0, 0, 1:4] = (0.0, 0.0, 0.5)
    scene = ComposedScene(recording, np.arange(1))
    scene.model.vis.map.zfar = 2.0  # under 8 m
    renderer = SceneRenderer(scene, offscreen_size=(160, 120))
    renderer.set_follow(True)
    renderer.camera.distance = 12.0

    renderer.render(160, 120, hud=False)
    image = renderer.read_pixels(160, 120).astype(int)
    renderer.close()

    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    assert ((blue > 120) & (blue > red + 50) & (blue > green + 20)).any()  # its torso
    assert scene.model.vis.map.zfar == 2.0  # the model's own, as it was


def test_a_world_far_from_the_sun_casts_its_shadow_where_the_camera_looks(
    gl_context, make_recording, small_model
):
    sun = '<light name="sun" pos="0 0 4" dir="0 0 -1"/>'
    spot = '<light name="spot" pos="2 -2 3" mode="targetbody" target="robot"/>'
    model_xml = small_model.replace(spot, "").replace(
        sun, '<light name="sun" pos="0 0 4" dir="0.5 0 -1" directional="true"/>'
    )
    recording = make_recording(
        frames=1, worlds=1, model_xml=model_xml, replicated_bodies=("robot",)
    )
    recording.qpos[0, 0, 1:4] = (200.0, 0.0, 0.5)  # MuJoCo's shadows reach 16 m
    images = []
    for shadows in (False, True):
        scene = ComposedScene(recording, np.arange(1))
        graphics = replace(QUALITY, shadows=shadows, reflections=False)
        renderer = SceneRenderer(scene, (160, 120), graphics=graphics)
        renderer.set_follow(True)
        renderer.camera.distance, renderer.camera.elevation = 3.0, -89.0
        renderer.render(160, 120, hud=False)
        images.append(renderer.read_pixels(160, 120).astype(int).sum(axis=2))
        renderer.close()

    assert (images[0] - images[1] > 60).sum() > 50  # its shadow, on the floor
