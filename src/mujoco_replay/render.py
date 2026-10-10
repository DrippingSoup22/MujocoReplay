"""Drawing a composed scene, its markers, and the overlay, in a window or offscreen.

MuJoCo draws in two steps. ``mjv_updateScene`` turns the model and data into a
list of shapes to draw, an ``MjvScene``, seen from a camera (``MjvCamera``)
with visual options (``MjvOption``). ``mjr_render`` then draws that list with
an ``MjrContext``: the OpenGL resources of one model (its meshes, textures,
shaders, and fonts), which can only be made while an OpenGL context is
current. The ``mjr_`` overlay functions draw text and rectangles on top. A
``SceneRenderer`` holds these objects for the scene being shown.
"""

import json

import mujoco
import numpy as np

from mujoco_replay.recording import Recording
from mujoco_replay.scene import ComposedScene
from mujoco_replay.settings import QUALITY, Graphics
from mujoco_replay.ui import ascii_text, text_width

# The highlighted world's markers; the ghosts' are ghost grey.
MARKER_RGBA = np.array([0.95, 0.15, 0.65, 0.9], dtype=np.float32)
# Facets around and along MuJoCo's round shapes. MuJoCo's own 28 by 16 drew
# 3.5 times the triangles of 16 by 8 with no visible difference.
FINE_SHAPES = (16, 8)
COARSE_SHAPES = (12, 6)
# A marker's beacon, in shares of the scene's extent: its height, the radius
# of its pole, and the radius of its head.
BEACON = (0.5, 0.006, 0.04)
# A ring marker: the line segments of its circle, their width in pixels at
# font scale 100, and how far above the marker it is drawn, in shares of the
# scene's extent, so that the floor under a marker on the ground cannot hide
# it.
RING_SEGMENTS = 64
RING_PIXELS = 3
RING_LIFT = 0.02
# Where framing touches a ring: its four points along the horizontal axes.
RING_EDGES = np.array(
    [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]
)
# The reflection given to a floor that has none, for the reflections switch:
# enough to see, too little to show mirrored legs as extra ghosts.
FLOOR_REFLECTANCE = 0.08
# The area MuJoCo's shadows cover, in model extents (``vis.map.shadowclip``):
# its default of 1 leaves worlds away from the first without shadows, and
# speckles small models with shadows that fall on the shapes casting them.
SHADOW_CLIP = 4.0
# Room in the scene for the shapes MuJoCo adds itself, such as light glyphs.
SPARE_SHAPES = 200
# The far clipping plane is at least this many camera distances away, and
# never nearer than the model's own (``vis.map.zfar``), so that zooming out
# keeps the worlds and the floor in view; MuJoCo draws the sky at 0.7 of it.
FAR_DISTANCES = 4.0
# MuJoCo makes a floor that is drawn everywhere with the OpenGL resources, to
# reach 1.05 far planes around the camera. Once the far plane is more than
# this many times the one the floor was made for, the floor would end before
# the sky, so the resources are made again, for twice the far plane.
FLOOR_SLACK = 1.5
# A plane faces up when the height of its normal is above this.
LEVEL = 0.99
# The camera stays this many degrees above the horizon, so above the floor.
LOWEST_ELEVATION = -2.0
# How far from the middle of the worlds, in model extents, framing still looks
# for a world; one whose simulation diverged lies beyond, or is not finite.
FRAMED_EXTENTS = 1000
TRACK_RGBA = (0.12, 0.12, 0.12, 0.75)
FILLED_RGBA = (0.85, 0.85, 0.85, 0.9)
EVENT_RGB = (1.0, 0.72, 0.2)
EPISODE_RGB = (0.25, 0.6, 1.0)
TEXT_RGB = (0.92, 0.93, 0.95)
SHADE_RGBA = (0.07, 0.08, 0.10, 0.75)
_IDENTITY = np.eye(3).ravel()


def offscreen_context(width: int, height: int) -> mujoco.GLContext:
    """Make a current OpenGL context without a visible window, and return it.

    This is MuJoCo's own helper: a hidden GLFW window by default, or EGL or
    OSMesa on a Linux machine without a display when ``MUJOCO_GL`` says so.
    """
    context = mujoco.GLContext(width, height)
    context.make_current()
    return context


class SceneRenderer:
    """Draws a composed scene into the current OpenGL context.

    Make an OpenGL context current first: the viewer's window, or
    ``offscreen_context``. With ``offscreen_size``, everything is drawn into
    MuJoCo's offscreen buffer of that size, which ``read_pixels`` reads back.
    Without it, the scene is drawn offscreen at the graphics' share of the
    window and scaled up into the window, and the overlay is drawn on top at
    the window's full resolution.
    """

    def __init__(
        self,
        scene: ComposedScene,
        offscreen_size: tuple[int, int] | None = None,
        font_scale: int = 150,
        graphics: Graphics = QUALITY,
    ) -> None:
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        # Sites mark points for sensors and attachments; their copies would be
        # drawn opaque in every ghost, so none is drawn.
        self.option.sitegroup[:] = 0
        self.markers_visible = True
        self.follow = False
        self.timeline_height = 0  # of the timeline last drawn, for clicks on it
        self.drew_scene = False  # whether the last render drew the scene afresh
        self._generation = 0  # counts the scenes and contexts shown
        self._picture: tuple | None = None  # what the scene drawn last shows
        self.graphics = graphics
        self.scene: ComposedScene | None = None
        self._offscreen_size = offscreen_size
        global_ = scene.model.vis.global_
        self._buffer = offscreen_size or (global_.offwidth, global_.offheight)
        self._font_scale = font_scale
        self._context: mujoco.MjrContext | None = None
        self._floor_far = 0.0  # the far plane the floors were made for, in extents
        self._shapes: mujoco.MjvScene | None = None
        self.show(scene)
        self.frame_all()

    def show(self, scene: ComposedScene) -> None:
        """Draw ``scene`` from now on; a new model gets new OpenGL resources."""
        new_model = self.scene is None or scene.model is not self.scene.model
        if new_model:
            _add_shadows_and_reflections(scene.model)
            _spread_floors(scene.model, scene.data)
            self._floor_far = 0.0
            self._make_context(scene.model)
        markers = len(scene.recording.marker_names or ())
        rings = (scene.recording.marker_shapes or ()).count("ring")
        shapes = (len(scene.worlds) + 2) * markers  # a beacon is two more
        shapes += RING_SEGMENTS * rings  # the highlighted world's alone
        needed = scene.model.ngeom + shapes + SPARE_SHAPES
        if new_model or self._shapes.maxgeom < needed:
            self._shapes = mujoco.MjvScene(scene.model, maxgeom=needed)
        self.scene = scene
        self._generation += 1

    def set_graphics(self, graphics: Graphics) -> None:
        """Draw from now on as ``graphics`` says; some changes remake the context."""
        remake = (graphics.antialiasing, graphics.fine_shapes) != (
            self.graphics.antialiasing,
            self.graphics.fine_shapes,
        )
        self.graphics = graphics
        if remake:
            self._make_context(self.scene.model)

    def render(
        self,
        width: int,
        height: int,
        status: str = "",
        flash: str = "",
        hud: bool = True,
        side: list[str] | None = None,
        corner: str = "",
        left: int = 0,
        inset: int = 0,
        message: str = "",
        hint: str = "",
        fresh: bool = False,
        top: int = 0,
    ) -> None:
        """Draw the scene as posed now, with the overlay unless ``hud`` is off.

        The scene fills the window right of ``left`` pixels and below the
        ``top`` pixels, for the panel and the tabs, and the overlay keeps
        ``inset`` pixels clear on the left. ``side`` holds lines for
        the top right, such as the setup; ``corner`` is a line for the bottom
        right, such as the frame rate; ``message`` is shown at the top and
        ``hint`` large in the middle. These four show without the overlay too.

        In a window, a scene that shows what it showed last time, when only
        the overlay or the panel changed, is not drawn again: the picture in
        the offscreen buffer is copied into the window once more, unless
        ``fresh`` asks for a drawing, as timing one does.
        """
        scene, shapes, context, camera = (
            self.scene,
            self._shapes,
            self._context,
            self.camera,
        )
        if self.follow:
            self._look_at(scene.world_centre(scene.highlight))
        picture = (
            self._generation,
            scene.frame_index,
            scene.highlight,
            scene.ghosts,
            self.markers_visible,
            self.graphics,
            tuple(camera.lookat),
            camera.distance,
            camera.azimuth,
            camera.elevation,
            width,
            height,
            left,
            top,
        )
        windowed = self._offscreen_size is None
        self.drew_scene = not windowed or fresh or picture != self._picture
        self._picture = picture
        if self.drew_scene:
            self._update_shapes()
            context = self._context  # made again if the floors were too short
            flags = shapes.flags
            flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.graphics.shadows
            flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.graphics.reflections
            if self.graphics.shadows:
                self._keep_shadows_clean()
                self._aim_shadows()
            if self.markers_visible:
                self._add_markers()
        below = max(1, height - top)
        viewport = mujoco.MjrRect(left, 0, max(1, width - left), below)
        if windowed:  # draw a share offscreen, scale it up
            share = self.graphics.resolution / 100
            drawn = mujoco.MjrRect(
                0,
                0,
                max(1, round(viewport.width * share)),
                max(1, round(below * share)),
            )
            self._fit_buffer(width, height)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
            if self.drew_scene:
                mujoco.mjr_render(drawn, shapes, context)
            mujoco.mjr_blitBuffer(drawn, viewport, 1, 0, context)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, context)
        else:
            mujoco.mjr_render(viewport, shapes, context)
        area = mujoco.MjrRect(inset, 0, max(1, width - inset), below)
        line = context.charHeight
        above, info_width, info_height = (
            self._overlay(area, status, flash) if hud else (0, 0, 0)
        )
        self.timeline_height = above
        if hud and status:
            above += 2 * line  # the playback line's own row
        if side:  # beside or below the top-left lines, clear of the rest
            bottom = above + (2 * line if corner else 0)
            ceiling = area.height - (2 * line if message else 0)
            beside = mujoco.MjrRect(
                area.left + info_width,
                bottom,
                max(1, area.width - info_width),
                max(1, ceiling - bottom),
            )
            under = mujoco.MjrRect(
                area.left,
                bottom,
                area.width,
                max(1, ceiling - info_height - bottom),
            )
            self._side((beside, under), side)
        if corner:  # above the playback line and the timeline, if any
            raised = mujoco.MjrRect(area.left, above, area.width, area.height - above)
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                raised,
                ascii_text(corner),
                "",
                context,
            )
        if message:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOP,
                area,
                ascii_text(message),
                "",
                context,
            )
        if hint:
            self._hint(area, hint)

    def _make_context(self, model: mujoco.MjModel) -> None:
        """Make the OpenGL resources for ``model``, as fine as the graphics ask.

        A floor drawn everywhere is made here, to reach around the camera as
        far as the model's own far plane, or twice the far plane that last
        outgrew the floor, whichever is farther.
        """
        if self._context is not None:
            self._context.free()
        quality = model.vis.quality
        quality.offsamples = 4 if self.graphics.antialiasing else 0
        detail = FINE_SHAPES if self.graphics.fine_shapes else COARSE_SHAPES
        quality.numslices, quality.numstacks = detail
        model.vis.global_.offwidth, model.vis.global_.offheight = self._buffer
        self._generation += 1
        own = model.vis.map.zfar
        self._floor_far = max(own, self._floor_far)
        model.vis.map.zfar = self._floor_far
        try:
            self._context = mujoco.MjrContext(model, self._font_scale)
        except mujoco.FatalError as error:
            raise RuntimeError(
                f"OpenGL is not available ({error}). On Linux without a "
                "display, set MUJOCO_GL=egl or MUJOCO_GL=osmesa."
            ) from None
        finally:
            model.vis.map.zfar = own
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)

    def _update_shapes(self) -> None:
        """Make MuJoCo's list of shapes for the scene as posed, from the camera.

        The far clipping plane is FAR_DISTANCES camera distances away, or the
        model's own if farther, for this list alone: MuJoCo's own would cut
        off the worlds and the floor when the camera is zoomed out. The floor
        is made again when the far plane outgrows it.
        """
        model = self.scene.model
        own = model.vis.map.zfar
        far = max(own, FAR_DISTANCES * self.camera.distance / model.stat.extent)
        if far > FLOOR_SLACK * self._floor_far:
            self._floor_far = 2 * far
            self._make_context(model)
        model.vis.map.zfar = far
        try:
            # MuJoCo moves a floor drawn everywhere under the camera of the
            # previous update; a camera that jumped, to another world, would
            # leave the floor behind in this drawing.
            mujoco.mjv_updateCamera(model, self.scene.data, self.camera, self._shapes)
            mujoco.mjv_updateScene(
                model,
                self.scene.data,
                self.option,
                None,
                self.camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                self._shapes,
            )
        finally:
            model.vis.map.zfar = own

    def _fit_buffer(self, width: int, height: int) -> None:
        """Grow the offscreen buffer to the window, which may have grown."""
        if width <= self._buffer[0] and height <= self._buffer[1]:
            return
        self._buffer = (max(width, self._buffer[0]), max(height, self._buffer[1]))
        global_ = self.scene.model.vis.global_
        global_.offwidth, global_.offheight = self._buffer
        mujoco.mjr_resizeOffscreen(*self._buffer, self._context)

    @property
    def context(self) -> mujoco.MjrContext:
        """The OpenGL resources, for drawing more on top, such as the panel."""
        return self._context

    def read_pixels(self, width: int, height: int) -> np.ndarray:
        """The last drawing as an RGB image, ``(height, width, 3)``, top row first."""
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(
            rgb, None, mujoco.MjrRect(0, 0, width, height), self._context
        )
        return np.flipud(rgb)

    def frame_all(self) -> None:
        """Look at every drawn world and its markers from a raised angle.

        The highlighted world's rings are taken in whole. A world whose
        positions diverged, to infinity or far beyond the others, is left
        out, so that it cannot take the camera with it. Following goes on, if
        on: from this distance and angle, the camera then looks at the
        highlighted world.
        """
        scene = self.scene
        extent = scene.model.stat.extent
        points = [scene.world_centre(copy) for copy in range(len(scene.worlds))]
        markers = scene.marker_positions()
        if markers is not None:
            points.extend(markers.reshape(-1, 3))
            shown = zip(
                markers[scene.highlight],
                scene.marker_radii()[scene.highlight],
                scene.recording.marker_shapes,
                strict=True,
            )
            for centre, radius, kind in shown:
                if kind == "ring":
                    points.extend(centre + radius * RING_EDGES)
        points = np.array(points)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):  # around the median, or the best world if none is near it
            for anchor in (np.median(points, axis=0), points[0]):
                near = (
                    np.linalg.norm(points - anchor, axis=1) <= FRAMED_EXTENTS * extent
                )
                if near.any():
                    break
            points = points[near]
        self.camera.elevation = -25.0
        self.camera.azimuth = 120.0
        if not len(points):
            return
        low, high = points.min(axis=0), points.max(axis=0)
        self.camera.lookat[:] = (low + high) / 2
        self.camera.distance = 0.8 * np.linalg.norm(high - low) + 1.2 * extent

    def copy_at(
        self, x: float, y: float, width: int, height: int, left: int, top: int = 0
    ) -> int | None:
        """The drawn world under a point of the last drawing, if any, as its copy.

        ``x`` and ``y`` are pixels from the bottom left of a window ``width`` by
        ``height`` whose scene starts ``left`` pixels in and ends ``top``
        pixels below its top, as ``render`` drew it.
        """
        scene, across, below = self.scene, max(1, width - left), max(1, height - top)
        point = np.zeros(3)
        geom, flex, skin = (np.zeros(1, dtype=np.int32) for _ in range(3))
        body = mujoco.mjv_select(
            scene.model,
            scene.data,
            self.option,
            across / below,
            (x - left) / across,
            y / below,
            self._shapes,
            point,
            geom,
            flex,
            skin,
        )
        return scene.copy_of_body(body)

    def look_from_above(self) -> None:
        """Look straight down, keeping the look-at point and the distance."""
        self.camera.elevation = -90.0

    def centre_on_highlight(self) -> None:
        """Look at the highlighted world, keeping the distance, the angle, and
        whether the camera follows it."""
        self._look_at(self.scene.world_centre(self.scene.highlight))

    def set_follow(self, follow: bool) -> None:
        """Keep looking at the highlighted world's centre of mass, or stop there.

        The free camera's look-at point moves with the world each frame. That
        is what MuJoCo's tracking camera does for one body; a world can have
        several root bodies.
        """
        self.follow = follow

    def move_camera(self, action: int, dx: float, dy: float) -> None:
        """Rotate, pan, or zoom by a mouse motion, as fractions of the height.

        Panning stops following, which would undo it; the camera stays above
        the floor.
        """
        if action in (mujoco.mjtMouse.mjMOUSE_MOVE_V, mujoco.mjtMouse.mjMOUSE_MOVE_H):
            self.follow = False
        mujoco.mjv_moveCamera(self.scene.model, action, dx, dy, self.camera)
        self.camera.elevation = min(self.camera.elevation, LOWEST_ELEVATION)

    def _look_at(self, point: np.ndarray) -> None:
        """Move the look-at point there, unless the point is not finite."""
        if np.isfinite(point).all():
            self.camera.lookat[:] = point

    def close(self) -> None:
        if self._context is not None:
            self._context.free()
            self._context = None

    def _keep_shadows_clean(self) -> None:
        """Mark the ghosts' and the floors' shapes as decoration, which casts
        no shadow.

        MuJoCo draws translucent shapes with full shadows, which would cover
        the floor in grey; and a plane can only shade itself, which speckles
        it. Both still receive shadows.
        """
        ghosts = self.scene.ghost_geoms
        geom, plane = int(mujoco.mjtObj.mjOBJ_GEOM), int(mujoco.mjtGeom.mjGEOM_PLANE)
        decoration = int(mujoco.mjtCatBit.mjCAT_DECOR)
        shapes = self._shapes
        for index in range(shapes.ngeom):
            shape = shapes.geoms[index]
            if shape.objtype == geom and (shape.type == plane or ghosts[shape.objid]):
                shape.category = decoration

    def _aim_shadows(self) -> None:
        """Cast the directional lights' shadows where the camera looks.

        MuJoCo draws such a light's shadows in a square, SHADOW_CLIP model
        extents each way, around the line from the light's position along its
        direction, so that worlds that walk away from it lose their shadows.
        A directional light lights everything the same from anywhere: moved
        in the list of shapes onto the line through the look-at point, at its
        distance along it, it lights as before and casts its shadows there.
        """
        lookat = self.camera.lookat
        directional = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        for index in range(self._shapes.nlight):
            light = self._shapes.lights[index]
            if light.type != directional or not light.castshadow:
                continue
            direction = light.dir / max(np.linalg.norm(light.dir), 1e-9)
            depth = (lookat - light.pos) @ direction
            if depth > 0:  # the light shines toward the look-at point
                light.pos[:] = lookat - depth * direction

    def _add_markers(self) -> None:
        """The markers, in the scene's spare slots, at the current frame's radii.

        A sphere marker is a sphere per drawn world, with a beacon sized to
        the scene over the highlighted world's, carrying the marker's name,
        so that a small target is found from any distance. A ring marker is
        drawn for the highlighted world alone: rings as wide as a world's
        range, one per drawn world, would cover the view.
        """
        scene = self.scene
        positions = scene.marker_positions()
        if positions is None:
            return
        recording = scene.recording
        positions = positions.astype(np.float64)
        radii = scene.marker_radii().astype(np.float64)
        extent = float(scene.model.stat.extent)
        height, pole, head = (share * extent for share in BEACON)
        sphere, cylinder = mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER
        ghost_rgba = scene.ghost_rgba
        for copy, (markers, sizes) in enumerate(zip(positions, radii, strict=True)):
            highlighted = copy == scene.highlight
            if not (highlighted or scene.ghosts_visible):
                continue
            rgba = MARKER_RGBA if highlighted else ghost_rgba
            for position, radius, name, kind in zip(
                markers,
                sizes,
                recording.marker_names,
                recording.marker_shapes,
                strict=True,
            ):
                if kind == "ring":
                    if highlighted:
                        self._add_ring(position, radius, extent, name)
                    continue
                self._add_shape(sphere, (radius,) * 3, position, rgba)
                if highlighted:
                    up = np.array([0.0, 0.0, height])
                    size = (pole, height / 2, 0.0)
                    self._add_shape(cylinder, size, position + up / 2, rgba)
                    self._add_shape(sphere, (head,) * 3, position + up, rgba, name)

    def _add_ring(self, centre, radius: float, extent: float, name: str) -> None:
        """A circle lying flat around ``centre``, as line segments a few pixels
        wide, with the marker's name where it meets the horizontal axis."""
        angles = np.linspace(0.0, 2 * np.pi, RING_SEGMENTS + 1)
        flat = np.zeros_like(angles)
        circle = np.stack([np.cos(angles), np.sin(angles), flat], axis=1)
        points = centre + radius * circle + (0.0, 0.0, RING_LIFT * extent)
        width = RING_PIXELS * self._font_scale / 100
        if self._offscreen_size is None:  # drawn at a share of the window
            width *= self.graphics.resolution / 100
        line = mujoco.mjtGeom.mjGEOM_LINE
        for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
            label = name if index == 0 else ""
            shape = self._add_shape(line, (0.0, 0.0, 0.0), start, MARKER_RGBA, label)
            if shape is not None:
                mujoco.mjv_connector(shape, line, width, start, end)

    def _add_shape(
        self, kind, size, position, rgba, label: str = ""
    ) -> mujoco.MjvGeom | None:
        """One decorative shape, which casts no shadow, if there is room for it."""
        shapes = self._shapes
        if shapes.ngeom == shapes.maxgeom:
            return None
        shape = shapes.geoms[shapes.ngeom]
        mujoco.mjv_initGeom(
            shape, kind, np.asarray(size, dtype=np.float64), position, _IDENTITY, rgba
        )
        shape.category = mujoco.mjtCatBit.mjCAT_DECOR
        if label:
            shape.label = ascii_text(label)
        shapes.ngeom += 1
        return shape

    def _overlay(
        self,
        area: mujoco.MjrRect,
        status: str,
        flash: str,
    ) -> tuple[int, int, int]:
        """The text corners and the timeline; the middle stays clear.

        Returns the height the timeline takes at the bottom, and the width and
        the height the top-left lines take. Lines too long for the area are
        cut.
        """
        context = self._context
        line = context.charHeight
        normal, big = mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtFont.mjFONT_BIG
        grid = mujoco.mjtGridPos
        above = self._timeline(area)
        raised = mujoco.MjrRect(area.left, above, area.width, area.height - above)
        fit = area.width - 2 * line
        info = [_cut(context, ascii_text(text), fit) for text in info_lines(self.scene)]
        mujoco.mjr_overlay(
            normal, grid.mjGRID_TOPLEFT, area, "\n".join(info), "", context
        )
        if status:
            status = _cut(context, ascii_text(status), fit)
            mujoco.mjr_overlay(
                normal, grid.mjGRID_BOTTOMLEFT, raised, status, "", context
            )
        if flash:  # above the playback line, which can reach the middle
            lift = above + 2 * context.charHeight
            clear = mujoco.MjrRect(area.left, lift, area.width, area.height - lift)
            mujoco.mjr_overlay(
                big, grid.mjGRID_BOTTOM, clear, ascii_text(flash), "", context
            )
        width = max(text_width(context, text) for text in info) + 2 * line
        return above, width, len(info) * (line + line // 3) + line // 2

    def _side(self, areas: tuple[mujoco.MjrRect, ...], lines: list[str]) -> None:
        """Lines at the top right of whichever area shows them best, in columns.

        MuJoCo's overlay holds 500 characters at most, so each line is drawn
        on its own. The area that leaves out the fewest lines, then cuts the
        fewest, is used, the first one on a tie.
        """
        context = self._context
        line = context.charHeight
        step = line + line // 4
        layouts = [(self._columns(area, lines), area) for area in areas]
        (kept, _, _), area = min(layouts, key=lambda layout: layout[0][1:])
        x = area.left + area.width - sum(width for _, width in kept)
        top = area.bottom + area.height
        for column, width in kept:
            height = len(column) * step + line // 2
            mujoco.mjr_rectangle(
                mujoco.MjrRect(x, top - height, width, height), *SHADE_RGBA
            )
            for index, text in enumerate(column):
                rect = mujoco.MjrRect(
                    x + line // 2,
                    top - line // 4 - (index + 1) * step,
                    text_width(context, text) + 4,
                    step,
                )
                _write(context, rect, text)
            x += width

    def _columns(
        self, area: mujoco.MjrRect, lines: list[str]
    ) -> tuple[list[tuple[list[str], int]], int, int]:
        """How lines fit an area in columns: the columns that fit, each with its
        width, then how many lines are left out and how many are cut.

        The columns the lines need share the area's width, down to ten
        characters' height each; a line too long for its column is cut, and
        lines beyond the columns that fit are counted on the last one shown.
        """
        context = self._context
        line = context.charHeight
        step = line + line // 4
        rows = max(1, (area.height - line // 2) // step)
        needed = -(-len(lines) // rows)  # columns, rounded up
        widest = min(max(area.width // needed, 10 * line), area.width) - line
        whole = [ascii_text(text) for text in lines]
        texts = [_cut(context, text, widest) for text in whole]
        cut = sum(text != full for text, full in zip(texts, whole, strict=True))
        columns = [texts[start : start + rows] for start in range(0, len(texts), rows)]
        kept: list[tuple[list[str], int]] = []
        for column in columns:
            width = max(text_width(context, text) for text in column) + line
            if sum(width for _, width in kept) + width > area.width:
                break
            kept.append((column, width))
        shown = sum(len(column) for column, _ in kept)
        if kept and shown < len(texts):
            kept[-1][0][-1] = f"... {len(texts) - shown + 1} more"
        return kept, len(texts) - shown, cut

    def frame_at(self, x: float, area_left: int, area_width: int) -> int:
        """The frame under ``x`` on the timeline along an area's bottom."""
        frames = self.scene.recording.frame_count
        share = (x - area_left) / max(1, area_width)
        return int(np.clip(share * frames, 0, frames - 1))

    def _timeline(self, area: mujoco.MjrRect) -> int:
        """Draw the timeline along the area's bottom; return the height it takes.

        An event label that would cover the one before it is left out; its
        tick stays.
        """
        scene, context = self.scene, self._context
        recording = scene.recording
        frames = recording.frame_count
        left, width = area.left, area.width
        bar = max(6, context.charHeight // 2)

        def x_at(frame: int) -> int:  # frames run from 0 to ``frames`` inclusive
            return left + min(width - 2, round(frame / frames * width))

        mujoco.mjr_rectangle(mujoco.MjrRect(left, 0, width, bar), *TRACK_RGBA)
        filled = round((scene.frame_index + 1) / frames * width)
        mujoco.mjr_rectangle(mujoco.MjrRect(left, 0, filled, bar), *FILLED_RGBA)
        world = scene.worlds[scene.highlight]
        for frame in np.flatnonzero(recording.episode_start[:, world]):
            tick = mujoco.MjrRect(x_at(frame), 0, 2, max(2, bar // 2))
            mujoco.mjr_rectangle(tick, *EPISODE_RGB, 1.0)
        if recording.event_frames is None:
            return bar
        row = context.charHeight + 4
        free = left  # where the next label may start
        for frame, label in sorted(
            zip(recording.event_frames, recording.event_labels, strict=True)
        ):
            x = x_at(frame)
            mujoco.mjr_rectangle(mujoco.MjrRect(x, 0, 2, bar + 3), *EVENT_RGB, 1.0)
            label = ascii_text(label)
            size = text_width(context, label) + 8
            start = int(np.clip(x - size // 2, left, max(left, left + width - size)))
            if start < free:
                continue
            rect = mujoco.MjrRect(start, bar + 3, size, row)
            _write(context, rect, label, shade=(0.1, 0.1, 0.1, 0.7), rgb=EVENT_RGB)
            free = start + size + 4
        return bar + 3 + row

    def _hint(self, area: mujoco.MjrRect, hint: str) -> None:
        """A large line in the middle of the area, on a dark band."""
        context, hint = self._context, ascii_text(hint)
        width = text_width(context, hint, big=True) + 2 * context.charHeight
        height = context.charHeightBig + context.charHeight
        rect = mujoco.MjrRect(
            area.left + (area.width - width) // 2,
            area.bottom + (area.height - height) // 2,
            width,
            height,
        )
        _write(context, rect, hint, mujoco.mjtFont.mjFONT_BIG, (0.07, 0.08, 0.1, 0.8))


def info_lines(scene: ComposedScene) -> list[str]:
    """The overlay's top-left lines: the file, the frame, the highlight, the set."""
    recording = scene.recording
    world = scene.worlds[scene.highlight]
    lines = [recording.title]
    if recording.frame_info is not None:
        values = recording.frame_info[scene.frame_index]
        lines.append(
            " | ".join(
                f"{name} {_number(value)}"
                for name, value in zip(recording.frame_info_names, values, strict=True)
            )
        )
    rank, ranked = world_rank(recording, world)
    highlighted = [
        f"world {recording.world_ids[world]}",
        f"rank {rank:,} of {ranked:,}",
        f"{recording.score_name} {recording.score[world]:+.3g}",
    ]
    if recording.level is not None:
        level = f"level {recording.level[world]}"
        if recording.level_count is not None:
            level += f" of {recording.level_count}"
        highlighted.append(level)
    lines.append(" | ".join(highlighted))
    ghosts = len(scene.worlds) - 1
    hidden = " (hidden)" if ghosts and not scene.ghosts_visible else ""
    lines.append(
        f"{len(scene.worlds)} of {recording.world_count:,} worlds drawn | "
        f"{ghosts} ghosts{hidden}"
    )
    return lines


def world_rank(recording: Recording, world: int) -> tuple[int, int]:
    """A world's rank, 1 for the best, and among how many worlds.

    The producer's ranks among all its worlds when the file gives them, else
    the rank by score among the file's own worlds.
    """
    if recording.rank is not None:
        return int(recording.rank[world]), recording.ranked_worlds
    order = np.argsort(-recording.score, kind="stable")
    return int(np.flatnonzero(order == world)[0]) + 1, recording.world_count


def setup_lines(setup: dict, path: str = "") -> list[str]:
    """A setup object as ``key = value`` lines.

    The values of a nested object follow a heading of its dotted path,
    indented, so that long paths are not repeated on every line.
    """
    values = {key: value for key, value in setup.items() if not isinstance(value, dict)}
    lines = [path] if path and values else []
    for key, value in values.items():
        shown = value if isinstance(value, str) else json.dumps(value)
        lines.append(f"{'  ' if path else ''}{key} = {shown}")
    for key, value in setup.items():
        if isinstance(value, dict):
            lines += setup_lines(value, f"{path}.{key}" if path else key)
    return lines


def _write(
    context: mujoco.MjrContext,
    rect: mujoco.MjrRect,
    text: str,
    font: int = mujoco.mjtFont.mjFONT_NORMAL,
    shade: tuple[float, ...] = (0, 0, 0, 0),
    rgb: tuple[float, ...] = TEXT_RGB,
) -> None:
    """Text in the middle of a rectangle, on a shade (by default none)."""
    mujoco.mjr_label(rect, font, text, *shade, *rgb, context)


def _cut(context: mujoco.MjrContext, text: str, width: int) -> str:
    """The text, shortened with "..." when it is wider than ``width`` pixels.

    One pass over the characters, so that a very long title or setup value
    cannot slow every frame down.
    """
    if text_width(context, text) <= width:
        return text
    widths, room = context.charWidth, width - text_width(context, "...")
    used = 0
    for index, char in enumerate(text):
        used += int(widths[ord(char)]) if ord(char) < len(widths) else 0
        if used > room:
            return text[:index] + "..."
    return text


def _number(value: float) -> str:
    """Whole numbers with thousands separators, others to four digits."""
    return f"{int(value):,}" if float(value).is_integer() else f"{value:.4g}"


def _add_shadows_and_reflections(model: mujoco.MjModel) -> None:
    """Give a scene a shadow and a floor reflection for the switches to show.

    When no light casts shadows, the first one does, over an area wide enough
    for the worlds; a floor (a plane with a material) that does not reflect
    reflects slightly. Whether either is drawn is still the graphics' choice,
    through MuJoCo's render flags.
    """
    if model.nlight and not model.light_castshadow.any():
        model.light_castshadow[0] = 1
    model.vis.map.shadowclip = max(model.vis.map.shadowclip, SHADOW_CLIP)
    for geom in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE):
        material = model.geom_matid[geom]
        if material >= 0 and model.mat_reflectance[material] == 0:
            model.mat_reflectance[material] = FLOOR_REFLECTANCE


def _spread_floors(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Draw the floor everywhere, as the simulation has it.

    A plane collides everywhere, but MuJoCo draws it at its size, so that a
    world that walks past the floor's edge seems to float. A plane of size 0
    is drawn everywhere: MuJoCo moves it under the camera by whole texture
    repeats, so that its pattern stays in place. The floor, the lowest of the
    static scene's planes that face up, is given size 0, and its texture the
    repeats that keep its scale, where a plane of size 0 repeats it every 2 /
    repeats. A floor keeps its size if those repeats would change other
    shapes, which share its material. ``data`` must be posed.

    MuJoCo hazes the horizon of a floor drawn everywhere, in white unless the
    model says otherwise, which would draw a bright band across a dark sky:
    the haze of a floor spread here takes the sky's colour at the horizon.
    """
    static = model.body_weldid[model.geom_bodyid] == 0
    up = data.geom_xmat[:, 8] > LEVEL  # the height of the plane's normal
    planes = np.flatnonzero(
        (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) & static & up
    )
    if not len(planes):
        return
    heights = data.geom_xpos[planes, 2]
    lowest = np.isclose(heights, heights.min(), rtol=0, atol=1e-6 * model.stat.extent)
    spread = False
    for geom in planes[lowest]:
        size = model.geom_size[geom, :2].copy()
        if not (size > 0).all():
            continue  # drawn everywhere already
        material = model.geom_matid[geom]
        if material >= 0:
            repeats = model.mat_texrepeat[material]
            kept = np.where(repeats > 0, repeats, 1.0)  # MuJoCo takes 0 as 1
            if not model.mat_texuniform[material]:  # repeats over the plane
                kept = kept / size
            if not np.allclose(kept, repeats):
                if np.count_nonzero(model.geom_matid == material) > 1:
                    continue
                model.mat_texrepeat[material] = kept
        model.geom_size[geom, :2] = 0
        spread = True
    horizon = _sky_at_the_horizon(model) if spread else None
    if horizon is not None:
        model.vis.rgba.haze[:3] = horizon


def _sky_at_the_horizon(model: mujoco.MjModel) -> np.ndarray | None:
    """The mean colour of the sky MuJoCo draws, around the horizon, if any.

    A sky is a cube map, its six faces one above the other; MuJoCo turns it
    so that the cube's third and fourth faces are up and down, and the middle
    rows of the others are the horizon.
    """
    skies = np.flatnonzero(model.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    if not len(skies):
        return None
    sky = skies[0]  # the one MuJoCo draws
    width, height = model.tex_width[sky], model.tex_height[sky]
    channels = model.tex_nchannel[sky]
    if height != 6 * width or channels < 3:
        return None
    start = model.tex_adr[sky]
    faces = model.tex_data[start : start + height * width * channels]
    faces = faces.reshape(6, width, width, channels)
    middle = faces[[0, 1, 4, 5], width // 2 - 1 : width // 2 + 1, :, :3]
    return middle.reshape(-1, 3).mean(axis=0) / 255
