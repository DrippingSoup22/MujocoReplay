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
import unicodedata

import mujoco
import numpy as np

from mujoco_replay.scene import GHOST_RGBA, ComposedScene
from mujoco_replay.settings import QUALITY, Graphics
from mujoco_replay.ui import text_width

# The highlighted world's markers; the ghosts' are ghost grey.
MARKER_RGBA = np.array([0.95, 0.15, 0.65, 0.9], dtype=np.float32)
# Facets around and along MuJoCo's round shapes: its defaults, and a coarse set
# that draws about a quarter of the triangles.
FINE_SHAPES = (28, 16)
COARSE_SHAPES = (12, 6)
# A marker's beacon, in shares of the scene's extent: its height, the radius
# of its pole, and the radius of its head.
BEACON = (0.5, 0.006, 0.04)
# The reflection given to a floor that has none, for the reflections switch.
FLOOR_REFLECTANCE = 0.15
# Room in the scene for the shapes MuJoCo adds itself, such as light glyphs.
SPARE_SHAPES = 200
TRACK_RGBA = (0.12, 0.12, 0.12, 0.75)
FILLED_RGBA = (0.85, 0.85, 0.85, 0.9)
EVENT_RGB = (1.0, 0.72, 0.2)
EPISODE_RGB = (0.25, 0.6, 1.0)
TEXT_RGB = (0.92, 0.93, 0.95)
# MuJoCo's fonts hold ASCII only; these characters have close equivalents.
ASCII_EQUIVALENTS = str.maketrans(
    {"·": "|", "×": "x", "–": "-", "—": "-", "…": "...", "’": "'", "“": '"', "”": '"'}
)
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
        self.graphics = graphics
        self.scene: ComposedScene | None = None
        self._offscreen_size = offscreen_size
        global_ = scene.model.vis.global_
        self._buffer = offscreen_size or (global_.offwidth, global_.offheight)
        self._font_scale = font_scale
        self._context: mujoco.MjrContext | None = None
        self._shapes: mujoco.MjvScene | None = None
        self.show(scene)
        self.frame_all()

    def show(self, scene: ComposedScene) -> None:
        """Draw ``scene`` from now on; a new model gets new OpenGL resources."""
        new_model = self.scene is None or scene.model is not self.scene.model
        if new_model:
            _add_shadows_and_reflections(scene.model)
            self._make_context(scene.model)
        markers = len(scene.recording.marker_names or ())
        shapes = (len(scene.worlds) + 2) * markers  # a beacon is two more
        needed = scene.model.ngeom + shapes + SPARE_SHAPES
        if new_model or self._shapes.maxgeom < needed:
            self._shapes = mujoco.MjvScene(scene.model, maxgeom=needed)
        self.scene = scene

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
        setup: bool = False,
        corner: str = "",
        inset: int = 0,
        message: str = "",
        hint: str = "",
    ) -> None:
        """Draw the scene as posed now, with the overlay unless ``hud`` is off.

        ``corner`` is a line for the bottom right, such as the frame rate. The
        overlay keeps ``inset`` pixels clear on the left, for the panel.
        ``message`` is shown at the top and ``hint`` large in the middle, even
        without the overlay.
        """
        scene, shapes, context = self.scene, self._shapes, self._context
        if self.follow:
            self.camera.lookat[:] = scene.world_centre(scene.highlight)
        mujoco.mjv_updateScene(
            scene.model,
            scene.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            shapes,
        )
        shapes.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.graphics.shadows
        shapes.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.graphics.reflections
        self._quiet_ghosts()
        if self.markers_visible:
            self._add_markers()
        viewport = mujoco.MjrRect(0, 0, width, height)
        if self._offscreen_size is None:  # draw a share offscreen, scale it up
            share = self.graphics.resolution / 100
            drawn = mujoco.MjrRect(
                0, 0, max(1, round(width * share)), max(1, round(height * share))
            )
            self._fit_buffer(width, height)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
            mujoco.mjr_render(drawn, shapes, context)
            mujoco.mjr_blitBuffer(drawn, viewport, 1, 0, context)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, context)
        else:
            mujoco.mjr_render(viewport, shapes, context)
        area = mujoco.MjrRect(inset, 0, max(1, width - inset), height)
        above = self._overlay(area, status, flash, setup) if hud else 0
        if corner:  # above the timeline, if any
            raised = mujoco.MjrRect(area.left, above, area.width, area.height - above)
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                raised,
                _ascii(corner),
                "",
                context,
            )
        if message:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOP,
                area,
                _ascii(message),
                "",
                context,
            )
        if hint:
            self._hint(area, hint)

    def _make_context(self, model: mujoco.MjModel) -> None:
        """Make the OpenGL resources for ``model``, as fine as the graphics ask."""
        if self._context is not None:
            self._context.free()
        quality = model.vis.quality
        quality.offsamples = 4 if self.graphics.antialiasing else 0
        detail = FINE_SHAPES if self.graphics.fine_shapes else COARSE_SHAPES
        quality.numslices, quality.numstacks = detail
        model.vis.global_.offwidth, model.vis.global_.offheight = self._buffer
        try:
            self._context = mujoco.MjrContext(model, self._font_scale)
        except mujoco.FatalError as error:
            raise RuntimeError(
                f"OpenGL is not available ({error}). On Linux without a "
                "display, set MUJOCO_GL=egl or MUJOCO_GL=osmesa."
            ) from None
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)

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
        """Look at every drawn world and its markers from a raised angle."""
        scene = self.scene
        points = [scene.world_centre(copy) for copy in range(len(scene.worlds))]
        markers = scene.marker_positions()
        if markers is not None:
            points.extend(markers.reshape(-1, 3))
        low, high = np.min(points, axis=0), np.max(points, axis=0)
        self.follow = False
        self.camera.lookat[:] = (low + high) / 2
        spread = np.linalg.norm(high - low)
        self.camera.distance = 0.8 * spread + 1.2 * scene.model.stat.extent
        self.camera.elevation = -25.0
        self.camera.azimuth = 120.0

    def centre_on_highlight(self) -> None:
        """Look at the highlighted world, keeping the distance and angle."""
        self.follow = False
        self.camera.lookat[:] = self.scene.world_centre(self.scene.highlight)

    def set_follow(self, follow: bool) -> None:
        """Keep looking at the highlighted world's centre of mass, or stop there.

        The free camera's look-at point moves with the world each frame. That
        is what MuJoCo's tracking camera does for one body; a world can have
        several root bodies.
        """
        self.follow = follow

    def move_camera(self, action: int, dx: float, dy: float) -> None:
        """Rotate, pan, or zoom by a mouse motion, as fractions of the height."""
        mujoco.mjv_moveCamera(self.scene.model, action, dx, dy, self.camera)

    def close(self) -> None:
        if self._context is not None:
            self._context.free()
            self._context = None

    def _quiet_ghosts(self) -> None:
        """Mark the ghosts' shapes as decoration, which casts no shadow.

        MuJoCo draws translucent shapes with full shadows, which would cover
        the floor in grey; it casts none from decoration.
        """
        ghosts = self.scene.ghost_geoms
        if not ghosts.any():
            return
        geom = int(mujoco.mjtObj.mjOBJ_GEOM)
        decoration = int(mujoco.mjtCatBit.mjCAT_DECOR)
        shapes = self._shapes
        for index in range(shapes.ngeom):
            shape = shapes.geoms[index]
            if shape.objtype == geom and ghosts[shape.objid]:
                shape.category = decoration

    def _add_markers(self) -> None:
        """The markers, in the scene's spare slots: a sphere of each marker's
        radius per drawn world, and over the highlighted world's markers a
        beacon sized to the scene, carrying the marker's name, so that a small
        target is found from any distance."""
        scene = self.scene
        positions = scene.marker_positions()
        if positions is None:
            return
        recording = scene.recording
        radii = recording.marker_radius.astype(np.float64)
        extent = float(scene.model.stat.extent)
        height, pole, head = (share * extent for share in BEACON)
        sphere, cylinder = mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER
        for copy, markers in enumerate(positions):
            highlighted = copy == scene.highlight
            if not (highlighted or scene.ghosts_visible):
                continue
            rgba = MARKER_RGBA if highlighted else GHOST_RGBA
            for position, radius, name in zip(
                markers.astype(np.float64), radii, recording.marker_names, strict=True
            ):
                self._add_shape(sphere, (radius, radius, radius), position, rgba)
                if highlighted:
                    up = np.array([0.0, 0.0, height])
                    size = (pole, height / 2, 0.0)
                    self._add_shape(cylinder, size, position + up / 2, rgba)
                    self._add_shape(sphere, (head,) * 3, position + up, rgba, name)

    def _add_shape(self, kind, size, position, rgba, label: str = "") -> None:
        """One decorative shape, which casts no shadow, if there is room for it."""
        shapes = self._shapes
        if shapes.ngeom == shapes.maxgeom:
            return
        shape = shapes.geoms[shapes.ngeom]
        mujoco.mjv_initGeom(
            shape, kind, np.asarray(size, dtype=np.float64), position, _IDENTITY, rgba
        )
        shape.category = mujoco.mjtCatBit.mjCAT_DECOR
        if label:
            shape.label = _ascii(label)
        shapes.ngeom += 1

    def _overlay(
        self,
        area: mujoco.MjrRect,
        status: str,
        flash: str,
        setup: bool,
    ) -> int:
        """The text corners and the timeline; the middle stays clear.

        Returns the height the timeline takes at the bottom.
        """
        context = self._context
        normal, big = mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtFont.mjFONT_BIG
        grid = mujoco.mjtGridPos
        above = self._timeline(area)
        raised = mujoco.MjrRect(area.left, above, area.width, area.height - above)
        info = "\n".join(info_lines(self.scene))
        mujoco.mjr_overlay(normal, grid.mjGRID_TOPLEFT, area, _ascii(info), "", context)
        if status:
            mujoco.mjr_overlay(
                normal, grid.mjGRID_BOTTOMLEFT, raised, _ascii(status), "", context
            )
        if flash:  # above the playback line, which can reach the middle
            lift = above + 2 * context.charHeight
            clear = mujoco.MjrRect(area.left, lift, area.width, area.height - lift)
            mujoco.mjr_overlay(
                big, grid.mjGRID_BOTTOM, clear, _ascii(flash), "", context
            )
        if setup:
            rows = max(1, area.height // context.charHeight - 2)
            lines = setup_lines(self.scene.recording.setup) or ["(no setup)"]
            if len(lines) > rows:
                lines = lines[: rows - 1] + [f"... {len(lines) - rows + 1} more"]
            mujoco.mjr_overlay(
                normal,
                grid.mjGRID_TOPRIGHT,
                area,
                _ascii("\n".join(lines)),
                "",
                context,
            )
        return above

    def _timeline(self, area: mujoco.MjrRect) -> int:
        """Draw the timeline along the area's bottom; return the height it takes."""
        scene, context = self.scene, self._context
        recording = scene.recording
        frames = recording.frame_count
        left, width = area.left, area.width
        bar = max(4, area.height // 100)

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
        for frame, label in zip(
            recording.event_frames, recording.event_labels, strict=True
        ):
            x = x_at(frame)
            mujoco.mjr_rectangle(mujoco.MjrRect(x, 0, 2, bar + 3), *EVENT_RGB, 1.0)
            label = _ascii(label)
            size = text_width(context, label) + 8
            start = int(np.clip(x - size // 2, left, max(left, left + width - size)))
            mujoco.mjr_label(
                mujoco.MjrRect(start, bar + 3, size, row),
                mujoco.mjtFont.mjFONT_NORMAL,
                label,
                0.1,
                0.1,
                0.1,
                0.7,
                *EVENT_RGB,
                context,
            )
        return bar + 3 + row

    def _hint(self, area: mujoco.MjrRect, hint: str) -> None:
        """A large line in the middle of the area, on a dark band."""
        context, hint = self._context, _ascii(hint)
        width = text_width(context, hint, big=True) + 2 * context.charHeight
        height = context.charHeightBig + context.charHeight
        rect = mujoco.MjrRect(
            area.left + (area.width - width) // 2,
            area.bottom + (area.height - height) // 2,
            width,
            height,
        )
        mujoco.mjr_label(
            rect,
            mujoco.mjtFont.mjFONT_BIG,
            hint,
            0.07,
            0.08,
            0.1,
            0.8,
            *TEXT_RGB,
            context,
        )


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
    rank = int(np.flatnonzero(np.argsort(-recording.score, kind="stable") == world)[0])
    highlighted = [
        f"world {recording.world_ids[world]}",
        f"rank {rank + 1} of {recording.world_count:,}",
        f"{recording.score_name} {recording.score[world]:+.3g}",
    ]
    if recording.level is not None:
        highlighted.append(f"level {recording.level[world]} of {recording.level.max()}")
    lines.append(" | ".join(highlighted))
    ghosts = len(scene.worlds) - 1
    hidden = " (hidden)" if ghosts and not scene.ghosts_visible else ""
    lines.append(
        f"{len(scene.worlds)} of {recording.world_count:,} worlds drawn | "
        f"{ghosts} ghosts{hidden}"
    )
    return lines


def setup_lines(setup: dict, prefix: str = "") -> list[str]:
    """A setup object flattened to ``key = value`` lines, nested keys dotted."""
    lines = []
    for key, value in setup.items():
        if isinstance(value, dict):
            lines += setup_lines(value, f"{prefix}{key}.")
        else:
            shown = value if isinstance(value, str) else json.dumps(value)
            lines.append(f"{prefix}{key} = {shown}")
    return lines


def _number(value: float) -> str:
    """Whole numbers with thousands separators, others to four digits."""
    return f"{int(value):,}" if float(value).is_integer() else f"{value:.4g}"


def _ascii(text: str) -> str:
    """Text that MuJoCo's ASCII-only fonts can draw: accents dropped, others "?"."""
    decomposed = unicodedata.normalize("NFKD", text.translate(ASCII_EQUIVALENTS))
    return "".join(
        char if char.isascii() else "" if unicodedata.combining(char) else "?"
        for char in decomposed
    )


def _add_shadows_and_reflections(model: mujoco.MjModel) -> None:
    """Give a scene a shadow and a floor reflection for the switches to show.

    When no light casts shadows, the first one does; a floor (a plane with a
    material) that does not reflect reflects slightly. Whether either is
    drawn is still the graphics' choice, through MuJoCo's render flags.
    """
    if model.nlight and not model.light_castshadow.any():
        model.light_castshadow[0] = 1
    for geom in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE):
        material = model.geom_matid[geom]
        if material >= 0 and model.mat_reflectance[material] == 0:
            model.mat_reflectance[material] = FLOOR_REFLECTANCE
