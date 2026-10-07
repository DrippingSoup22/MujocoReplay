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

# The highlighted world's markers; the ghosts' are ghost grey.
MARKER_RGBA = np.array([0.95, 0.15, 0.65, 0.9], dtype=np.float32)
# Room in the scene for the shapes MuJoCo adds itself, such as light glyphs.
SPARE_SHAPES = 200
TRACK_RGBA = (0.12, 0.12, 0.12, 0.75)
FILLED_RGBA = (0.85, 0.85, 0.85, 0.9)
EVENT_RGB = (1.0, 0.72, 0.2)
EPISODE_RGB = (0.25, 0.6, 1.0)
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
    ``offscreen_context``. With ``offscreen_size``, drawing goes to MuJoCo's
    offscreen buffer of that size, which ``read_pixels`` reads back.
    """

    def __init__(
        self,
        scene: ComposedScene,
        offscreen_size: tuple[int, int] | None = None,
        font_scale: int = 150,
    ) -> None:
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.markers_visible = True
        self.follow = False
        self.scene: ComposedScene | None = None
        self._offscreen_size = offscreen_size
        self._font_scale = font_scale
        self._context: mujoco.MjrContext | None = None
        self._shapes: mujoco.MjvScene | None = None
        self.show(scene)
        self.frame_all()

    def show(self, scene: ComposedScene) -> None:
        """Draw ``scene`` from now on; a new model gets new OpenGL resources."""
        new_model = self.scene is None or scene.model is not self.scene.model
        if new_model:
            if self._context is not None:
                self._context.free()
            if self._offscreen_size is not None:
                width, height = self._offscreen_size
                scene.model.vis.global_.offwidth = width
                scene.model.vis.global_.offheight = height
            try:
                self._context = mujoco.MjrContext(scene.model, self._font_scale)
            except mujoco.FatalError as error:
                raise RuntimeError(
                    f"OpenGL is not available ({error}). On Linux without a "
                    "display, set MUJOCO_GL=egl or MUJOCO_GL=osmesa."
                ) from None
            if self._offscreen_size is not None:
                mujoco.mjr_setBuffer(
                    mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context
                )
        markers = len(scene.recording.marker_names or ())
        needed = scene.model.ngeom + len(scene.worlds) * markers + SPARE_SHAPES
        if new_model or self._shapes.maxgeom < needed:
            self._shapes = mujoco.MjvScene(scene.model, maxgeom=needed)
        self.scene = scene

    def render(
        self,
        width: int,
        height: int,
        status: str = "",
        flash: str = "",
        hud: bool = True,
        setup: bool = False,
    ) -> None:
        """Draw the scene as posed now, with the overlay unless ``hud`` is off."""
        scene = self.scene
        if self.follow:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = scene.root_body(scene.highlight)
        mujoco.mjv_updateScene(
            scene.model,
            scene.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._shapes,
        )
        self._quiet_ghosts()
        if self.markers_visible:
            self._add_markers()
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self._shapes, self._context)
        if hud:
            self._overlay(viewport, status, flash, setup)

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
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = (low + high) / 2
        spread = np.linalg.norm(high - low)
        self.camera.distance = 0.8 * spread + 1.2 * scene.model.stat.extent
        self.camera.elevation = -25.0
        self.camera.azimuth = 120.0

    def centre_on_highlight(self) -> None:
        """Look at the highlighted world, keeping the distance and angle."""
        self.follow = False
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = self.scene.world_centre(self.scene.highlight)

    def set_follow(self, follow: bool) -> None:
        """Track the highlighted world's centre of mass, or stop where it is."""
        self.follow = follow
        if not follow:  # the tracking camera left its last look-at point behind
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE

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
        """One sphere per drawn world and marker, in the scene's spare slots."""
        scene, shapes = self.scene, self._shapes
        positions = scene.marker_positions()
        if positions is None:
            return
        radii = scene.recording.marker_radius.astype(np.float64)
        for copy, markers in enumerate(positions):
            if copy == scene.highlight:
                rgba = MARKER_RGBA
            elif scene.ghosts_visible:
                rgba = GHOST_RGBA
            else:
                continue
            for position, radius in zip(markers, radii, strict=True):
                if shapes.ngeom == shapes.maxgeom:
                    return
                shape = shapes.geoms[shapes.ngeom]
                mujoco.mjv_initGeom(
                    shape,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.full(3, radius),
                    position.astype(np.float64),
                    _IDENTITY,
                    rgba,
                )
                shape.category = mujoco.mjtCatBit.mjCAT_DECOR
                shapes.ngeom += 1

    def _overlay(
        self, viewport: mujoco.MjrRect, status: str, flash: str, setup: bool
    ) -> None:
        """The text corners and the timeline; the middle stays clear."""
        context = self._context
        normal, big = mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtFont.mjFONT_BIG
        grid = mujoco.mjtGridPos
        above = self._timeline(viewport.width, viewport.height)
        raised = mujoco.MjrRect(0, above, viewport.width, viewport.height - above)
        info = "\n".join(info_lines(self.scene))
        mujoco.mjr_overlay(
            normal, grid.mjGRID_TOPLEFT, viewport, _ascii(info), "", context
        )
        if status:
            mujoco.mjr_overlay(
                normal, grid.mjGRID_BOTTOMLEFT, raised, _ascii(status), "", context
            )
        if flash:  # above the playback line, which can reach the middle
            lift = above + 2 * context.charHeight
            clear = mujoco.MjrRect(0, lift, viewport.width, viewport.height - lift)
            mujoco.mjr_overlay(
                big, grid.mjGRID_BOTTOM, clear, _ascii(flash), "", context
            )
        if setup:
            rows = max(1, viewport.height // context.charHeight - 2)
            lines = setup_lines(self.scene.recording.setup) or ["(no setup)"]
            if len(lines) > rows:
                lines = lines[: rows - 1] + [f"... {len(lines) - rows + 1} more"]
            mujoco.mjr_overlay(
                normal,
                grid.mjGRID_TOPRIGHT,
                viewport,
                _ascii("\n".join(lines)),
                "",
                context,
            )

    def _timeline(self, width: int, height: int) -> int:
        """Draw the timeline along the bottom; return the height it takes."""
        scene, context = self.scene, self._context
        recording = scene.recording
        frames = recording.frame_count
        bar = max(4, height // 100)

        def x_at(frame: int) -> int:  # frames run from 0 to ``frames`` inclusive
            return min(width - 2, round(frame / frames * width))

        mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, bar), *TRACK_RGBA)
        filled = round((scene.frame_index + 1) / frames * width)
        mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, filled, bar), *FILLED_RGBA)
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
            text = _ascii(label)
            text_width = sum(int(context.charWidth[ord(char)]) for char in text) + 8
            left = int(np.clip(x - text_width // 2, 0, max(0, width - text_width)))
            mujoco.mjr_label(
                mujoco.MjrRect(left, bar + 3, text_width, row),
                mujoco.mjtFont.mjFONT_NORMAL,
                text,
                0.1,
                0.1,
                0.1,
                0.7,
                *EVENT_RGB,
                context,
            )
        return bar + 3 + row


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
