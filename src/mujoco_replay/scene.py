"""The composite scene: a model's static part once, its moving part once per world.

A recording's model describes one world. To draw many worlds in one picture,
the model is composed again with ``MjSpec``, MuJoCo's editable form of a model:
the static scene (the floor, the lights, every root body without joints) once,
and one copy of the replicated bodies per drawn world, their names prefixed
``w0_``, ``w1_``, and so on. Each frame, every world's recorded ``qpos`` is
written into its copy's joints and MuJoCo's kinematics computes where every
body is; nothing is simulated. The highlighted world keeps the model's colours
and the others are grey ghosts. docs/design.md explains each step.
"""

import contextlib
import hashlib
import math
import os
from pathlib import Path

import mujoco
import numpy as np

from mujoco_replay.recording import Recording, RecordingError

# Raised whenever the composition changes what it compiles, so that composites
# cached by an older version are not used; and how many composites are kept.
CACHE_VERSION = 2
CACHE_ENTRIES = 16

# The ghosts' grey, and their alpha at each strength when up to FADE_FROM
# worlds are drawn; with more, it falls with the square root of their number,
# so that the ghosts in front of the highlighted world leave it readable. A
# ghost shape's alpha is the ghosts' alpha times the shape's own, so that a
# shape the model hides stays hidden.
GHOST_RGB = (0.55, 0.55, 0.55)
GHOST_ALPHAS = {"hidden": 0.0, "faint": 0.07, "normal": 0.15, "strong": 0.35}
FADE_FROM = 8
# Parts of a model that drawing does not need. They name bodies, joints, and
# sites that the composition removes or renames, so they are deleted before
# compiling.
UNUSED_PARTS = (
    "actuators",
    "sensors",
    "tendons",
    "equalities",
    "pairs",
    "excludes",
    "keys",
    "tuples",
    "skins",
    "flexes",
)
# Position coordinates per joint type, indexed by mjtJoint: free, ball, slide,
# hinge.
QPOS_SIZES = (7, 4, 1, 1)
# A shape with this colour is drawn in its material's colour instead.
DEFAULT_GEOM_RGBA = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)


class ComposedScene:
    """One recording's drawn worlds composed into one model, posed frame by frame.

    ``worlds`` are indices into the file's worlds, best first, as
    ``choose_worlds`` returns them; copy ``k`` of the replicated bodies shows
    world ``worlds[k]``. ``highlight`` is the copy drawn in colour. ``model``
    and ``data`` are ordinary MuJoCo objects for the renderer to draw;
    ``ghost_geoms`` marks the shapes currently drawn as ghosts, and
    ``ghost_rgba`` is the ghosts' colour, which their markers share.
    """

    def __init__(
        self, recording: Recording, worlds: np.ndarray, cache: Path | None = None
    ) -> None:
        """Compose the scene, or load it from the ``cache`` folder when given.

        ``cached`` tells which happened. The cache holds compiled composites
        in MuJoCo's binary format, keyed by the model, the replicated bodies,
        the number of copies, and the MuJoCo version.
        """
        self.recording = recording
        self.worlds = np.asarray(worlds, dtype=np.int64)
        self.highlight = 0
        self.ghosts = "normal"  # a strength of GHOST_ALPHAS
        self.frame_index = 0

        spec = self._parse()
        original = self._compile(self._parse(), "model_xml")
        if original.nq != recording.position_count:
            raise self._error(
                f"qpos has {recording.position_count} positions per world, "
                f"but the model in model_xml has {original.nq}"
            )
        roots = self._replicated_roots(spec)
        bodies, joints = _subtree_names(spec, roots)
        prefixes = [f"w{copy}_" for copy in range(len(self.worlds))]
        stored = None
        if cache is not None:
            stored = cache / f"{_cache_key(recording, roots, len(prefixes))}.mjb"
        self.model = _load_composite(stored)
        self.cached = self.model is not None
        if self.model is None:
            self.model = self._compose(spec, roots, bodies, prefixes)
            _save_composite(self.model, stored)
        self.data = mujoco.MjData(self.model)

        self._map_positions(original, joints)
        # Without replicated bodies, the world body stands in for each copy.
        self._root_bodies = [
            np.array([self.model.body(prefix + name).id for name in roots] or [0])
            for prefix in prefixes
        ]
        self._body_copy = np.full(self.model.nbody, -1)
        for copy in range(len(self.worlds)):
            for name in bodies:
                self._body_copy[self.model.body(f"w{copy}_{name}").id] = copy
        self._geom_copy = self._body_copy[self.model.geom_bodyid]
        self._natural_rgba = self.model.geom_rgba.copy()
        self._natural_matid = self.model.geom_matid.copy()
        self._shape_alpha = self._drawn_alpha()
        self.ghost_geoms = np.zeros(self.model.ngeom, dtype=bool)
        self._apply_colours()
        self.set_frame(0)

    def set_frame(self, frame_index: int) -> None:
        """Pose every copy as its world is at ``frame_index``."""
        self.frame_index = frame_index
        rows = self.recording.qpos[frame_index, self.worlds]
        self.data.qpos[self._targets] = rows[:, self._columns]
        self.data.qpos[self._static_targets] = rows[
            self.highlight, self._static_columns
        ]
        mujoco.mj_kinematics(self.model, self.data)
        # Centres of mass, which the camera looks at; then the lights and
        # cameras that hang on bodies.
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def fits(self, recording: Recording, worlds: np.ndarray) -> bool:
        """Whether another recording's worlds can be shown on this composite.

        The composite depends only on the model and the number of copies, so
        consecutive files of one run reuse it instead of composing again.
        """
        own = self.recording
        return (
            recording.model_xml == own.model_xml
            and recording.assets == own.assets
            and recording.replicated_bodies == own.replicated_bodies
            and recording.position_count == own.position_count
            and len(worlds) == len(self.worlds)
        )

    def show(self, recording: Recording, worlds: np.ndarray) -> None:
        """Show another recording that ``fits``, from its first frame."""
        self.recording = recording
        self.worlds = np.asarray(worlds, dtype=np.int64)
        self.set_frame(0)

    def set_highlight(self, copy: int) -> None:
        """Draw copy ``copy`` in colour and the others as ghosts."""
        self.highlight = copy % len(self.worlds)
        self._apply_colours()
        self.set_frame(self.frame_index)  # the static joints follow the highlight

    def set_ghosts(self, strength: str) -> None:
        """Draw every copy but the highlighted one at a strength of GHOST_ALPHAS."""
        self.ghosts = strength
        self._apply_colours()

    @property
    def ghosts_visible(self) -> bool:
        return self.ghosts != "hidden"

    @property
    def ghost_rgba(self) -> np.ndarray:
        """The ghosts' colour at their strength, faded for many worlds."""
        fade = min(1.0, math.sqrt(FADE_FROM / len(self.worlds)))
        return np.array([*GHOST_RGB, GHOST_ALPHAS[self.ghosts] * fade], np.float32)

    def copy_of_body(self, body: int) -> int | None:
        """The copy a body of the composite belongs to; none for the static scene."""
        copy = int(self._body_copy[body]) if body >= 0 else -1
        return copy if copy >= 0 else None

    def marker_positions(self) -> np.ndarray | None:
        """Each drawn world's markers at the current frame, ``(copies, M, 3)``."""
        if self.recording.marker_positions is None:
            return None
        return self.recording.marker_positions[self.frame_index, self.worlds]

    def world_centre(self, copy: int) -> np.ndarray:
        """Copy ``copy``'s centre of mass at the current frame, over all its roots."""
        roots = self._root_bodies[copy]
        mass = self.model.body_subtreemass[roots]
        if not mass.sum() > 0:  # massless bodies: their mean position instead
            return self.data.xpos[roots].mean(axis=0)
        return mass @ self.data.subtree_com[roots] / mass.sum()

    def _compose(
        self,
        spec: mujoco.MjSpec,
        roots: list[str],
        bodies: list[str],
        prefixes: list[str],
    ) -> mujoco.MjModel:
        """The composite: ``spec`` as the static scene, plus a copy per prefix."""
        _delete_unused_parts(spec)
        for name in roots:
            spec.delete(spec.body(name))
        static = {body.name for body in spec.bodies}
        for prefix in prefixes:
            source = self._parse()
            # Lights on the moving bodies light the scene once, from the best world.
            _reduce_to(source, roots, keep_lights=prefix == "w0_")
            frame = spec.worldbody.add_frame()
            spec.attach(source, frame=frame, prefix=prefix)
        _keep_targets(spec, set(bodies), static, prefixes)
        return self._compile(spec, "the composed scene")

    def _parse(self) -> mujoco.MjSpec:
        """A fresh spec of the recorded model, with every body and joint named.

        Compiling would otherwise merge bodies without joints into their
        parents (``fusestatic``), inside the spec, so that bodies named by
        the recorded model would be missing from the copies.
        """
        try:
            spec = mujoco.MjSpec.from_string(
                self.recording.model_xml, assets=self.recording.assets or None
            )
        except ValueError as error:
            raise self._error(f"model_xml cannot be parsed: {error}") from None
        spec.compiler.fusestatic = False
        _name_unnamed(spec)
        return spec

    def _compile(self, spec: mujoco.MjSpec, what: str) -> mujoco.MjModel:
        try:
            return spec.compile()
        except ValueError as error:
            raise self._error(f"{what} does not compile: {error}") from None

    def _error(self, problem: str) -> RecordingError:
        return RecordingError(f"{self.recording.title or 'recording'}: {problem}")

    def _replicated_roots(self, spec: mujoco.MjSpec) -> list[str]:
        """The file's replicated bodies, or every root body that has a joint."""
        roots = {body.name: body for body in spec.worldbody.bodies}
        if self.recording.replicated_bodies is None:
            return [
                name
                for name, body in roots.items()
                if body.find_all(mujoco.mjtObj.mjOBJ_JOINT)
            ]
        for name in self.recording.replicated_bodies:
            if name not in roots:
                raise self._error(
                    f"replicated_bodies names {name!r}, "
                    "which is not a root body of the model"
                )
        return list(self.recording.replicated_bodies)

    def _map_positions(self, original: mujoco.MjModel, joints: set[str]) -> None:
        """Where each recorded ``qpos`` entry goes in the composite's ``qpos``.

        Joints are matched by name, never by position in ``qpos``. A replicated
        joint goes to its copy in every world; a joint of the static scene goes
        to its one place and takes the highlighted world's value.
        """
        columns: list[int] = []
        targets: list[list[int]] = [[] for _ in self.worlds]
        static_columns: list[int] = []
        static_targets: list[int] = []
        for joint in range(original.njnt):
            name = original.joint(joint).name
            span = range(QPOS_SIZES[original.jnt_type[joint]])
            start = int(original.jnt_qposadr[joint])
            if name in joints:
                columns.extend(start + i for i in span)
                for copy, copy_targets in enumerate(targets):
                    address = int(self.model.joint(f"w{copy}_{name}").qposadr[0])
                    copy_targets.extend(address + i for i in span)
            else:
                address = int(self.model.joint(name).qposadr[0])
                static_columns.extend(start + i for i in span)
                static_targets.extend(address + i for i in span)
        self._columns = np.array(columns, dtype=np.int64)
        self._targets = np.array(targets, dtype=np.int64).reshape(
            len(self.worlds), len(columns)
        )
        self._static_columns = np.array(static_columns, dtype=np.int64)
        self._static_targets = np.array(static_targets, dtype=np.int64)

    def _drawn_alpha(self) -> np.ndarray:
        """Each shape's alpha as MuJoCo draws it, material included."""
        alpha = self._natural_rgba[:, 3].copy()
        from_material = (self._natural_matid >= 0) & np.all(
            self._natural_rgba == DEFAULT_GEOM_RGBA, axis=1
        )
        alpha[from_material] = self.model.mat_rgba[
            self._natural_matid[from_material], 3
        ]
        return alpha

    def _apply_colours(self) -> None:
        """Natural colours for the highlighted copy, ghost grey for the others."""
        own = self._geom_copy == self.highlight
        self.ghost_geoms = (self._geom_copy >= 0) & ~own
        self.model.geom_rgba[own] = self._natural_rgba[own]
        self.model.geom_matid[own] = self._natural_matid[own]
        # MuJoCo leaves shapes with alpha 0, such as hidden ghosts, out entirely.
        ghost_rgba = np.tile(self.ghost_rgba, (int(self.ghost_geoms.sum()), 1))
        ghost_rgba[:, 3] *= self._shape_alpha[self.ghost_geoms]
        self.model.geom_rgba[self.ghost_geoms] = ghost_rgba
        self.model.geom_matid[self.ghost_geoms] = -1  # no material or texture


def _name_unnamed(spec: mujoco.MjSpec) -> None:
    """Give every unnamed body and joint a name, the same one on every parse.

    Joints are matched by name between the recorded model and the composite,
    and attaching a body leaves its unnamed parts unnamed.
    """
    for kind, items in (("body", spec.bodies), ("joint", spec.joints)):
        taken = {item.name for item in items}
        for index, item in enumerate(items):
            if not item.name:
                name = f"{kind}{index}"
                while name in taken:
                    name = f"_{name}"
                item.name = name
                taken.add(name)


def _subtree_names(spec: mujoco.MjSpec, roots: list[str]) -> tuple[list[str], set[str]]:
    """The names of the bodies and of the joints under the given root bodies."""
    bodies: list[str] = []
    joints: set[str] = set()
    for name in roots:
        root = spec.body(name)
        bodies.append(name)
        bodies.extend(body.name for body in root.find_all(mujoco.mjtObj.mjOBJ_BODY))
        joints.update(joint.name for joint in root.find_all(mujoco.mjtObj.mjOBJ_JOINT))
    return bodies, joints


def _cache_key(recording: Recording, roots: list[str], copies: int) -> str:
    """A name for one composite: the same model and copies give the same name.

    Each part is hashed with its length, so that two different sets of parts
    can never run together into the same bytes.
    """
    parts = [str(CACHE_VERSION), mujoco.__version__, recording.model_xml]
    parts += [repr(roots), str(copies)]
    for name in sorted(recording.assets):
        parts += [name, recording.assets[name]]
    digest = hashlib.sha256()
    for part in parts:
        data = part if isinstance(part, bytes) else part.encode()
        digest.update(len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()[:32]


def _load_composite(path: Path | None) -> mujoco.MjModel | None:
    """A cached composite, or nothing when there is none or it cannot be read.

    Python reads the file and MuJoCo the bytes: MuJoCo's own file access may
    not open a path with characters outside the system's code page.
    """
    if path is None or not path.exists():
        return None
    try:
        model = mujoco.MjModel.from_binary_path(
            path.name, {path.name: path.read_bytes()}
        )
    except (OSError, ValueError):
        return None
    with contextlib.suppress(OSError):
        path.touch()  # recently used: kept when the cache is trimmed
    return model


def _save_composite(model: mujoco.MjModel, path: Path | None) -> None:
    """Store a composite and trim the cache; a failure costs only the cache."""
    if path is None:
        return
    partial = path.with_suffix(".partial")
    try:
        content = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, content)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(content.tobytes())
        os.replace(partial, path)  # never a half-written file under the real name
        stored = sorted(path.parent.glob("*.mjb"), key=os.path.getmtime)
        for old in stored[:-CACHE_ENTRIES]:
            old.unlink()
    except (OSError, ValueError):
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)


def _keep_targets(
    spec: mujoco.MjSpec, moving: set[str], static: set[str], prefixes: list[str]
) -> None:
    """Keep every camera and light aimed at the body it was aimed at.

    A static one aimed at a moving body would aim at a body that no longer
    exists; it follows the best world's copy instead. One inside a copy aimed
    at a static body was given the copy's prefix by attaching; it loses it.
    """
    for item in [*spec.cameras, *spec.lights]:
        target = item.targetbody
        if target in moving:
            item.targetbody = f"w0_{target}"
            continue
        for prefix in prefixes:
            if target.startswith(prefix) and target[len(prefix) :] in static:
                item.targetbody = target[len(prefix) :]
                break


def _reduce_to(spec: mujoco.MjSpec, roots: list[str], keep_lights: bool) -> None:
    """Strip a spec down to the given root bodies and the assets they use.

    Attaching a spec copies everything in it, so whatever a copy does not need
    goes first: the other root bodies, the world body's own shapes, sites,
    cameras, and lights, the unused parts, and every material, texture, mesh,
    and height field the kept bodies do not use.
    """
    _delete_unused_parts(spec)
    world = spec.worldbody
    for body in list(world.bodies):
        if body.name not in roots:
            spec.delete(body)
    for item in [*world.geoms, *world.sites, *world.cameras, *world.lights]:
        spec.delete(item)
    if not keep_lights:
        for light in list(spec.lights):
            spec.delete(light)
    materials = {geom.material for geom in spec.geoms}
    materials.update(site.material for site in spec.sites)
    used = {
        "materials": materials,
        "textures": {
            texture
            for material in spec.materials
            if material.name in materials
            for texture in material.textures
        },
        "meshes": {geom.meshname for geom in spec.geoms},
        "hfields": {geom.hfieldname for geom in spec.geoms},
    }
    for group, names in used.items():
        for item in list(getattr(spec, group)):
            if item.name not in names:
                spec.delete(item)


def _delete_unused_parts(spec: mujoco.MjSpec) -> None:
    for group in UNUSED_PARTS:
        for item in list(getattr(spec, group)):
            spec.delete(item)
