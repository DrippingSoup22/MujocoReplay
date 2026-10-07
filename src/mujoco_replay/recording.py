"""The recording file: what it holds, and writing and reading it.

A recording is one NumPy ``.npz`` file holding the positions of ``K`` worlds of
one MuJoCo model over ``T`` frames, with the model itself and the facts needed
to replay it. ``Recording`` is the in-memory form, with every optional part
filled with its default; ``write_recording`` and ``read_recording`` move it to
and from a file. The reader is the one place an outside file enters the
program, so it checks every key there; the rest of the tool trusts a
``Recording``. The keys, shapes, and rules are fixed in
docs/recording-format.md.
"""

import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1
REQUIRED_KEYS = ("format_version", "model_xml", "frame_seconds", "qpos")
ASSET_PREFIX = "asset/"
# What each key holds: its kind of values and its number of dimensions.
KEYS = {
    "format_version": ("whole", 0),
    "model_xml": ("text", 0),
    "frame_seconds": ("number", 0),
    "qpos": ("number", 3),
    "replicated_bodies": ("text", 1),
    "score": ("number", 1),
    "score_name": ("text", 0),
    "world_ids": ("whole", 1),
    "level": ("whole", 1),
    "level_count": ("whole", 0),
    "rank": ("whole", 1),
    "ranked_worlds": ("whole", 0),
    "episode_start": ("truth", 2),
    "marker_names": ("text", 1),
    "marker_positions": ("number", 4),
    "marker_radius": ("number", 1),
    "frame_info_names": ("text", 1),
    "frame_info": ("number", 2),
    "event_frames": ("whole", 1),
    "event_labels": ("text", 1),
    "title": ("text", 0),
    "setup_json": ("text", 0),
}
# The NumPy dtype kinds that each kind of value accepts.
DTYPE_KINDS = {"whole": "iu", "number": "iuf", "truth": "biu", "text": "U"}
# Optional keys that come in groups: a group is written and read as a whole.
MARKER_KEYS = ("marker_names", "marker_positions", "marker_radius")
INFO_KEYS = ("frame_info_names", "frame_info")
EVENT_KEYS = ("event_frames", "event_labels")
RANK_KEYS = ("rank", "ranked_worlds")
# The shortest time between frames, and the deepest nesting of the setup.
MIN_FRAME_SECONDS = 1e-6
MAX_SETUP_DEPTH = 32


class RecordingError(ValueError):
    """A recording file that does not follow the format."""


@dataclass(frozen=True)
class Recording:
    """One recording in memory; see docs/recording-format.md for each field.

    Only ``model_xml``, ``frame_seconds``, and ``qpos`` are required. A ``None``
    for ``score``, ``world_ids``, or ``episode_start`` becomes the default on
    creation; ``level``, ``level_count``, and the marker, info, event, and
    rank groups stay ``None`` when absent. Arrays are cast to the format's
    types.
    """

    model_xml: str
    frame_seconds: float
    qpos: np.ndarray  # (T, K, nq) float32
    assets: dict[str, bytes] = field(default_factory=dict)
    replicated_bodies: tuple[str, ...] | None = None
    score: np.ndarray | None = None  # (K,) float32
    score_name: str = "score"
    world_ids: np.ndarray | None = None  # (K,) int64
    level: np.ndarray | None = None  # (K,) int64
    episode_start: np.ndarray | None = None  # (T, K) bool
    marker_names: tuple[str, ...] | None = None
    marker_positions: np.ndarray | None = None  # (T, K, M, 3) float32
    marker_radius: np.ndarray | None = None  # (M,) float32
    frame_info_names: tuple[str, ...] | None = None
    frame_info: np.ndarray | None = None  # (T, I) float64
    event_frames: np.ndarray | None = None  # (E,) int64
    event_labels: tuple[str, ...] | None = None
    title: str = ""
    setup: dict[str, Any] = field(default_factory=dict)
    level_count: int | None = None
    rank: np.ndarray | None = None  # (K,) int64
    ranked_worlds: int | None = None

    def __post_init__(self) -> None:
        """Cast the arrays and fill the defaults of the per-world fields.

        A group given in part raises ``ValueError``, so that the writer never
        writes a file the reader would refuse.
        """
        for group in (MARKER_KEYS, INFO_KEYS, EVENT_KEYS, RANK_KEYS):
            given = [name for name in group if getattr(self, name) is not None]
            if given and len(given) < len(group):
                missing = next(name for name in group if name not in given)
                raise ValueError(f"{missing} is missing while {given[0]} is given")
        set_field = object.__setattr__
        set_field(self, "frame_seconds", float(self.frame_seconds))
        set_field(self, "qpos", np.asarray(self.qpos, dtype=np.float32))
        frames, worlds = self.qpos.shape[:2]
        casts = {
            "score": (np.zeros(worlds), np.float32),
            "world_ids": (np.arange(worlds), np.int64),
            "level": (None, np.int64),
            "episode_start": (np.zeros((frames, worlds)), np.bool_),
            "marker_positions": (None, np.float32),
            "marker_radius": (None, np.float32),
            "frame_info": (None, np.float64),
            "event_frames": (None, np.int64),
            "rank": (None, np.int64),
        }
        for name, (default, dtype) in casts.items():
            value = getattr(self, name)
            if value is None:
                value = default
            if value is not None:
                value = np.asarray(value, dtype=dtype)
            set_field(self, name, value)
        for name in (
            "replicated_bodies",
            "marker_names",
            "frame_info_names",
            "event_labels",
        ):
            value = getattr(self, name)
            if isinstance(value, str):  # one name, not its letters
                value = (value,)
            if value is not None:
                set_field(self, name, tuple(str(item) for item in value))
        for name in ("level_count", "ranked_worlds"):
            value = getattr(self, name)
            if value is not None:
                set_field(self, name, int(value))

    @property
    def frame_count(self) -> int:
        return self.qpos.shape[0]

    @property
    def world_count(self) -> int:
        return self.qpos.shape[1]

    @property
    def position_count(self) -> int:
        return self.qpos.shape[2]


def write_recording(path: Path | str, recording: Recording) -> None:
    """Save a recording as an ``.npz`` file, uncompressed.

    The file is written under another name first and then renamed, so that a
    reader never finds it half written.
    """
    arrays: dict[str, Any] = {
        "format_version": np.int64(FORMAT_VERSION),
        "model_xml": np.array(recording.model_xml),
        "frame_seconds": np.float64(recording.frame_seconds),
        "qpos": recording.qpos,
        "score": recording.score,
        "score_name": np.array(recording.score_name),
        "world_ids": recording.world_ids,
        "episode_start": recording.episode_start,
        "title": np.array(recording.title),
        "setup_json": np.array(json.dumps(recording.setup, default=_plain)),
    }
    for name, content in recording.assets.items():
        arrays[ASSET_PREFIX + name] = np.frombuffer(content, dtype=np.uint8)
    if recording.replicated_bodies is not None:
        arrays["replicated_bodies"] = _strings(recording.replicated_bodies)
    if recording.level is not None:
        arrays["level"] = recording.level
    if recording.level_count is not None:
        arrays["level_count"] = np.int64(recording.level_count)
    if recording.rank is not None:
        arrays["rank"] = recording.rank
        arrays["ranked_worlds"] = np.int64(recording.ranked_worlds)
    if recording.marker_names is not None:
        arrays["marker_names"] = _strings(recording.marker_names)
        arrays["marker_positions"] = recording.marker_positions
        arrays["marker_radius"] = recording.marker_radius
    if recording.frame_info_names is not None:
        arrays["frame_info_names"] = _strings(recording.frame_info_names)
        arrays["frame_info"] = recording.frame_info
    if recording.event_frames is not None:
        arrays["event_frames"] = recording.event_frames
        arrays["event_labels"] = _strings(recording.event_labels)
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as file:
        np.savez(file, **arrays)
    os.replace(partial, path)


def read_recording(path: Path | str) -> Recording:
    """Load and check a recording file; a bad file raises ``RecordingError``."""
    path = Path(path)
    try:
        return _read(path)
    except RecordingError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        EOFError,
        MemoryError,
        zipfile.BadZipFile,
    ) as error:  # a damaged or half-written file, whatever breaks first
        raise RecordingError(f"{path} cannot be read as a recording: {error}") from None


def _read(path: Path) -> Recording:
    def fail(key: str, problem: str) -> RecordingError:
        return RecordingError(f"{path}: {key} {problem}")

    with np.load(path, allow_pickle=False) as file:
        if not isinstance(file, np.lib.npyio.NpzFile):
            raise RecordingError(f"{path} is not an .npz file of arrays")
        arrays = {}
        for key in file.files:
            try:
                arrays[key] = file[key]
            except ValueError as error:  # such as an array of Python objects
                raise fail(key, f"cannot be read: {error}") from None

    for key in REQUIRED_KEYS:
        if key not in arrays:
            raise fail(key, "is missing")
    for key, (kind, dimensions) in KEYS.items():
        if key not in arrays:
            continue
        if arrays[key].dtype.kind not in DTYPE_KINDS[kind]:
            raise fail(key, f"must hold {kind} values, holds {arrays[key].dtype}")
        if arrays[key].ndim != dimensions:
            shape = {0: "a single value", 1: "a list"}.get(dimensions)
            raise fail(key, f"must be {shape or f'{dimensions}-dimensional'}")
    if int(arrays["format_version"]) != FORMAT_VERSION:
        raise fail("format_version", f"must be {FORMAT_VERSION}")
    for group in (MARKER_KEYS, INFO_KEYS, EVENT_KEYS, RANK_KEYS):
        present = [key in arrays for key in group]
        if any(present) and not all(present):
            missing = [
                key for key, here in zip(group, present, strict=True) if not here
            ]
            present_key = group[present.index(True)]
            raise fail(missing[0], f"is missing while {present_key} is present")

    qpos = arrays["qpos"]
    frames, worlds = qpos.shape[:2]
    if not frames or not worlds:
        raise fail("qpos", "must hold at least one frame and one world")
    frame_seconds = float(arrays["frame_seconds"])
    if not MIN_FRAME_SECONDS <= frame_seconds < np.inf:
        raise fail("frame_seconds", f"must be finite and at least {MIN_FRAME_SECONDS}")

    expected_shapes = {
        "score": (worlds,),
        "world_ids": (worlds,),
        "level": (worlds,),
        "rank": (worlds,),
        "episode_start": (frames, worlds),
    }
    if "marker_names" in arrays:
        markers = len(arrays["marker_names"])
        expected_shapes["marker_positions"] = (frames, worlds, markers, 3)
        expected_shapes["marker_radius"] = (markers,)
    if "frame_info_names" in arrays:
        expected_shapes["frame_info"] = (frames, len(arrays["frame_info_names"]))
    if "event_frames" in arrays:
        expected_shapes["event_labels"] = (len(arrays["event_frames"]),)
    for key, shape in expected_shapes.items():
        if key in arrays and arrays[key].shape != shape:
            raise fail(key, f"must have shape {shape}, has {arrays[key].shape}")
    if "level" in arrays and arrays["level"].min() < 1:
        raise fail("level", "must be 1 or more")
    if "level_count" in arrays:
        if "level" not in arrays:
            raise fail("level_count", "needs level")
        if arrays["level_count"] < arrays["level"].max():
            raise fail("level_count", "must be at least the highest level")
    if "rank" in arrays:
        ranks, ranked = arrays["rank"], int(arrays["ranked_worlds"])
        if ranked < worlds:
            raise fail("ranked_worlds", f"must be at least the {worlds} worlds")
        if ranks.min() < 1 or ranks.max() > ranked:
            raise fail("rank", f"must lie between 1 and ranked_worlds, {ranked}")
        if len(np.unique(ranks)) < worlds:
            raise fail("rank", "must give each world its own rank")
    if "replicated_bodies" in arrays:
        names = [str(name) for name in arrays["replicated_bodies"]]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise fail("replicated_bodies", f"names {repeated[0]!r} more than once")
    if "event_frames" in arrays and arrays["event_frames"].size:
        events = arrays["event_frames"]
        if events.min() < 0 or events.max() > frames:
            raise fail("event_frames", f"must lie between 0 and {frames}")
    setup: Any = {}
    if "setup_json" in arrays:
        try:
            setup = json.loads(str(arrays["setup_json"]))
        except (json.JSONDecodeError, RecursionError) as error:
            raise fail("setup_json", f"is not valid JSON: {error}") from None
        if not isinstance(setup, dict):
            raise fail("setup_json", "must be a JSON object")
        if _deeper_than(setup, MAX_SETUP_DEPTH):
            raise fail("setup_json", f"must nest at most {MAX_SETUP_DEPTH} levels")

    def optional(key: str) -> Any:
        return arrays.get(key)

    def text(key: str, default: str) -> str:
        return str(arrays[key]) if key in arrays else default

    return Recording(
        model_xml=str(arrays["model_xml"]),
        frame_seconds=frame_seconds,
        qpos=qpos,
        assets={
            key.removeprefix(ASSET_PREFIX): value.tobytes()
            for key, value in arrays.items()
            if key.startswith(ASSET_PREFIX)
        },
        replicated_bodies=optional("replicated_bodies"),
        score=optional("score"),
        score_name=text("score_name", "score"),
        world_ids=optional("world_ids"),
        level=optional("level"),
        episode_start=optional("episode_start"),
        marker_names=optional("marker_names"),
        marker_positions=optional("marker_positions"),
        marker_radius=optional("marker_radius"),
        frame_info_names=optional("frame_info_names"),
        frame_info=optional("frame_info"),
        event_frames=optional("event_frames"),
        event_labels=optional("event_labels"),
        title=text("title", path.stem) or path.stem,
        setup=setup,
        level_count=optional("level_count"),
        rank=optional("rank"),
        ranked_worlds=optional("ranked_worlds"),
    )


def _deeper_than(value: Any, limit: int) -> bool:
    """Whether a JSON value nests objects or lists more than ``limit`` deep."""
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, dict | list):
            if depth == limit:
                return True
            children = item.values() if isinstance(item, dict) else item
            stack.extend((child, depth + 1) for child in children)
    return False


def in_name_order(paths: list[str]) -> list[str]:
    """Paths sorted by name with numbers by value, so cycle_2 comes before cycle_10."""

    def key(path: str) -> list[int | str]:
        parts = re.split(r"(\d+)", path.lower())
        return [int(part) if part.isdigit() else part for part in parts]

    return sorted(paths, key=key)


def _plain(value: Any) -> Any:
    """A setup value that JSON cannot hold, as one it can: NumPy numbers and
    arrays as numbers and lists, anything else, such as a path, as text."""
    if isinstance(value, np.generic | np.ndarray):
        return value.tolist()
    return str(value)


def _strings(items: tuple[str, ...]) -> np.ndarray:
    """A tuple of strings as a NumPy unicode array, also when empty."""
    return np.array(list(items), dtype=str)
