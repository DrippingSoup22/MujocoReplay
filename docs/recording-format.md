# Recording format

A recording is one NumPy `.npz` file that holds the positions of many worlds of
one MuJoCo model over a stretch of time, together with everything needed to
replay it: the model itself, which bodies belong to each world, a score per
world, markers, events, and the setup of the run that produced it. A recording
is self-contained: nothing outside the file is needed to replay it. This
document is the contract between producers, such as the sibling `Centipede`
project, and this tool. The writer and the reader live in
`src/mujoco_replay/recording.py`, and the rank selection shared with producers
in `src/mujoco_replay/selection.py`.

## Why `.npz`

An `.npz` file is a zip of NumPy arrays. It needs only NumPy, is written in one
call, can be read partially, and is opened without pickling
(`allow_pickle=False`), so a file cannot run code when it is loaded. Strings
are stored as NumPy unicode arrays and binary assets as `uint8` arrays.

## Notation

| Symbol | Meaning |
| --- | --- |
| `T` | Number of frames |
| `K` | Number of worlds in the file |
| `nq` | Number of position coordinates of the model (`MjModel.nq`), in MuJoCo's own `qpos` layout |
| `M` | Number of markers per world |
| `E` | Number of events |
| `I` | Number of per-frame info values |

## Keys

Required keys:

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `format_version` | int, scalar | `1` for this document |
| `model_xml` | str, scalar | The complete model as one MJCF document, with includes already resolved. `mujoco.MjSpec.from_file(path).to_xml()` produces exactly this (verified with MuJoCo 3.12.0: includes and inline meshes come out in one document) |
| `frame_seconds` | float, scalar | Simulated time between consecutive frames, in seconds |
| `qpos` | float32, `(T, K, nq)` | Every world's position coordinates at every frame, in the compiled model's `qpos` order |

Optional keys:

| Key | Type and shape | Default | Meaning |
| --- | --- | --- | --- |
| `asset/<name>` | uint8, `(n,)` | none | Contents of a file the model refers to (mesh, texture, height field), under the name the XML uses. Not needed when the model's assets are inline, as Centipede's are |
| `replicated_bodies` | str, `(R,)` | every root body whose subtree has a joint | Names of the root bodies (direct children of the world body) that exist once per world. Everything else in the model is the static scene, drawn once |
| `score` | float32, `(K,)` | zeros | Each world's score over the recording; higher is better. Used to rank worlds |
| `score_name` | str, scalar | `"score"` | What the score is, for the overlay |
| `world_ids` | int64, `(K,)` | `0 … K−1` | The producer's index of each world |
| `level` | int64, `(K,)` | none | Each world's level, `1` for the best band, as `selected_ranks` assigns them (see below). Shown in the overlay when present |
| `episode_start` | bool, `(T, K)` | all false | True at frames where that world's episode began, so the frame shows a freshly reset pose. The first frame need not be flagged |
| `marker_names` | str, `(M,)` | none | Names of per-world points to draw, such as a target |
| `marker_positions` | float32, `(T, K, M, 3)` | none | Each marker's world position at each frame, in metres |
| `marker_radius` | float32, `(M,)` | none | Radius of the sphere drawn for each marker, in metres |
| `frame_info_names` | str, `(I,)` | none | Names of values that describe each frame, such as the number of updates done so far |
| `frame_info` | float64, `(T, I)` | none | Those values at each frame; shown in the overlay |
| `event_frames` | int64, `(E,)` | none | Frames at which something happened, from `0` to `T` inclusive; `T` means after the last frame |
| `event_labels` | str, `(E,)` | none | What happened, such as `"update 16"`; shown as a tick on the timeline and flashed when playback passes the frame |
| `title` | str, scalar | the file name | One line naming the recording, for the window title and overlay |
| `setup_json` | str, scalar | `"{}"` | A JSON object describing the run's setup, nested tables allowed; shown on demand as flattened `key = value` lines |

Rules:

- `qpos` uses the layout of the model compiled from `model_xml`.
- Joints that are not inside a replicated body, such as a moving obstacle in
  the static scene, take their values from the highlighted world's row.
- Marker, event, and info keys come in groups: a group is either complete or
  absent.
- Shapes must agree with each other; `qpos` holds at least one frame and one
  world; `frame_seconds` is positive; event frames lie in `[0, T]`; levels are
  at least `1`; `replicated_bodies` names each body once; `setup_json` parses
  as a JSON object. The reader rejects anything else with an error naming the file and
  the key.
- The checks that need the model are made when the scene is composed:
  `model_xml` must parse and compile, its `nq` must match `qpos`, and
  `replicated_bodies` must name root bodies of the model. The error names the
  recording's title and the key.
- Text is drawn in MuJoCo's font, which holds ASCII only; other characters in
  the title, names, and labels are replaced when drawn (`·` by `|`, accents
  dropped, anything else by `?`).

These are the only checks: an outside file enters the program at the reader
and at the scene composition, and the tool's own modules trust each other.

## Writer and reader

`recording.py` offers:

- `Recording`: a frozen dataclass with one field per key above, defaults
  filled in, and `frame_count`, `world_count`, and `position_count` properties.
- `write_recording(path, recording)`: writes the file with `numpy.savez`
  (uncompressed; floating-point poses compress poorly, and the file stays
  simple to read partially).
- `read_recording(path) -> Recording`: reads and checks the file.

Importing `mujoco_replay.recording` or `mujoco_replay.selection` imports only
NumPy and the standard library, so a training machine without a display can
write recordings.

`selection.py` offers `selected_ranks(world_count, levels, per_level)`, which
both producers and this tool use to pick worlds by rank, so that one rule
exists:

- Ranks run from `0` (best score) to `world_count − 1` (worst). If
  `world_count ≤ levels × per_level`, every rank is selected.
- Otherwise the ranks are split into `levels` contiguous bands of as equal a
  size as possible, band `i` covering ranks `⌈i·W/L⌉` to `⌈(i+1)·W/L⌉ − 1`;
  a rank's level is therefore `⌊rank·L/W⌋ + 1`, which also holds when there
  are fewer worlds than levels.
- From each band, `per_level` ranks are taken evenly spaced from the band's
  first to its last rank, both included (`first + round(j·(n−1)/(per_level−1))`
  for `j = 0 … per_level−1`; with `per_level = 1` only the first). The best
  world and the worst world are therefore always selected.
- The result is the sorted list of selected ranks and each one's level, `1`
  for the first band. A producer orders its worlds by score, descending and
  stable, and keeps the worlds at those ranks.

A producer that keeps a fixed set of worlds instead (for continuity across
consecutive recordings) still stores `score` and may store `level` computed
from each world's rank among all its worlds.

## What Centipede writes

Centipede records training windows and evaluation episodes; its `plan.md` and
`docs/configuration.md` describe the settings. Its files follow these
conventions, which are not part of the format:

| Item | Training window | Evaluation |
| --- | --- | --- |
| File | `runs/<run>/recordings/cycle_NNNN.npz`: the window collected in cycle `NNNN`, by the agents after `NNNN − 1` updates | `runs/<run>/evaluations/<stem>_<actor>_seed<seed>.npz`: every world's first episode, and what followed it until the last world's first episode ended |
| Worlds | Selected by `selected_ranks` from all worlds, ranked by the window's summed reward; or the first worlds, when continuity across windows is wanted | All |
| `score` | Sum of every segment's rewards over the window (`score_name = "summed reward"`) | The same, over the recording |
| Markers | `target`: the head's target, drawn on the ground with the arrival radius | The same |
| `frame_info` | `updates` (done so far) and `steps per world` (collected so far) | none |
| Events | `update NNNN` at frame `T` | none |
| `setup_json` | The run's complete configuration, the cycle, the device, the code version | The checkpoint, seed, actor, and configuration |

## Size

One world-frame costs `4·nq` bytes: 276 bytes for a model with 69 position
coordinates. For that model:

| Recording | Size |
| --- | --- |
| 32 worlds, 256 frames | 2.3 MB |
| 64 worlds, 256 frames | 4.5 MB |
| 1,024 worlds, 256 frames | 72 MB |
| 8 worlds, 8,192 frames | 18 MB |

Producers keep files small by recording few windows and selecting worlds, not
by lowering precision or dropping frames: float16 loses millimetres a few metres
from the origin, and skipped frames make playback jerky.
