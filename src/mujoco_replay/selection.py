"""Choosing worlds by rank, so that every level of score is represented.

``selected_ranks`` is the one rule producers and the viewer share: a producer
keeps the worlds at these ranks when it writes a file, and the viewer draws
the worlds at these ranks when a file holds more than it shows. It is
described in docs/recording-format.md.
"""

import numpy as np

from mujoco_replay.recording import Recording


def selected_ranks(
    world_count: int, levels: int, per_level: int
) -> tuple[np.ndarray, np.ndarray]:
    """The ranks to keep, sorted, and each one's level (1 for the best band).

    Ranks run from 0 (best) to ``world_count - 1`` (worst). When the worlds
    fit, every rank is kept. Otherwise the ranks are split into ``levels``
    bands of as equal a size as possible (band ``i`` starts at rank
    ``ceil(i * world_count / levels)``), and ``per_level`` ranks are taken
    from each band, evenly spaced from its first to its last rank, so the best
    and the worst world are always kept.
    """
    if world_count <= levels * per_level:
        ranks = np.arange(world_count)
        return ranks, level_of_ranks(ranks, world_count, levels)
    chosen = []
    for band_index in range(levels):
        first = -(-band_index * world_count // levels)
        last = -(-(band_index + 1) * world_count // levels) - 1
        if last < first:  # an empty band, when there are fewer worlds than levels
            continue
        if per_level == 1:
            chosen.append(first)
        else:
            chosen.extend(
                first + int(round(j * (last - first) / (per_level - 1)))
                for j in range(per_level)
            )
    ranks = np.unique(np.array(chosen, dtype=np.int64))
    return ranks, level_of_ranks(ranks, world_count, levels)


def level_of_ranks(ranks: np.ndarray, world_count: int, levels: int) -> np.ndarray:
    """The band each rank falls in, from 1 for the best band."""
    return (np.asarray(ranks, dtype=np.int64) * levels // world_count) + 1


def choose_worlds(
    recording: Recording,
    levels: int,
    per_level: int,
    world_ids: list[int] | None = None,
    all_worlds: bool = False,
) -> np.ndarray:
    """Indices into the file's worlds to draw, best first.

    By default the worlds at ``selected_ranks`` of the file's scores; with
    ``all_worlds`` every world; with ``world_ids`` exactly those producer
    indices, in rank order. Ties keep the file's order.
    """
    order = np.argsort(-recording.score, kind="stable")
    if all_worlds:
        return order
    if world_ids is not None:
        wanted = set(world_ids)
        return np.array(
            [index for index in order if int(recording.world_ids[index]) in wanted],
            dtype=np.int64,
        )
    ranks, _ = selected_ranks(recording.world_count, levels, per_level)
    return order[ranks]
