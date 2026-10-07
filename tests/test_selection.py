"""Tests for choosing worlds by rank, and for the producer half's light imports."""

import subprocess
import sys

import numpy as np

from mujoco_replay.recording import Recording
from mujoco_replay.selection import choose_worlds, selected_ranks


def test_every_rank_is_kept_when_the_worlds_fit():
    ranks, levels = selected_ranks(6, levels=4, per_level=2)

    assert ranks.tolist() == [0, 1, 2, 3, 4, 5]
    assert levels.tolist() == [1, 1, 2, 3, 3, 4]


def test_each_level_contributes_evenly_spaced_ranks_including_best_and_worst():
    ranks, levels = selected_ranks(100, levels=4, per_level=3)

    assert ranks.tolist() == [0, 12, 24, 25, 37, 49, 50, 62, 74, 75, 87, 99]
    assert levels.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]


def test_uneven_bands_and_one_per_level_take_each_bands_first_rank():
    ranks, levels = selected_ranks(10, levels=3, per_level=1)

    assert ranks.tolist() == [0, 4, 7]
    assert levels.tolist() == [1, 2, 3]
    many, _ = selected_ranks(1000, levels=4, per_level=8)
    assert len(many) == 32 and many[0] == 0 and many[-1] == 999
    assert np.all(np.diff(many) > 0)


def test_choose_worlds_follows_the_scores_best_first():
    recording = Recording(
        "<mujoco/>",
        0.02,
        np.zeros((1, 5, 1)),
        score=[0.1, 0.9, 0.5, 0.9, -1.0],
        world_ids=[10, 11, 12, 13, 14],
    )

    assert choose_worlds(recording, levels=2, per_level=1).tolist() == [1, 0]
    assert choose_worlds(recording, 2, 1, all_worlds=True).tolist() == [1, 3, 2, 0, 4]
    assert choose_worlds(recording, 2, 1, world_ids=[14, 10]).tolist() == [0, 4]


def test_the_producer_half_imports_neither_mujoco_nor_graphics():
    script = (
        "import sys, mujoco_replay, mujoco_replay.recording, mujoco_replay.selection;"
        "print(sorted(m for m in sys.modules if m in ('mujoco', 'glfw', 'OpenGL')))"
    )
    output = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert output == "[]"
