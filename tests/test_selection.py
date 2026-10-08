"""Tests for choosing worlds by rank, and for the producer half's light imports."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from mujoco_replay.recording import Recording
from mujoco_replay.selection import choose_worlds, selected_ranks, world_counts


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


def test_a_count_of_worlds_takes_the_best_of_as_many_rank_bands():
    # Rank order, best first: 1, 3, 6, 2, 5, 7, 0, 4.
    recording = Recording(
        "<mujoco/>",
        0.02,
        np.zeros((1, 8, 1)),
        score=[0.1, 0.9, 0.5, 0.8, -1.0, 0.3, 0.7, 0.2],
        world_ids=[10, 11, 12, 13, 14, 15, 16, 17],
    )

    assert choose_worlds(recording, 1).tolist() == [1]
    assert choose_worlds(recording, 2).tolist() == [1, 5]  # best of each half
    assert choose_worlds(recording, 4).tolist() == [1, 6, 5, 0]
    assert choose_worlds(recording, 16).tolist() == [1, 3, 6, 2, 5, 7, 0, 4]
    assert choose_worlds(recording, 2, world_ids=[14, 10]).tolist() == [0, 4]


def test_a_kept_world_is_drawn_in_place_of_the_best_of_its_band():
    # Rank order, best first: 1, 3, 6, 2, 5, 7, 0, 4.
    recording = Recording(
        "<mujoco/>",
        0.02,
        np.zeros((1, 8, 1)),
        score=[0.1, 0.9, 0.5, 0.8, -1.0, 0.3, 0.7, 0.2],
    )

    assert choose_worlds(recording, 1, keep=7).tolist() == [7]
    assert choose_worlds(recording, 2, keep=2).tolist() == [2, 5]  # the upper half
    assert choose_worlds(recording, 2, keep=0).tolist() == [1, 0]  # the lower half
    assert choose_worlds(recording, 8, keep=4).tolist() == [1, 3, 6, 2, 5, 7, 0, 4]


def test_the_counts_offered_double_up_to_128_and_the_files_worlds():
    assert world_counts(1) == [1]
    assert world_counts(3) == [1, 2, 3]
    assert world_counts(32) == [1, 2, 4, 8, 16, 32]
    assert world_counts(48) == [1, 2, 4, 8, 16, 32, 48]
    assert world_counts(1024) == [1, 2, 4, 8, 16, 32, 64, 128]


def test_the_producer_half_imports_neither_mujoco_nor_graphics():
    script = (
        "import sys, mujoco_replay, mujoco_replay.recording, mujoco_replay.selection;"
        "print(sorted(m for m in sys.modules if m in ('mujoco', 'glfw', 'OpenGL')))"
    )
    source = Path(__file__).resolve().parents[1] / "src"  # this copy, not another
    environment = {**os.environ, "PYTHONPATH": str(source)}
    output = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    ).stdout.strip()

    assert output == "[]"
