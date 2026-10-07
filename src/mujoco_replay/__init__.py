"""Replays recorded poses of many worlds of one MuJoCo model in one scene.

``recording`` and ``selection`` are the producer-facing half, importing only
NumPy; the viewer and video modules import MuJoCo and the window library only
when they run. See docs/recording-format.md and docs/design.md.
"""
