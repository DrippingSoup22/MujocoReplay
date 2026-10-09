"""Tests for the command: what it does before its options are parsed."""

from mujoco_replay import cli


def test_the_file_picker_mode_runs_before_any_option_is_parsed(monkeypatch):
    shown = []
    monkeypatch.setattr(cli, "pick_files", lambda output: shown.append(output) or 0)

    assert cli.main([cli.PICK_FILES, "chosen.txt"]) == 0
    assert shown == ["chosen.txt"]
