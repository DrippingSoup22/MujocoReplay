"""Tests for the settings: the presets, the mode, and keeping them between runs."""

import json

from mujoco_replay.settings import (
    PERFORMANCE,
    QUALITY,
    Settings,
    load_settings,
    save_settings,
)


def test_a_preset_names_the_mode_until_one_switch_changes():
    settings = Settings().with_mode("quality")

    assert settings.graphics == QUALITY and settings.mode == "quality"
    assert settings.with_graphics(shadows=False).mode == "custom"
    assert settings.with_mode("performance").graphics == PERFORMANCE


def test_saved_settings_come_back_and_a_bad_file_gives_the_defaults(tmp_path):
    path = tmp_path / "folder" / "settings.json"
    chosen = Settings(worlds=32, cache=False).with_graphics(resolution=50)

    save_settings(chosen, path)
    assert load_settings(path) == chosen

    path.write_text("{not json")
    assert load_settings(path) == Settings()
    assert load_settings(tmp_path / "missing.json") == Settings()


def test_a_wrong_value_falls_back_to_its_default_alone(tmp_path):
    path = tmp_path / "settings.json"
    saved = {"worlds": 1000, "cache": "yes", "frame_rate": True}
    saved["graphics"] = {"resolution": 60, "shadows": True}
    path.write_text(json.dumps(saved))

    settings = load_settings(path)

    assert (settings.worlds, settings.cache, settings.frame_rate) == (16, True, True)
    assert settings.graphics.resolution == PERFORMANCE.resolution
    assert settings.graphics.shadows


def test_a_file_that_starts_with_a_byte_order_mark_is_read(tmp_path):
    path = tmp_path / "settings.json"  # as Windows PowerShell 5.1 writes UTF-8
    path.write_text(json.dumps({"worlds": 32}), encoding="utf-8-sig")

    assert load_settings(path).worlds == 32
