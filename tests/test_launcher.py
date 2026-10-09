"""The MujocoReplay program: the command, with its failures shown in a message
box when there is no console to print them to."""

import sys
import tomllib
from importlib import import_module
from pathlib import Path

import pytest

from mujoco_replay import cli, launcher


@pytest.fixture
def shown(monkeypatch):
    """The texts the program would show in message boxes."""
    texts = []
    monkeypatch.setattr(launcher, "_show", texts.append)
    return texts


def no_console(monkeypatch):
    """No error stream, as under Windows' pythonw; pytest sets its own error
    stream before each test, so a fixture could not take it away."""
    monkeypatch.setattr(sys, "stderr", None)


def test_the_install_makes_the_program_from_main():
    """A wrong entry would make a program that fails without a word."""
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    module, name = project["project"]["gui-scripts"]["MujocoReplay"].split(":")
    assert getattr(import_module(module), name) is launcher.main


def test_with_a_console_the_program_is_the_command(monkeypatch, shown):
    monkeypatch.setattr(cli, "main", lambda: 7)
    assert launcher.main() == 7
    assert shown == []


def test_without_a_console_the_commands_message_is_shown(monkeypatch, shown, tmp_path):
    no_console(monkeypatch)
    missing = str(tmp_path / "missing.npz")
    monkeypatch.setattr(sys, "argv", ["MujocoReplay", missing])
    assert launcher.main() == 1
    assert len(shown) == 1
    assert shown[0].startswith(f"mujoco-replay: {missing} cannot be read")
    assert sys.stderr is None


def test_without_a_console_a_wrong_option_is_shown(monkeypatch, shown):
    no_console(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["MujocoReplay", "--worlds", "0"])
    assert launcher.main() == 2
    assert "argument --worlds: 0 is not from 1 to" in shown[0]


def test_without_a_console_an_unforeseen_error_is_shown(monkeypatch, shown):
    no_console(monkeypatch)

    def fail():
        raise ValueError("an unforeseen failure")

    monkeypatch.setattr(cli, "main", fail)
    assert launcher.main() == 1
    assert shown[0].startswith("Traceback")
    assert shown[0].endswith("ValueError: an unforeseen failure")


def test_without_a_console_a_clean_run_shows_nothing(monkeypatch, shown):
    no_console(monkeypatch)
    monkeypatch.setattr(cli, "main", lambda: 0)
    assert launcher.main() == 0
    assert shown == []


def test_without_a_console_the_file_pickers_process_is_the_command(monkeypatch, shown):
    """The window reads why the picker failed; a message box would hide it."""
    no_console(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["MujocoReplay", cli.PICK_FILES, "chosen.txt"])
    monkeypatch.setattr(cli, "main", lambda: 1)
    assert launcher.main() == 1
    assert shown == []
