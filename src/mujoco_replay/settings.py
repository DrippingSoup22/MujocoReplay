"""The viewer's settings: what it draws and how finely, kept between runs.

Two presets set the graphics at once. Quality draws shadows, reflections,
anti-aliasing, and fine shapes at full resolution; Performance turns them all
off and draws fewer pixels, for weak graphics cards. Changing one switch
afterwards makes the mode "custom". The settings are saved as JSON in the
user's settings folder, so that the next run starts where this one ended; a
file that cannot be read, or a value out of place in it, falls back to the
default.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from mujoco_replay.selection import DEFAULT_WORLDS, MAX_WORLDS

# Shares of the window's width and height that are drawn, then scaled up.
RESOLUTIONS = (50, 75, 100)
# How strongly the worlds other than the highlighted one are drawn.
GHOST_STRENGTHS = ("hidden", "faint", "normal", "strong")


@dataclass(frozen=True)
class Graphics:
    """How finely the scene is drawn; docs/design.md says what each costs."""

    shadows: bool
    reflections: bool
    antialiasing: bool
    fine_shapes: bool
    resolution: int  # percent, one of RESOLUTIONS


QUALITY = Graphics(
    shadows=True, reflections=True, antialiasing=True, fine_shapes=True, resolution=100
)
PERFORMANCE = Graphics(
    shadows=False,
    reflections=False,
    antialiasing=False,
    fine_shapes=False,
    resolution=75,
)
PRESETS = {"quality": QUALITY, "performance": PERFORMANCE}


@dataclass(frozen=True)
class Settings:
    """Everything the panel sets that lasts from one run to the next."""

    graphics: Graphics = PERFORMANCE
    worlds: int = DEFAULT_WORLDS
    ghosts: str = "normal"  # one of GHOST_STRENGTHS
    cache: bool = True
    frame_rate: bool = False
    panel: bool = True
    overlay: bool = True

    @property
    def mode(self) -> str:
        """``quality``, ``performance``, or ``custom`` when no preset matches."""
        for name, preset in PRESETS.items():
            if self.graphics == preset:
                return name
        return "custom"

    def with_mode(self, name: str) -> "Settings":
        return replace(self, graphics=PRESETS[name])

    def with_graphics(self, **changes) -> "Settings":
        return replace(self, graphics=replace(self.graphics, **changes))


def user_folder(kind: str) -> Path:
    """This tool's ``settings`` or ``cache`` folder, where the system keeps such."""
    cache = kind == "cache"
    if sys.platform == "win32":
        variable = "LOCALAPPDATA" if cache else "APPDATA"
        base = Path(os.environ.get(variable) or Path.home() / "AppData" / "Roaming")
        return base / "MujocoReplay" / ("cache" if cache else "")
    if sys.platform == "darwin":
        place = "Caches" if cache else "Application Support"
        return Path.home() / "Library" / place / "MujocoReplay"
    variable = "XDG_CACHE_HOME" if cache else "XDG_CONFIG_HOME"
    default = Path.home() / (".cache" if cache else ".config")
    return Path(os.environ.get(variable) or default) / "mujoco-replay"


def settings_path() -> Path:
    return user_folder("settings") / "settings.json"


def load_settings(path: Path | None = None) -> Settings:
    """The saved settings, with the default wherever a value is missing or wrong."""
    try:
        saved = json.loads((path or settings_path()).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return Settings()
    if not isinstance(saved, dict):
        return Settings()
    defaults = Settings()
    graphics = saved.get("graphics")
    graphics = graphics if isinstance(graphics, dict) else {}
    chosen = {
        item.name: graphics[item.name]
        for item in fields(Graphics)
        if type(graphics.get(item.name)) is type(getattr(PERFORMANCE, item.name))
    }
    if chosen.get("resolution") not in RESOLUTIONS:
        chosen.pop("resolution", None)
    values = {
        item.name: saved[item.name]
        for item in fields(Settings)
        if item.name != "graphics"
        and type(saved.get(item.name)) is type(getattr(defaults, item.name))
    }
    if not 1 <= values.get("worlds", DEFAULT_WORLDS) <= MAX_WORLDS:
        values.pop("worlds")
    if values.get("ghosts", "normal") not in GHOST_STRENGTHS:
        values.pop("ghosts")
    return replace(defaults, graphics=replace(PERFORMANCE, **chosen), **values)


def save_settings(settings: Settings, path: Path | None = None) -> None:
    """Write the settings; a folder that cannot be written is silently skipped."""
    path = path or settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    except OSError:
        pass
