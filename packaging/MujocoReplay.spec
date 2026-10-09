# PyInstaller's recipe for the executable: pyinstaller packaging/MujocoReplay.spec
#
# Two programs in one folder: MujocoReplay, the window without a console,
# which carries the icon, and MujocoReplay-console, the same command with a
# console, for the terminal and render, which carries none, so that the
# window is the one program in the folder that looks like the application.
# They share one folder of Python, NumPy, MuJoCo, and GLFW, so that nothing
# needs installing. MuJoCo loads its library and its plugins, and GLFW its
# library, from their packages' own folders, so those are collected in place;
# imageio reads its version from its package's metadata, which is copied; and
# the icon is written from the package's pixel art.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT / "src"))
from mujoco_replay.icon import ico_bytes  # noqa: E402

ICON = Path(workpath) / "MujocoReplay.ico"
ICON.parent.mkdir(parents=True, exist_ok=True)
ICON.write_bytes(ico_bytes())
LIBRARIES = collect_dynamic_libs("mujoco") + collect_dynamic_libs("glfw")
METADATA = copy_metadata("imageio") + copy_metadata("imageio-ffmpeg")

parts = []
for script, name, console, program_icon in (
    ("launcher.py", "MujocoReplay", False, str(ICON)),
    ("__main__.py", "MujocoReplay-console", True, "NONE"),
):
    found = Analysis(
        [str(ROOT / "src" / "mujoco_replay" / script)],
        pathex=[str(ROOT / "src")],
        binaries=LIBRARIES,
        datas=METADATA,
    )
    program = EXE(
        PYZ(found.pure),
        found.scripts,
        exclude_binaries=True,
        name=name,
        console=console,
        icon=program_icon,
        upx=False,
    )
    parts += [program, found.binaries, found.datas]

COLLECT(*parts, name="MujocoReplay", upx=False)
