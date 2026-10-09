# PyInstaller's recipe for the executable: pyinstaller packaging/MujocoReplay.spec
#
# Two programs in one folder, as pip makes them: MujocoReplay, the window
# without a console, and mujoco-replay, the command, for render and the
# terminal. They share one folder of Python, NumPy, MuJoCo, and GLFW, so
# that nothing needs installing. MuJoCo loads its library and its plugins, and
# GLFW its library, from their packages' own folders, so those are collected
# in place; imageio reads its version from its package's metadata, which is
# copied; and the programs' icon is written from the package's pixel art.

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
for script, name, console in (
    ("launcher.py", "MujocoReplay", False),
    ("__main__.py", "mujoco-replay", True),
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
        icon=str(ICON),
        upx=False,
    )
    parts += [program, found.binaries, found.datas]

COLLECT(*parts, name="MujocoReplay", upx=False)
