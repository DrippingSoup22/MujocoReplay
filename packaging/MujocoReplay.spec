# PyInstaller's recipe for the executable: pyinstaller packaging/MujocoReplay.spec
#
# One program, MujocoReplay, the window without a console, which carries the
# icon, in a folder with the Python, NumPy, MuJoCo, and GLFW it runs on, so
# that nothing needs installing. The command line, for render and the
# terminal, comes with the Python package instead. MuJoCo loads its library
# and its plugins, and GLFW its library, from their packages' own folders, so
# those are collected in place; imageio reads its version from its package's
# metadata, which is copied; and the icon is written from the package's pixel
# art.

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

found = Analysis(
    [str(ROOT / "src" / "mujoco_replay" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=LIBRARIES,
    datas=METADATA,
)
program = EXE(
    PYZ(found.pure),
    found.scripts,
    exclude_binaries=True,
    name="MujocoReplay",
    console=False,
    icon=str(ICON),
    upx=False,
)
COLLECT(program, found.binaries, found.datas, name="MujocoReplay", upx=False)
