"""The application's icon: pixel art, kept as text and scaled by whole pixels.

Three figures stand on the checkered floor of a simulated world: one model in
three worlds, two in ghost grey and the best in colour, as the window draws
them, with a play sign for the replay. They stand for a model of any kind,
since any model replays the same way. The picture is 32 by 32 pixels, with a
simpler one of 16 by 16 for the smallest sizes, each letter a pixel of the
palette's colour; scaling repeats whole pixels, so every size stays sharp.
The viewer gives its window this icon; ``python -m mujoco_replay.icon [PATH]``
writes it as a Windows icon file, for a shortcut, and prints where. Only NumPy
and the standard library are imported.
"""

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

# The sizes a Windows icon file holds: multiples of the two pictures.
ICO_SIZES = (16, 32, 48, 64, 128, 256)
# Each letter's colour: red, green, blue, and opacity.
PALETTE = {
    ".": (0, 0, 0, 0),  # outside the tile
    "K": (12, 14, 20, 255),  # the outline
    "b": (40, 52, 70, 255),  # the sky, high
    "B": (30, 38, 52, 255),  # the sky, low
    "F": (64, 80, 100, 255),  # the floor's light squares
    "f": (46, 58, 76, 255),  # its dark squares
    "g": (96, 106, 122, 255),  # a ghost
    "O": (247, 147, 30, 255),  # the best world
    "o": (196, 100, 18, 255),  # its shade
    "Y": (255, 214, 102, 255),  # its light
    "W": (240, 242, 246, 255),  # the play sign
}
ICON = (
    "...KKKKKKKKKKKKKKKKKKKKKKKKKK...",
    "..KbbbbbbbbbbbbbbbbbbbbbbbbbbK..",
    ".KbbbbbbbbbbbbbbbbbbbbbbbbbbbbK.",
    "KbbbbbbbbbbbbbbbbbbbbbbbWbbbbbbK",
    "KbbbbbbbbbbbbbbbbbbbbbbbWWbbbbbK",
    "KbbbbbbbbbbbbbbbbbbbbbbbWWWbbbbK",
    "KbbbbbbbbbbbbbbbbbbbbbbbWWWWbbbK",
    "KbbbbbbbbbbbbbbbbbbbbbbbWWWbbbbK",
    "KbbbbbbbbbbbbbbbbbbbbbbbWWbbbbbK",
    "KbbbbbbbbbbbbbbbbbbbbbbbWbbbbbbK",
    "KBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBK",
    "KBBBBBgggBBgBBgggBBgBBYOOBBBBBBK",
    "KBBBBgggggBgBgggggBgBYOOOOBBBBBK",
    "KBBBBgggggBBgggggggBBYOOOOBBBBBK",
    "KBBBBBgggBBBBBgggBBBBBYOOBBBBBBK",
    "KBBBBBBgBBBBBBBgBBBBBBBOBBBBBBBK",
    "KBBBBgggggBBBgggggBBBOOOOoBBBBBK",
    "KBBBgBgggBgBBgggggBBOOOOOOoBBBBK",
    "KBBgBBgggBBgBBgggBBoBBOOoBBoBBBK",
    "KBBBBBgggBBBBBgggBBoBBOOoBBoBBBK",
    "KBBBBBgggBBBBBgggBBBBBOOoBBBBBBK",
    "KBBBBggBggBBBBgggBBBBBOOoBBBBBBK",
    "KBBBBgBBBgBBBBgBgBBBBBoBoBBBBBBK",
    "KBBBggBBBggBBggBggBBBOoBOoBBBBBK",
    "KBBBgBBBBBgBBgBBBgBBBoBBBoBBBBBK",
    "KFFggfffFFggggffFggFOoffFOoFfffK",
    "KFFFffffFFFFffffFFFFffffFFFFfffK",
    "KFFFffffFFFFffffFFFFffffFFFFfffK",
    "KfffFFFFffffFFFFffffFFFFffffFFFK",
    ".KffFFFFffffFFFFffffFFFFffffFFK.",
    "..KfFFFFffffFFFFffffFFFFffffFK..",
    "...KKKKKKKKKKKKKKKKKKKKKKKKKK...",
)
SMALL_ICON = (
    "..KKKKKKKKKKKK..",
    ".KbbbbbbbbbbbbK.",
    "KbbbbbbbbbbbbbbK",
    "KbbbbbbbbbbbbbbK",
    "KBBgBBBgBBBYBBBK",
    "KBgggBgggBOOOBBK",
    "KBBgBBBgBBBOBBBK",
    "KBBgBBBgBBBOBBBK",
    "KBgBgBgBgBOBoBBK",
    "KBgBgBgBgBOBoBBK",
    "KFFffFFffFFffFFK",
    "KFFffFFffFFffFFK",
    "KffFFffFFffFFffK",
    "KffFFffFFffFFffK",
    ".KFFffFFffFFffK.",
    "..KKKKKKKKKKKK..",
)


def draw(size: int) -> np.ndarray:
    """The icon as an RGBA image, ``(size, size, 4)`` bytes, top row first.

    A multiple of 32 pixels shows the full picture, other sizes the simpler
    one; each pixel becomes a square of whole pixels where the size allows.
    """
    rows = ICON if size % len(ICON) == 0 else SMALL_ICON
    picture = np.array([[PALETTE[char] for char in row] for row in rows], np.uint8)
    nearest = np.arange(size) * len(rows) // size
    return picture[nearest][:, nearest]


def ico_bytes(sizes: tuple[int, ...] = ICO_SIZES) -> bytes:
    """A Windows icon file holding the icon at ``sizes``, each as a PNG."""
    images = [_png(draw(size)) for size in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries = b""
    for size, image in zip(sizes, images, strict=True):
        side = size % 256  # 0 stands for 256
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(image), offset)
        offset += len(image)
    return header + entries + b"".join(images)


def main(arguments: list[str] | None = None) -> int:
    """Write the icon file to the path given, or next to the settings; print
    where, so that a shortcut can be pointed at it."""
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments:
        path = Path(arguments[0])
    else:
        from mujoco_replay.settings import user_folder

        path = user_folder("settings") / "MujocoReplay.ico"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ico_bytes())
    print(path.resolve())
    return 0


def _png(rgba: np.ndarray) -> bytes:
    """An RGBA image as a PNG file, each row unfiltered."""
    height, width = rgba.shape[:2]
    rows = np.concatenate(
        [np.zeros((height, 1), np.uint8), rgba.reshape(height, -1)], 1
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 9))
        + chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    sys.exit(main())
