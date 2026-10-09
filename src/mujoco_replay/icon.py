"""The application's icon, drawn with NumPy at any size.

A centipede, the robot these recordings were first made of, walks over a dark
tile in the window's colours, in front of two grey ghosts of itself, as the
window draws the best world among the others; a magenta point ahead of it is
its target, and a play button says that this is a replay. The shapes are
signed distances, so every size is drawn sharp and smooth, and the smallest
sizes leave out the details they could not show. The viewer gives its window
this icon; ``python -m mujoco_replay.icon [PATH]`` writes it as a Windows icon
file, for a shortcut, and prints where. Only NumPy and the standard library
are imported.
"""

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

# The sizes a Windows icon file holds, and the colours.
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
TILE_TOP, TILE_BOTTOM = (0.17, 0.21, 0.26), (0.06, 0.07, 0.09)
OUTLINE = (0.05, 0.06, 0.08)
GHOST = (0.62, 0.64, 0.68)
BODY_TOP, BODY_BOTTOM = (1.0, 0.68, 0.26), (0.90, 0.38, 0.08)
LEGS_TOP, LEGS_BOTTOM = (0.80, 0.33, 0.07), (0.62, 0.22, 0.05)
TARGET = (0.95, 0.15, 0.65)  # the window's marker magenta
BADGE = (0.20, 0.45, 0.80)  # the panel's accent blue
PLAY = (0.96, 0.97, 0.99)


def draw(size: int) -> np.ndarray:
    """The icon as an RGBA image, ``(size, size, 4)`` bytes, top row first."""
    centres = (np.arange(size) + 0.5) / size
    x, y = np.meshgrid(centres, centres)  # in shares of the size, y down
    canvas = np.zeros((size, size, 4))
    tile = _rounded_square(x, y, 0.46, 0.11)
    _paint(canvas, _gradient(y, TILE_TOP, TILE_BOTTOM, 0.04, 0.96), _cover(tile, size))
    if size >= 48:  # the floor's squares, faintly
        squares = (np.floor(x * 8) + np.floor(y * 8)) % 2 == 0
        floor = np.clip((y - 0.35) / 0.3, 0, 1) * squares * _cover(tile + 0.02, size)
        _paint(canvas, np.array(TILE_TOP), floor * 0.18)
    small = size <= 24
    shape = _Centipede(segments=6 if small else 8, small=small)
    if not small:
        for dx, dy, phase in ((-0.04, -0.13, 1.6), (0.05, 0.12, -1.2)):
            body, legs = shape.distances(x, y, dx, dy, phase, antennae=False)
            _paint(canvas, np.array(GHOST), _cover(np.minimum(body, legs), size) * 0.18)
    body, legs = shape.distances(x, y, 0.0, 0.0, 0.3, antennae=size >= 48)
    _paint(
        canvas, np.array(OUTLINE), _cover(np.minimum(body, legs) - 0.016, size) * 0.85
    )
    _paint(canvas, _gradient(y, LEGS_TOP, LEGS_BOTTOM, 0.3, 0.8), _cover(legs, size))
    _paint(canvas, _gradient(y, BODY_TOP, BODY_BOTTOM, 0.3, 0.75), _cover(body, size))
    if size >= 48:
        _paint(canvas, np.array(TARGET), _cover(_circle(x, y, 0.80, 0.24, 0.035), size))
    cx, cy, radius = (0.74, 0.74, 0.20) if small else (0.76, 0.76, 0.16)
    _paint(canvas, np.array(OUTLINE), _cover(_circle(x, y, cx, cy, radius), size))
    _paint(canvas, np.array(BADGE), _cover(_circle(x, y, cx, cy, radius - 0.028), size))
    play = _play_triangle(x, y, cx + 0.012, cy, 0.85 * radius) - 0.006
    _paint(canvas, np.array(PLAY), _cover(play, size))
    return np.uint8(np.round(np.clip(canvas, 0, 1) * 255))


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


class _Centipede:
    """A centipede seen from above, heading up and to the right."""

    def __init__(self, segments: int, small: bool) -> None:
        self.segments = segments
        self.radius = 0.055 if small else 0.042  # of the body
        self.leg = 0.07 if small else 0.085  # length
        self.thickness = 0.016 if small else 0.011  # of a leg

    def distances(self, x, y, dx, dy, phase, antennae):
        """The body's and the legs' distances, shifted by ``dx`` and ``dy``,
        with the body's wave at ``phase``."""
        t = np.linspace(0, 1, self.segments)
        spine = np.stack(
            [
                0.17 + dx + 0.49 * t,
                0.66 + dy - 0.24 * t + 0.05 * np.sin(1.8 * np.pi * t + phase),
            ],
            axis=1,
        )
        radius = self.radius
        body = np.full(x.shape, np.inf)
        legs = np.full(x.shape, np.inf)
        for start, end in zip(spine[:-1], spine[1:], strict=True):
            body = np.minimum(body, _segment(x, y, start, end, radius))
            ahead = (end - start) / np.linalg.norm(end - start)
            for side in (-1, 1):
                out = side * np.array([-ahead[1], ahead[0]])
                tip = start + self.leg * (out - 0.35 * ahead)
                legs = np.minimum(legs, _segment(x, y, start, tip, self.thickness))
        head, ahead = (
            spine[-1],
            (spine[-1] - spine[-2]) / np.linalg.norm(spine[-1] - spine[-2]),
        )
        centre = head + 0.6 * radius * ahead
        body = np.minimum(body, _circle(x, y, *centre, 1.3 * radius))
        if antennae:
            base = head + 1.2 * radius * ahead
            for side in (-1, 1):
                out = side * np.array([-ahead[1], ahead[0]])
                tip = base + 0.07 * (0.8 * ahead + 0.6 * out)
                legs = np.minimum(legs, _segment(x, y, base, tip, 0.8 * self.thickness))
        return body, legs


def _rounded_square(x, y, half, corner):
    """The distance to a square with rounded corners in the middle."""
    qx, qy = np.abs(x - 0.5) - half + corner, np.abs(y - 0.5) - half + corner
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0) - corner


def _circle(x, y, cx, cy, radius):
    return np.hypot(x - cx, y - cy) - radius


def _segment(x, y, start, end, radius):
    """The distance to a line from ``start`` to ``end`` with round ends."""
    (ax, ay), (ex, ey) = start, end - start
    along = np.clip(((x - ax) * ex + (y - ay) * ey) / (ex * ex + ey * ey), 0, 1)
    return np.hypot(x - ax - ex * along, y - ay - ey * along) - radius


def _play_triangle(x, y, cx, cy, side):
    """The distance to a triangle pointing right, centred on ``cx, cy``."""
    height = side * np.sqrt(3) / 2
    corners = np.array(
        [
            (cx - height / 3, cy - side / 2),
            (cx + 2 * height / 3, cy),
            (cx - height / 3, cy + side / 2),
        ]
    )
    inside = np.full(x.shape, -np.inf)
    nearest = np.full(x.shape, np.inf)
    for start, end in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        ex, ey = end - start
        length = np.hypot(ex, ey)
        # The edges run clockwise on the screen, so the inside is on the right.
        inside = np.maximum(
            inside, ((x - start[0]) * ey - (y - start[1]) * ex) / length
        )
        nearest = np.minimum(nearest, _segment(x, y, start, end, 0.0))
    return np.where(inside > 0, nearest, inside)


def _gradient(y, top, bottom, start, end):
    """A colour from ``top`` to ``bottom`` between two heights."""
    share = np.clip((y - start) / (end - start), 0, 1)[..., None]
    return np.asarray(top) + (np.asarray(bottom) - np.asarray(top)) * share


def _cover(distance, size):
    """How much of each pixel a shape covers, from its distance in shares."""
    return np.clip(0.5 - distance * size, 0, 1)


def _paint(canvas, colour, alpha):
    """Lay a colour over the canvas with the given coverage."""
    alpha = alpha[..., None]
    canvas[..., :3] = canvas[..., :3] * (1 - alpha) + colour * alpha
    canvas[..., 3:] = canvas[..., 3:] * (1 - alpha) + alpha


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
