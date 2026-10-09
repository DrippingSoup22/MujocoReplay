"""Tests for the icon: drawn at any size, and written as a Windows icon file."""

import struct
import zlib

import numpy as np

from mujoco_replay import icon


def test_the_icon_has_clear_corners_a_solid_tile_and_its_colours_at_every_size():
    for size in (16, 32, 48, 64, 256):
        image = icon.draw(size)

        assert image.shape == (size, size, 4) and image.dtype == np.uint8
        assert image[0, 0, 3] == image[-1, -1, 3] == 0  # the tile's round corners
        assert image[size // 2, size // 4, 3] == 255
        red, green, blue = image[..., :3].reshape(-1, 3).astype(int).T
        assert ((red > 200) & (blue < 100)).any(), size  # the best world, orange
        assert ((red > 80) & (red < 130) & (blue > red)).any(), size  # its ghosts


def test_every_size_repeats_whole_pixels_of_one_of_the_two_pictures():
    big, small = icon.draw(32), icon.draw(16)

    assert (icon.draw(256) == big.repeat(8, 0).repeat(8, 1)).all()
    assert (icon.draw(48) == small.repeat(3, 0).repeat(3, 1)).all()
    assert len({len(row) for row in icon.ICON + icon.SMALL_ICON}) == 2


def test_the_icon_file_holds_each_size_as_a_png_of_the_drawing(tmp_path, capsys):
    path = tmp_path / "MujocoReplay.ico"

    assert icon.main([str(path)]) == 0

    data = path.read_bytes()
    assert struct.unpack_from("<HHH", data) == (0, 1, len(icon.ICO_SIZES))
    for index, size in enumerate(icon.ICO_SIZES):
        entry = struct.unpack_from("<BBBBHHII", data, 6 + 16 * index)
        side, length, offset = entry[0], entry[6], entry[7]
        png = data[offset : offset + length]
        assert side == size % 256 and png.startswith(b"\x89PNG\r\n\x1a\n")
        assert struct.unpack_from(">II", png, 16) == (size, size)
    first = data[struct.unpack_from("<I", data, 6 + 12)[0] :]
    rows = np.frombuffer(zlib.decompress(_chunk(first, b"IDAT")), np.uint8)
    pixels = rows.reshape(16, 1 + 16 * 4)[:, 1:].reshape(16, 16, 4)
    assert (pixels == icon.draw(16)).all()
    assert capsys.readouterr().out.strip() == str(path.resolve())


def _chunk(png: bytes, kind: bytes) -> bytes:
    """The data of a PNG's first chunk of a kind."""
    at = 8
    while True:
        length = struct.unpack_from(">I", png, at)[0]
        if png[at + 4 : at + 8] == kind:
            return png[at + 8 : at + 8 + length]
        at += 12 + length
