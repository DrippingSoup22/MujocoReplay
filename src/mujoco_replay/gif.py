"""Animated GIFs: what the window shows, recorded, and a video's frames, written.

A GIF holds up to 256 colours per frame and a delay per frame in hundredths
of a second. Each frame is reduced to its own palette with Pillow's fast
octree, which drew Centipede's scene as well as median cut in a third of the
size and a seventh of the time (3.5 ms for 640 by 360). Pillow merges a frame
that repeats the one before into it, so a paused scene costs nothing. Pillow
comes with the ``video`` extra, and with the executable; it is imported when a
GIF is first made, so the window runs without it.
"""

import queue
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# The window's recording: frames at most this wide, this many a second, for at
# most this long, which holds about 200 MB of frames.
RECORD_WIDTH = 640
RECORD_FPS = 15
RECORD_SECONDS = 60
SHORTEST_DELAY = 0.02  # browsers draw shorter delays as a tenth of a second


def delays(times: Sequence[float], end: float) -> list[int]:
    """Each frame's delay in milliseconds, from the times it was shown.

    Frame ``i`` lasts from ``times[i]`` to the next frame's time, the last one
    to ``end``. A GIF counts hundredths of a second, so the delays round the
    running time, not each delay, and do not drift; none is shorter than
    SHORTEST_DELAY.
    """
    marks = np.round((np.append(times, end) - times[0]) * 100).astype(int)
    shortest = round(SHORTEST_DELAY * 100)
    return [10 * max(shortest, int(gap)) for gap in np.diff(marks)]


def write_gif(path: Path | str, frames: list, milliseconds: Sequence[int]) -> None:
    """Write paletted frames, each with its delay, as a GIF that loops."""
    frames[0].save(
        path,
        "GIF",
        save_all=True,
        append_images=frames[1:],
        duration=list(milliseconds),
        loop=0,
    )


def paletted(rgb: np.ndarray, size: tuple[int, int] | None = None):
    """An RGB image, ``(height, width, 3)``, as a frame of its own 256 colours,
    first resized to ``size`` (width, height) if given."""
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(rgb))
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BOX)
    return image.quantize(256, method=Image.Quantize.FASTOCTREE)


class GifWriter:
    """Frames of one size at a fixed rate into a GIF, as an MP4 writer takes
    them: ``append_data`` each frame, and ``close`` writes the file."""

    def __init__(self, path: Path | str, fps: float) -> None:
        import PIL  # noqa: F401  (fails here, before any drawing, without Pillow)

        self.path, self.fps = Path(path), fps
        self._frames: list = []

    def append_data(self, rgb: np.ndarray) -> None:
        self._frames.append(paletted(rgb))

    def close(self) -> None:
        count = len(self._frames)
        times = np.arange(count) / self.fps
        write_gif(self.path, self._frames, delays(times, count / self.fps))

    def __enter__(self) -> "GifWriter":
        return self

    def __exit__(self, kind, error, trace) -> None:
        if kind is None and self._frames:
            self.close()


class GifRecorder:
    """What the window shows, recorded into a GIF at ``path``.

    The window hands over a frame with the time it was shown; a thread of its
    own shrinks it to at most RECORD_WIDTH and reduces its colours, so that
    the window keeps drawing. Every frame takes the first one's size, should
    the window change. ``stop`` ends the recording, and the thread writes the
    file; ``finished`` tells when, and ``error`` why it failed, if it did.
    Making one without Pillow raises ``ImportError``.
    """

    def __init__(self, path: Path | str, now: float) -> None:
        import PIL  # noqa: F401  (fails at once, before any recording, without it)

        self.path, self.started = Path(path), now
        self.recording, self.error = True, ""
        self._times: list[float] = []
        self._frames: list = []
        self._size: tuple[int, int] | None = None
        self._queue: queue.Queue = queue.Queue()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def seconds(self, now: float) -> float:
        """How long it has recorded."""
        return now - self.started

    def add(self, rgb: np.ndarray, now: float) -> None:
        """A frame the window shows from ``now`` on, ``(height, width, 3)``."""
        if self.recording:
            self._queue.put((rgb, now))

    def stop(self, now: float) -> None:
        """End the recording at ``now``; the file is written next."""
        if self.recording:
            self.recording = False
            self._queue.put((None, now))

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def wait(self) -> None:
        """Wait until the file is written, or failed."""
        self._thread.join()

    def _work(self) -> None:
        try:
            while True:
                rgb, now = self._queue.get()
                if rgb is None:
                    break
                if self._size is None:
                    height, width = rgb.shape[:2]
                    scale = min(1.0, RECORD_WIDTH / width)
                    self._size = (
                        max(1, round(width * scale)),
                        max(1, round(height * scale)),
                    )
                self._frames.append(paletted(rgb, self._size))
                self._times.append(now)
            if not self._frames:
                raise RuntimeError("no frame was recorded")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_gif(self.path, self._frames, delays(self._times, now))
        except Exception as error:  # a full disk, a folder that cannot be made
            self.error = str(error) or type(error).__name__
        finally:
            self._frames = []
            self._done.set()
