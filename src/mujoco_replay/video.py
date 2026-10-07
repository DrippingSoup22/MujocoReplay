"""Writing a playlist to an MP4 video, drawn offscreen exactly as in the window.

Playback runs on a clock of video frames instead of the wall clock: video frame
``i`` shows the scene at ``i / fps`` seconds, so each recorded frame lasts
``seconds_per_frame`` in the video. A recorded frame is drawn once and its
pixels repeated, and redrawn only when the event flash appears or goes. The
frames go to ``imageio`` with the ``ffmpeg`` plugin of ``imageio-ffmpeg``,
which bundles its own encoder; both come with the ``video`` extra and are
imported only here.
"""

import math
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from mujoco_replay.playback import Playback
from mujoco_replay.recording import Recording
from mujoco_replay.render import SceneRenderer, offscreen_context
from mujoco_replay.scene import ComposedScene

FLASH_SECONDS = 1.0


def export(
    recordings: list[Recording],
    worlds: list[np.ndarray],
    path: Path | str,
    seconds_per_frame: float = 0.3,
    fps: int = 30,
    size: tuple[int, int] = (1280, 720),
    hud: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Write the playlist to ``path`` as an MP4; return the video frames written.

    ``worlds`` holds, per recording, the indices of the worlds to draw, best
    first. The size is rounded down to even numbers, as the encoder needs.
    ``progress``, when given, is called with the frames done and the total.
    """
    import imageio.v2 as imageio

    width, height = size[0] // 2 * 2, size[1] // 2 * 2
    recorded = sum(recording.frame_count for recording in recordings)
    total = math.ceil(recorded * seconds_per_frame * fps - 1e-9)
    playback = Playback(recordings, seconds_per_frame, now=0.0)
    shown_file, flash, flash_until = 0, "", -1.0
    drawn, pixels = None, None
    with ExitStack() as cleanup:  # releases in reverse order, also on errors
        context = offscreen_context(width, height)
        cleanup.callback(context.free)
        scene = ComposedScene(recordings[0], worlds[0])
        renderer = SceneRenderer(scene, (width, height), _font_scale(height))
        cleanup.callback(renderer.close)
        writer = cleanup.enter_context(
            imageio.get_writer(
                path, format="FFMPEG", fps=fps, quality=8, macro_block_size=2
            )
        )
        for index in range(total):
            now = index / fps
            labels = playback.advance(now)
            if labels:
                flash, flash_until = "  ".join(labels), now + FLASH_SECONDS
            if playback.file_index != shown_file:
                shown_file = playback.file_index
                scene = _scene_for(scene, recordings[shown_file], worlds[shown_file])
                renderer.show(scene)
            showing = flash if now < flash_until else ""
            key = (shown_file, playback.frame_index, showing)
            if key != drawn:
                scene.set_frame(playback.frame_index)
                renderer.render(width, height, playback.status(), showing, hud=hud)
                pixels, drawn = renderer.read_pixels(width, height), key
            writer.append_data(pixels)
            if progress is not None:
                progress(index + 1, total)
    return total


def _scene_for(
    previous: ComposedScene, recording: Recording, worlds: np.ndarray
) -> ComposedScene:
    """The next file's scene: the previous composite when it fits, else a new one."""
    if previous.fits(recording, worlds):
        previous.show(recording, worlds)
        previous.set_highlight(0)
        return previous
    return ComposedScene(recording, worlds)


def _font_scale(height: int) -> int:
    """MuJoCo's font scale for a video height: 150 at 720 rows, 200 at 1080."""
    return int(np.clip(50 * round(height / 240), 100, 300))
