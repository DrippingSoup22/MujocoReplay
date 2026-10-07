"""Where playback stands: which file and frame, playing or paused, and how fast.

Pure logic without graphics, driven by the viewer and the video exporter. The
speed is wall-clock seconds per recorded frame, slow enough by default to see
each step. Advancing uses wall time, so a slow renderer skips frames at fast
speeds instead of slowing down. docs/design.md lists the keys that drive it.
"""

import math

from mujoco_replay.recording import Recording

DEFAULT_SECONDS_PER_FRAME = 0.3
# Speed presets in seconds per recorded frame. Real time and its multiples are
# added per file, since they depend on the file's frame_seconds.
FIXED_PRESETS = (3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05)
REAL_TIME_MULTIPLES = (1, 2, 4)


class Playback:
    """A playlist of recordings and a position in it."""

    def __init__(
        self,
        recordings: list[Recording],
        seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME,
        now: float = 0.0,
    ) -> None:
        self.recordings = recordings
        self.file_index = 0
        self.frame_index = 0
        self.playing = True
        self.seconds_per_frame = seconds_per_frame
        self._clock = now
        self._owed = 0.0  # wall time not yet turned into frames

    @property
    def recording(self) -> Recording:
        return self.recordings[self.file_index]

    @property
    def presets(self) -> list[float]:
        """The speed presets for the current file, slowest first."""
        frame_seconds = self.recording.frame_seconds
        real_time = {frame_seconds / multiple for multiple in REAL_TIME_MULTIPLES}
        return sorted(set(FIXED_PRESETS) | real_time, reverse=True)

    @property
    def real_time_multiple(self) -> float:
        return self.recording.frame_seconds / self.seconds_per_frame

    def advance(self, now: float) -> list[str]:
        """Move on by the wall time since the last call; return events passed."""
        elapsed, self._clock = now - self._clock, now
        if not self.playing:
            return []
        self._owed += elapsed
        # The tolerance counts an exact multiple whole despite rounding.
        frames = int(self._owed / self.seconds_per_frame + 1e-9)
        self._owed -= frames * self.seconds_per_frame
        return self._forward(frames)

    def sync(self, now: float) -> None:
        """Forget the wall time since the last call, after a slow pause."""
        self._clock = now

    def seconds_to_next_frame(self, now: float) -> float:
        """Wall time until playing moves on by a frame; infinite while paused."""
        if not self.playing:
            return math.inf
        return max(0.0, self.seconds_per_frame - self._owed - (now - self._clock))

    def toggle(self) -> None:
        """Play or pause; playing again at the very end replays the file."""
        if not self.playing and self._at_end():
            self.frame_index = 0
        self.playing = not self.playing
        self._owed = 0.0

    def step(self, frames: int) -> list[str]:
        """Pause and move ``frames`` frames, across files; return events passed."""
        self.playing = False
        self._owed = 0.0
        if frames > 0:
            return self._forward(frames)
        for _ in range(-frames):
            if self.frame_index > 0:
                self.frame_index -= 1
            elif self.file_index > 0:
                self.file_index -= 1
                self.frame_index = self.recording.frame_count - 1
        return []

    def faster(self) -> None:
        quicker = [value for value in self.presets if value < self.seconds_per_frame]
        if quicker:
            self._set_speed(quicker[0])

    def slower(self) -> None:
        slower = [value for value in self.presets if value > self.seconds_per_frame]
        if slower:
            self._set_speed(slower[-1])

    def default_speed(self) -> None:
        self._set_speed(DEFAULT_SECONDS_PER_FRAME)

    def seek(self, frame: int) -> None:
        """Pause at a frame of the current file, as a click on the timeline does."""
        self.playing = False
        self._owed = 0.0
        self.frame_index = min(max(frame, 0), self.recording.frame_count - 1)

    def first_frame(self) -> None:
        self.frame_index = 0

    def last_frame(self) -> None:
        self.frame_index = self.recording.frame_count - 1

    def restart(self) -> None:
        """Play the current file from its first frame."""
        self.frame_index = 0
        self.playing = True
        self._owed = 0.0

    def next_file(self) -> None:
        if self.file_index + 1 < len(self.recordings):
            self.file_index += 1
            self.frame_index = 0

    def previous_file(self) -> None:
        if self.file_index > 0:
            self.file_index -= 1
            self.frame_index = 0

    def status(self) -> str:
        """The playback line of the overlay."""
        recording = self.recording
        parts = []
        if len(self.recordings) > 1:
            parts.append(f"file {self.file_index + 1} of {len(self.recordings)}")
        parts.append(f"frame {self.frame_index + 1} / {recording.frame_count}")
        parts.append(f"{self.frame_index * recording.frame_seconds:.2f} s")
        parts.append(
            f"{self.seconds_per_frame:.3g} s per frame "
            f"({self.real_time_multiple:.2g}x real time)"
        )
        if not self.playing:
            parts.append("paused")
        return " | ".join(parts)

    def _forward(self, frames: int) -> list[str]:
        """Move forward frame by frame, into the next file at a file's end."""
        passed: list[str] = []
        for _ in range(frames):
            if self.frame_index + 1 < self.recording.frame_count:
                self.frame_index += 1
                passed += self._events_at(self.frame_index)
            elif self.file_index + 1 < len(self.recordings):
                passed += self._events_at(self.recording.frame_count)
                self.file_index += 1
                self.frame_index = 0
                passed += self._events_at(0)
            else:
                if self.playing:  # the end of the playlist, reached while playing
                    passed += self._events_at(self.recording.frame_count)
                self.playing = False
                break
        return passed

    def _events_at(self, frame: int) -> list[str]:
        """The labels of the current file's events at ``frame``."""
        recording = self.recording
        if recording.event_frames is None:
            return []
        return [
            label
            for event_frame, label in zip(
                recording.event_frames, recording.event_labels, strict=True
            )
            if event_frame == frame
        ]

    def _at_end(self) -> bool:
        return (
            self.file_index == len(self.recordings) - 1
            and self.frame_index == self.recording.frame_count - 1
        )

    def _set_speed(self, seconds_per_frame: float) -> None:
        self.seconds_per_frame = seconds_per_frame
        self._owed = 0.0
