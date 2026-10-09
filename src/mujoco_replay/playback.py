"""Where playback stands: which file and frame, playing or paused, and how fast.

Pure logic without graphics, driven by the viewer and the video exporter. The
files are a video's playlist, where playing runs on from the end of one file
into the next, or the window's tabs, each played on its own unless Play next
runs them on too, and Loop starts them again; showing another file keeps the
frame, so that two files can be compared at the same moment.
The speed is wall-clock seconds per recorded frame, slow enough by default to
see each step. Advancing uses wall time, so a slow renderer skips frames at
fast speeds instead of slowing down. docs/design.md lists the keys that drive
it.
"""

import math

from mujoco_replay.recording import Recording

DEFAULT_SECONDS_PER_FRAME = 0.1
# Speed presets in seconds per recorded frame. Real time and its multiples are
# added per file, since they depend on the file's frame_seconds.
FIXED_PRESETS = (3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05)
REAL_TIME_MULTIPLES = (1, 2, 4)


class Playback:
    """A playlist of recordings and a position in it.

    With ``run_on``, playing and stepping go on from the end of a file into
    the next, as a video does; without, they stay in the file shown. With
    ``loop``, playing starts again at the end: the file shown, or, running
    on, the first file after the last.
    """

    def __init__(
        self,
        recordings: list[Recording],
        seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME,
        now: float = 0.0,
        run_on: bool = True,
        loop: bool = False,
    ) -> None:
        self.recordings = recordings
        self.run_on = run_on
        self.loop = loop
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
        """Play or pause; playing again at the end replays the file."""
        if not self.playing and self._at_end():
            self.frame_index = 0
        self.playing = not self.playing
        self._owed = 0.0

    def step(self, frames: int) -> list[str]:
        """Pause and move ``frames`` frames; return events passed."""
        self.playing = False
        self._owed = 0.0
        if frames > 0:
            return self._forward(frames)
        for _ in range(-frames):
            if self.frame_index > 0:
                self.frame_index -= 1
            elif self.run_on and self.file_index > 0:
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

    def show_file(self, index: int) -> None:
        """Show another file at the same frame, playing or paused as before."""
        self.file_index = index
        self._fit_frame()

    def remove_file(self, index: int) -> None:
        """Close a file; when it was shown, the one after it takes its place,
        or the one before when it was the last. The last file may go too."""
        del self.recordings[index]
        if index < self.file_index or self.file_index == len(self.recordings):
            self.file_index -= 1
        if self.recordings:
            self._fit_frame()

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
        """Move forward frame by frame, into the next file at a file's end
        when playback runs on."""
        passed: list[str] = []
        for _ in range(frames):
            if self.frame_index + 1 < self.recording.frame_count:
                self.frame_index += 1
                passed += self._events_at(self.frame_index)
            elif self.run_on and self.file_index + 1 < len(self.recordings):
                passed += self._events_at(self.recording.frame_count)
                self.file_index += 1
                self.frame_index = 0
                passed += self._events_at(0)
            elif self.loop and self.playing:  # the end: from the start again
                passed += self._events_at(self.recording.frame_count)
                if self.run_on:
                    self.file_index = 0
                self.frame_index = 0
                passed += self._events_at(0)
            else:
                if self.playing:  # the end, reached while playing
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

    def _fit_frame(self) -> None:
        """Keep the frame in the file shown: a shorter file shows its last
        frame, paused, so that no mode leaves it at once for another."""
        last = self.recording.frame_count - 1
        if self.frame_index > last:
            self.frame_index, self.playing = last, False

    def _at_end(self) -> bool:
        last = not self.run_on or self.file_index == len(self.recordings) - 1
        return last and self.frame_index == self.recording.frame_count - 1

    def _set_speed(self, seconds_per_frame: float) -> None:
        self.seconds_per_frame = seconds_per_frame
        self._owed = 0.0
