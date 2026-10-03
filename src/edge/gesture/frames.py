"""Frame sources: where 8×8 depth frames come from.

A frame is a tuple of 64 distances in millimetres, row-major, row 0 first,
exactly as the sensor delivers it. Orientation (which way is "up" for the
person in front of the mirror) is NOT handled here but in the classifier, so
recordings stay raw and can be re-evaluated after a mounting change.

Three sources share one interface — an iterator of ``(t, frame)`` with `t` in
seconds on a monotonic clock:

    SyntheticSource   generated swipes; walking skeleton and unit tests
    ReplaySource      a JSONL recording; regression tests on real hand data
    Sen0628Source     the DFRobot SEN0628 on UART5; needs the hardware

The SEN0628 with firmware V1.3 reports invalid zones as 4000 mm, so 4000 means
"nothing measured" rather than "far away".
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO, Protocol

from edge.events.publisher import GESTURES as GESTURE_NAMES

ROWS = COLS = 8
ZONES = ROWS * COLS
INVALID_MM = 4000

Frame = tuple[int, ...]


class FrameSourceError(RuntimeError):
    """Raised for problems the user can act on (missing file, no sensor)."""


class FrameSource(Protocol):
    name: str

    def frames(self) -> Iterator[tuple[float, Frame]]: ...


def validate(frame: Sequence[int]) -> Frame:
    """Return the frame as a tuple of ints, or raise if it is not 8×8."""
    if len(frame) != ZONES:
        raise ValueError(f"expected {ZONES} zones, got {len(frame)}")
    return tuple(int(v) for v in frame)


def decode_le16(raw: Sequence[int]) -> Frame:
    """Decode 128 bytes of (low, high) pairs into 64 distances.

    The DFRobot library returns the 8×8 data as byte pairs, low byte first.
    Kept separate from the serial code so it can be tested without hardware.
    """
    if len(raw) != 2 * ZONES:
        raise ValueError(f"expected {2 * ZONES} bytes, got {len(raw)}")
    return tuple(raw[i] | (raw[i + 1] << 8) for i in range(0, len(raw), 2))


# --------------------------------------------------------------------------- #
# Synthetic swipes
# --------------------------------------------------------------------------- #

def _blank(background_mm: int) -> list[int]:
    return [background_mm] * ZONES


def synthetic_swipe(gesture: str, *, steps: int = 8, hand_mm: int = 200,
                    background_mm: int = INVALID_MM) -> list[Frame]:
    """Frames of a 2×3 "hand" crossing the field in the canonical orientation.

    Canonical means: column 0 is the viewer's left, row 0 is up. A
    `swipe_right` therefore moves from column 0 towards column 7.
    """
    if gesture not in GESTURE_NAMES:
        raise ValueError(f"unknown gesture {gesture!r}; known: {GESTURE_NAMES}")
    horizontal = gesture in ("swipe_left", "swipe_right")
    forward = gesture in ("swipe_right", "swipe_down")
    # 2 zones along the motion axis, 3 across: the centroid moves cleanly and
    # the blob is clearly bigger than single-zone noise.
    along, across = 2, 3
    span = (COLS if horizontal else ROWS) - along
    frames = []
    for i in range(steps):
        pos = round(i * span / (steps - 1))
        if not forward:
            pos = span - pos
        f = _blank(background_mm)
        for a in range(along):
            for c in range(across):
                major, minor = pos + a, 3 + c - 1   # centred across the field
                row, col = (minor, major) if horizontal else (major, minor)
                f[row * COLS + col] = hand_mm
        frames.append(tuple(f))
    return frames


class SyntheticSource:
    """Plays a script of gestures, separated by empty frames.

    `realtime=True` sleeps between frames so the timing matches a real sensor;
    in tests it stays off and `t` advances virtually.
    """

    name = "synthetic"

    def __init__(self, script: Sequence[str], *, fps: float = 15.0,
                 gap_frames: int = 15, realtime: bool = False,
                 background_mm: int = INVALID_MM):
        for gesture in script:
            if gesture not in GESTURE_NAMES:
                raise FrameSourceError(
                    f"unknown gesture {gesture!r}; known: {', '.join(GESTURE_NAMES)}")
        self.script = list(script)
        self.fps = fps
        self.gap_frames = gap_frames
        self.realtime = realtime
        self.background_mm = background_mm

    def frames(self) -> Iterator[tuple[float, Frame]]:
        dt = 1.0 / self.fps
        t = time.monotonic()
        empty = tuple(_blank(self.background_mm))

        def emit(frame: Frame) -> tuple[float, Frame]:
            nonlocal t
            if self.realtime:
                time.sleep(dt)
                t = time.monotonic()
            else:
                t += dt
            return t, frame

        for _ in range(self.gap_frames):
            yield emit(empty)
        for gesture in self.script:
            for frame in synthetic_swipe(gesture,
                                         background_mm=self.background_mm):
                yield emit(frame)
            for _ in range(self.gap_frames):
                yield emit(empty)


# --------------------------------------------------------------------------- #
# Recordings
# --------------------------------------------------------------------------- #

class Recorder:
    """Writes frames as JSONL: {"t": seconds, "d": [64 distances]} per line.

    Recordings contain no image data — only distances — so they can live in
    tests/fixtures/ once they are curated.
    """

    def __init__(self, path: Path | str):
        self._fh: IO[str] = open(path, "w", encoding="utf-8")  # noqa: SIM115
        self._t0: float | None = None

    def write(self, t: float, frame: Frame) -> None:
        if self._t0 is None:
            self._t0 = t
        self._fh.write(json.dumps({"t": round(t - self._t0, 4), "d": frame},
                                  separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._fh.close()


class ReplaySource:
    """Replays a JSONL recording. Timestamps are taken from the file."""

    name = "replay"

    def __init__(self, path: Path | str, *, realtime: bool = False):
        self.path = Path(path)
        if not self.path.is_file():
            raise FrameSourceError(f"recording not found: {self.path}")
        self.realtime = realtime

    def frames(self) -> Iterator[tuple[float, Frame]]:
        previous: float | None = None
        with self.path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    t, frame = float(row["t"]), validate(row["d"])
                except (ValueError, KeyError, TypeError) as exc:
                    raise FrameSourceError(
                        f"{self.path}:{lineno}: not a frame record ({exc})"
                    ) from exc
                if self.realtime and previous is not None:
                    time.sleep(max(0.0, t - previous))
                previous = t
                yield t, frame


# --------------------------------------------------------------------------- #
# Hardware: DFRobot SEN0628 on UART5
# --------------------------------------------------------------------------- #

class Sen0628Source:
    """The SEN0628 over UART5 (/dev/ttyAMA5, 115200 baud).

    Not wired up yet, on purpose. The handover from the MagicMirror project
    describes the DFRobot library (DFRobot_matrixLidar_uart, set_Ranging_Mode(8),
    get_all_data() returning 64 values as low/high byte pairs) and notes that it
    opens /dev/ttyAMA0 hard-coded — so it needs a small subclass for
    /dev/ttyAMA5. The exact constructor and attribute names have to be checked
    against the library source on the device before this adapter is written;
    guessing them here would produce code that looks finished and is not.

    Step 3 of the integration plan: verify the API on the Pi, implement
    `frames()` using `decode_le16`, record a few hundred real swipes with
    `--record`, and turn the best ones into tests/fixtures.
    """

    name = "sen0628"

    def __init__(self, port: str = "/dev/ttyAMA5", baud: int = 115200):
        self.port = port
        self.baud = baud

    def frames(self) -> Iterator[tuple[float, Frame]]:
        raise FrameSourceError(
            "SEN0628 adapter not implemented yet — verify the DFRobot library "
            "API on the Pi first (integration plan, step 3). Use --source "
            "synthetic or --source replay until then.")
