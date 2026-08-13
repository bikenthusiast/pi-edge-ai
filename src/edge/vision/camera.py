"""Camera capture and detection debouncing.

Two things live here, and only one of them needs hardware:

* `CameraStream` — a threaded grabber. Needs OpenCV and a real device.
* `Debouncer`    — decides which detections become events. Pure logic, fully
  testable on the Mac, and the reason `events.db` stays a sane size.

OpenCV is imported lazily so this module can be imported (and `Debouncer`
tested) on a machine without a camera stack.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Self


class CameraError(RuntimeError):
    """Raised when a capture device cannot be opened or read."""


class CameraStream:
    """Reads frames on a background thread so inference never blocks capture.

    Without this, camera latency and inference time add up. With it, the
    inference loop always works on the most recent frame and the two costs
    overlap instead of stacking.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 fps: int = 30, fourcc: str = "MJPG"):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise CameraError(
                "OpenCV is not available. On the Pi install it via apt "
                "(python3-opencv) and create the venv with "
                "--system-site-packages."
            ) from exc

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        # MJPG matters: with raw YUYV a 640x480 webcam is limited by USB
        # bandwidth to roughly 5-10 FPS.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Queue depth 1: always hand out the freshest frame, never a stale one.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise CameraError(
                f"Could not open camera {index}. Check `v4l2-ctl --list-devices` "
                "and that your user is in the 'video' group."
            )

        self._frame = None
        self._running = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        """Latest frame, or None if the first one has not arrived yet."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait_for_frame(self, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.01)
        raise CameraError(f"No frame within {timeout:.0f}s — is the device busy?")

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


@dataclass
class Detection:
    """A debounced detection, ready to be written to the event log."""

    label: str
    score: float          # peak score observed during the episode
    duration_ms: float


class Debouncer:
    """Turns a per-frame classification stream into discrete events.

    A 15 FPS pipeline that logged every frame would write over a million rows a
    day. This emits one event per *episode* instead: a label has to hold for
    `min_frames` consecutive frames to open an episode, and the episode closes
    once the label has been absent for `cooldown_s`.

    Hysteresis is deliberate: `enter_score` is higher than `exit_score`, so a
    detection hovering around the threshold produces one stable episode rather
    than a burst of on/off events.
    """

    def __init__(self, enter_score: float = 0.60, exit_score: float = 0.40,
                 min_frames: int = 3, cooldown_s: float = 2.0):
        if exit_score > enter_score:
            raise ValueError("exit_score must not exceed enter_score")

        self.enter_score = enter_score
        self.exit_score = exit_score
        self.min_frames = min_frames
        self.cooldown_s = cooldown_s

        self._candidate: str | None = None
        self._streak = 0
        self._streak_started_at = 0.0
        self._peak_score = 0.0
        self._active: str | None = None
        self._started_at = 0.0
        self._last_seen = 0.0

    def update(self, label: str | None, score: float,
               now: float | None = None) -> Detection | None:
        """Feed one frame's top prediction. Returns an event when one closes."""
        now = time.monotonic() if now is None else now

        # Does this frame keep the active episode alive?
        if self._active is not None:
            if label == self._active and score >= self.exit_score:
                self._last_seen = now
                self._peak_score = max(self._peak_score, score)
            elif now - self._last_seen >= self.cooldown_s:
                return self._close(now)
            return None

        # No episode running: count consecutive frames above the entry bar.
        if label is not None and score >= self.enter_score:
            if label == self._candidate:
                self._streak += 1
            else:
                self._streak = 1
                self._streak_started_at = now
            self._candidate = label
            if self._streak >= self.min_frames:
                self._active = label
                # Backdate to the first frame of the streak: the object was
                # already there, we just needed min_frames to be sure. Using
                # `now` here would silently shorten every episode by the
                # confirmation window.
                self._started_at = self._streak_started_at
                self._last_seen = now
                self._peak_score = score
                self._streak = 0
        else:
            self._candidate = None
            self._streak = 0
        return None

    def flush(self, now: float | None = None) -> Detection | None:
        """Close any open episode — call this when the loop shuts down."""
        if self._active is None:
            return None
        return self._close(time.monotonic() if now is None else now)

    def _close(self, now: float) -> Detection:
        detection = Detection(
            label=self._active,
            # Peak, not last or mean: the log should answer "how sure were we
            # at best", and a mean would be dragged down by the fade-out.
            score=self._peak_score,
            duration_ms=(self._last_seen - self._started_at) * 1000,
        )
        self._active = None
        self._candidate = None
        self._streak = 0
        self._peak_score = 0.0
        return detection

    @property
    def active_label(self) -> str | None:
        return self._active
