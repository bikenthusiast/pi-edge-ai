"""Rule-based swipe classifier for 8×8 depth frames.

One gesture per hand movement, not per frame — the same idea as the vision
`Debouncer`, but the decision depends on the *trajectory* of an episode rather
than on a label holding steady, so it is a separate class.

How it works:

1. A zone counts as "hand" if it measured something closer than `near_mm`.
   4000 mm (invalid, firmware V1.3) and 0 never count.
2. A hand episode opens when at least `min_zones` zones are hand, and closes
   after `release_frames` frames without one.
3. On close, the centroid's start-to-end travel decides: the dominant axis must
   move at least `min_travel` zones and clearly more than the other axis, and
   the episode must be neither too short (noise) nor too long (someone
   standing in front of the mirror, which must never trigger anything).
4. After every closed episode, `cooldown_s` suppresses the next one, so the
   return movement of a hand does not fire the opposite gesture.

Orientation: the sensor's row 0 / column 0 depend on how it is mounted in the
frame. `rotate` (0/90/180/270, clockwise) and `mirror` map sensor coordinates
to the viewer's: column 0 = viewer's left, row 0 = up. Calibrate once by doing
a swipe to the right and checking the output.

This is version "rules-v1"; it is stored with every event as the model name,
so a later learned classifier can be told apart in the log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from edge.gesture.frames import COLS, INVALID_MM, ROWS, Frame, validate

MODEL_NAME = "rules-v1"


@dataclass(frozen=True)
class Gesture:
    name: str
    confidence: float     # 0..1, from the travel distance
    duration_ms: float


@dataclass
class _Episode:
    started: float
    last_seen: float
    path: list[tuple[float, float]] = field(default_factory=list)


class SwipeClassifier:
    def __init__(self, *, near_mm: int = 400, min_zones: int = 3,
                 release_frames: int = 2, min_travel: float = 3.0,
                 dominance: float = 1.5, min_duration_s: float = 0.1,
                 max_duration_s: float = 1.5, cooldown_s: float = 1.0,
                 rotate: int = 0, mirror: bool = False):
        if rotate not in (0, 90, 180, 270):
            raise ValueError("rotate must be 0, 90, 180 or 270")
        if not 0 < near_mm < INVALID_MM:
            raise ValueError(f"near_mm must be between 0 and {INVALID_MM}")
        self.near_mm = near_mm
        self.min_zones = min_zones
        self.release_frames = release_frames
        self.min_travel = min_travel
        self.dominance = dominance
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.cooldown_s = cooldown_s
        self.rotate = rotate
        self.mirror = mirror

        self._episode: _Episode | None = None
        self._misses = 0
        self._blocked_until = float("-inf")

    # -- coordinates --------------------------------------------------------

    def _orient(self, row: int, col: int) -> tuple[int, int]:
        """Sensor (row, col) → viewer (row, col)."""
        n = ROWS - 1
        for _ in range(self.rotate // 90):          # clockwise quarter turns
            row, col = col, n - row
        if self.mirror:
            col = n - col
        return row, col

    def _centroid(self, frame: Frame) -> tuple[float, float] | None:
        hits = [divmod(i, COLS) for i, mm in enumerate(frame)
                if 0 < mm < self.near_mm]
        if len(hits) < self.min_zones:
            return None
        oriented = [self._orient(r, c) for r, c in hits]
        x = sum(c for _, c in oriented) / len(oriented)
        y = sum(r for r, _ in oriented) / len(oriented)
        return x, y

    # -- stream -------------------------------------------------------------

    def update(self, frame: Frame, t: float) -> Gesture | None:
        """Feed one frame. Returns a gesture when an episode closes as one."""
        centroid = self._centroid(validate(frame))

        if self._episode is None:
            if centroid is not None and t >= self._blocked_until:
                self._episode = _Episode(started=t, last_seen=t, path=[centroid])
                self._misses = 0
            return None

        if centroid is not None:
            self._episode.path.append(centroid)
            self._episode.last_seen = t
            self._misses = 0
            return None

        self._misses += 1
        if self._misses < self.release_frames:
            return None
        return self._close()

    def flush(self) -> Gesture | None:
        """Evaluate an open episode, e.g. when the source ends."""
        return self._close() if self._episode is not None else None

    def _close(self) -> Gesture | None:
        episode, self._episode = self._episode, None
        assert episode is not None
        self._blocked_until = episode.last_seen + self.cooldown_s
        return self._judge(episode)

    def _judge(self, episode: _Episode) -> Gesture | None:
        duration = episode.last_seen - episode.started
        if not self.min_duration_s <= duration <= self.max_duration_s:
            return None
        (x0, y0), (x1, y1) = episode.path[0], episode.path[-1]
        dx, dy = x1 - x0, y1 - y0
        horizontal = abs(dx) >= abs(dy)
        major, minor = (dx, dy) if horizontal else (dy, dx)
        travel = abs(major)
        if travel < self.min_travel or travel < self.dominance * abs(minor):
            return None
        if horizontal:
            name = "swipe_right" if dx > 0 else "swipe_left"
        else:
            name = "swipe_down" if dy > 0 else "swipe_up"
        max_travel = COLS - 1
        return Gesture(name=name,
                       confidence=round(min(1.0, travel / max_travel), 3),
                       duration_ms=round(duration * 1000, 1))
