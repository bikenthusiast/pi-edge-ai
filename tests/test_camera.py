"""Debouncer tests run anywhere; camera tests need a device and are marked.

Time is injected via the `now` parameter rather than slept through, so the
whole file runs in milliseconds and never flakes on a loaded machine.
"""

from __future__ import annotations

import pytest

from edge.vision.camera import CameraStream, Debouncer, Detection


@pytest.fixture
def deb() -> Debouncer:
    return Debouncer(enter_score=0.6, exit_score=0.4, min_frames=3,
                     cooldown_s=2.0)


def test_single_frame_does_not_open_an_episode(deb):
    assert deb.update("person", 0.9, now=0.0) is None
    assert deb.active_label is None


def test_episode_opens_after_min_frames(deb):
    for i in range(3):
        assert deb.update("person", 0.9, now=i * 0.1) is None
    assert deb.active_label == "person"


def test_streak_resets_when_the_label_changes(deb):
    deb.update("person", 0.9, now=0.0)
    deb.update("person", 0.9, now=0.1)
    deb.update("cat", 0.9, now=0.2)      # breaks the run
    deb.update("person", 0.9, now=0.3)
    assert deb.active_label is None


def test_low_scores_never_open_an_episode(deb):
    for i in range(10):
        deb.update("person", 0.55, now=i * 0.1)   # below enter_score
    assert deb.active_label is None


def test_hysteresis_keeps_a_wobbling_detection_as_one_episode(deb):
    """0.45 is below enter_score but above exit_score: the episode survives."""
    for i in range(3):
        deb.update("person", 0.9, now=i * 0.1)
    assert deb.active_label == "person"

    for i in range(20):
        assert deb.update("person", 0.45, now=0.3 + i * 0.1) is None
    assert deb.active_label == "person"


def test_episode_closes_after_the_cooldown(deb):
    for i in range(3):
        deb.update("person", 0.9, now=i * 0.1)

    assert deb.update(None, 0.0, now=1.0) is None      # inside cooldown
    event = deb.update(None, 0.0, now=2.3)             # past cooldown

    assert isinstance(event, Detection)
    assert event.label == "person"
    assert event.duration_ms == pytest.approx(200.0, abs=1.0)


def test_one_episode_yields_exactly_one_event(deb):
    events = []
    for i in range(3):
        events.append(deb.update("person", 0.9, now=i * 0.1))
    for i in range(30):
        events.append(deb.update("person", 0.9, now=0.3 + i * 0.1))
    events.append(deb.update(None, 0.0, now=10.0))

    assert len([e for e in events if e is not None]) == 1


def test_a_new_episode_can_start_after_one_closes(deb):
    for i in range(3):
        deb.update("person", 0.9, now=i * 0.1)
    assert deb.update(None, 0.0, now=5.0) is not None

    for i in range(3):
        deb.update("cat", 0.9, now=6.0 + i * 0.1)
    assert deb.active_label == "cat"


def test_flush_closes_an_open_episode(deb):
    for i in range(3):
        deb.update("person", 0.9, now=i * 0.1)

    event = deb.flush(now=1.0)
    assert event is not None and event.label == "person"
    assert deb.flush(now=2.0) is None      # nothing left to close


def test_exit_score_above_enter_score_is_rejected():
    with pytest.raises(ValueError, match="exit_score"):
        Debouncer(enter_score=0.4, exit_score=0.9)


# --- hardware -------------------------------------------------------------- #

@pytest.mark.hardware
def test_camera_delivers_a_frame():
    with CameraStream(0) as stream:
        frame = stream.wait_for_frame(timeout=5.0)
        assert frame.ndim == 3
        assert frame.shape[2] == 3


@pytest.mark.hardware
def test_camera_frames_advance():
    """A frozen stream returns identical buffers — this catches that."""
    import numpy as np

    with CameraStream(0) as stream:
        first = stream.wait_for_frame()
        import time
        time.sleep(0.5)
        second = stream.wait_for_frame()
        assert not np.array_equal(first, second)


def test_duration_includes_the_confirmation_window(deb):
    """The episode is backdated to the first frame of the streak, not to the
    frame that confirmed it — otherwise every episode loses min_frames of
    duration and short events report 0 ms."""
    for i in range(3):
        deb.update("person", 0.9, now=i * 0.1)       # streak starts at 0.0
    for i in range(5):
        deb.update("person", 0.9, now=0.3 + i * 0.1)  # last seen at 0.7

    event = deb.flush(now=1.0)
    assert event.duration_ms == pytest.approx(700.0, abs=1.0)


def test_event_reports_the_peak_score_not_the_threshold(deb):
    for i in range(3):
        deb.update("person", 0.70, now=i * 0.1)
    deb.update("person", 0.94, now=0.3)      # peak
    deb.update("person", 0.45, now=0.4)      # fading, still above exit_score

    event = deb.flush(now=1.0)
    assert event.score == pytest.approx(0.94)
