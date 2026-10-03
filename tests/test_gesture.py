"""Gesture tests: classifier, frame sources, payload contract, CLI.

No sensor, no broker. The swipes are synthetic; once real recordings exist
(integration step 3) they go to tests/fixtures/ and get a replay test here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from edge.events.publisher import (
    GESTURES,
    MqttPublisher,
    NullPublisher,
    gesture_payload,
)
from edge.events.store import EventStore
from edge.gesture import run
from edge.gesture.classifier import SwipeClassifier
from edge.gesture.frames import (
    INVALID_MM,
    ZONES,
    FrameSourceError,
    Recorder,
    ReplaySource,
    Sen0628Source,
    SyntheticSource,
    decode_le16,
    synthetic_swipe,
)

CONTRACT = Path(__file__).parent / "fixtures" / "contract" / "gesture_v1.json"
EMPTY = (INVALID_MM,) * ZONES
DT = 1 / 15


def feed(clf: SwipeClassifier, frames, t0: float = 0.0):
    """Feed frames at 15 FPS, then two empty frames to release the hand."""
    out, t = [], t0
    for frame in [*frames, EMPTY, EMPTY]:
        t += DT
        if (g := clf.update(frame, t)) is not None:
            out.append(g)
    return out, t


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", GESTURES)
def test_each_swipe_is_recognised_once(name):
    gestures, _ = feed(SwipeClassifier(), synthetic_swipe(name))
    assert [g.name for g in gestures] == [name]
    assert 0.5 < gestures[0].confidence <= 1.0


def test_hand_held_still_is_not_a_gesture():
    # Someone standing in front of the mirror: hand-sized blob, no movement.
    still = [synthetic_swipe("swipe_right")[3]] * 10
    gestures, _ = feed(SwipeClassifier(), still)
    assert gestures == []


def test_hand_held_too_long_is_not_a_gesture():
    slow = [f for f in synthetic_swipe("swipe_right", steps=8) for _ in range(5)]
    gestures, _ = feed(SwipeClassifier(max_duration_s=1.5), slow)
    assert gestures == []


def test_single_noisy_zone_is_ignored():
    noisy = list(EMPTY)
    noisy[10] = 150
    gestures, _ = feed(SwipeClassifier(), [tuple(noisy)] * 5)
    assert gestures == []


def test_far_objects_and_invalid_zones_do_not_count():
    far = tuple([1200] * ZONES)   # a person 1.2 m away fills the field
    gestures, _ = feed(SwipeClassifier(near_mm=400), [far] * 8)
    assert gestures == []


def test_cooldown_swallows_the_return_movement():
    clf = SwipeClassifier(cooldown_s=1.0)
    first, t = feed(clf, synthetic_swipe("swipe_right"))
    # The hand swings back immediately — that must not fire swipe_left.
    back, t = feed(clf, synthetic_swipe("swipe_left"), t0=t)
    assert [g.name for g in first] == ["swipe_right"]
    assert back == []
    # After the cooldown, a new gesture counts again.
    later, _ = feed(clf, synthetic_swipe("swipe_left"), t0=t + 2.0)
    assert [g.name for g in later] == ["swipe_left"]


@pytest.mark.parametrize(("rotate", "mirror", "expected"), [
    (0, False, "swipe_right"),
    (180, False, "swipe_left"),
    (0, True, "swipe_left"),
    (90, False, "swipe_down"),
    (270, False, "swipe_up"),
])
def test_orientation_maps_sensor_to_viewer(rotate, mirror, expected):
    clf = SwipeClassifier(rotate=rotate, mirror=mirror)
    gestures, _ = feed(clf, synthetic_swipe("swipe_right"))
    assert [g.name for g in gestures] == [expected]


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        SwipeClassifier(rotate=45)
    with pytest.raises(ValueError):
        SwipeClassifier(near_mm=INVALID_MM)


def test_frame_of_wrong_size_is_rejected():
    with pytest.raises(ValueError):
        SwipeClassifier().update((100,) * 63, 0.0)


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #

def test_synthetic_source_plays_the_script():
    clf = SwipeClassifier()
    names = [g.name for t, f in SyntheticSource(["swipe_up", "swipe_left"]).frames()
             if (g := clf.update(f, t))]
    assert names == ["swipe_up", "swipe_left"]


def test_synthetic_source_rejects_unknown_gestures():
    with pytest.raises(FrameSourceError):
        SyntheticSource(["wave"])


def test_record_and_replay_roundtrip(tmp_path):
    path = tmp_path / "rec.jsonl"
    recorder = Recorder(path)
    for t, frame in SyntheticSource(["swipe_down"]).frames():
        recorder.write(t, frame)
    recorder.close()

    clf = SwipeClassifier()
    names = [g.name for t, f in ReplaySource(path).frames()
             if (g := clf.update(f, t))]
    assert names == ["swipe_down"]


def test_replay_reports_the_broken_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"t":0,"d":[1,2,3]}\n')
    with pytest.raises(FrameSourceError, match=":1:"):
        list(ReplaySource(path).frames())


def test_decode_le16_low_byte_first():
    raw = [0xA0, 0x0F] * ZONES             # 0x0FA0 = 4000
    assert decode_le16(raw) == (4000,) * ZONES
    with pytest.raises(ValueError):
        decode_le16([0] * 10)


def test_sen0628_adapter_fails_loudly_until_implemented():
    with pytest.raises(FrameSourceError, match="step 3"):
        next(Sen0628Source().frames())


# --------------------------------------------------------------------------- #
# Contract: docs/mqtt.md, topic edge/<device>/gesture, schema v1
# --------------------------------------------------------------------------- #

def test_payload_matches_the_shared_contract_fixture():
    # The same file lives in the MagicMirror repo (tests/fixtures/gesture_v1.json)
    # and is what the bridge is tested against. Change both, or bump "v".
    expected = json.loads(CONTRACT.read_text())
    produced = json.loads(gesture_payload(
        event_id=7, ts=datetime(2026, 10, 3, 8, 30, tzinfo=UTC),
        gesture="swipe_up", confidence=0.857, source="sen0628",
        duration_ms=466.66, model="rules-v1"))
    assert produced == expected


def test_unknown_gesture_cannot_be_published():
    with pytest.raises(ValueError):
        gesture_payload(event_id=1, ts=datetime.now(UTC), gesture="wave",
                        confidence=1.0, source="x", duration_ms=None, model=None)


def test_gesture_is_published_with_qos1_and_not_retained():
    class FakeClient:
        """Records calls; enough of paho's surface for the constructor."""

        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return lambda *a, **kw: self.calls.append((name, a, kw))

    client = FakeClient()
    pub = MqttPublisher("broker", device="mirror", client=client)
    pub.publish_gesture(event_id=1, ts=datetime(2026, 10, 3, tzinfo=UTC),
                        gesture="swipe_left", confidence=0.9, source="sen0628",
                        duration_ms=300.0, model="rules-v1")
    name, args, kwargs = client.calls[-1]
    assert name == "publish"
    assert args[0] == "edge/mirror/gesture"
    assert kwargs == {"qos": 1, "retain": False}


def test_null_publisher_accepts_gestures():
    NullPublisher().publish_gesture(gesture="swipe_up")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_stores_one_row_per_gesture(tmp_path, capsys):
    db = tmp_path / "events.db"
    code = run.main(["--source", "synthetic", "--script", "swipe_up,swipe_down",
                     "--db", str(db), "--quiet"])
    assert code == 0
    store = EventStore(db)
    rows = store.query()
    store.close()
    assert sorted(e.label for e in rows) == ["swipe_down", "swipe_up"]
    assert {e.source for e in rows} == {"synthetic"}


def test_cli_replay_without_file_is_a_user_error(capsys):
    assert run.main(["--source", "replay", "--quiet"]) == 1
    assert "--file" in capsys.readouterr().err
