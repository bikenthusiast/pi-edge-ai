"""MQTT publisher tests. No broker, no network: paho is replaced by a fake.

The fake records calls in order, which matters here — the Last Will has to be
registered before connecting, or a crash right after start goes unannounced.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta, timezone

import pytest

from edge.events.publisher import (
    MqttPublisher,
    NullPublisher,
    PublisherError,
    event_payload,
    presence_payload,
    status_payload,
    topic,
)

T0 = datetime(2026, 9, 19, 8, 30, tzinfo=UTC)


class FakeInfo:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def wait_for_publish(self, timeout=None):
        if self.fail:
            raise RuntimeError("not connected")


class FakeClient:
    def __init__(self, fail_wait: bool = False):
        self.calls: list[tuple] = []
        self.fail_wait = fail_wait
        self.on_connect = None

    def __getattr__(self, name):
        # Record any other method call (will_set, connect_async, ...).
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record

    def publish(self, topic, payload, qos=0, retain=False):
        self.calls.append(("publish", (topic, payload),
                           {"qos": qos, "retain": retain}))
        return FakeInfo(fail=self.fail_wait)

    def names(self) -> list[str]:
        return [name for name, *_ in self.calls]

    def published(self) -> list[tuple[str, dict, bool]]:
        return [(args[0], json.loads(args[1]), kwargs["retain"])
                for name, args, kwargs in self.calls if name == "publish"]


class ReasonCode:
    def __init__(self, failure: bool):
        self.is_failure = failure


@pytest.fixture
def fake() -> FakeClient:
    return FakeClient()


@pytest.fixture
def pub(fake) -> MqttPublisher:
    return MqttPublisher("broker", device="pi4", client=fake)


# --- topics and payloads --------------------------------------------------- #

def test_topic_layout():
    assert topic("pi4", "events") == "edge/pi4/events"
    assert topic("pi4", "status", prefix="home") == "home/pi4/status"


@pytest.mark.parametrize("bad", ["", "pi/4", "pi+", "#"])
def test_topic_rejects_wildcards_and_separators(bad):
    with pytest.raises(ValueError, match="invalid device"):
        topic(bad, "events")


def test_event_payload_is_versioned_json():
    data = json.loads(event_payload(
        event_id=7, ts=T0, label="tabby", score=0.912345, source="camera0",
        duration_ms=1234.56, model="v2.tflite"))

    assert data == {
        "v": 1, "id": 7, "ts": "2026-09-19T08:30:00.000+00:00",
        "label": "tabby", "score": 0.9123, "source": "camera0",
        "duration_ms": 1234.6, "model": "v2.tflite",
    }


def test_payload_timestamps_are_normalised_to_utc():
    berlin = timezone(timedelta(hours=2))
    data = json.loads(presence_payload(
        label=None, source="camera0",
        ts=datetime(2026, 9, 19, 10, 30, tzinfo=berlin)))
    assert data["ts"] == "2026-09-19T08:30:00.000+00:00"


def test_presence_payload_marks_absence_explicitly():
    here = json.loads(presence_payload(label="tabby", source="c0", ts=T0))
    gone = json.loads(presence_payload(label=None, source="c0", ts=T0))
    assert (here["present"], here["label"]) == (True, "tabby")
    assert (gone["present"], gone["label"]) == (False, None)


def test_status_payload():
    assert json.loads(status_payload(True)) == {"v": 1, "online": True}


# --- publisher behaviour --------------------------------------------------- #

def test_last_will_is_registered_before_connecting(pub, fake):
    names = fake.names()
    assert names.index("will_set") < names.index("connect_async")
    assert names.index("connect_async") < names.index("loop_start")

    (_, args, kwargs), = [c for c in fake.calls if c[0] == "will_set"]
    assert args[0] == "edge/pi4/status"
    assert json.loads(args[1])["online"] is False
    assert kwargs["retain"] is True


def test_credentials_are_only_set_when_given(fake):
    MqttPublisher("broker", device="pi4", client=fake)
    assert "username_pw_set" not in fake.names()

    other = FakeClient()
    MqttPublisher("broker", device="pi4", username="edge", password="s3cret",
                  client=other)
    assert "username_pw_set" in other.names()


def test_online_is_announced_on_every_successful_connect(pub, fake):
    pub._on_connect(fake, None, None, ReasonCode(failure=False))
    pub._on_connect(fake, None, None, ReasonCode(failure=True))
    pub._on_connect(fake, None, None, ReasonCode(failure=False))

    status = [p for p in fake.published() if p[0] == "edge/pi4/status"]
    assert len(status) == 2
    assert all(payload["online"] and retained for _, payload, retained in status)


def test_events_are_not_retained(pub, fake):
    pub.publish_event(event_id=1, ts=T0, label="tabby", score=0.9,
                      source="camera0", duration_ms=500.0, model="v2.tflite")
    ((topic_, payload, retained),) = fake.published()
    assert topic_ == "edge/pi4/events"
    assert payload["id"] == 1
    assert retained is False


def test_presence_is_retained(pub, fake):
    pub.publish_presence(label="tabby", source="camera0", ts=T0)
    ((topic_, _payload, retained),) = fake.published()
    assert topic_ == "edge/pi4/presence"
    assert retained is True


def test_close_announces_offline_then_disconnects(pub, fake):
    pub.close()
    topic_, payload, retained = fake.published()[-1]
    assert (topic_, payload["online"], retained) == ("edge/pi4/status", False, True)

    names = fake.names()
    assert names.index("disconnect") < names.index("loop_stop")
    assert names.index("publish") < names.index("disconnect")


def test_close_survives_an_unreachable_broker():
    fake = FakeClient(fail_wait=True)
    MqttPublisher("broker", device="pi4", client=fake).close()   # must not raise
    assert "loop_stop" in fake.names()


def test_missing_paho_gives_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "paho", None)
    monkeypatch.setitem(sys.modules, "paho.mqtt", None)
    monkeypatch.setitem(sys.modules, "paho.mqtt.client", None)
    with pytest.raises(PublisherError, match="requirements-pi.txt"):
        MqttPublisher("broker", device="pi4")


def test_null_publisher_accepts_everything():
    null = NullPublisher()
    null.publish_event(event_id=1, label="x")
    null.publish_presence(label=None)
    null.close()
