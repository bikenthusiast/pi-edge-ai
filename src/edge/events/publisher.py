"""MQTT publisher: pushes closed events and live presence to a broker.

SQLite stays the source of truth. MQTT is a best-effort notification channel
for consumers such as the MagicMirror module: if the broker is unreachable, the
pipeline keeps running and every event is still in events.db (see ADR 0003).

Topics, all under ``<prefix>/<device>/``:

    events    one message per closed episode          QoS 1, not retained
    presence  what is in view right now, or nothing   QoS 1, retained
    status    {"online": true|false}, false via LWT   QoS 1, retained

paho-mqtt is imported lazily, like OpenCV and LiteRT elsewhere, so the topic
and payload helpers stay testable on the Mac without it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_PREFIX = "edge"
_FORBIDDEN_IN_TOPIC_LEVEL = set("/+#")


class PublisherError(RuntimeError):
    """Raised for configuration problems the user can act on."""


# --------------------------------------------------------------------------- #
# Pure helpers — no network, no paho
# --------------------------------------------------------------------------- #

def topic(device: str, kind: str, prefix: str = DEFAULT_PREFIX) -> str:
    """Build a topic. Rejects names that would silently turn into wildcards."""
    for name, value in (("device", device), ("prefix", prefix)):
        if not value or _FORBIDDEN_IN_TOPIC_LEVEL & set(value):
            raise ValueError(f"invalid {name} for an MQTT topic: {value!r}")
    return f"{prefix}/{device}/{kind}"


def _iso(moment: datetime) -> str:
    # Same convention as the event store, so a consumer can match an MQTT
    # message to its row by timestamp as well as by id.
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="milliseconds")


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps({"v": SCHEMA_VERSION, **payload}, separators=(",", ":"))


def event_payload(*, event_id: int, ts: datetime, label: str, score: float,
                  source: str, duration_ms: float | None,
                  model: str | None) -> str:
    return _dump({
        "id": event_id,
        "ts": _iso(ts),
        "label": label,
        "score": round(float(score), 4),
        "source": source,
        "duration_ms": None if duration_ms is None else round(duration_ms, 1),
        "model": model,
    })


def presence_payload(*, label: str | None, source: str, ts: datetime) -> str:
    return _dump({
        "ts": _iso(ts),
        "source": source,
        "present": label is not None,
        "label": label,
    })


def status_payload(online: bool) -> str:
    return _dump({"online": online})


# --------------------------------------------------------------------------- #
# Publishers
# --------------------------------------------------------------------------- #

class NullPublisher:
    """Used when no broker is configured. Keeps the pipeline free of `if`s."""

    def publish_event(self, **_fields: Any) -> None:
        pass

    def publish_presence(self, **_fields: Any) -> None:
        pass

    def close(self) -> None:
        pass


class MqttPublisher:
    """Non-blocking publisher. Connects and reconnects on paho's own thread.

    The vision loop must never wait on the network, so nothing here blocks
    except `close()`, which waits briefly for the final offline message.
    """

    def __init__(self, host: str, port: int = 1883, *, device: str,
                 prefix: str = DEFAULT_PREFIX, username: str | None = None,
                 password: str | None = None, max_queued: int = 1000,
                 client: Any = None):
        self.events_topic = topic(device, "events", prefix)
        self.presence_topic = topic(device, "presence", prefix)
        self.status_topic = topic(device, "status", prefix)

        if client is None:
            try:
                import paho.mqtt.client as mqtt
            except ImportError as exc:
                raise PublisherError(
                    "paho-mqtt is not installed. Run "
                    "`.venv/bin/pip install -r requirements-pi.txt` on the Pi, "
                    "or start the pipeline without --mqtt-host."
                ) from exc
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                 client_id=f"pi-edge-ai-{device}")

        self._client = client
        if username:
            client.username_pw_set(username, password)
        # Bounded queue: during a long broker outage paho would otherwise
        # buffer every event in RAM. SQLite has them anyway.
        client.max_queued_messages_set(max_queued)
        # Last Will: the broker announces "offline" if the Pi dies or loses
        # the network, so the mirror can grey out stale presence.
        client.will_set(self.status_topic, status_payload(False),
                        qos=1, retain=True)
        client.on_connect = self._on_connect
        client.connect_async(host, port, keepalive=30)
        client.loop_start()

    def _on_connect(self, client: Any, _userdata: Any, _flags: Any,
                    reason_code: Any, _properties: Any = None) -> None:
        # Runs on every (re)connect, which is exactly when "online" is news.
        if not getattr(reason_code, "is_failure", False):
            client.publish(self.status_topic, status_payload(True),
                           qos=1, retain=True)

    def publish_event(self, **fields: Any) -> None:
        self._client.publish(self.events_topic, event_payload(**fields),
                             qos=1, retain=False)

    def publish_presence(self, **fields: Any) -> None:
        # Retained: a mirror that boots later still learns what is in view.
        self._client.publish(self.presence_topic, presence_payload(**fields),
                             qos=1, retain=True)

    def close(self, timeout: float = 2.0) -> None:
        info = self._client.publish(self.status_topic, status_payload(False),
                                    qos=1, retain=True)
        try:
            info.wait_for_publish(timeout=timeout)
        except (RuntimeError, ValueError):
            pass  # broker unreachable: the Last Will covers this case
        self._client.disconnect()
        self._client.loop_stop()
