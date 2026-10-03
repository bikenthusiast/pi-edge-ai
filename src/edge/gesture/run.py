"""Gesture pipeline: frames → classifier → events.db + MQTT.

    # Walking skeleton, no sensor needed: two synthetic swipes in real time
    python -m edge.gesture.run --source synthetic --script swipe_up,swipe_down \\
        --realtime --mqtt-host localhost

    # Regression run on a recording
    python -m edge.gesture.run --source replay --file tests/fixtures/x.jsonl

    # Later, on the mirror: live sensor, recording every frame for the dataset
    python -m edge.gesture.run --source sen0628 --record swipes.jsonl \\
        --mqtt-host localhost

Same rules as the vision pipeline (ADR 0003): SQLite is the source of truth,
MQTT is best effort, and one timestamp is used for both so a consumer can match
a message to its row by id and ts.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from edge.events.publisher import MqttPublisher, NullPublisher, PublisherError
from edge.events.store import EventStore
from edge.gesture.classifier import MODEL_NAME, SwipeClassifier
from edge.gesture.frames import (
    FrameSource,
    FrameSourceError,
    Recorder,
    ReplaySource,
    Sen0628Source,
    SyntheticSource,
)


class GracefulExit:
    """SIGINT/SIGTERM → flag, so the loop can close the store and send offline.

    Same as in edge.vision.pipeline, duplicated on purpose: importing that
    module would pull in NumPy, Pillow and the camera code, and the gesture
    pipeline needs none of them.
    """

    def __init__(self) -> None:
        self.stop = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, *_args) -> None:
        self.stop = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recognise swipe gestures from 8×8 depth frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", choices=("synthetic", "replay", "sen0628"),
                   default="synthetic")
    p.add_argument("--script", default="swipe_up",
                   help="synthetic: comma-separated gestures to play")
    p.add_argument("--file", type=Path, help="replay: JSONL recording")
    p.add_argument("--port", default="/dev/ttyAMA5", help="sen0628: UART")
    p.add_argument("--realtime", action="store_true",
                   help="synthetic/replay: pace frames like a real sensor")
    p.add_argument("--record", type=Path,
                   help="write every frame to this JSONL file")
    p.add_argument("--db", type=Path, default=Path("events.db"))
    p.add_argument("--near-mm", type=int, default=400,
                   help="zones closer than this count as hand")
    p.add_argument("--cooldown", type=float, default=1.0)
    p.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                   help="sensor mounting, clockwise")
    p.add_argument("--mirror", action="store_true",
                   help="flip left/right after rotating")
    p.add_argument("--mqtt-host", help="omit to run without MQTT")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--device", default=socket.gethostname(),
                   help="device name in the MQTT topics: edge/<device>/…")
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def make_source(args: argparse.Namespace) -> FrameSource:
    if args.source == "synthetic":
        script = [g.strip() for g in args.script.split(",") if g.strip()]
        return SyntheticSource(script, realtime=args.realtime)
    if args.source == "replay":
        if args.file is None:
            raise FrameSourceError("--source replay needs --file")
        return ReplaySource(args.file, realtime=args.realtime)
    return Sen0628Source(args.port)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        source = make_source(args)
        classifier = SwipeClassifier(near_mm=args.near_mm,
                                     cooldown_s=args.cooldown,
                                     rotate=args.rotate, mirror=args.mirror)
        publisher = (
            MqttPublisher(args.mqtt_host, args.mqtt_port, device=args.device,
                          username=os.environ.get("MQTT_USERNAME"),
                          password=os.environ.get("MQTT_PASSWORD"))
            if args.mqtt_host else NullPublisher()
        )
    except (FrameSourceError, PublisherError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    store = EventStore(args.db)
    recorder = Recorder(args.record) if args.record else None
    lifecycle = GracefulExit()
    started = time.monotonic()
    count = 0

    def emit(gesture) -> None:
        nonlocal count
        closed_at = datetime.now(UTC)
        event_id = store.record(gesture.name, gesture.confidence,
                                source=source.name,
                                duration_ms=gesture.duration_ms,
                                model=MODEL_NAME, ts=closed_at)
        publisher.publish_gesture(event_id=event_id, ts=closed_at,
                                  gesture=gesture.name,
                                  confidence=gesture.confidence,
                                  source=source.name,
                                  duration_ms=gesture.duration_ms,
                                  model=MODEL_NAME)
        count += 1
        if not args.quiet:
            print(f"{closed_at:%H:%M:%S}  #{event_id:<5} {gesture.name:<12} "
                  f"conf={gesture.confidence:.2f}  {gesture.duration_ms:.0f} ms")

    if not args.quiet:
        print(f"Source: {source.name}   Model: {MODEL_NAME}   Events: {args.db}")
        if args.mqtt_host:
            print(f"MQTT  : {args.mqtt_host}:{args.mqtt_port} "
                  f"→ edge/{args.device}/gesture")

    exit_code = 0
    try:
        for t, frame in source.frames():
            if lifecycle.stop:
                break
            if args.max_seconds and time.monotonic() - started > args.max_seconds:
                break
            if recorder:
                recorder.write(t, frame)
            gesture = classifier.update(frame, t)
            if gesture:
                emit(gesture)
        gesture = classifier.flush()
        if gesture:
            emit(gesture)
    except FrameSourceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if recorder:
            recorder.close()
        store.close()
        publisher.close()

    if not args.quiet:
        print(f"{count} gesture(s).")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
