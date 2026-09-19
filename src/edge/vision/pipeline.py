"""Live pipeline: camera → classifier → debouncer → event log (→ MQTT).

This is the piece that actually runs on the Pi for hours at a time.

    python -m edge.vision.pipeline --min-score 0.6 --db events.db
    python -m edge.vision.pipeline --mqtt-host localhost   # also publish

MQTT credentials come from MQTT_USERNAME / MQTT_PASSWORD in the environment,
never from the command line, where they would show up in `ps` and shell history.

Design: the loop never writes per frame. Only the debouncer's closed episodes
become rows, which is what keeps the database readable and the SD card alive.
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

import numpy as np
from PIL import Image

from edge.events.publisher import (
    MqttPublisher,
    NullPublisher,
    PublisherError,
)
from edge.events.store import EventStore
from edge.vision.camera import CameraError, CameraStream, Debouncer, Detection
from edge.vision.model import (
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    ModelError,
    TFLiteModel,
    load_labels,
    preprocess,
)


class GracefulExit:
    """Turns SIGINT/SIGTERM into a flag so the loop can flush and close."""

    def __init__(self) -> None:
        self.stop = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, *_args) -> None:
        self.stop = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the live classification pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--db", type=Path, default=Path("events.db"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.60,
                        help="score needed to open an episode")
    parser.add_argument("--exit-score", type=float, default=0.40,
                        help="score below which an episode may close")
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--source", default="camera0",
                        help="stored with each event, for multi-camera setups")
    parser.add_argument("--mqtt-host",
                        help="publish events to this MQTT broker "
                             "(omit to run without MQTT)")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--device", default=socket.gethostname(),
                        help="device name in the MQTT topics: edge/<device>/…")
    parser.add_argument("--max-seconds", type=float,
                        help="stop after N seconds (useful for smoke tests)")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        labels = load_labels(args.labels)
        model = TFLiteModel(args.model, num_threads=args.threads)
    except ModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    debouncer = Debouncer(
        enter_score=args.min_score,
        exit_score=args.exit_score,
        min_frames=args.min_frames,
        cooldown_s=args.cooldown,
    )
    try:
        publisher = (
            MqttPublisher(args.mqtt_host, args.mqtt_port, device=args.device,
                          username=os.environ.get("MQTT_USERNAME"),
                          password=os.environ.get("MQTT_PASSWORD"))
            if args.mqtt_host else NullPublisher()
        )
    except (PublisherError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    store = EventStore(args.db)
    lifecycle = GracefulExit()

    try:
        stream = CameraStream(args.camera)
    except CameraError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        store.close()
        publisher.close()
        return 1

    frames = 0
    started = time.monotonic()
    model_name = Path(args.model).name
    present: str | None = None

    def log_event(detection: Detection) -> None:
        # One timestamp for both sinks, so the MQTT message and the SQLite row
        # describe the same moment and can be matched by id and ts.
        closed_at = datetime.now(UTC)
        event_id = store.record(detection.label, detection.score,
                                source=args.source,
                                duration_ms=detection.duration_ms,
                                model=model_name, ts=closed_at)
        publisher.publish_event(event_id=event_id, ts=closed_at,
                                label=detection.label, score=detection.score,
                                source=args.source,
                                duration_ms=detection.duration_ms,
                                model=model_name)

    def update_presence() -> None:
        # Presence changes when an episode opens or closes — not per frame.
        nonlocal present
        if debouncer.active_label != present:
            present = debouncer.active_label
            publisher.publish_presence(label=present, source=args.source,
                                       ts=datetime.now(UTC))
    if not args.quiet:
        # Say which weights are loaded. A long-running service that does not
        # announce this leaves you guessing after a model swap.
        print(f"Model : {model_name} "
              f"({model.width}x{model.height}, {model.dtype.__name__}, "
              f"{args.threads} threads)")
        print(f"Events: {args.db}   Stop with Ctrl+C.")
        if args.mqtt_host:
            print(f"MQTT  : {args.mqtt_host}:{args.mqtt_port} "
                  f"→ edge/{args.device}/…")

    try:
        while not lifecycle.stop:
            if args.max_seconds and time.monotonic() - started > args.max_seconds:
                break

            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # BGR (OpenCV) → RGB (what the model was trained on). Getting this
            # wrong raises no error, it just degrades accuracy silently.
            rgb = frame[:, :, ::-1]
            image = preprocess(Image.fromarray(rgb), model.width, model.height,
                               fit="crop")
            model.invoke(model.make_tensor(image))
            scores = model.output(0)

            best = int(np.argmax(scores))
            event = debouncer.update(labels[best], float(scores[best]))
            frames += 1
            update_presence()

            if event is not None:
                log_event(event)
                if not args.quiet:
                    print(f"  logged: {event.label} "
                          f"({event.duration_ms / 1000:.1f}s)")

        final = debouncer.flush()
        if final is not None:
            log_event(final)
        update_presence()
    finally:
        stream.release()
        publisher.close()
        elapsed = time.monotonic() - started
        total = len(store.query(limit=10_000))
        store.close()
        if not args.quiet:
            print(f"\n{frames} frames in {elapsed:.1f}s "
                  f"({frames / max(elapsed, 1e-6):.1f} FPS), "
                  f"{total} events in the log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())