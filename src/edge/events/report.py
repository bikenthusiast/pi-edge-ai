"""Read the event log from the command line.

    python -m edge.events.report --hours 24
    python -m edge.events.report --label person --limit 20 --raw

The summary view is what the agent's tool will call later, so it is worth
having a human-readable version now: if the numbers look wrong here, they will
look wrong to the model too.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from edge.events.store import EventStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise or list events from the log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", type=Path, default=Path("events.db"))
    parser.add_argument("--hours", type=float, default=24.0,
                        help="look back this many hours")
    parser.add_argument("--label", help="restrict to one label")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--raw", action="store_true",
                        help="list individual events instead of the summary")
    parser.add_argument("--models", action="store_true",
                        help="show which model produced which rows")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.db.is_file():
        print(f"No event log at {args.db}. Has the pipeline run yet?",
              file=sys.stderr)
        return 1

    store = EventStore(args.db)
    since = datetime.now(UTC) - timedelta(hours=args.hours)

    try:
        if args.models:
            rows = store.models_used(since=since)
            if not rows:
                print("No events in that window.")
                return 0
            print(f"{'model':<40} {'count':>6}  last seen")
            print("-" * 66)
            for row in rows:
                last = datetime.fromisoformat(str(row["last_seen"]))
                print(f"{str(row['model'])[:40]:<40} {row['n']:>6}  "
                      f"{last:%m-%d %H:%M}")
            if len(rows) > 1:
                print("\nMore than one model in this window — scores are not "
                      "comparable across them.")
            return 0

        if args.raw:
            events = store.query(since=since, label=args.label,
                                 min_score=args.min_score, limit=args.limit)
            if not events:
                print("No events in that window.")
                return 0
            print(f"{'time (UTC)':<21} {'label':<24} {'score':>6} {'dur':>7}  model")
            print("-" * 88)
            for event in events:
                duration = (f"{event.duration_ms / 1000:.1f}s"
                            if event.duration_ms else "-")
                model = (event.model or "-")[:26]
                print(f"{event.ts:%Y-%m-%d %H:%M:%S}   {event.label[:24]:<24} "
                      f"{event.score * 100:5.1f}% {duration:>7}  {model}")
            return 0

        rows = store.summarise(since=since)
        if not rows:
            print(f"No events in the last {args.hours:.0f} h.")
            return 0

        print(f"Last {args.hours:.0f} h — {sum(r['n'] for r in rows)} events\n")
        print(f"{'label':<28} {'count':>6} {'avg':>7}  first seen")
        print("-" * 62)
        for row in rows:
            first = datetime.fromisoformat(row["first_seen"])
            print(f"{str(row['label'])[:28]:<28} {row['n']:>6} "
                  f"{row['avg_score'] * 100:6.1f}%  {first:%m-%d %H:%M}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())