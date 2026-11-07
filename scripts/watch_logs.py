#!/usr/bin/env python
"""Compactly monitor arbitrary HMQR tqdm log files without scrolling history."""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD_SEPARATOR = re.compile(r"[\r\n]+")


def last_record(path: Path, max_bytes: int = 256 * 1024) -> str:
    if not path.exists():
        return "(log not created yet)"
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(0, size - max_bytes))
        text = stream.read().decode("utf-8", errors="replace")
    records = [part.strip() for part in RECORD_SEPARATOR.split(text) if part.strip()]
    return records[-1] if records else "(waiting for output)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="log paths relative to the project root")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    paths = [Path(value) if Path(value).is_absolute() else ROOT / value for value in args.logs]
    try:
        while True:
            width = os.get_terminal_size().columns if sys.stdout.isatty() else 120
            rows = [f"HMQR budgeted runs  {time.strftime('%Y-%m-%d %H:%M:%S')}"]
            for path in paths:
                latest = last_record(path).replace("\x1b", "?")
                if len(latest) > width:
                    latest = latest[: max(1, width - 1)] + "…"
                rows.extend(("", path.name, latest))
            sys.stdout.write("\x1b[H\x1b[J" + "\n".join(rows) + "\n")
            sys.stdout.flush()
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
