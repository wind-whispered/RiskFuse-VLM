#!/usr/bin/env python
"""Follow four-GPU HMQR logs without printing every tqdm refresh as a new line."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD_SEPARATOR = re.compile(r"[\r\n]+")


def _last_record(path: Path, max_bytes: int = 256 * 1024) -> tuple[int, str]:
    """Return the file size and its last CR/LF-delimited display record."""
    if not path.exists():
        return 0, "(log not created yet)"
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(0, size - max_bytes))
        text = stream.read().decode("utf-8", errors="replace")
    records = [part.strip() for part in RECORD_SEPARATOR.split(text) if part.strip()]
    return size, records[-1] if records else "(waiting for output)"


@dataclass
class LogFollower:
    path: Path
    offset: int = 0
    pending: str = ""
    latest: str = "(waiting for output)"

    @classmethod
    def from_path(cls, path: Path) -> "LogFollower":
        offset, latest = _last_record(path)
        return cls(path=path, offset=offset, latest=latest)

    def poll(self) -> bool:
        """Read only newly appended bytes and retain the newest display record."""
        if not self.path.exists():
            return False
        size = self.path.stat().st_size
        if size < self.offset:  # The launcher truncated or replaced the log.
            self.offset = 0
            self.pending = ""
        if size == self.offset:
            return False

        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        text = self.pending + chunk.decode("utf-8", errors="replace")
        parts = RECORD_SEPARATOR.split(text)
        self.pending = parts.pop() if text and text[-1] not in "\r\n" else ""
        records = [part.strip() for part in parts if part.strip()]

        # tqdm commonly leaves the newest refresh unterminated. It is still the
        # value users want to see, so display it while retaining it as pending.
        candidate = self.pending.strip() or (records[-1] if records else "")
        if candidate and candidate != self.latest:
            self.latest = candidate
            return True
        return False


def _load_status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        # status.json is atomically replaced, but tolerate transient filesystem
        # issues instead of terminating a long-running monitor.
        return None


def _job_key(job: dict) -> tuple:
    return job.get("gpu"), job.get("backbone"), job.get("log")


def _state(job: dict) -> str:
    rc = job.get("returncode")
    return "RUNNING" if rc is None else ("DONE" if rc == 0 else f"FAILED({rc})")


def _shorten(text: str, width: int) -> str:
    text = text.replace("\x1b", "?")
    if width <= 0 or len(text) <= width:
        return text
    return text[: max(1, width - 1)] + "\u2026"


def _render(status: dict | None, followers: dict[tuple, LogFollower]) -> None:
    width = os.get_terminal_size().columns if sys.stdout.isatty() else 120
    rows = [
        f"HMQR live progress  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * min(width, 120),
    ]
    if status is None:
        rows.append("Waiting for multi-GPU status file ...")
    else:
        rows.append(
            f"phase={status.get('phase')}  launcher_updated={status.get('updated_at')}"
        )
        for job in status.get("jobs", []):
            follower = followers[_job_key(job)]
            label = (
                f"GPU {job.get('gpu')}  {str(job.get('backbone')):20.20s} "
                f"pid={job.get('pid')}  {_state(job)}"
            )
            rows.extend(("", label, _shorten(follower.latest, width)))
    # Repaint the same terminal rows. Unlike print-per-poll, tqdm history does
    # not accumulate and the cursor remains at the bottom of the dashboard.
    sys.stdout.write("\x1b[H\x1b[J" + "\n".join(rows) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", default="results/logs/multigpu/status.json")
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="seconds between incremental reads (default: 0.5)",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--exit-when-done", action="store_true",
        help="exit when the currently displayed phase completes",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    status_path = Path(args.status)
    if not status_path.is_absolute():
        status_path = ROOT / status_path

    followers: dict[tuple, LogFollower] = {}
    last_signature = None
    first_render = True
    try:
        while True:
            status = _load_status(status_path)
            jobs = status.get("jobs", []) if status else []
            active_keys = {_job_key(job) for job in jobs}
            followers = {key: value for key, value in followers.items() if key in active_keys}
            for job in jobs:
                key = _job_key(job)
                if key not in followers:
                    followers[key] = LogFollower.from_path(Path(job["log"]))

            changed = False
            for follower in followers.values():
                changed = follower.poll() or changed
            signature = (
                status.get("phase") if status else None,
                tuple((_job_key(job), job.get("returncode")) for job in jobs),
            )
            if first_render or changed or signature != last_signature:
                _render(status, followers)
                first_render = False
                last_signature = signature

            done = bool(jobs) and all(job.get("returncode") is not None for job in jobs)
            if args.once or (done and args.exit_when_done):
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
