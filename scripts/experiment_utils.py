from __future__ import annotations

import csv
import os
import platform
import resource
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = usage.ru_maxrss

    if platform.system() == "Darwin":
        return float(max_rss / (1024 * 1024))

    return float(max_rss / 1024)


def runtime_context():
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pid": os.getpid(),
    }


@contextmanager
def timed_block():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def append_row_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    fieldnames = list(row.keys())

    if file_exists:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            original_header = next(reader, None)
            existing_rows = list(reader)

        if original_header:
            fieldnames = list(original_header)
            for key in fieldnames:
                if key not in row:
                    row[key] = None
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

            if fieldnames != original_header:
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(fieldnames)
                    for existing_row in existing_rows:
                        writer.writerow(existing_row + [""] * (len(fieldnames) - len(existing_row)))

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
