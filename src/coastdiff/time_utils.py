"""Timestamp parsing: a single timestamp, or a continuous range with an interval."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_TS_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
               "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
_INTERVAL_RE = re.compile(r"^(\d+)\s*([hHdD])$")


def parse_timestamp(s: str) -> datetime:
    s = s.strip()
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp {s!r}; try e.g. '2025-07-15 00:00'")


def parse_interval(s: str) -> timedelta:
    """'6h' -> 6 hours, '1d' -> 1 day. ERA5 is hourly, so anything
    finer than 1h isn't meaningful."""
    m = _INTERVAL_RE.match(s.strip())
    if not m:
        raise ValueError(f"Could not parse interval {s!r}; expected e.g. '6h' or '1d'")
    n, unit = int(m.group(1)), m.group(2).lower()
    return timedelta(hours=n) if unit == "h" else timedelta(days=n)


def generate_range(start: str, end: str, interval: str) -> list[datetime]:
    t0, t1, step = parse_timestamp(start), parse_timestamp(end), parse_interval(interval)
    if t1 < t0:
        raise ValueError(f"--end ({end}) is before --start ({start})")
    out, t = [], t0
    while t <= t1:
        out.append(t)
        t += step
    return out
