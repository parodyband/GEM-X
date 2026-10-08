"""Smooth display of an irregular pose stream.

Poses arrive once per inference (~25 Hz, with jitter). Drawing each one as
it lands makes the character step. Instead each sample is placed on the local
clock using the smallest transit time seen so far, and the display runs a
little behind real time so there is always a newer sample to interpolate
towards. The delay adapts to about 1.5 sample intervals.
"""

from __future__ import annotations

import time
from collections import deque
from statistics import median


class Playback:
    MIN_DELAY = 0.025
    MAX_DELAY = 0.25

    def __init__(self, maxlen: int = 64):
        self.samples: deque[tuple[float, object]] = deque(maxlen=maxlen)  # (source time, payload)
        self._intervals: deque[float] = deque(maxlen=32)
        self.offset: float | None = None  # local clock - source clock
        self.last_arrival = 0.0

    def clear(self) -> None:
        self.samples.clear()
        self._intervals.clear()
        self.offset = None

    def add(self, t: float, payload, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        if self.samples and t <= self.samples[-1][0]:
            self.clear()  # stream restarted or went backwards
        if self.samples:
            self._intervals.append(t - self.samples[-1][0])
        transit = now - t
        if self.offset is None or transit < self.offset:
            self.offset = transit
        else:
            self.offset += 0.01 * (transit - self.offset)  # follow slow clock drift
        self.samples.append((t, payload))
        self.last_arrival = now

    @property
    def delay(self) -> float:
        if not self._intervals:
            return self.MIN_DELAY
        return min(max(1.5 * median(self._intervals), self.MIN_DELAY), self.MAX_DELAY)

    def stale(self, now: float | None = None, seconds: float = 0.5) -> bool:
        now = time.perf_counter() if now is None else now
        return not self.samples or now - self.last_arrival > seconds

    def at(self, now: float | None = None):
        """(payload_a, payload_b, alpha) to show now, or None when empty."""
        if not self.samples:
            return None
        now = time.perf_counter() if now is None else now
        target = now - self.offset - self.delay
        s = self.samples
        if target <= s[0][0]:
            return s[0][1], s[0][1], 0.0
        if target >= s[-1][0]:
            return s[-1][1], s[-1][1], 0.0
        for i in range(len(s) - 1, 0, -1):
            t0, p0 = s[i - 1]
            t1, p1 = s[i]
            if t0 <= target <= t1:
                return p0, p1, (target - t0) / (t1 - t0)
        return s[-1][1], s[-1][1], 0.0
