
import threading
import time
from contextlib import contextmanager

from config import (
    API_INTERVAL,
    SOURCE_BURST_SIZE,
    SOURCE_BURST_PAUSE,
)


class ArxivRateLimiter:
    """Coordinate arXiv requests within one process."""

    def __init__(self):
        self.api_interval = API_INTERVAL
        self.source_burst_size = SOURCE_BURST_SIZE
        self.source_burst_pause = SOURCE_BURST_PAUSE

        # Shared lock protects both request types and burst state.
        self._lock = threading.Lock()

        self._last_request_finished = None
        self._last_source_start = None
        self._source_requests_in_burst = 0

    @contextmanager
    def request(self, kind):
        if kind not in ("api", "source"):
            raise ValueError(f"Unknown request type: {kind}")

        with self._lock:
            if kind == "api":
                now = time.monotonic()
                if self._last_request_finished is not None:
                    # API calls wait a full interval after any prior
                    # arXiv request, including source downloads.
                    delay = self.api_interval - (
                        now - self._last_request_finished
                    )
                    if delay > 0:
                        time.sleep(delay)

            else:
                now = time.monotonic()

                # Reset the burst after 1 second without a source request.
                if (
                    self._last_source_start is not None
                    and now - self._last_source_start >= 1.0
                ):
                    self._source_requests_in_burst = 0

                # If four requests have already started, pause.
                if self._source_requests_in_burst >= self.source_burst_size:
                    time.sleep(self.source_burst_pause)
                    self._source_requests_in_burst = 0

                self._source_requests_in_burst += 1
                self._last_source_start = time.monotonic()

            # Hold the lock throughout the HTTP request.
            # This prevents concurrent requests from bypassing
            # the shared pacing policy.
            try:
                yield
            finally:
                self._last_request_finished = time.monotonic()


arxiv_limiter = ArxivRateLimiter()



class S2RateLimiter:
    """Space request starts at least `interval` seconds apart."""

    def __init__(self, interval):
        self.interval = interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_allowed)
            self._next_allowed = start + self.interval

        delay = start - now
        if delay > 0:
            time.sleep(delay)

    def penalize(self, seconds):
        """Push the next request back (after a 429 or a server error)."""
        with self._lock:
            self._next_allowed = max(
                self._next_allowed, time.monotonic() + seconds
            )


s2_limiter = S2RateLimiter(S2_MIN_INTERVAL)
