
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
        self._last_source_start = None

        self.api_interval = API_INTERVAL
        self.source_burst_size = SOURCE_BURST_SIZE
        self.source_burst_pause = SOURCE_BURST_PAUSE

        # Shared lock protects both request types and burst state.
        self._lock = threading.Lock()

        self._last_api_start = None
        self._source_requests_in_burst = 0

    @contextmanager
    def request(self, kind):
        if kind not in ("api", "source"):
            raise ValueError(f"Unknown request type: {kind}")

        with self._lock:
            if kind == "api":
                # Preserve the existing API pacing.
                now = time.monotonic()

                if self._last_api_start is not None:
                    delay = self.api_interval - (
                        now - self._last_api_start
                    )

                    if delay > 0:
                        time.sleep(delay)

                self._last_api_start = time.monotonic()

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
            yield


arxiv_limiter = ArxivRateLimiter()
