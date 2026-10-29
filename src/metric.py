
import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from config import METRICS_FILE
from logger import get_logger

logger = get_logger(__name__)

class Metrics:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.started = time.perf_counter()
        self.lock = threading.Lock()
        self.counts = defaultdict(int)
        self.timings = defaultdict(
            lambda: {
                "calls": 0,
                "total": 0.0,
                "max": 0.0,
            }
        )

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self.lock:
                stats = self.timings[name]
                stats["calls"] += 1
                stats["total"] += elapsed
                stats["max"] = max(stats["max"], elapsed)

    def inc(self, name, amount=1):
        with self.lock:
            self.counts[name] += amount

    def report(self):
        elapsed = time.perf_counter() - self.started

        with self.lock:
            counts = dict(self.counts)
            timings = {
                name: {
                    "calls": s["calls"],
                    "total_seconds": round(s["total"], 3),
                    "average_seconds": round(
                        s["total"] / s["calls"], 3
                    ),
                    "max_seconds": round(s["max"], 3),
                }
                for name, s in self.timings.items()
                if s["calls"]
            }

        result = {
            "elapsed_seconds": round(elapsed, 3),
            "counters": counts,
            "timings": timings,
        }

        self.output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        temporary = self.output_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.output_path)

        logger.info("Metrics report:\n%s", json.dumps(result, indent=2))
        return result


metrics = Metrics(METRICS_FILE)