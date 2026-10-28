import threading
import time


class PipelineMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._workers = {}
        self._completed = {}
        self._last_completion = self._started

    def state(self, worker, state, task=None, **details):
        now = time.monotonic()

        with self._lock:
            previous = self._workers.get(worker, {})
            same_operation = (
                previous.get("state") == state
                and previous.get("task") == task
            )

            self._workers[worker] = {
                "state": state,
                "task": task,
                "since": previous["since"] if same_operation else now,
                "details": details,
            }

    def completed(self, stage):
        with self._lock:
            self._completed[stage] = self._completed.get(stage, 0) + 1
            self._last_completion = time.monotonic()

    def snapshot(self, queues=None):
        now = time.monotonic()

        with self._lock:
            workers = {
                name: {
                    "state": info["state"],
                    "task": info["task"],
                    "duration_seconds": round(now - info["since"], 1),
                    "details": dict(info["details"]),
                }
                for name, info in self._workers.items()
            }
            completed = dict(self._completed)
            last_completion = self._last_completion

        queue_status = {}
        for name, task_queue in (queues or {}).items():
            queue_status[name] = {
                "size": task_queue.qsize(),
                "capacity": task_queue.maxsize,
                "unfinished": task_queue.unfinished_tasks,
            }

        return {
            "elapsed_seconds": round(now - self._started, 1),
            "seconds_since_completion": round(now - last_completion, 1),
            "completed": completed,
            "workers": workers,
            "queues": queue_status,
        }


monitor = PipelineMonitor()
