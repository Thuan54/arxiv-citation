from datetime import datetime, timezone
import json
import threading
import time

from logger import get_logger


logger = get_logger("pipeline.worker")


class PipelineMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._workers = {}
        self._completed = {}
        self._last_completion = self._started

    @staticmethod
    def _timestamp():
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def state(self, worker, state, task=None, **details):
        now = time.monotonic()
        timestamp = self._timestamp()

        with self._lock:
            previous = self._workers.get(worker, {})
            same_operation = (
                previous.get("state") == state
                and previous.get("task") == task
            )
            previous_state = previous.get("state")
            previous_task = previous.get("task")
            previous_state_duration = (
                round(now - previous["since"], 1)
                if previous
                else None
            )

            if task is not None and task != previous_task:
                task_started_at = timestamp
                last_progress_at = timestamp
                task_finished_at = None
            elif task is not None:
                task_started_at = previous.get("task_started_at")
                last_progress_at = timestamp
                task_finished_at = previous.get("task_finished_at")
            elif previous_task is not None:
                task_started_at = previous.get("task_started_at")
                last_progress_at = timestamp
                task_finished_at = timestamp
            else:
                task_started_at = previous.get("task_started_at")
                last_progress_at = previous.get("last_progress_at")
                task_finished_at = previous.get("task_finished_at")

            self._workers[worker] = {
                "state": state,
                "task": task,
                "since": previous["since"] if same_operation else now,
                "task_started_at": task_started_at,
                "last_progress_at": last_progress_at,
                "task_finished_at": task_finished_at,
                "details": details,
            }

        if not same_operation:
            logger.info(
                "WORKER_STATE worker=%s state=%s task=%s "
                "previous_state=%s previous_state_duration_seconds=%s "
                "details=%s",
                worker,
                state,
                task if task is not None else previous_task,
                previous_state,
                previous_state_duration,
                json.dumps(details, ensure_ascii=False, sort_keys=True),
            )

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
                    "task_started_at": info["task_started_at"],
                    "last_progress_at": info["last_progress_at"],
                    "task_finished_at": info["task_finished_at"],
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
