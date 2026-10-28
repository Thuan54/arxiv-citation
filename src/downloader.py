import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import requests

from config import DOWNLOAD_TIMEOUT, RAW_DIR, USER_AGENT
from helper import log_failure, format_error
from monitor import monitor
from rate_limiter import arxiv_limiter
from metric import metrics
from state import set_source

logger = logging.getLogger(__name__)
STOP = None

MAX_DOWNLOAD_ATTEMPTS = 3
CHUNK_SIZE = 64 * 1024


def download_source(
    session,
    url,
    headers,
    timeout,
    directory,
    arxiv_id=None,
    worker_name=None,
    monitor_task=None,
):
    """Download and validate a gzip source archive with retries."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    last_error = None

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        tmp_path = None

        try:
            if worker_name:
                monitor.state(
                    worker_name,
                    "waiting_for_rate_limiter",
                    task=monitor_task or arxiv_id,
                    attempt=attempt,
                )
            with arxiv_limiter.request("source"):
                if worker_name:
                    monitor.state(
                        worker_name,
                        "http_download",
                        task=monitor_task or arxiv_id,
                        attempt=attempt,
                    )
                with metrics.measure("source_download"):
                    with session.get(
                        url,
                        headers=headers,
                        stream=True,
                        timeout=timeout,
                    ) as response:
                        response.raise_for_status()

                        content_length = response.headers.get(
                            "Content-Length"
                        )
                        expected_size = (
                            int(content_length)
                            if content_length is not None
                            else None
                        )

                        downloaded_size = 0

                        with tempfile.NamedTemporaryFile(
                            mode="wb",
                            suffix=".gz",
                            dir=directory,
                            delete=False,
                        ) as temp:
                            tmp_path = Path(temp.name)

                            for chunk in response.iter_content(
                                chunk_size=CHUNK_SIZE
                            ):
                                if not chunk:
                                    continue

                                downloaded_size += len(chunk)
                                temp.write(chunk)

                        if (
                            expected_size is not None
                            and downloaded_size != expected_size
                        ):
                            raise IOError(
                                f"Incomplete download: received "
                                f"{downloaded_size} of "
                                f"{expected_size} bytes"
                            )

            # A successful download must produce a real temporary file.
            if tmp_path is None or not tmp_path.is_file():
                raise IOError("Download produced no temporary file")

            # Validate gzip magic bytes.
            with tmp_path.open("rb") as source:
                if source.read(2) != b"\x1f\x8b":
                    raise ValueError(
                        f"Response is not gzip data: {url}"
                    )

            metrics.inc("source_downloads_succeeded")
            return tmp_path

        
        except (
            requests.exceptions.RequestException,
            OSError,
            ValueError,
        ) as exc:
            last_error = exc

            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

            retryable = attempt < MAX_DOWNLOAD_ATTEMPTS

            log_failure(
                arxiv_id or url,
                "Source download attempt",
                format_error(exc),
                exception=exc,
                attempt=attempt,
                url=url,
                details={
                    "max_attempts": MAX_DOWNLOAD_ATTEMPTS,
                    "retryable": retryable,
                    "timeout_seconds": timeout,
                },
            )

            if retryable:
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s",
                    attempt,
                    MAX_DOWNLOAD_ATTEMPTS,
                    url,
                    format_error(exc),
                )
                if worker_name:
                    monitor.state(
                        worker_name,
                        "retry_backoff",
                        task=monitor_task or arxiv_id,
                        attempt=attempt,
                        delay_seconds=attempt * 2,
                    )
                time.sleep(attempt * 2)
              

    raise RuntimeError(
        f"Download failed after {MAX_DOWNLOAD_ATTEMPTS} attempts: "
        f"{format_error(last_error)}"
    ) from last_error


def download_task(worker_id, task, extraction_queue, session):
    """Download one source and enqueue it for extraction."""
    arxiv_id = task["arxiv_id"]
    version = task["version"]
    version_id = task["version_id"]

    destination = RAW_DIR / arxiv_id / version
    destination.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://arxiv.org/src/{version_id}"

    # Reuse completed extractions without downloading again.
    if (
        destination.is_dir()
        and (destination / ".complete").is_file()
    ):
        monitor.state(
            f"downloader-{worker_id}",
            "waiting_to_enqueue_extraction",
            task=version_id,
            reused=True,
        )
        extraction_queue.put({
            "task": task,
            "tmp_path": None,
            "reused": True,
        })
        metrics.inc("source_versions_reused")
        return

    tmp_path = None

    try:
        tmp_path = download_source(
            session=session,
            url=url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "identity",
            },
            timeout=DOWNLOAD_TIMEOUT,
            directory=destination.parent,
            arxiv_id=arxiv_id,
            worker_name=f"downloader-{worker_id}",
            monitor_task=version_id,
        )

        # Transfer ownership of the temporary file to the
        # extraction worker.
        monitor.state(
            f"downloader-{worker_id}",
            "waiting_to_enqueue_extraction",
            task=version_id,
            reused=False,
        )
        extraction_queue.put({
            "task": task,
            "tmp_path": tmp_path,
            "reused": False,
        })
        tmp_path = None

        metrics.inc("source_downloads_queued_for_extraction")

    except Exception as exc:
        log_failure(
            arxiv_id,
            f"Source download {version}",
            format_error(exc),
            exception=exc,
            url=url,
            details={
                "version": version,
                "worker_id": worker_id,
                "stage": "download",
            },
        )
        metrics.inc("source_versions_failed")

        set_source(
            arxiv_id,
            version,
            None,
            "failed",
            [],
        )

        logger.error(
            "Download failed | paper=%s | version=%s | %s",
            arxiv_id,
            version,
            format_error(exc),
        )

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def downloader_worker(worker_id, download_queue, extraction_queue):
    """Download worker: download and enqueue, without extracting."""
    session = requests.Session()
    logger.info("Downloader %s started", worker_id)

    try:
        while True:
            monitor.state(
                f"downloader-{worker_id}",
                "waiting_for_download_task",
            )
            task = download_queue.get()

            try:
                if task is STOP:
                    monitor.state(
                        f"downloader-{worker_id}",
                        "stopped",
                    )
                    return

                monitor.state(
                    f"downloader-{worker_id}",
                    "processing_download",
                    task=task["version_id"],
                )
                download_task(
                    worker_id,
                    task,
                    extraction_queue,
                    session,
                )
                monitor.completed("download_tasks")

            finally:
                download_queue.task_done()

    finally:
        session.close()
        logger.info("Downloader %s stopped", worker_id)
