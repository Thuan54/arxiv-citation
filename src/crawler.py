import json
import threading
from queue import Queue
from dotenv import load_dotenv


from config import (
    NUM_DOWNLOADERS, NUM_EXTRACTORS, QUEUE_SIZE,
    DOWNLOAD_QUEUE_SIZE, EXTRACTION_QUEUE_SIZE,
)
from discovery import iter_version_tasks
from downloader import ( 
    downloader_worker,
    STOP,
)
from extractor import extraction_worker
from metadata import metadata_worker
from metric import metrics
from logger import get_logger, setup_logging
from monitor import monitor


load_dotenv()
logger = get_logger(__name__)
health_logger = get_logger("pipeline.health")
HEALTH_REPORT_INTERVAL = 5


def forward_download_tasks(
    dispatch_queue,
    download_queue,
    worker_count,
):
    """Move discovered downloads to the bounded worker queue."""
    stopped_workers = 0

    while stopped_workers < worker_count:
        monitor.state(
            "download-dispatcher",
            "waiting_for_download_dispatch_task",
        )
        task = dispatch_queue.get()

        try:
            if task is STOP:
                monitor.state(
                    "download-dispatcher",
                    "waiting_to_enqueue_download_stop",
                    stopped_workers=stopped_workers + 1,
                )
                download_queue.put(STOP)
                stopped_workers += 1
                continue

            monitor.state(
                "download-dispatcher",
                "waiting_to_enqueue_download",
                task=task["version_id"],
            )
            download_queue.put(task)
        finally:
            dispatch_queue.task_done()

    monitor.state("download-dispatcher", "stopped")


def health_report(stop_event, queues):
    while True:
        health_logger.info(
            "PIPELINE_HEALTH %s",
            json.dumps(monitor.snapshot(queues), ensure_ascii=False),
        )

        if stop_event.wait(HEALTH_REPORT_INTERVAL):
            return


def main():
    setup_logging(only_logger="pipeline")
    metadata_queue = Queue(maxsize=QUEUE_SIZE)
    download_dispatch_queue = Queue(maxsize=DOWNLOAD_QUEUE_SIZE)
    download_queue = Queue(maxsize=DOWNLOAD_QUEUE_SIZE)
    extraction_queue = Queue(maxsize=EXTRACTION_QUEUE_SIZE)

    errors = []
    errors_lock = threading.Lock()

    def record_error(exc):
        with errors_lock:
            errors.append(exc)

        logger.error(
            "Fatal pipeline error: %s: %s",
            type(exc).__name__,
            str(exc).splitlines()[0] or "No error details",
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def produce():
        monitor.state("discovery", "starting")
        try:
            for task in iter_version_tasks():
                monitor.state(
                    "discovery",
                    "waiting_to_enqueue_metadata",
                    task=task["version_id"],
                )
                metadata_queue.put(task)

                monitor.state(
                    "discovery",
                    "waiting_to_enqueue_download_dispatch",
                    task=task["version_id"],
                )
                download_dispatch_queue.put(task)

        except Exception as exc:
            record_error(exc)

        finally:
            # Metadata can finish independently of source extraction.
            monitor.state("discovery", "waiting_to_enqueue_metadata_stop")
            metadata_queue.put(STOP)

            # Stop download producers. Extraction remains active
            # until every downloader has finished.
            monitor.state("discovery", "waiting_to_enqueue_download_stops")
            for _ in range(NUM_DOWNLOADERS):
                download_dispatch_queue.put(STOP)

    metadata_thread = threading.Thread(
        target=metadata_worker,
        args=(metadata_queue,),
        name="metadata-worker",
    )

    downloader_threads = [
        threading.Thread(
            target=downloader_worker,
            args=(worker_id, download_queue, extraction_queue),
            name=f"downloader-{worker_id}",
        )
        for worker_id in range(NUM_DOWNLOADERS)
    ]

    extraction_threads = [
        threading.Thread(
            target=extraction_worker,
            args=(worker_id, extraction_queue),
            name=f"extractor-{worker_id}",
        )
        for worker_id in range(NUM_EXTRACTORS)
    ]

    # Start consumers before the producer to avoid queue deadlocks.
    metadata_thread.start()

    for worker in extraction_threads:
        worker.start()

    for worker in downloader_threads:
        worker.start()

    producer = threading.Thread(
        target=produce,
        name="discovery",
    )
    download_dispatcher = threading.Thread(
        target=forward_download_tasks,
        args=(
            download_dispatch_queue,
            download_queue,
            NUM_DOWNLOADERS,
        ),
        name="download-dispatcher",
    )
    monitor_stop = threading.Event()
    health_thread = threading.Thread(
        target=health_report,
        args=(
            monitor_stop,
            {
                "metadata": metadata_queue,
                "download_dispatch": download_dispatch_queue,
                "download": download_queue,
                "extraction": extraction_queue,
            },
        ),
        name="health-monitor",
        daemon=True,
    )

    if monitor.enabled:
        health_thread.start()
    download_dispatcher.start()
    producer.start()

    producer.join()
    download_dispatch_queue.join()
    download_dispatcher.join()

    # All discovered download tasks and stop signals are queued.
    monitor.state("crawler", "waiting_for_downloads")
    download_queue.join()

    for worker in downloader_threads:
        worker.join()

    # No downloader can enqueue more extraction work now.
    monitor.state("crawler", "waiting_to_enqueue_extraction_stops")
    for _ in range(NUM_EXTRACTORS):
        extraction_queue.put(STOP)

    monitor.state("crawler", "waiting_for_extractions")
    extraction_queue.join()

    for worker in extraction_threads:
        worker.join()

    monitor.state("crawler", "waiting_for_metadata")
    metadata_queue.join()
    metadata_thread.join()

    if monitor.enabled:
        monitor_stop.set()
        health_thread.join()

    metrics.report()
    logger.info(
        "Crawl complete. Run `python src/retry_s2_references.py` to enrich "
        "Semantic Scholar data and export tables."
    )

    if errors:
        raise RuntimeError(
            f"Pipeline encountered {len(errors)} fatal error(s)"
        ) from errors[0]


if __name__ == "__main__":
    main()