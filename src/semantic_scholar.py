import json
import threading
import time
from queue import Empty

import requests

from config import (
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_API_TIMEOUT,
    SEMANTIC_SCHOLAR_BATCH_WAIT,
    SEMANTIC_SCHOLAR_REFERENCE_PAGE_SIZE,
    SEMANTIC_SCHOLAR_REQUEST_INTERVAL,
    BATCH_SIZE,
)
from downloader import STOP
from helper import log_failure
from logger import get_logger
from metric import metrics
from monitor import monitor
from state import set_references


PAPER_REFERENCES_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/"
    "ARXIV:{arxiv_id}/references"
)
PAPER_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
REFERENCE_FIELDS = (
    "references.paperId,references.externalIds,references.title,"
    "references.authors,references.year,references.venue,"
    "references.publicationDate,referenceCount"
)
logger = get_logger(__name__)


class SemanticScholarReferenceWorker:
    def __init__(self):
        self._request_lock = threading.Lock()
        self._last_request_started = None

    def _wait_for_request_slot(self):
        now = time.monotonic()
        if self._last_request_started is not None:
            delay = SEMANTIC_SCHOLAR_REQUEST_INTERVAL - (
                now - self._last_request_started
            )
            if delay > 0:
                time.sleep(delay)

    def _request_headers(self):
        if SEMANTIC_SCHOLAR_API_KEY:
            return {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
        return {}

    def fetch_references_batch(self, arxiv_ids, session):
        with self._request_lock:
            self._wait_for_request_slot()
            response = session.post(
                PAPER_BATCH_URL,
                params={"fields": REFERENCE_FIELDS},
                json={
                    "ids": [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]
                },
                timeout=SEMANTIC_SCHOLAR_API_TIMEOUT,
                headers=self._request_headers(),
            )
            self._last_request_started = time.monotonic()
            response.raise_for_status()
            payload = response.json()

        if not isinstance(payload, list) or len(payload) != len(arxiv_ids):
            raise ValueError(
                "Semantic Scholar returned an invalid batch response "
                f"for {len(arxiv_ids)} papers"
            )

        results = {}
        incomplete_ids = []
        for arxiv_id, paper in zip(arxiv_ids, payload):
            if paper is None:
                raise ValueError(
                    f"Semantic Scholar did not find paper {arxiv_id}"
                )
            if not isinstance(paper, dict):
                raise ValueError(
                    f"Semantic Scholar returned invalid paper data for {arxiv_id}"
                )

            references = paper.get("references")
            reference_count = paper.get("referenceCount")
            if not isinstance(references, list):
                if reference_count == 0:
                    references = []
                else:
                    incomplete_ids.append(arxiv_id)
                    continue

            results[arxiv_id] = [
                json.dumps(
                    {"citedPaper": reference},
                    ensure_ascii=False,
                )
                for reference in references
                if isinstance(reference, dict)
            ]
            if (
                isinstance(reference_count, int)
                and len(references) < reference_count
            ):
                incomplete_ids.append(arxiv_id)

        for arxiv_id in incomplete_ids:
            results[arxiv_id] = self.fetch_references(arxiv_id, session)

        return results

    def _get_page(self, session, arxiv_id, offset):
        with self._request_lock:
            self._wait_for_request_slot()

            response = session.get(
                PAPER_REFERENCES_URL.format(arxiv_id=arxiv_id),
                params={
                    "offset": offset,
                    "limit": SEMANTIC_SCHOLAR_REFERENCE_PAGE_SIZE,
                    "fields": (
                        "contexts,intents,isInfluential,"
                        "citedPaper.paperId,citedPaper.externalIds,"
                        "citedPaper.title,citedPaper.authors,"
                        "citedPaper.year,citedPaper.venue,"
                        "citedPaper.publicationDate"
                    ),
                },
                timeout=SEMANTIC_SCHOLAR_API_TIMEOUT,
                headers=self._request_headers(),
            )
            self._last_request_started = time.monotonic()
            response.raise_for_status()
            payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError(
                f"Semantic Scholar returned invalid response for {arxiv_id}"
            )
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError(
                f"Semantic Scholar returned invalid references for {arxiv_id}"
            )
        return data, payload.get("next")

    def fetch_references(self, arxiv_id, session):
        references = []
        offset = 0

        while True:
            page, next_offset = self._get_page(
                session,
                arxiv_id,
                offset,
            )
            for item in page:
                cited_paper = item.get("citedPaper")
                if cited_paper is None:
                    continue
                references.append(json.dumps(item, ensure_ascii=False))

            if next_offset is None:
                break
            if not isinstance(next_offset, int) or next_offset <= offset:
                raise ValueError(
                    "Semantic Scholar returned an invalid pagination offset "
                    f"for {arxiv_id}: {next_offset!r}"
                )
            offset = next_offset

        return references


def semantic_scholar_reference_worker(task_queue):
    logger.info("Semantic Scholar reference worker started")
    client = SemanticScholarReferenceWorker()
    session = requests.Session()

    try:
        while True:
            first_task = task_queue.get()
            if first_task is STOP:
                task_queue.task_done()
                return

            tasks = [first_task]
            stop_after_batch = False
            try:
                while len(tasks) < BATCH_SIZE:
                    try:
                        task = task_queue.get(
                            timeout=SEMANTIC_SCHOLAR_BATCH_WAIT
                        )
                    except Empty:
                        break
                    if task is STOP:
                        task_queue.task_done()
                        stop_after_batch = True
                        break
                    tasks.append(task)

                arxiv_ids = [task["arxiv_id"] for task in tasks]
                for arxiv_id in arxiv_ids:
                    monitor.state(
                        "semantic-scholar-worker",
                        "fetching_references",
                        task=arxiv_id,
                    )

                try:
                    references_by_id = client.fetch_references_batch(
                        arxiv_ids,
                        session,
                    )
                except (requests.RequestException, ValueError) as exc:
                    for arxiv_id in arxiv_ids:
                        _mark_reference_failure(arxiv_id, exc)
                else:
                    for arxiv_id in arxiv_ids:
                        references = references_by_id[arxiv_id]
                        set_references(arxiv_id, references, "success")
                        metrics.inc("semantic_scholar_references_fetched")
                        metrics.inc(
                            "semantic_scholar_reference_records",
                            len(references),
                        )
                        logger.info(
                            "Fetched %s Semantic Scholar references for %s",
                            len(references),
                            arxiv_id,
                        )
            finally:
                for task in tasks:
                    task_queue.task_done()
                    monitor.state(
                        "semantic-scholar-worker",
                        "waiting_for_paper",
                    )
                    monitor.completed("semantic_scholar_papers")

            if stop_after_batch:
                return
    finally:
        session.close()
        logger.info("Semantic Scholar reference worker stopped")


def _mark_reference_failure(arxiv_id, exc):
    log_failure(
        arxiv_id,
        "Semantic Scholar references",
        str(exc),
        exception=(
            exc if isinstance(exc, requests.RequestException) else None
        ),
        details={"error_type": type(exc).__name__},
    )
    metrics.inc("semantic_scholar_reference_failures")
    set_references(arxiv_id, [], "failed")
    logger.warning(
        "Semantic Scholar reference lookup failed for %s: %s",
        arxiv_id,
        exc,
    )
