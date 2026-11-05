"""Crawl reference metadata from the Semantic Scholar Graph API.

Papers are resolved in batches through POST /graph/v1/paper/batch (up to 500
ids per request) instead of one GET per paper, so a crawl needs a small
fraction of the requests and stays inside the rate limits.

Semantic Scholar keeps a single reference list per arXiv paper and does not
distinguish versions, so one lookup per paper is attached to every version
when the record is written.
"""
import queue
import threading
import time

import requests

from config import (
    S2_API_KEY, S2_MAX_ATTEMPTS, S2_MIN_INTERVAL, S2_TIMEOUT, USER_AGENT,
)
from downloader import STOP
from helper import format_error, log_failure
from logger import get_logger
from metric import metrics
from monitor import monitor
from state import set_references


logger = get_logger(__name__)

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

# The API accepts up to 500 ids per call but returns at most ~10 MB, and
# reference lists are large, so stay well below that. A batch that is
# rejected or times out is split in half automatically (see _resolve).
S2_BATCH_SIZE = 50
# Send a partly filled batch if no new paper arrives for this long.
S2_BATCH_WAIT = 10.0
BATCH_TIMEOUT = max(S2_TIMEOUT, 60)

# Flat field names only: nested forms such as "authors.name" are rejected.
# "paperId" of every reference is always returned by the API.
FIELDS = ",".join([
    "venue",
    "references.title",
    "references.authors",
    "references.year",
    "references.publicationDate",
    "references.venue",
    "references.externalIds",
])

# A list this long may have been cut off by the API.
POSSIBLY_TRUNCATED_AT = 1000
GENERIC_VENUES = {"", "arxiv", "arxiv.org"}


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


class SplitBatch(Exception):
    """The batch as a whole failed in a way that smaller batches may avoid."""


def _retry_after_seconds(response, attempt):
    try:
        value = float(response.headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        value = 0.0

    if value <= 0:
        value = min(60.0, 5.0 * 2 ** (attempt - 1))

    return min(value, 120.0)


def post_batch(session, arxiv_ids):
    """POST one batch; return the list of results aligned with arxiv_ids.

    Entries are null for papers Semantic Scholar does not know. Raises
    SplitBatch when halving the batch could help, any other exception when
    it could not (bad key, rate limit that never cleared, ...).
    """
    headers = {"User-Agent": USER_AGENT}

    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    body = {"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]}
    can_split = len(arxiv_ids) > 1
    last_error = None

    for attempt in range(1, S2_MAX_ATTEMPTS + 1):
        s2_limiter.wait()

        try:
            with metrics.measure("s2_request"):
                response = session.post(
                    BATCH_URL,
                    params={"fields": FIELDS},   # query string, not the body
                    json=body,
                    headers=headers,
                    timeout=BATCH_TIMEOUT,
                )
        except requests.exceptions.Timeout as exc:
            last_error = exc
            if can_split:
                raise SplitBatch(str(exc)) from exc
            s2_limiter.penalize(attempt * 2)
            continue
        except requests.exceptions.RequestException as exc:
            last_error = exc
            s2_limiter.penalize(attempt * 2)
            continue

        status = response.status_code

        if status == 429:
            metrics.inc("s2_rate_limited")
            last_error = requests.exceptions.HTTPError(
                "HTTP 429", response=response
            )
            s2_limiter.penalize(_retry_after_seconds(response, attempt))
            continue

        if status >= 500:
            last_error = requests.exceptions.HTTPError(
                f"HTTP {status}", response=response
            )
            s2_limiter.penalize(attempt * 2)
            continue

        if status in (400, 413):
            error = requests.exceptions.HTTPError(
                f"HTTP {status}", response=response
            )
            if can_split:
                raise SplitBatch(f"HTTP {status}") from error
            raise error

        # 401/403 (bad key), 404 and other 4xx: smaller batches won't help.
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as exc:
            last_error = exc
            s2_limiter.penalize(attempt * 2)
            continue

        if not isinstance(data, list) or len(data) != len(arxiv_ids):
            error = RuntimeError("Unexpected batch response shape")
            if can_split:
                raise SplitBatch(str(error)) from error
            raise error

        return data

    error = RuntimeError(
        f"Semantic Scholar request failed after {S2_MAX_ATTEMPTS} "
        f"attempts: {format_error(last_error)}"
    )
    status = getattr(getattr(last_error, "response", None), "status_code", None)

    # Still rate limited: splitting would only send even more requests.
    if can_split and status != 429:
        raise SplitBatch(str(error)) from last_error

    raise error from last_error


def normalize_reference(item):
    """Flatten one reference to the fields the `references` table needs."""
    if not isinstance(item, dict):
        return None

    cited = item.get("citedPaper") or item
    external = cited.get("externalIds") or {}

    return {
        "s2_id": cited.get("paperId"),
        # Kept exactly as returned, so old IDs like hep-th/9901001 survive.
        "arxiv_id": external.get("ArXiv"),
        "title": cited.get("title"),
        "authors": [
            author["name"]
            for author in (cited.get("authors") or [])
            if isinstance(author, dict) and author.get("name")
        ],
        "year": cited.get("year"),
        "submission_date": cited.get("publicationDate"),
        "venue": cited.get("venue") or None,
        "external_ids": external,
    }


def clean_venue(value):
    if not isinstance(value, str):
        return None

    value = " ".join(value.split())
    return None if value.lower() in GENERIC_VENUES else value


def _parse_item(item):
    """Turn one batch entry into (status, references, venue, error)."""
    if item is None:
        return ("not_found", None, None, None)

    if not isinstance(item, dict):
        return ("failed", None, None, RuntimeError("Malformed batch entry"))

    raw = item.get("references") or []
    references = [ref for ref in map(normalize_reference, raw) if ref]

    metrics.inc("s2_papers_succeeded")
    metrics.inc("s2_references_total", len(references))
    metrics.inc(
        "s2_references_with_arxiv_id",
        sum(1 for ref in references if ref["arxiv_id"]),
    )

    if len(raw) >= POSSIBLY_TRUNCATED_AT:
        metrics.inc("s2_reference_lists_maybe_truncated")

    return ("ok", references, clean_venue(item.get("venue")), None)


def _resolve(session, arxiv_ids, results):
    try:
        data = post_batch(session, arxiv_ids)

    except SplitBatch:
        # Too big, too slow, or one bad id spoiling the call: halve it.
        metrics.inc("s2_batch_splits")
        middle = len(arxiv_ids) // 2
        _resolve(session, arxiv_ids[:middle], results)
        _resolve(session, arxiv_ids[middle:], results)
        return

    except Exception as exc:
        for arxiv_id in arxiv_ids:
            results[arxiv_id] = ("failed", None, None, exc)
        return

    metrics.inc("s2_batches")

    for arxiv_id, item in zip(arxiv_ids, data):
        results[arxiv_id] = _parse_item(item)


def crawl_papers(session, arxiv_ids):
    """Look up many papers. Returns {arxiv_id: (status, refs, venue, error)}.

    status is "ok", "not_found" (Semantic Scholar has no record) or
    "failed" (request problem; `error` holds the exception).
    """
    results = {}
    _resolve(session, list(arxiv_ids), results)
    return results


def process_batch(session, tasks):
    arxiv_ids = [task["arxiv_id"] for task in tasks]

    monitor.state(
        "s2-worker",
        "requesting",
        task=f"{arxiv_ids[0]} (+{len(arxiv_ids) - 1})",
        batch_size=len(arxiv_ids),
    )

    try:
        results = crawl_papers(session, arxiv_ids)
    except Exception as exc:
        results = {
            arxiv_id: ("failed", None, None, exc) for arxiv_id in arxiv_ids
        }

    for arxiv_id in arxiv_ids:
        status, references, venue, error = results.get(
            arxiv_id,
            ("failed", None, None, RuntimeError("No result returned")),
        )

        if status == "not_found":
            metrics.inc("s2_papers_not_found")
            log_failure(
                arxiv_id,
                "Semantic Scholar references",
                "Paper not found on Semantic Scholar",
            )
        elif status == "failed":
            metrics.inc("s2_papers_failed")
            log_failure(
                arxiv_id,
                "Semantic Scholar references",
                format_error(error),
                exception=error,
            )

        # Always report back, even on failure: state.py will not write a
        # paper's record until its reference crawl has finished.
        set_references(arxiv_id, references, venue)

    monitor.completed("s2_batches")


def references_worker(task_queue):
    """Collect one task per paper, crawl them in batches."""
    session = requests.Session()
    pending = []
    logger.info("Semantic Scholar worker started")

    try:
        while True:
            monitor.state(
                "s2-worker",
                "waiting_for_task",
                pending_batch_size=len(pending),
            )

            try:
                task = task_queue.get(
                    timeout=S2_BATCH_WAIT if pending else None
                )
            except queue.Empty:
                # Quiet for a while: don't hold finished papers back.
                process_batch(session, pending)
                pending = []
                continue

            try:
                if task is not STOP:
                    pending.append(task)

                if pending and (task is STOP or len(pending) >= S2_BATCH_SIZE):
                    process_batch(session, pending)
                    pending = []

                if task is STOP:
                    monitor.state("s2-worker", "stopped")
                    return

            finally:
                task_queue.task_done()

    finally:
        session.close()
        logger.info("Semantic Scholar worker stopped")
