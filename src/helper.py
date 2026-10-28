import json
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import FAILURES_FILE, ROOT

_failure_lock = threading.Lock()


def relative_source_path(destination):
    """Return a project-relative path ending in a slash."""
    return (
        Path(destination)
        .resolve()
        .relative_to(ROOT.resolve())
        .as_posix()
        + "/"
    )


def clean_arxiv_id(value):
    return re.sub(r"v\d+$", "", value.strip())


def parse_versioned_id(entry_id):
    """Return (base arXiv ID, version label) from an Atom entry ID."""
    match = re.search(r"/abs/(.+?)(?:v(\d+))?$", entry_id.strip())

    if not match:
        return None

    base_id = match.group(1)
    version = f"v{match.group(2) or '1'}"

    return base_id, version


def get_processed_ids(path):
    """Load completed paper IDs from the output JSONL file."""
    processed = set()

    if not path.exists():
        return processed

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                arxiv_id = record.get("arxiv_id")

                if arxiv_id:
                    processed.add(arxiv_id)

            except (json.JSONDecodeError, AttributeError):
                continue

    return processed


def log_failure(
    arxiv_id,
    step,
    message,
    *,
    exception=None,
    attempt=None,
    url=None,
    details=None,
):
    FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arxiv_id": arxiv_id,
        "failed_step": step,
        "error_message": str(message),
    }

    if exception is not None:
        record["exception_type"] = type(exception).__name__

        if isinstance(exception, requests.exceptions.HTTPError):
            response = exception.response
            if response is not None:
                record["http_status"] = response.status_code
                record["response_url"] = response.url

        if exception.__traceback__ is not None:
            record["traceback"] = "".join(
                traceback.format_exception(
                    type(exception),
                    exception,
                    exception.__traceback__,
                )
            )

    if attempt is not None:
        record["attempt"] = attempt

    if url is not None:
        record["url"] = url

    if details:
        record["details"] = details

    with _failure_lock:
        with FAILURES_FILE.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def format_error(exc):
    """Return a short, readable description of a crawler error."""
    if isinstance(exc, requests.exceptions.Timeout):
        return "Request timed out"

    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        if response is not None:
            return f"HTTP {response.status_code}"
        return "HTTP request failed"

    if isinstance(exc, requests.exceptions.ConnectionError):
        return "Connection failed"

    if isinstance(exc, requests.exceptions.RequestException):
        return "Request failed"

    if isinstance(exc, OSError):
        return f"File I/O error: {exc}"

    message = str(exc).strip().splitlines()[0]
    if len(message) > 180:
        message = message[:177] + "..."

    return message or "No error details available"
