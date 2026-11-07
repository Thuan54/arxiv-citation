import json
import threading

from config import JSONL_FILE
from logger import get_logger

_lock = threading.Lock()
_write_lock = threading.Lock()
_papers = {}
logger = get_logger(__name__)


def initialize_paper(arxiv_id, versions):
    with _lock:
        _papers[arxiv_id] = {
            "expected": set(versions),
            "versions": {
                version: {
                    "version": version,
                    "metadata": None,
                    "metadata_status": None,
                    "latex_source_path": None,
                    # Filled by the post-crawl Semantic Scholar enrichment.
                    "s2_references": None,
                }
                for version in versions
            },
            "metadata_done": set(),
            "source_done": set(),
        }


def _finish_if_ready(arxiv_id):
    paper = _papers.get(arxiv_id)

    if paper is None:
        return None

    expected = paper["expected"]

    if paper["metadata_done"] != expected or paper["source_done"] != expected:
        return None

    ordered = sorted(expected, key=lambda v: int(v[1:]))

    # revised_dates of a version = dates of v2..that version. Only now are
    # all versions known, so this is the earliest it can be computed.
    revised_dates = []

    for version in ordered:
        item = paper["versions"][version]
        metadata = item["metadata"]

        if metadata is None:
            continue

        if version != "v1" and metadata.get("version_date"):
            revised_dates.append(metadata["version_date"])

        metadata["revised_dates"] = list(revised_dates)

    record = {
        "arxiv_id": arxiv_id,
        "versions": [paper["versions"][version] for version in ordered],
    }

    del _papers[arxiv_id]
    return record


def _write_record(record):
    JSONL_FILE.parent.mkdir(parents=True, exist_ok=True)

    with _write_lock:
        with JSONL_FILE.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    logger.info(
        "Paper %s (%s versions) written to %s",
        record["arxiv_id"],
        len(record["versions"]),
        JSONL_FILE,
    )


def set_metadata(arxiv_id, version, metadata, status=None):
    with _lock:
        paper = _papers.get(arxiv_id)

        if paper is None or version not in paper["expected"]:
            return

        paper["versions"][version]["metadata"] = metadata
        paper["versions"][version]["metadata_status"] = status
        paper["metadata_done"].add(version)

        record = _finish_if_ready(arxiv_id)

    if record is not None:
        _write_record(record)


def set_source(
    arxiv_id,
    version,
    path,
    status=None,
):
    with _lock:
        paper = _papers.get(arxiv_id)

        if paper is None or version not in paper["expected"]:
            return

        item = paper["versions"][version]
        item["latex_source_path"] = path
        item["source_status"] = status
        paper["source_done"].add(version)

        record = _finish_if_ready(arxiv_id)

    if record is not None:
        _write_record(record)
