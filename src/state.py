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
                    "raw_references": [],
                    "source_status": None,
                    "reference_status": None,
                }
                for version in versions
            },
            "metadata_done": set(),
            "source_done": set(),
            "references_done": set(),
        }


def _finish_if_ready(arxiv_id):
    paper = _papers.get(arxiv_id)

    if paper is None:
        return None

    expected = paper["expected"]

    if (
        paper["metadata_done"] != expected
        or paper["source_done"] != expected
        or paper["references_done"] != expected
    ):
        return None

    record = {
        "arxiv_id": arxiv_id,
        "versions": [
            paper["versions"][version]
            for version in sorted(
                expected,
                key=lambda v: int(v[1:]),
            )
        ],
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


def set_references(arxiv_id, references, status):
    with _lock:
        paper = _papers.get(arxiv_id)
        if paper is None:
            return

        for version in paper["expected"]:
            item = paper["versions"][version]
            item["reference_status"] = status
            if references:
                item["raw_references"] = references
                item["reference_source"] = "semantic_scholar"
            paper["references_done"].add(version)

        record = _finish_if_ready(arxiv_id)

    if record is not None:
        _write_record(record)
