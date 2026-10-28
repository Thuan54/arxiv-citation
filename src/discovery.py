
import re
import xml.etree.ElementTree as ET

import requests

from config import (
    API_INTERVAL, API_TIMEOUT, BATCH_SIZE, IDS_FILE, JSONL_FILE, USER_AGENT,
)
from helper import clean_arxiv_id, get_processed_ids, log_failure
from metric import metrics
from rate_limiter import arxiv_limiter
from state import initialize_paper


API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def iter_id_batches():
    """Read IDs incrementally, keeping only one input batch in memory."""
    if not IDS_FILE.exists():
        raise FileNotFoundError(IDS_FILE)

    processed = get_processed_ids(JSONL_FILE)
    seen = set()
    batch = []

    with IDS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            arxiv_id = clean_arxiv_id(line)

            if not arxiv_id:
                continue

            if arxiv_id in processed or arxiv_id in seen:
                continue

            seen.add(arxiv_id)
            batch.append(arxiv_id)

            if len(batch) >= BATCH_SIZE:
                metrics.inc("input_papers", len(batch))
                yield batch
                batch = []

    if batch:
        metrics.inc("input_papers", len(batch))
        yield batch


def discover_batch(batch):
    """Discover versions for one bounded batch of paper IDs."""
    with arxiv_limiter.request("api"):
      with metrics.measure("discovery_api"):
          response = requests.get(
              API_URL,
              params={
                  "id_list": ",".join(batch),
                  "max_results": len(batch),
              },
              headers={"User-Agent": USER_AGENT},
              timeout=API_TIMEOUT,
          )
          response.raise_for_status()

    root = ET.fromstring(response.text)
    latest_versions = {}

    for entry in root.findall("atom:entry", ATOM_NS):
        node = entry.find("atom:id", ATOM_NS)

        if node is None or not node.text:
            continue

        match = re.search(
            r"/abs/(.+?)(?:v(\d+))?$",
            node.text.strip(),
        )

        if not match:
            continue

        arxiv_id = match.group(1)
        version_text = match.group(2)

        if not version_text:
            log_failure(
                arxiv_id,
                "Discovery",
                "API entry has no version suffix",
            )
            metrics.inc("discovery_version_unknown")
            continue

        latest_versions[arxiv_id] = int(version_text)

    for arxiv_id in batch:
        latest = latest_versions.get(arxiv_id)

        if latest is None:
            log_failure(
                arxiv_id,
                "Discovery",
                "No entry returned by arXiv API",
            )
            metrics.inc("discovery_missing")
            continue

        versions = [f"v{i}" for i in range(1, latest + 1)]
        initialize_paper(arxiv_id, versions)

        metrics.inc("papers_discovered")
        metrics.inc("versions_scheduled", len(versions))

        for version in versions:
            yield {
                "arxiv_id": arxiv_id,
                "version": version,
                "version_id": f"{arxiv_id}{version}",
            }


def iter_version_tasks():
    """Yield version tasks without retaining the full task list."""
    for batch in iter_id_batches():
        try:
            yield from discover_batch(batch)

        except Exception as exc:
            metrics.inc("discovery_batches_failed")

            for arxiv_id in batch:
                log_failure(arxiv_id, "Discovery", exc)


print("[Discovery] Finished")