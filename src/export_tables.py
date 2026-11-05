import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from config import (
    API_TIMEOUT,
    BATCH_SIZE,
    JSONL_FILE,
    PAPERS_TABLE_FILE,
    REFERENCES_TABLE_FILE,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_API_TIMEOUT,
    USER_AGENT,
)
from helper import clean_arxiv_id, log_failure, parse_versioned_id
from rate_limiter import arxiv_limiter


ARXIV_API_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_BATCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_ID_PATTERNS = (
    re.compile(
        r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)"
        r"(?P<id>\d{4}\.\d{4,5})(?:v\d+)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)"
        r"(?P<id>[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?",
        re.IGNORECASE,
    ),
)


def _read_paper_records(path):
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc
    return records


def _version_number(version):
    match = re.fullmatch(r"v(\d+)", version)
    return int(match.group(1)) if match else 0


def build_papers_table(records):
    rows = []
    for record in records:
        versions = sorted(
            record.get("versions", []),
            key=lambda item: _version_number(item.get("version", "")),
        )
        metadata_by_version = [
            item.get("metadata") or {}
            for item in versions
        ]

        initial_metadata = next(
            (
                metadata
                for metadata in metadata_by_version
                if metadata.get("submission_date")
            ),
            {},
        )
        latest_metadata = next(
            (
                metadata
                for metadata in reversed(metadata_by_version)
                if metadata
            ),
            {},
        )

        initial_date = initial_metadata.get("submission_date")
        revised_dates = []
        seen_dates = set()
        for version, metadata in zip(versions, metadata_by_version):
            if _version_number(version.get("version", "")) <= 1:
                continue
            revised_date = metadata.get("updated_date")
            if (
                revised_date
                and revised_date != initial_date
                and revised_date not in seen_dates
            ):
                revised_dates.append(revised_date)
                seen_dates.add(revised_date)

        venue = next(
            (
                metadata.get("venue")
                for metadata in reversed(metadata_by_version)
                if metadata.get("venue")
            ),
            None,
        )
        rows.append({
            "arxiv_id": record["arxiv_id"],
            "title": latest_metadata.get("title"),
            "authors": latest_metadata.get("authors") or [],
            "submission_date": initial_date,
            "revised_dates": revised_dates,
            "venue": venue,
        })
    return rows


def extract_citation_arxiv_ids(raw_reference):
    identifiers = []
    try:
        semantic_reference = json.loads(raw_reference)
    except (TypeError, json.JSONDecodeError):
        semantic_reference = None

    if isinstance(semantic_reference, dict):
        cited_paper = semantic_reference.get("citedPaper") or {}
        external_ids = cited_paper.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv") or external_ids.get("arXiv")
        if arxiv_id:
            identifiers.append(clean_arxiv_id(str(arxiv_id)))

    for pattern in ARXIV_ID_PATTERNS:
        for match in pattern.finditer(raw_reference):
            arxiv_id = match.group("id")
            if "/" in arxiv_id:
                arxiv_id = arxiv_id.lower()
            if arxiv_id not in identifiers:
                identifiers.append(arxiv_id)
    return identifiers


def _fetch_arxiv_metadata(arxiv_ids):
    metadata_by_id = {}
    for start in range(0, len(arxiv_ids), BATCH_SIZE):
        batch = arxiv_ids[start:start + BATCH_SIZE]
        try:
            with arxiv_limiter.request("api"):
                response = requests.get(
                    ARXIV_API_URL,
                    params={
                        "id_list": ",".join(batch),
                        "max_results": len(batch),
                    },
                    headers={"User-Agent": USER_AGENT},
                    timeout=API_TIMEOUT,
                )
                response.raise_for_status()

            root = ET.fromstring(response.content)
            for entry in root.findall("atom:entry", ATOM_NS):
                entry_id = entry.findtext("atom:id", namespaces=ATOM_NS)
                parsed_id = parse_versioned_id(entry_id or "")
                if parsed_id is None:
                    continue
                arxiv_id, _ = parsed_id

                def text(tag):
                    return (
                        entry.findtext(f"atom:{tag}", namespaces=ATOM_NS)
                        or ""
                    ).strip()

                metadata_by_id[arxiv_id] = {
                    "title": " ".join(text("title").split()) or None,
                    "authors": [
                        name
                        for author in entry.findall("atom:author", ATOM_NS)
                        if (
                            name := author.findtext(
                                "atom:name",
                                namespaces=ATOM_NS,
                            )
                        )
                    ],
                    "submission_date": text("published")[:10] or None,
                }
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            for arxiv_id in batch:
                log_failure(
                    arxiv_id,
                    "Reference arXiv metadata",
                    exc,
                    exception=exc,
                )
    return metadata_by_id


def _fetch_semantic_scholar_ids(arxiv_ids):
    semantic_ids = {}
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    for start in range(0, len(arxiv_ids), 500):
        batch = arxiv_ids[start:start + 500]
        try:
            response = requests.post(
                SEMANTIC_SCHOLAR_BATCH_URL,
                params={
                    "fields": "paperId,externalIds",
                },
                json={
                    "ids": [f"ARXIV:{arxiv_id}" for arxiv_id in batch],
                },
                headers=headers,
                timeout=SEMANTIC_SCHOLAR_API_TIMEOUT,
            )
            response.raise_for_status()
            results = response.json()
            if not isinstance(results, list) or len(results) != len(batch):
                raise ValueError(
                    "Semantic Scholar returned an unexpected batch response"
                )

            for arxiv_id, result in zip(batch, results):
                if result and result.get("paperId"):
                    semantic_ids[arxiv_id] = result["paperId"]
        except (requests.RequestException, ValueError) as exc:
            for arxiv_id in batch:
                log_failure(
                    arxiv_id,
                    "Semantic Scholar reference lookup",
                    exc,
                    exception=exc if isinstance(
                        exc, requests.RequestException
                    ) else None,
                )
    return semantic_ids


def build_references_table(records, papers_table):
    paper_by_id = {paper["arxiv_id"]: paper for paper in papers_table}
    references_by_paper = {}
    for record in records:
        citing_id = record["arxiv_id"]
        cited_ids = references_by_paper.setdefault(citing_id, set())
        for version in record.get("versions", []):
            for raw_reference in version.get("raw_references") or []:
                cited_ids.update(extract_citation_arxiv_ids(raw_reference))

    all_cited_ids = sorted({
        cited_id
        for cited_ids in references_by_paper.values()
        for cited_id in cited_ids
    })
    missing_local_ids = [
        arxiv_id
        for arxiv_id in all_cited_ids
        if arxiv_id not in paper_by_id
    ]
    external_metadata = _fetch_arxiv_metadata(missing_local_ids)
    semantic_ids = _fetch_semantic_scholar_ids(all_cited_ids)

    rows = []
    for citing_id in sorted(references_by_paper):
        for cited_id in sorted(references_by_paper[citing_id]):
            paper = paper_by_id.get(cited_id)
            metadata = external_metadata.get(cited_id, {})
            rows.append({
                "arxiv_id": citing_id,
                "ref_arxiv_id": cited_id,
                "ref_title": (
                    paper.get("title") if paper else metadata.get("title")
                ),
                "ref_authors": (
                    paper.get("authors") if paper else metadata.get("authors", [])
                ),
                "ref_submission_date": (
                    paper.get("submission_date")
                    if paper
                    else metadata.get("submission_date")
                ),
                "ref_s2_id": semantic_ids.get(cited_id),
            })
    return rows


def _write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def export_tables(source_path=JSONL_FILE):
    """Write the normalized paper and arXiv-reference JSONL tables."""
    records = _read_paper_records(Path(source_path))
    papers_table = build_papers_table(records)
    references_table = build_references_table(records, papers_table)
    _write_jsonl_atomic(PAPERS_TABLE_FILE, papers_table)
    _write_jsonl_atomic(REFERENCES_TABLE_FILE, references_table)
    return {
        "papers": len(papers_table),
        "references": len(references_table),
    }
