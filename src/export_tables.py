import json
from pathlib import Path

from config import (
    JSONL_FILE,
    PAPERS_TABLE_FILE,
    REFERENCES_TABLE_FILE,
    VENUE_FROM_S2_FALLBACK,
)
from helper import clean_arxiv_id
from s2_reference_data import load_s2_references


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


def build_papers_table(records, s2_references=None):
    s2_references = s2_references or {}
    rows = []
    for record in records:
        versions = sorted(
            record.get("versions", []),
            key=lambda item: _version_number(item.get("version", "")),
        )
        enrichment = s2_references.get(record["arxiv_id"])
        metadata_by_version = []
        for item in versions:
            metadata = dict(item.get("metadata") or {})
            if (
                enrichment is not None
                and VENUE_FROM_S2_FALLBACK
                and not metadata.get("venue")
                and enrichment.get("venue")
            ):
                metadata["venue"] = enrichment["venue"]
            metadata_by_version.append(metadata)

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


def build_references_table(records, s2_references=None):
    s2_references = s2_references or {}
    references_by_paper = {}
    for record in records:
        citing_id = record["arxiv_id"]
        enrichment = s2_references.get(citing_id)
        if enrichment is None:
            continue

        cited_by_id = references_by_paper.setdefault(citing_id, {})
        for reference in enrichment.get("s2_references") or []:
            arxiv_id = reference.get("arxiv_id")
            if not isinstance(arxiv_id, str) or not arxiv_id.strip():
                continue
            arxiv_id = clean_arxiv_id(arxiv_id)
            if "/" in arxiv_id:
                arxiv_id = arxiv_id.lower()
            cited_by_id.setdefault(arxiv_id, reference)

    rows = []
    for citing_id in sorted(references_by_paper):
        for cited_id, reference in sorted(
            references_by_paper[citing_id].items()
        ):
            rows.append({
                "arxiv_id": citing_id,
                "ref_arxiv_id": cited_id,
                "ref_title": reference.get("title"),
                "ref_authors": reference.get("authors") or [],
                "ref_submission_date": reference.get("submission_date"),
                "ref_s2_id": reference.get("s2_id"),
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
    s2_references = load_s2_references()
    papers_table = build_papers_table(records, s2_references)
    references_table = build_references_table(records, s2_references)
    _write_jsonl_atomic(PAPERS_TABLE_FILE, papers_table)
    _write_jsonl_atomic(REFERENCES_TABLE_FILE, references_table)
    return {
        "papers": len(papers_table),
        "references": len(references_table),
    }
