import argparse
import json
from pathlib import Path
import re

try:
    from config import (
        JSONL_FILE,
        PAPERS_TABLE_FILE,
        REFERENCES_TABLE_FILE,
        VENUE_FROM_S2_FALLBACK,
    )
    from s2_reference_data import load_s2_references
    from helper import clean_arxiv_id
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        JSONL_FILE,
        PAPERS_TABLE_FILE,
        REFERENCES_TABLE_FILE,
        VENUE_FROM_S2_FALLBACK,
    )
    from s2_reference_data import load_s2_references
    from helper import clean_arxiv_id


def _iter_paper_records(path):
    """Yield records one by one from path."""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc


def _read_paper_records(path):
    return list(_iter_paper_records(path))


def _version_number(version):
    match = re.fullmatch(r"v(\d+)", version)
    return int(match.group(1)) if match else 0


def build_paper_row(record, s2_references=None):
    """Build normalized paper row for a single paper record."""
    if s2_references is None:
        s2_references = {}

    enrichment = s2_references.get(record.get("arxiv_id", ""))
    versions = sorted(
        record.get("versions", []),
        key=lambda item: _version_number(item.get("version", "")),
    )
    metadata_by_version = []
    for version in versions:
        metadata = dict(version.get("metadata") or {})
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
    initial_date = initial_metadata.get("submission_date")

    latest_metadata = next(
        (
            metadata
            for metadata in reversed(metadata_by_version)
            if metadata.get("title") or metadata.get("authors")
        ),
        metadata_by_version[-1] if metadata_by_version else {},
    )

    revised_dates = []
    seen_dates = set()
    for version, metadata in zip(versions, metadata_by_version):
        if _version_number(version.get("version", "")) <= 1:
            continue
        revised_date = (
            metadata.get("version_date") or metadata.get("updated_date")
        )
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

    return {
        "arxiv_id": record["arxiv_id"],
        "title": latest_metadata.get("title"),
        "authors": latest_metadata.get("authors") or [],
        "submission_date": initial_date,
        "revised_dates": revised_dates,
        "venue": venue,
    }


def build_papers_table(records, s2_references=None):
    if s2_references is None:
        s2_references = {}

    return [
        build_paper_row(record, s2_references)
        for record in sorted(records, key=lambda item: item.get("arxiv_id", ""))
    ]


def build_paper_reference_rows(record, s2_references=None):
    """Build normalized reference rows for a single paper record."""
    if not s2_references:
        return []

    citing_id = record.get("arxiv_id", "")
    enrichment = s2_references.get(citing_id)
    if enrichment is None:
        return []

    cited_by_id = {}
    for reference in enrichment.get("s2_references") or []:
        arxiv_id = reference.get("arxiv_id")
        if not isinstance(arxiv_id, str) or not arxiv_id.strip():
            continue
        arxiv_id = clean_arxiv_id(arxiv_id)
        if "/" in arxiv_id:
            arxiv_id = arxiv_id.lower()
        cited_by_id.setdefault(arxiv_id, reference)

    rows = []
    for cited_id, reference in sorted(cited_by_id.items()):
        rows.append({
            "arxiv_id": citing_id,
            "ref_arxiv_id": cited_id,
            "ref_title": reference.get("title"),
            "ref_authors": reference.get("authors") or [],
            "ref_submission_date": reference.get("submission_date"),
            "ref_s2_id": reference.get("s2_id"),
        })
    return rows


def build_references_table(records, s2_references=None):
    s2_references = s2_references or {}
    rows = []
    for record in sorted(records, key=lambda item: item.get("arxiv_id", "")):
        rows.extend(build_paper_reference_rows(record, s2_references))
    return rows


def _load_processed_arxiv_ids(path):
    """Load already processed arxiv_ids from an existing papers_table.jsonl."""
    if not path.exists():
        return set()
    processed = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "arxiv_id" in row:
                    processed.add(str(row["arxiv_id"]))
            except json.JSONDecodeError:
                continue
    return processed


def export_tables(
    source_path=JSONL_FILE,
    papers_table_path=PAPERS_TABLE_FILE,
    references_table_path=REFERENCES_TABLE_FILE,
    resume=True,
):
    """Write normalized paper and arXiv-reference JSONL tables incrementally."""
    source_path = Path(source_path)
    papers_table_path = Path(papers_table_path)
    references_table_path = Path(references_table_path)

    papers_table_path.parent.mkdir(parents=True, exist_ok=True)
    references_table_path.parent.mkdir(parents=True, exist_ok=True)

    s2_references = load_s2_references()

    processed_arxiv_ids = set()
    if resume and papers_table_path.exists():
        processed_arxiv_ids = _load_processed_arxiv_ids(papers_table_path)
        if processed_arxiv_ids:
            print(f"Resuming export: {len(processed_arxiv_ids)} papers already processed.")

    mode = "a" if resume else "w"
    new_papers = 0
    new_references = 0

    with papers_table_path.open(mode, encoding="utf-8") as papers_file, \
            references_table_path.open(mode, encoding="utf-8") as refs_file:

        for record in _iter_paper_records(source_path):
            arxiv_id = str(record.get("arxiv_id", ""))
            if not arxiv_id:
                continue

            if resume and arxiv_id in processed_arxiv_ids:
                print(f"[ALREADY PROCESSED] Paper {arxiv_id} - skipping")
                continue

            paper_row = build_paper_row(record, s2_references)
            ref_rows = build_paper_reference_rows(record, s2_references)

            # Write references first, then paper row, flushing both
            for ref in ref_rows:
                refs_file.write(json.dumps(ref, ensure_ascii=False) + "\n")
            refs_file.flush()

            papers_file.write(json.dumps(paper_row, ensure_ascii=False) + "\n")
            papers_file.flush()

            processed_arxiv_ids.add(arxiv_id)
            new_papers += 1
            new_references += len(ref_rows)
            print(
                f"[PROCESSED] Paper {arxiv_id} "
                f"(venue: {bool(paper_row['venue'])}, references: {len(ref_rows)})"
            )

    total_references = 0
    if references_table_path.exists():
        with references_table_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_references += 1

    return {
        "papers": len(processed_arxiv_ids),
        "references": total_references,
        "new_papers": new_papers,
        "new_references": new_references,
        "s2_rows_loaded": len(s2_references),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export normalized paper and arXiv-reference JSONL tables incrementally."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=JSONL_FILE,
        help="Path to input papers.jsonl (default: config.JSONL_FILE)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite existing output files instead of resuming from already processed papers.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(
            f"Input papers file not found: {args.source}."
        )
    stats = export_tables(source_path=args.source, resume=not args.no_resume)
    print("\nExport complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()



def main():
    parser = argparse.ArgumentParser(
        description="Export normalized paper and arXiv-reference JSONL tables incrementally."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=JSONL_FILE,
        help="Path to input papers.jsonl (default: config.JSONL_FILE)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite existing output files instead of resuming from already processed papers.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(
            f"Input papers file not found: {args.source}."
        )
    stats = export_tables(source_path=args.source, resume=not args.no_resume)
    print("\nExport complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()