"""Enrich completed crawl output with Semantic Scholar data.

Run after the crawler has finished. Papers without `s2_references` are looked
up in batches of up to 500. Results are saved in a compact sidecar file, leaving
the crawler output unchanged. Failed lookups are retried on the next run:

    export S2_API_KEY=...
    python src/s2_references.py --dry-run
    python src/s2_references.py
    python src/s2_references.py --include-empty

After enrichment, normalized tables are exported.
"""
import argparse
import json
from collections import Counter

import requests

from config import (
    JSONL_FILE,
    S2_API_KEY,
    S2_REFERENCES_FILE,
    VENUE_FROM_S2_FALLBACK,
)
from export_tables import export_tables
from helper import format_error, log_failure
from semantic_scholar import S2_BATCH_SIZE, crawl_papers
from s2_reference_data import load_s2_references


def needs_retry(record, include_empty, enrichment):
    if enrichment is not None:
        refs = enrichment["s2_references"]
        return include_empty and refs == []

    versions = record.get("versions", [])
    refs = versions[0].get("s2_references") if versions else None

    if refs is None:
        return True

    return include_empty and refs == []


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="look papers up and report, but do not write")
    parser.add_argument("--include-empty", action="store_true",
                        help="also retry papers whose list came back empty")
    args = parser.parse_args()

    if not JSONL_FILE.exists():
        raise SystemExit(f"Not found: {JSONL_FILE}")

    if not S2_API_KEY:
        print("Warning: S2_API_KEY is not set; expect heavy throttling.")

    enrichments = load_s2_references(S2_REFERENCES_FILE)
    total = 0
    todo = []
    with JSONL_FILE.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {JSONL_FILE} at line {line_number}"
                ) from exc
            total += 1
            arxiv_id = record["arxiv_id"]
            if needs_retry(
                record,
                args.include_empty,
                enrichments.get(arxiv_id),
            ):
                todo.append(arxiv_id)
    print(f"{len(todo)} of {total} papers need a Semantic Scholar retry")

    results = {}
    session = requests.Session()

    for start in range(0, len(todo), S2_BATCH_SIZE):
        chunk = todo[start:start + S2_BATCH_SIZE]
        results.update(crawl_papers(session, chunk))
        print(f"  looked up {min(start + S2_BATCH_SIZE, len(todo))}/{len(todo)}")

    session.close()

    recovered = 0
    outcome = Counter()
    for arxiv_id, result in results.items():
        status, references, venue, error = result

        if status == "not_found":
            log_failure(
                arxiv_id,
                "Semantic Scholar references",
                "Paper not found on Semantic Scholar",
            )
            outcome["still not found on Semantic Scholar"] += 1
            continue

        if status == "failed":
            log_failure(
                arxiv_id,
                "Semantic Scholar references",
                format_error(error),
                exception=error,
            )
            outcome[f"still failing: {format_error(error)}"] += 1
            continue

        recovered += 1
        outcome["recovered" if references else "found, but no references listed"] += 1
        enrichments[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "s2_references": references,
            "venue": venue if VENUE_FROM_S2_FALLBACK else None,
        }

    print(f"Updated {recovered} papers")
    for reason, count in outcome.most_common():
        print(f"  {count:6d}  {reason}")

    if args.dry_run:
        print("(dry run: nothing written)")
        return

    S2_REFERENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = S2_REFERENCES_FILE.with_name(S2_REFERENCES_FILE.name + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as out:
        for enrichment in enrichments.values():
            out.write(json.dumps(enrichment, ensure_ascii=False) + "\n")
    tmp_file.replace(S2_REFERENCES_FILE)
    print(f"Wrote {S2_REFERENCES_FILE}")
    print(f"Exported tables: {export_tables()}")


if __name__ == "__main__":
    main()