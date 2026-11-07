"""Enrich completed crawl output with Semantic Scholar data.

Run after the crawler has finished. Papers with null `s2_references` are
looked up in batches of up to 500. Failed lookups remain null, so rerunning
this script retries them without repeating the crawler:

    export S2_API_KEY=...
    python src/retry_s2_references.py --dry-run
    python src/retry_s2_references.py
    python src/retry_s2_references.py --include-empty

After enrichment, normalized tables are exported. The original paper file is
kept as papers.jsonl.bak.
"""
import argparse
import json
import shutil
from collections import Counter

import requests

from config import JSONL_FILE, S2_API_KEY, VENUE_FROM_S2_FALLBACK
from export_tables import export_tables
from helper import format_error, log_failure
from semantic_scholar import S2_BATCH_SIZE, crawl_papers


def needs_retry(record, include_empty):
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

    with JSONL_FILE.open("r", encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]

    todo = [
        record["arxiv_id"]
        for record in records
        if needs_retry(record, args.include_empty)
    ]
    print(f"{len(todo)} of {len(records)} papers need a Semantic Scholar retry")

    results = {}
    session = requests.Session()

    for start in range(0, len(todo), S2_BATCH_SIZE):
        chunk = todo[start:start + S2_BATCH_SIZE]
        results.update(crawl_papers(session, chunk))
        print(f"  looked up {min(start + S2_BATCH_SIZE, len(todo))}/{len(todo)}")

    session.close()

    recovered = 0
    outcome = Counter()

    for record in records:
        result = results.get(record["arxiv_id"])

        if result is None:
            continue

        status, references, venue, error = result

        if status == "not_found":
            log_failure(
                record["arxiv_id"],
                "Semantic Scholar references",
                "Paper not found on Semantic Scholar",
            )
            outcome["still not found on Semantic Scholar"] += 1
            continue

        if status == "failed":
            log_failure(
                record["arxiv_id"],
                "Semantic Scholar references",
                format_error(error),
                exception=error,
            )
            outcome[f"still failing: {format_error(error)}"] += 1
            continue

        recovered += 1
        outcome["recovered" if references else "found, but no references listed"] += 1

        for version in record["versions"]:
            version["s2_references"] = references
            metadata = version.get("metadata")

            if (
                metadata is not None
                and VENUE_FROM_S2_FALLBACK
                and not metadata.get("venue")
                and venue
            ):
                metadata["venue"] = venue

    print(f"Updated {recovered} papers")
    for reason, count in outcome.most_common():
        print(f"  {count:6d}  {reason}")

    if args.dry_run:
        print("(dry run: nothing written)")
        return

    tmp_file = JSONL_FILE.with_name(JSONL_FILE.name + ".tmp")
    bak_file = JSONL_FILE.with_name(JSONL_FILE.name + ".bak")

    with tmp_file.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    shutil.copy2(JSONL_FILE, bak_file)
    tmp_file.replace(JSONL_FILE)
    print(f"Wrote {JSONL_FILE} (backup: {bak_file.name})")
    print(f"Exported tables: {export_tables()}")


if __name__ == "__main__":
    main()