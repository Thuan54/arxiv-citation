"""Re-crawl Semantic Scholar references for papers where they are missing.

Fixes an existing papers.jsonl without re-running the whole crawler: only
papers whose `s2_references` is null (crawl failed or paper not found) are
looked up again, in batches. Run from the project root:

    export S2_API_KEY=...
    python src/retry_s2_ref.py --dry-run             # report only
    python src/retry_s2_ref.py                       # update papers.jsonl
    python src/retry_s2_ref.py --include-empty       # also retry [] results

The original file is kept as papers.jsonl.bak. The file is rewritten with one
record per line, so a papers.jsonl with merged lines is repaired at the same
time (unreadable fragments, if any, go to papers.jsonl.corrupt).
"""
import argparse
import json
import shutil
from collections import Counter
import time
import requests

from config import JSONL_FILE, S2_API_KEY, VENUE_FROM_S2_FALLBACK
from helper import format_error
from jsonl_io import describe, iter_records
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

    report = {}
    records = list(iter_records(JSONL_FILE, report))
    problems = describe(report)
    if problems:
        print(f"Note: {problems}")

    ids = Counter(record.get("arxiv_id") for record in records)
    duplicated = sum(1 for count in ids.values() if count > 1)
    if duplicated:
        print(f"Note: {duplicated} arxiv_id(s) appear more than once (kept as they are)")

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
            outcome["still not found on Semantic Scholar"] += 1
            continue

        if status == "failed":
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

    if report["corrupt"]:
        corrupt_file = JSONL_FILE.with_name(JSONL_FILE.name + ".corrupt")
        with corrupt_file.open("a", encoding="utf-8") as out:
            for line_no, fragment in report["corrupt"]:
                out.write(json.dumps({"line": line_no, "fragment": fragment}, ensure_ascii=False) + "\n")
        print(f"Saved {len(report['corrupt'])} unreadable fragment(s) to {corrupt_file.name}")

    with tmp_file.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    shutil.copy2(JSONL_FILE, bak_file)
    tmp_file.replace(JSONL_FILE)
    print(f"Wrote {JSONL_FILE} (backup: {bak_file.name})")


if __name__ == "__main__":
    main()