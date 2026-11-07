"""Re-extract raw_references in data/metadata/papers.jsonl from data/raw.

Uses the sources already on disk, so nothing is downloaded and no arXiv API
requests are made. Run from the project root (same folder you run crawler.py
from):

    python backfill_references.py --dry-run   # report only, change nothing
    python backfill_references.py             # rewrite papers.jsonl

The original file is kept as papers.jsonl.bak.
"""
import argparse
import json
import shutil
from collections import Counter

from config import JSONL_FILE, ROOT
from reference_extractor import extract_raw_references


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the summary without modifying papers.jsonl",
    )
    args = parser.parse_args()

    if not JSONL_FILE.exists():
        raise SystemExit(f"Not found: {JSONL_FILE}")

    tmp_file = JSONL_FILE.with_name(JSONL_FILE.name + ".tmp")
    bak_file = JSONL_FILE.with_name(JSONL_FILE.name + ".bak")

    scanned = changed = 0
    refs_before = refs_after = 0
    nonempty_before = nonempty_after = 0
    still_empty = Counter()
    missing_dirs = 0

    out = None if args.dry_run else tmp_file.open("w", encoding="utf-8")

    try:
        with JSONL_FILE.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue

                record = json.loads(line)

                for version in record.get("versions", []):
                    old = version.get("raw_references") or []
                    refs_before += len(old)
                    nonempty_before += bool(old)

                    path = version.get("latex_source_path")
                    directory = ROOT / path if path else None

                    if directory is None or not directory.is_dir():
                        # Failed download/extraction: nothing to rescan.
                        missing_dirs += 1
                        refs_after += len(old)
                        nonempty_after += bool(old)
                        continue

                    scanned += 1
                    report = {}
                    new = extract_raw_references(directory, report)

                    if new != old:
                        changed += 1
                        version["raw_references"] = new

                    refs_after += len(new)
                    nonempty_after += bool(new)

                    if not new:
                        if report["total_files"] == 0:
                            still_empty["no files left (PDF-only?)"] += 1
                        elif not report["tex_files"] and not report["bbl_files"]:
                            still_empty["no .tex/.bbl files"] += 1
                        else:
                            still_empty["TeX present, no bibliography found"] += 1

                if out:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if out:
            out.close()

    if not args.dry_run:
        shutil.copy2(JSONL_FILE, bak_file)
        tmp_file.replace(JSONL_FILE)

    print(f"Versions rescanned from disk : {scanned}")
    print(f"Versions without source dir  : {missing_dirs} (left unchanged)")
    print(f"Versions whose refs changed  : {changed}")
    print(f"Versions with references     : {nonempty_before} -> {nonempty_after}")
    print(f"Total raw references         : {refs_before} -> {refs_after}")
    if still_empty:
        print("Still empty after rescan:")
        for reason, count in still_empty.most_common():
            print(f"  {count:6d}  {reason}")
    print("(dry run: nothing written)" if args.dry_run
          else f"Wrote {JSONL_FILE} (backup: {bak_file.name})")


if __name__ == "__main__":
    main()