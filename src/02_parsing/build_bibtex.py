"""Build the ``bibtex`` table (Milestone 2) from the Milestone 1 corpus.

    python build_bibtex.py --input data/metadata/papers.jsonl --out-dir data/parsed

Reads ``papers.jsonl`` one publication at a time (constant memory), parses the
``raw_references`` of every version, deduplicates them across versions and
writes:

``bibtex.jsonl``          the ``bibtex`` table: one row per unified reference
                          (arxiv_id, citation_key, entry_type, title, authors,
                          year, doi, journal, url)
``bibtex_entries.jsonl``  the working store, one line per publication: unified
                          entries with all fields + rendered BibTeX, the
                          per-version ``\\cite`` rename maps and statistics
``bibtex_stats.json``     corpus statistics and performance numbers (time, RAM)
``bibtex.parquet``        same table as parquet (only with ``--parquet``)
``bib/<arxiv_id>.bib``    standard BibTeX files (only with ``--bib-dir``)
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from bib_dedup import deduplicate_publication
from jsonl_io import describe, iter_records

# The table has a single venue column called ``journal``. For conference
# papers BibTeX uses ``booktitle`` instead; fall back to it so the column is
# not empty for them. Set to False for a strict BibTeX ``journal`` field.
JOURNAL_FALLBACK_TO_BOOKTITLE = True


def table_rows(result: dict):
    """Rows of the ``bibtex`` table for one deduplicated publication."""
    for entry in result["entries"]:
        f = entry["fields"]
        journal = f.get("journal")
        if journal is None and JOURNAL_FALLBACK_TO_BOOKTITLE:
            journal = f.get("booktitle")
        yield {
            "arxiv_id": result["arxiv_id"],
            "citation_key": entry["citation_key"],
            "entry_type": entry["entry_type"],
            "title": f.get("title"),
            "authors": [f"{g} {fam}".strip() for g, fam in entry["authors"]],
            "year": f.get("year"),
            "doi": f.get("doi"),
            "journal": journal,
            "url": f.get("url"),
        }


def versions_of(record: dict) -> dict:
    """{version number: raw references}; a version without sources stays (empty)."""
    return {
        int(v["version"].lstrip("v")): list(v.get("raw_references") or [])
        for v in record.get("versions", [])
    }


def max_rss_mb() -> float:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    except ImportError:                                  # Windows
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default="data/metadata/papers.jsonl")
    parser.add_argument("--out-dir", default="data/parsed")
    parser.add_argument("--limit", type=int, default=None, help="only the first N publications")
    parser.add_argument("--bib-dir", default=None, help="also write one .bib file per publication")
    parser.add_argument("--parquet", action="store_true", help="also write bibtex.parquet")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.bib_dir:
        Path(args.bib_dir).mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    totals, styles, types = Counter(), Counter(), Counter()
    coverage = Counter()
    rows_for_parquet = []
    publications = 0

    input_report = {}
    with open(out_dir / "bibtex_entries.jsonl", "w", encoding="utf-8") as store, \
            open(out_dir / "bibtex.jsonl", "w", encoding="utf-8") as table:

        for record in iter_records(args.input, input_report):
            if args.limit is not None and publications >= args.limit:
                break

            result = deduplicate_publication(record["arxiv_id"], versions_of(record))
            publications += 1

            store.write(json.dumps(result, ensure_ascii=False) + "\n")

            for row in table_rows(result):
                table.write(json.dumps(row, ensure_ascii=False) + "\n")
                if args.parquet:
                    rows_for_parquet.append(row)
                types[row["entry_type"]] += 1
                for column in ("title", "authors", "year", "doi", "journal", "url"):
                    coverage[column] += bool(row[column])

            if args.bib_dir and result["entries"]:
                text = "\n\n".join(e["bibtex"] for e in result["entries"]) + "\n"
                (Path(args.bib_dir) / f"{result['arxiv_id'].replace('/', '_')}.bib").write_text(text, encoding="utf-8")

            s = result["stats"]
            totals["publications"] += 1
            totals["versions"] += s["versions"]
            totals["empty_versions"] += s["empty_versions"]
            totals["publications_without_references"] += int(s["entries_after"] == 0)
            for k in ("entries_before", "entries_after", "within_version_duplicates",
                      "cross_version_duplicates", "renamed_cite_keys", "ambiguous_keys", "parse_errors"):
                totals[k] += s[k]
            styles.update(s["styles"])

    if args.parquet:
        try:
            import pandas as pd
            frame = pd.DataFrame(rows_for_parquet)
            frame["year"] = frame["year"].astype("Int64")        # nullable int, not float
            frame.to_parquet(out_dir / "bibtex.parquet", index=False)
        except ImportError:
            print("pandas/pyarrow not installed: skipping parquet output", file=sys.stderr)

    if describe(input_report):
        print(f"Note about {args.input}: {describe(input_report)}", file=sys.stderr)

    elapsed = time.perf_counter() - started
    n_rows = totals["entries_after"]
    stats = {
        **totals,
        "input_problems": {"multi_record_lines": len(input_report["multi_record_lines"]),
                           "corrupt_fragments": len(input_report["corrupt"])},
        "reduction_pct": round(100 * (1 - n_rows / totals["entries_before"]), 2) if totals["entries_before"] else 0.0,
        "column_coverage_pct": {k: round(100 * v / n_rows, 2) for k, v in coverage.items()} if n_rows else {},
        "entry_types": dict(types.most_common()),
        "bibliography_styles": dict(styles.most_common()),
        "performance": {
            "wall_seconds": round(elapsed, 3),
            "seconds_per_publication": round(elapsed / publications, 4) if publications else None,
            "max_rss_mb": round(max_rss_mb(), 1),
            "output_bytes": {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file()},
        },
    }
    (out_dir / "bibtex_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
