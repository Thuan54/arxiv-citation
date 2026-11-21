"""Build the ``bibtex`` table (Milestone 2) from the Milestone 1 corpus.

    python src/02_parsing/build_bibtex.py --input data/metadata/papers.jsonl --out-dir data/parsed

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

try:
    from .bib_dedup import deduplicate_publication
    from .jsonl_io import describe, iter_records
except ImportError:
    try:
        from bib_dedup import deduplicate_publication
        from jsonl_io import describe, iter_records
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    """{version number: raw references}; extracts from latex_source_path if missing."""
    try:
        from reference_extractor import extract_raw_references
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from reference_extractor import extract_raw_references

    root = Path.cwd()
    out = {}
    for v in record.get("versions", []):
        ver_num = int(str(v.get("version", "v1")).lstrip("v"))
        refs = v.get("raw_references")
        if not refs:
            source_path = v.get("latex_source_path")
            if source_path:
                directory = Path(source_path)
                if not directory.is_absolute():
                    directory = root / directory
                if directory.is_dir():
                    try:
                        refs = extract_raw_references(directory)
                    except (OSError, UnicodeError, ValueError):
                        refs = []
        out[ver_num] = list(refs or [])
    return out


def max_rss_mb() -> float:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    except ImportError:                                  # Windows
        return float("nan")


def _load_existing_state(entries_path: Path):
    """Restore processed publications and aggregated statistics from an existing bibtex_entries.jsonl."""
    if not entries_path.exists():
        return set(), Counter(), Counter(), Counter(), Counter(), 0

    processed_arxiv_ids = set()
    totals = Counter()
    styles = Counter()
    types = Counter()
    coverage = Counter()
    publications = 0

    with entries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            aid = str(record.get("arxiv_id", ""))
            if not aid:
                continue

            processed_arxiv_ids.add(aid)
            publications += 1
            s = record.get("stats", {})
            totals["publications"] += 1
            totals["versions"] += s.get("versions", 0)
            totals["empty_versions"] += s.get("empty_versions", 0)
            totals["publications_without_references"] += int(s.get("entries_after", 0) == 0)
            for k in (
                "entries_before",
                "entries_after",
                "within_version_duplicates",
                "cross_version_duplicates",
                "renamed_cite_keys",
                "ambiguous_keys",
                "parse_errors",
            ):
                totals[k] += s.get(k, 0)
            styles.update(s.get("styles", {}))

            for entry in record.get("entries", []):
                types[entry.get("entry_type")] += 1
                f_fields = entry.get("fields", {})
                journal = f_fields.get("journal")
                if journal is None and JOURNAL_FALLBACK_TO_BOOKTITLE:
                    journal = f_fields.get("booktitle")
                coverage["title"] += bool(f_fields.get("title"))
                coverage["authors"] += bool(entry.get("authors"))
                coverage["year"] += bool(f_fields.get("year"))
                coverage["doi"] += bool(f_fields.get("doi"))
                coverage["journal"] += bool(journal)
                coverage["url"] += bool(f_fields.get("url"))

    return processed_arxiv_ids, totals, styles, types, coverage, publications


def build_bibtex(
    input_file="data/metadata/papers.jsonl",
    out_dir="data/parsed",
    limit=None,
    bib_dir=None,
    parquet=False,
    resume=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if bib_dir:
        Path(bib_dir).mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    entries_file = out_dir / "bibtex_entries.jsonl"
    table_file = out_dir / "bibtex.jsonl"

    if resume and entries_file.exists():
        (
            processed_arxiv_ids,
            totals,
            styles,
            types,
            coverage,
            publications,
        ) = _load_existing_state(entries_file)
        if processed_arxiv_ids:
            print(f"Resuming bibtex build: {len(processed_arxiv_ids)} publications already processed.")
    else:
        processed_arxiv_ids = set()
        totals, styles, types = Counter(), Counter(), Counter()
        coverage = Counter()
        publications = 0

    mode = "a" if resume else "w"
    input_report = {}

    with open(entries_file, mode, encoding="utf-8") as store, \
            open(table_file, mode, encoding="utf-8") as table:

        for record in iter_records(input_file, input_report):
            if limit is not None and publications >= limit:
                break

            arxiv_id = str(record.get("arxiv_id", ""))
            if resume and arxiv_id in processed_arxiv_ids:
                print(f"[ALREADY PROCESSED] Publication {arxiv_id} - skipping")
                continue

            result = deduplicate_publication(record["arxiv_id"], versions_of(record))
            publications += 1

            store.write(json.dumps(result, ensure_ascii=False) + "\n")
            store.flush()

            rows_count = 0
            for row in table_rows(result):
                table.write(json.dumps(row, ensure_ascii=False) + "\n")
                types[row["entry_type"]] += 1
                for column in ("title", "authors", "year", "doi", "journal", "url"):
                    coverage[column] += bool(row[column])
                rows_count += 1
            table.flush()

            if bib_dir and result["entries"]:
                text = "\n\n".join(e["bibtex"] for e in result["entries"]) + "\n"
                (Path(bib_dir) / f"{result['arxiv_id'].replace('/', '_')}.bib").write_text(text, encoding="utf-8")

            s = result["stats"]
            totals["publications"] += 1
            totals["versions"] += s["versions"]
            totals["empty_versions"] += s["empty_versions"]
            totals["publications_without_references"] += int(s["entries_after"] == 0)
            for k in (
                "entries_before",
                "entries_after",
                "within_version_duplicates",
                "cross_version_duplicates",
                "renamed_cite_keys",
                "ambiguous_keys",
                "parse_errors",
            ):
                totals[k] += s[k]
            styles.update(s["styles"])

            processed_arxiv_ids.add(arxiv_id)
            print(
                f"[PROCESSED] Publication {arxiv_id} "
                f"(entries: {len(result['entries'])}, table rows: {rows_count})"
            )

    if parquet:
        try:
            import pandas as pd
            parquet_rows = []
            if table_file.exists():
                with open(table_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parquet_rows.append(json.loads(line))
            frame = pd.DataFrame(parquet_rows)
            if not frame.empty and "year" in frame.columns:
                frame["year"] = frame["year"].astype("Int64")        # nullable int, not float
            frame.to_parquet(out_dir / "bibtex.parquet", index=False)
        except ImportError:
            print("pandas/pyarrow not installed: skipping parquet output", file=sys.stderr)

    if describe(input_report):
        print(f"Note about {input_file}: {describe(input_report)}", file=sys.stderr)

    elapsed = time.perf_counter() - started
    n_rows = totals["entries_after"]
    stats = {
        **totals,
        "input_problems": {
            "multi_record_lines": len(input_report.get("multi_record_lines", [])),
            "corrupt_fragments": len(input_report.get("corrupt", [])),
        },
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
    print("\nBibTeX build complete:")
    print(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default="data/metadata/papers.jsonl")
    parser.add_argument("--out-dir", default="data/parsed")
    parser.add_argument("--limit", type=int, default=None, help="only the first N publications")
    parser.add_argument("--bib-dir", default=None, help="also write one .bib file per publication")
    parser.add_argument("--parquet", action="store_true", help="also write bibtex.parquet")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite existing output files instead of resuming from already processed publications",
    )
    args = parser.parse_args()

    build_bibtex(
        input_file=args.input,
        out_dir=args.out_dir,
        limit=args.limit,
        bib_dir=args.bib_dir,
        parquet=args.parquet,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()