"""Build Milestone 2 elements, hierarchy, and unified BibTeX tables.

Run from the repository root after Milestone 1:

    python src/02_parsing/parse_corpus.py
"""
import argparse
import json
import os
from pathlib import Path

try:
    from ..config import (
    ELEMENTS_FILE,
    HIERARCHY_FILE,
    JSONL_FILE,
    PARSED_VERSIONS_FILE,
    ROOT,
)
    from ..reference_extractor import extract_raw_references
    from ..s2_reference_data import load_s2_references

except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import (
    ELEMENTS_FILE,
    HIERARCHY_FILE,
    JSONL_FILE,
    PARSED_VERSIONS_FILE,
    ROOT,
)
    from reference_extractor import extract_raw_references
    from s2_reference_data import load_s2_references

try:
    from .hierarchy import parse_publication_sources
except ImportError:
    from hierarchy import parse_publication_sources



OUTPUT_FILES = {
    "elements": ELEMENTS_FILE,
    "hierarchy": HIERARCHY_FILE,
    "versions": PARSED_VERSIONS_FILE,
}


def _read_publications(path):
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc


def _source_directory(source_path):
    if not source_path:
        return None
    path = Path(source_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _load_processed_arxiv_ids(versions_path):
    """Read already processed arxiv_ids from existing versions.jsonl."""
    if not versions_path.exists():
        return set()
    processed = set()
    with versions_path.open("r", encoding="utf-8") as f:
        for line in f:
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


def parse_corpus(source_file=JSONL_FILE, resume=True):
    counts = {name: 0 for name in OUTPUT_FILES}
    s2_references = load_s2_references()

    for output_path in OUTPUT_FILES.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_arxiv_ids = set()
    if resume and OUTPUT_FILES["versions"].exists():
        processed_arxiv_ids = _load_processed_arxiv_ids(OUTPUT_FILES["versions"])
        if processed_arxiv_ids:
            print(f"Resuming parsing: {len(processed_arxiv_ids)} papers already processed.")

    mode = "a" if resume else "w"
    output_streams = {}

    try:
        for name, output_path in OUTPUT_FILES.items():
            output_streams[name] = output_path.open(mode, encoding="utf-8")

        for publication in _read_publications(Path(source_file)):
            arxiv_id = str(publication["arxiv_id"])
            if resume and arxiv_id in processed_arxiv_ids:
                print(f"[ALREADY PROCESSED] Paper {arxiv_id} - skipping")
                continue

            versions = sorted(
                publication.get("versions", []),
                key=lambda item: int(
                    str(item.get("version", "v1")).lstrip("v")
                ),
            )
            version_by_number = {
                int(str(item.get("version", "v1")).lstrip("v")): item
                for item in versions
            }
            version_directories = [
                (
                    str(item.get("version", "v1")),
                    _source_directory(item.get("latex_source_path")),
                )
                for item in versions
            ]
            references_by_version = {}
            try:
                elements, hierarchy, parsed_versions = (
                    parse_publication_sources(arxiv_id, version_directories)
                )
            except (OSError, UnicodeError, ValueError) as exc:
                parsed_versions = [
                    {
                        "arxiv_id": arxiv_id,
                        "version": int(str(version).lstrip("v")),
                        "status": "failed",
                        "element_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for version, _ in version_directories
                ]
                elements, hierarchy = [], []

            for version, directory in version_directories:
                version_number = int(version.lstrip("v"))
                source_version = version_by_number.get(version_number, {})
                existing_refs = source_version.get("raw_references")
                if existing_refs:
                    references_by_version[version_number] = {
                        "raw_references": existing_refs,
                        "reference_extraction_status": "complete",
                        "reference_files_scanned": 0,
                        "reference_extraction_errors": [],
                    }
                    continue

                report = {}
                if directory is not None and directory.is_dir():
                    try:
                        raw_references = extract_raw_references(
                            directory,
                            report,
                        )
                    except (OSError, UnicodeError, ValueError) as exc:
                        references_by_version[version_number] = {
                            "raw_references": [],
                            "reference_extraction_status": "failed",
                            "reference_files_scanned": report.get(
                                "total_files",
                                0,
                            ),
                            "reference_extraction_errors": [
                                f"{type(exc).__name__}: {exc}"
                            ],
                        }
                    else:
                        references_by_version[version_number] = {
                            "raw_references": raw_references,
                            "reference_extraction_status": (
                                "partial"
                                if report["read_failures"] else "complete"
                            ),
                            "reference_files_scanned": report["total_files"],
                            "reference_extraction_errors": report["read_failures"],
                        }
                else:
                    references_by_version[version_number] = {
                        "raw_references": existing_refs or [],
                        "reference_extraction_status": "source_unavailable",
                        "reference_files_scanned": 0,
                        "reference_extraction_errors": [],
                    }

            enrichment = s2_references.get(arxiv_id)
            for parsed_version in parsed_versions:
                version_number = parsed_version["version"]
                parsed_version.update(
                    references_by_version.get(
                        version_number,
                        {
                            "raw_references": [],
                            "reference_extraction_status": "failed",
                            "reference_files_scanned": 0,
                            "reference_extraction_errors": [],
                        },
                    )
                )
                source_version = version_by_number.get(version_number, {})
                parsed_version["s2_references"] = (
                    enrichment["s2_references"]
                    if enrichment is not None
                    else source_version.get("s2_references")
                )

            publication_rows = {
                "elements": elements,
                "hierarchy": hierarchy,
                "versions": parsed_versions,
            }
            for name, rows in publication_rows.items():
                stream = output_streams[name]
                for row in rows:
                    stream.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                stream.flush()
                counts[name] += len(rows)

            processed_arxiv_ids.add(arxiv_id)
            print(
                f"[PROCESSED] Paper {arxiv_id} "
                f"(versions: {len(parsed_versions)}, "
                f"elements: {len(elements)}, "
                f"hierarchy: {len(hierarchy)})"
            )

    finally:
        for stream in output_streams.values():
            if not stream.closed:
                stream.close()

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Build Milestone 2 elements, hierarchy, and versions incrementally."
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
            f"Milestone 1 input not found: {args.source}. "
            "Run the crawler first."
        )
    counts = parse_corpus(source_file=args.source, resume=not args.no_resume)
    print("\nMilestone 2 parsing complete:")
    for name, count in counts.items():
        print(f"  {name}: {count} rows")
    print(f"Output directory: {OUTPUT_FILES['elements'].parent}")


if __name__ == "__main__":
    main()
