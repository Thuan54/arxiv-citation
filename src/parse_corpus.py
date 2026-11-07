"""Build Milestone 2 elements, hierarchy, and unified BibTeX tables.

Run from the repository root after Milestone 1:

    python src/parse_corpus.py
"""
import json
import os
import tempfile
from pathlib import Path

from config import (
    ELEMENTS_FILE,
    HIERARCHY_FILE,
    JSONL_FILE,
    PARSED_VERSIONS_FILE,
    ROOT,
)
from hierarchy import parse_publication_sources
from reference_extractor import extract_raw_references


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


def parse_corpus(source_file=JSONL_FILE):
    temporary_files = {}
    output_streams = {}
    counts = {name: 0 for name in OUTPUT_FILES}
    try:
        for name, output_path in OUTPUT_FILES.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            )
            temporary_files[name] = Path(output.name)
            output_streams[name] = output

        for publication in _read_publications(Path(source_file)):
            arxiv_id = publication["arxiv_id"]
            versions = sorted(
                publication.get("versions", []),
                key=lambda item: int(
                    str(item.get("version", "v1")).lstrip("v")
                ),
            )
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
                    legacy_references = next(
                        (
                            item.get("raw_references", [])
                            for item in versions
                            if int(
                                str(item.get("version", "v1")).lstrip("v")
                            ) == version_number
                        ),
                        [],
                    )
                    references_by_version[version_number] = {
                        "raw_references": legacy_references,
                        "reference_extraction_status": "source_unavailable",
                        "reference_files_scanned": 0,
                        "reference_extraction_errors": [],
                    }

            for parsed_version in parsed_versions:
                parsed_version.update(
                    references_by_version.get(
                        parsed_version["version"],
                        {
                            "raw_references": [],
                            "reference_extraction_status": "failed",
                            "reference_files_scanned": 0,
                            "reference_extraction_errors": [],
                        },
                    )
                )
                source_version = next(
                    (
                        item for item in versions
                        if int(str(item.get("version", "v1")).lstrip("v"))
                        == parsed_version["version"]
                    ),
                    {},
                )
                parsed_version["s2_references"] = source_version.get(
                    "s2_references"
                )

            publication_rows = {
                "elements": elements,
                "hierarchy": hierarchy,
                "versions": parsed_versions,
            }
            for name, rows in publication_rows.items():
                for row in rows:
                    output_streams[name].write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                counts[name] += len(rows)

        for output in output_streams.values():
            output.close()
        for name, output_path in OUTPUT_FILES.items():
            os.replace(temporary_files[name], output_path)
    finally:
        for output in output_streams.values():
            if not output.closed:
                output.close()
        for path in temporary_files.values():
            path.unlink(missing_ok=True)
    return counts


def main():
    if not JSONL_FILE.exists():
        raise SystemExit(
            f"Milestone 1 input not found: {JSONL_FILE}. "
            "Run the crawler first."
        )
    counts = parse_corpus()
    print("Milestone 2 parsing complete:")
    for name, count in counts.items():
        print(f"  {name}: {count} rows")
    print(f"Output directory: {OUTPUT_FILES['elements'].parent}")


if __name__ == "__main__":
    main()
