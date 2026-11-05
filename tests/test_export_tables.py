import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import export_tables


class TableExportTest(unittest.TestCase):
    def setUp(self):
        self.records = [{
            "arxiv_id": "2501.12345",
            "versions": [
                {
                    "version": "v1",
                    "metadata": {
                        "title": "Citing paper",
                        "authors": ["A. Author"],
                        "submission_date": "2025-01-03",
                        "updated_date": "2025-01-03",
                        "venue": None,
                    },
                    "raw_references": [
                        r"\bibitem{ref} Work at arXiv:2407.00001v2.",
                        r"\bibitem{duplicate} arXiv:2407.00001.",
                        r"\bibitem{other} doi:10.1234/example",
                    ],
                },
                {
                    "version": "v2",
                    "metadata": {
                        "title": "Citing paper",
                        "authors": ["A. Author", "B. Author"],
                        "submission_date": "2025-01-03",
                        "updated_date": "2025-02-10",
                        "venue": "Example Conference",
                    },
                    "raw_references": [
                        r"\bibitem{ref} Work at arXiv:2407.00001v2.",
                    ],
                },
            ],
        }]

    def test_build_paper_table_combines_version_metadata(self):
        rows = export_tables.build_papers_table(self.records)
        self.assertEqual(rows, [{
            "arxiv_id": "2501.12345",
            "title": "Citing paper",
            "authors": ["A. Author", "B. Author"],
            "submission_date": "2025-01-03",
            "revised_dates": ["2025-02-10"],
            "venue": "Example Conference",
        }])

    def test_reference_table_keeps_only_unique_arxiv_citations(self):
        papers = export_tables.build_papers_table(self.records)
        with (
            patch.object(export_tables, "_fetch_arxiv_metadata", return_value={
                "2407.00001": {
                    "title": "Cited paper",
                    "authors": ["C. Author"],
                    "submission_date": "2024-07-01",
                },
            }) as fetch_arxiv,
            patch.object(
                export_tables,
                "_fetch_semantic_scholar_ids",
                return_value={"2407.00001": "s2-paper-id"},
            ) as fetch_semantic_scholar,
        ):
            rows = export_tables.build_references_table(
                self.records,
                papers,
            )

        self.assertEqual(rows, [{
            "arxiv_id": "2501.12345",
            "ref_arxiv_id": "2407.00001",
            "ref_title": "Cited paper",
            "ref_authors": ["C. Author"],
            "ref_submission_date": "2024-07-01",
            "ref_s2_id": "s2-paper-id",
        }])
        fetch_arxiv.assert_called_once_with(["2407.00001"])
        fetch_semantic_scholar.assert_called_once_with(["2407.00001"])

    def test_export_writes_jsonl_rows_to_both_table_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source_path = root / "source.jsonl"
            source_path.write_text(
                json.dumps(self.records[0]) + "\n",
                encoding="utf-8",
            )
            papers_path = root / "tables" / "papers.jsonl"
            references_path = root / "tables" / "references.jsonl"
            with (
                patch.object(export_tables, "PAPERS_TABLE_FILE", papers_path),
                patch.object(
                    export_tables,
                    "REFERENCES_TABLE_FILE",
                    references_path,
                ),
                patch.object(export_tables, "_fetch_arxiv_metadata", return_value={
                    "2407.00001": {
                        "title": "Cited paper",
                        "authors": ["C. Author"],
                        "submission_date": "2024-07-01",
                    },
                }),
                patch.object(
                    export_tables,
                    "_fetch_semantic_scholar_ids",
                    return_value={"2407.00001": "s2-paper-id"},
                ),
            ):
                counts = export_tables.export_tables(source_path)

            self.assertEqual(counts, {"papers": 1, "references": 1})
            paper_row = json.loads(papers_path.read_text(encoding="utf-8"))
            reference_row = json.loads(
                references_path.read_text(encoding="utf-8")
            )
            self.assertEqual(paper_row["arxiv_id"], "2501.12345")
            self.assertEqual(reference_row["ref_s2_id"], "s2-paper-id")


if __name__ == "__main__":
    unittest.main()
