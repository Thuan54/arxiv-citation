import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import state


class VersionStatusTest(unittest.TestCase):
    def test_metadata_and_source_status_are_written_to_record(self):
        with TemporaryDirectory() as temp:
            output_path = Path(temp) / "papers.jsonl"
            original_jsonl_file = state.JSONL_FILE
            state.JSONL_FILE = output_path
            arxiv_id = "status-test-paper"

            try:
                state.initialize_paper(arxiv_id, ["v1"])
                state.set_metadata(arxiv_id, "v1", None, "failed")
                state.set_source(
                    arxiv_id,
                    "v1",
                    None,
                    "success",
                )
                self.assertFalse(output_path.exists())
                state.set_references(arxiv_id, [], "failed")

                record = json.loads(output_path.read_text(encoding="utf-8"))
                version = record["versions"][0]
                self.assertEqual(version["metadata_status"], "failed")
                self.assertEqual(version["source_status"], "success")
                self.assertEqual(version["reference_status"], "failed")
                self.assertEqual(version["raw_references"], [])
            finally:
                state.JSONL_FILE = original_jsonl_file
                with state._lock:
                    state._papers.pop(arxiv_id, None)

    def test_semantic_scholar_references_survive_later_source_completion(self):
        with TemporaryDirectory() as temp:
            output_path = Path(temp) / "papers.jsonl"
            original_jsonl_file = state.JSONL_FILE
            state.JSONL_FILE = output_path
            arxiv_id = "semantic-reference-test"
            semantic_references = ['{"citedPaper":{"title":"Cited work"}}']

            try:
                state.initialize_paper(arxiv_id, ["v1"])
                state.set_metadata(
                    arxiv_id,
                    "v1",
                    {"title": "Citing work"},
                    "success",
                )
                state.set_references(
                    arxiv_id,
                    semantic_references,
                    "success",
                )
                state.set_source(
                    arxiv_id,
                    "v1",
                    None,
                    "success",
                )

                version = json.loads(
                    output_path.read_text(encoding="utf-8")
                )["versions"][0]
                self.assertEqual(
                    version["raw_references"],
                    semantic_references,
                )
                self.assertEqual(version["reference_status"], "success")
            finally:
                state.JSONL_FILE = original_jsonl_file
                with state._lock:
                    state._papers.pop(arxiv_id, None)


if __name__ == "__main__":
    unittest.main()
