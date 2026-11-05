import io
import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import extractor
import state
from config import ROOT


class SourceExtractionTest(unittest.TestCase):
    def test_source_extraction_does_not_read_bibliography(self):
        with TemporaryDirectory(
            prefix=".extractor-test-",
            dir=ROOT,
        ) as temp:
            temp_path = Path(temp)
            raw_dir = temp_path / "raw"
            arxiv_id = "process-test-paper"
            version = "v1"
            archive_path = temp_path / "source.tar.gz"
            tex_source = (
                r"\begin{thebibliography}{1}"
                r"\bibitem{key} Reference text"
                r"\end{thebibliography}"
            ).encode("utf-8")
            with tarfile.open(archive_path, "w:gz") as archive:
                tex_info = tarfile.TarInfo("main.tex")
                tex_info.size = len(tex_source)
                archive.addfile(tex_info, io.BytesIO(tex_source))
                image_info = tarfile.TarInfo("figure.png")
                image_info.size = 4
                archive.addfile(image_info, io.BytesIO(b"image"))

            original_jsonl_file = state.JSONL_FILE
            state.JSONL_FILE = temp_path / "papers.jsonl"
            state.initialize_paper(arxiv_id, [version])
            state.set_metadata(
                arxiv_id,
                version,
                {"title": "Test paper"},
                "success",
            )

            try:
                with patch.object(extractor, "RAW_DIR", raw_dir):
                    extractor.extraction_task(
                        {
                            "task": {
                                "arxiv_id": arxiv_id,
                                "version": version,
                                "version_id": f"{arxiv_id}{version}",
                            },
                            "tmp_path": archive_path,
                            "reused": False,
                        },
                        worker_id=0,
                    )

                state.set_references(
                    arxiv_id,
                    ['{"citedPaper":{"title":"Semantic Scholar citation"}}'],
                    "success",
                )
                record = json.loads(
                    state.JSONL_FILE.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    record["versions"][0]["raw_references"],
                    ['{"citedPaper":{"title":"Semantic Scholar citation"}}'],
                )
                output_dir = raw_dir / arxiv_id / version
                self.assertTrue((output_dir / ".complete").is_file())
                self.assertFalse((output_dir / "figure.png").exists())
            finally:
                state.JSONL_FILE = original_jsonl_file
                with state._lock:
                    state._papers.pop(arxiv_id, None)


if __name__ == "__main__":
    unittest.main()
