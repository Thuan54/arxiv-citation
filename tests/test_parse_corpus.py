import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import parse_corpus


class ParseCorpusTest(unittest.TestCase):
    def test_keeps_crawled_references_attached_to_their_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / "raw" / "2501.12345" / "v1"
            source_dir.mkdir(parents=True)
            (source_dir / "main.tex").write_text(
                r"""\documentclass{article}
\begin{document}
\section{Body}
The body text.
\end{document}
""",
                encoding="utf-8",
            )
            references_v1 = ['{"citedPaper":{"title":"First citation"}}']
            references_v2 = ['{"citedPaper":{"title":"Later citation"}}']
            input_path = root / "papers.jsonl"
            input_path.write_text(
                json.dumps({
                    "arxiv_id": "2501.12345",
                    "versions": [
                        {
                            "version": "v1",
                            "latex_source_path": str(source_dir),
                            "raw_references": references_v1,
                        },
                        {
                            "version": "v2",
                            "latex_source_path": None,
                            "raw_references": references_v2,
                        },
                    ],
                }) + "\n",
                encoding="utf-8",
            )
            output_paths = {
                name: root / f"{name}.jsonl"
                for name in ("elements", "hierarchy", "versions")
            }

            with patch.dict(
                parse_corpus.OUTPUT_FILES,
                output_paths,
                clear=True,
            ):
                counts = parse_corpus.parse_corpus(input_path)

            versions = [
                json.loads(line)
                for line in output_paths["versions"].read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                [version["raw_references"] for version in versions],
                [references_v1, references_v2],
            )
            self.assertEqual([version["status"] for version in versions], [
                "parsed",
                "empty",
            ])
            self.assertEqual(counts["versions"], 2)


if __name__ == "__main__":
    unittest.main()
