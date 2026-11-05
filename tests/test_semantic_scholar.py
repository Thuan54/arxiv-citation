import json
import queue
import unittest
from unittest.mock import MagicMock, patch

from export_tables import extract_citation_arxiv_ids
from semantic_scholar import SemanticScholarReferenceWorker


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


class SemanticScholarReferenceTest(unittest.TestCase):
    def test_batch_fetches_references_for_multiple_papers(self):
        response_data = [
            {
                "paperId": "s2-citing-one",
                "referenceCount": 1,
                "references": [{
                    "paperId": "s2-reference-one",
                    "externalIds": {"ArXiv": "2407.00001"},
                    "title": "Cited work one",
                }],
            },
            {
                "paperId": "s2-citing-two",
                "referenceCount": 1,
                "references": [{
                    "paperId": "s2-reference-two",
                    "externalIds": {"ArXiv": "2408.00002"},
                    "title": "Cited work two",
                }],
            },
        ]
        session = FakeSession([FakeResponse(response_data)])
        client = SemanticScholarReferenceWorker()

        result = client.fetch_references_batch(
            ["2501.12345", "2502.12345"],
            session,
        )

        self.assertEqual(len(session.calls), 1)
        url, request = session.calls[0]
        self.assertEqual(url, "https://api.semanticscholar.org/graph/v1/paper/batch")
        self.assertEqual(
            request["json"]["ids"],
            ["ARXIV:2501.12345", "ARXIV:2502.12345"],
        )
        self.assertIn("references.externalIds", request["params"]["fields"])
        self.assertEqual(
            extract_citation_arxiv_ids(result["2501.12345"][0]),
            ["2407.00001"],
        )
        self.assertEqual(
            extract_citation_arxiv_ids(result["2502.12345"][0]),
            ["2408.00002"],
        )

    def test_fetches_all_pages_and_serializes_references(self):
        reference = {
            "citedPaper": {
                "paperId": "s2-reference-id",
                "externalIds": {"ArXiv": "2407.00001"},
                "title": "Cited work",
            },
            "contexts": ["Mentioned in the introduction."],
            "intents": ["background"],
            "isInfluential": False,
        }
        session = FakeSession([
            FakeResponse({"data": [reference], "next": 1000}),
            FakeResponse({"data": [], "next": None}),
        ])
        client = SemanticScholarReferenceWorker()

        with patch("semantic_scholar.time.sleep"):
            raw_references = client.fetch_references(
                "2501.12345",
                session,
            )

        self.assertEqual(len(session.calls), 2)
        self.assertEqual(
            session.calls[0][1]["params"]["offset"],
            0,
        )
        self.assertEqual(
            session.calls[1][1]["params"]["offset"],
            1000,
        )
        self.assertEqual(json.loads(raw_references[0]), reference)
        self.assertEqual(
            extract_citation_arxiv_ids(raw_references[0]),
            ["2407.00001"],
        )

    def test_worker_stores_references_and_stops_cleanly(self):
        ref_record = json.dumps({
            "citedPaper": {
                "paperId": "s2-reference-id",
                "externalIds": {"ArXiv": "2407.00001"},
                "title": "Cited work",
            },
        })
        task_queue = queue.Queue()
        task_queue.put({"arxiv_id": "2501.12345"})
        task_queue.put({"arxiv_id": "2502.12345"})
        task_queue.put(None)
        references_by_id = {
            "2501.12345": [ref_record],
            "2502.12345": [],
        }

        with (
            patch(
                "semantic_scholar.SemanticScholarReferenceWorker.fetch_references_batch",
                return_value=references_by_id,
            ) as fetch_batch_mock,
            patch("semantic_scholar.requests.Session") as session_class,
            patch("semantic_scholar.set_references") as set_references,
        ):
            session_class.return_value = MagicMock()
            from semantic_scholar import semantic_scholar_reference_worker

            semantic_scholar_reference_worker(task_queue)

        fetch_batch_mock.assert_called_once_with(
            ["2501.12345", "2502.12345"],
            session_class.return_value,
        )
        self.assertEqual(set_references.call_count, 2)
        set_references.assert_any_call(
            "2501.12345",
            [ref_record],
            "success",
        )
        set_references.assert_any_call("2502.12345", [], "success")
        self.assertEqual(task_queue.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()
