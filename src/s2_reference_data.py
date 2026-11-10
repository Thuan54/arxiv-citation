"""Read Semantic Scholar enrichment rows stored separately from paper data."""
import json

from config import S2_REFERENCES_FILE


def load_s2_references(path=None):
    """Return enrichment rows keyed by arXiv ID, with later rows taking precedence."""
    if path is None:
        path = S2_REFERENCES_FILE

    data = {}
    if not path.exists():
        return data

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc
            data[row["arxiv_id"]] = row
    return data
