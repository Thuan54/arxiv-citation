"""Tolerant reader for ``papers.jsonl``.

A well-formed file has one JSON record per line. After a crash, a manual
``cat a.jsonl b.jsonl`` (first file without a trailing newline) or two crawler
processes writing at once, a line can hold several records or a cut-off
fragment, and a plain ``json.loads(line)`` then fails with "Extra data".
:func:`iter_records` recovers every complete record and reports the rest.
"""
import json

RECORD_START = '{"arxiv_id"'


def iter_records(path, report=None):
    """Yield every complete JSON record of `path`.

    `report` (a dict) is filled with:
      lines                number of non-empty lines
      multi_record_lines   line numbers that held more than one record
      corrupt              [(line_number, fragment)] text that is not a record
    """
    report = report if report is not None else {}
    report.setdefault("lines", 0)
    report.setdefault("multi_record_lines", [])
    report.setdefault("corrupt", [])
    decoder = json.JSONDecoder()

    with open(path, encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            text = line.strip()
            if not text:
                continue

            report["lines"] += 1
            pos, found = 0, 0

            while pos < len(text):
                try:
                    obj, end = decoder.raw_decode(text, pos)
                except json.JSONDecodeError:
                    # A cut-off record: keep it aside and resume at the next
                    # record start on the same line, if there is one.
                    nxt = text.find(RECORD_START, pos + 1)
                    stop = len(text) if nxt == -1 else nxt
                    report["corrupt"].append((line_no, text[pos:stop]))
                    pos = stop
                    continue

                if isinstance(obj, dict):
                    found += 1
                    yield obj
                else:
                    report["corrupt"].append((line_no, text[pos:end]))

                pos = end
                while pos < len(text) and text[pos].isspace():
                    pos += 1

            if found > 1:
                report["multi_record_lines"].append(line_no)


def describe(report) -> str:
    """One-paragraph summary of the problems found (empty if none)."""
    parts = []
    if report["multi_record_lines"]:
        lines = report["multi_record_lines"]
        parts.append(f"{len(lines)} line(s) held several records "
                     f"(first: line {lines[0]}); they are split on rewrite")
    if report["corrupt"]:
        parts.append(f"{len(report['corrupt'])} unreadable fragment(s) "
                     f"(first: line {report['corrupt'][0][0]}); saved to the .corrupt file")
    return "; ".join(parts)