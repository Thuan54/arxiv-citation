"""Extract bibliography entries from downloaded arXiv source directories."""
import codecs
import re
from collections import Counter
from pathlib import Path


INCLUDE_BIBTEX_FALLBACK = True
TEX_SUFFIXES = {".tex", ".ltx"}
BBL_SUFFIX = ".bbl"
BIB_SUFFIX = ".bib"
TEXT_SUFFIXES = TEX_SUFFIXES | {BBL_SUFFIX, BIB_SUFFIX}

COMMENT_PATTERN = re.compile(r"(?<!\\)%.*")
BIBITEM_START = re.compile(r"\\bibitem(?![A-Za-z@])")
ITEM_START = re.compile(r"\\item(?![A-Za-z@])")
BIBLIOGRAPHY_END = re.compile(
    r"\\end\s*\{(?:thebibliography|references)\}"
)
THEBIBLIOGRAPHY_BLOCK = re.compile(
    r"\\begin\s*\{thebibliography\}.*?\\end\s*\{thebibliography\}",
    re.DOTALL,
)
BIBLATEX_ENTRY = re.compile(r"\\entry\s*\{.*?\\endentry", re.DOTALL)
BIB_ENTRY_START = re.compile(r"^[ \t]*@([A-Za-z]+)[ \t]*\{", re.MULTILINE)
SKIPPED_BIB_TYPES = {"comment", "string", "preamble"}
MACRO_DEFINITION = re.compile(
    r"\\(?:re|provide)?newcommand\s*\{?\s*$|\\(?:[gex]?def|let)\s*$"
)


def _skip_group(text, pos, open_char, close_char, limit=None):
    stop = len(text) if limit is None else min(len(text), pos + limit)
    depth = 0
    i = pos

    while i < stop:
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1

    return None


def _skip_label(text, pos, limit=1000):
    stop = min(len(text), pos + limit)
    braces = 0
    i = pos + 1

    while i < stop:
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == "{":
            braces += 1
        elif char == "}":
            braces = max(0, braces - 1)
        elif char == "]" and braces == 0:
            return i + 1
        i += 1

    return None


def _bibitem_header_end(text, start):
    pos = start + len(r"\bibitem")
    while pos < len(text) and text[pos].isspace():
        pos += 1

    if pos < len(text) and text[pos] == "[":
        pos = _skip_label(text, pos)
        if pos is None:
            return None
        while pos < len(text) and text[pos].isspace():
            pos += 1

    if pos < len(text) and text[pos] == "{":
        return _skip_group(text, pos, "{", "}", limit=1000)
    return None


def _extract_bibitems(text):
    starts = []
    for match in BIBITEM_START.finditer(text):
        before = text[max(0, match.start() - 40):match.start()]
        if MACRO_DEFINITION.search(before):
            continue
        if _bibitem_header_end(text, match.start()) is not None:
            starts.append(match.start())

    entries = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        stop = BIBLIOGRAPHY_END.search(text, start, end)
        if stop:
            end = stop.start()
        entry = text[start:end].strip()
        if entry:
            entries.append(entry)
    return entries


def _extract_plain_items(text):
    entries = []
    for block in THEBIBLIOGRAPHY_BLOCK.finditer(text):
        body = block.group(0)
        if BIBITEM_START.search(body):
            continue
        starts = [match.start() for match in ITEM_START.finditer(body)]
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(body)
            entry = BIBLIOGRAPHY_END.sub("", body[start:end]).strip()
            if entry:
                entries.append(entry)
    return entries


def _extract_biblatex(text):
    return [match.group(0).strip() for match in BIBLATEX_ENTRY.finditer(text)]


def _extract_bibtex(text):
    entries = []
    pos = 0

    while True:
        match = BIB_ENTRY_START.search(text, pos)
        if match is None:
            break
        if match.group(1).lower() in SKIPPED_BIB_TYPES:
            pos = match.end()
            continue
        end = _skip_group(text, match.end() - 1, "{", "}", limit=200_000)
        if end is None:
            pos = match.end()
            continue
        entries.append(text[match.start():end].strip())
        pos = end

    return entries


def _read_text(path):
    data = path.read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _candidate_files(directory):
    tex_files, bbl_files, bib_files = [], [], []
    total = 0

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        total += 1
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        suffix = path.suffix.lower()
        if suffix in TEX_SUFFIXES:
            tex_files.append(path)
        elif suffix == BBL_SUFFIX:
            bbl_files.append(path)
        elif suffix == BIB_SUFFIX:
            bib_files.append(path)
    return total, tex_files, bbl_files, bib_files


def extract_raw_references(directory, report=None):
    """Extract bibliography source records without modifying crawler state."""
    directory = Path(directory)
    total, tex_files, bbl_files, bib_files = _candidate_files(directory)
    references = []
    seen = set()
    kinds = Counter()
    read_failures = []

    def add(entries, kind):
        for entry in entries:
            key = re.sub(r"\s+", " ", entry)
            if key in seen:
                continue
            seen.add(key)
            references.append(entry)
            kinds[kind] += 1

    for path in bbl_files + tex_files:
        try:
            text = _read_text(path)
        except OSError as exc:
            read_failures.append(f"{path}: {exc}")
            continue
        text = COMMENT_PATTERN.sub("", text)
        add(_extract_bibitems(text), "bibitem")
        add(_extract_plain_items(text), "item")
        if path.suffix.lower() == BBL_SUFFIX:
            add(_extract_biblatex(text), "biblatex")

    if not references and INCLUDE_BIBTEX_FALLBACK:
        for path in bib_files:
            try:
                text = _read_text(path)
            except OSError as exc:
                read_failures.append(f"{path}: {exc}")
                continue
            add(_extract_bibtex(text), "bibtex")

    if report is not None:
        report.update({
            "total_files": total,
            "tex_files": len(tex_files),
            "bbl_files": len(bbl_files),
            "bib_files": len(bib_files),
            "kinds": dict(kinds),
            "read_failures": read_failures,
        })

    return references


__all__ = ["TEXT_SUFFIXES", "extract_raw_references"]
