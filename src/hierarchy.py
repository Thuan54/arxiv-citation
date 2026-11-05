"""Parse LaTeX source into versioned, deduplicated hierarchy rows."""
import codecs
import re
from collections import defaultdict
from pathlib import Path


TEX_SUFFIXES = (".tex", ".ltx")
HEADING_LEVELS = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}
HEADING_COMMAND = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection|paragraph|"
    r"subparagraph)\*?"
)
ENVIRONMENT = re.compile(r"\\(begin|end)\s*\{([^{}]+)\}")
ITEM_COMMAND = re.compile(r"\\item\b(?:\s*\[[^\]]*\])?")
DISPLAY_MATH = re.compile(r"\$\$|\\\[")
INCLUDE_COMMAND = re.compile(
    r"\\(input|include)\s*(?:\{([^{}]+)\}|([^\s{}]+))"
)
REFERENCE_HEADING = re.compile(
    r"^(?:references|bibliography|works cited|literature cited)\b",
    re.IGNORECASE,
)
REFERENCE_ENVS = {"thebibliography", "bibliography", "references"}
LIST_ENVS = {"itemize", "enumerate", "description"}
FIGURE_ENVS = {"figure", "figure*", "table", "table*"}
MATH_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "displaymath", "flalign", "flalign*",
    "alignat", "alignat*", "eqnarray", "eqnarray*", "math",
}
PROTECTED_ENVIRONMENTS = REFERENCE_ENVS | FIGURE_ENVS | MATH_ENVS | {
    "tabular", "tabular*", "longtable",
}
ABBREVIATIONS = {
    "e.g.", "i.e.", "et al.", "etc.", "fig.", "eq.", "ref.", "sec.",
    "no.", "vs.", "dr.", "prof.", "mr.", "mrs.", "ms.", "st.",
}


def _read_text(path):
    data = path.read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def strip_comments(text):
    """Remove unescaped TeX comments while preserving line boundaries."""
    result = []
    for line in text.splitlines(keepends=True):
        i = 0
        while i < len(line):
            if line[i] == "%":
                backslashes = 0
                j = i - 1
                while j >= 0 and line[j] == "\\":
                    backslashes += 1
                    j -= 1
                if backslashes % 2 == 0:
                    newline = "\n" if line.endswith("\n") else ""
                    result.append(newline)
                    break
            result.append(line[i])
            i += 1
    return "".join(result)


def _balanced_group_end(text, opening):
    if opening >= len(text) or text[opening] != "{":
        return None
    depth = 0
    i = opening
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def find_document_root(directory):
    directory = Path(directory)
    candidates = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in TEX_SUFFIXES
    )
    roots = []
    for path in candidates:
        if re.search(r"\\documentclass\b", strip_comments(_read_text(path))):
            roots.append(path)
    if not roots:
        return None
    return next(
        (path for path in roots if path.name.lower() in {"main.tex", "paper.tex"}),
        roots[0],
    )


def gather_compilation_source(root_file):
    """Expand reachable input/include files in place; ignore unused TeX files."""
    root_file = Path(root_file).resolve()
    source_root = root_file.parent.resolve()
    loaded = set()

    def expand(path):
        path = Path(path).resolve()
        if path in loaded or not path.is_relative_to(source_root):
            return ""
        loaded.add(path)
        text = strip_comments(_read_text(path))
        include_only_match = re.search(
            r"\\includeonly\s*\{([^{}]*)\}", text
        )
        include_only = None
        if include_only_match:
            include_only = {
                item.strip().removesuffix(".tex").removesuffix(".ltx")
                for item in include_only_match.group(1).split(",")
                if item.strip()
            }
            text = (
                text[:include_only_match.start()]
                + text[include_only_match.end():]
            )
        pieces = []
        cursor = 0
        for match in INCLUDE_COMMAND.finditer(text):
            pieces.append(text[cursor:match.start()])
            command = match.group(1)
            requested = (match.group(2) or match.group(3)).strip()
            requested = requested.removesuffix(".tex").removesuffix(".ltx")
            if (
                command == "include"
                and include_only is not None
                and requested not in include_only
            ):
                cursor = match.end()
                continue
            child = (path.parent / requested)
            options = [child.with_suffix(suffix) for suffix in TEX_SUFFIXES]
            options.insert(0, child)
            include_path = next(
                (candidate for candidate in options if candidate.is_file()),
                None,
            )
            pieces.append(expand(include_path) if include_path else "")
            cursor = match.end()
        pieces.append(text[cursor:])
        return "".join(pieces)

    combined = expand(root_file)
    combined = re.sub(r"\\documentclass(?:\[[^\]]*\])?\s*\{[^{}]*\}", "", combined)
    combined = re.sub(r"\\begin\s*\{document\}", "", combined)
    combined = re.sub(r"\\end\s*\{document\}", "", combined)
    return combined


def _skip_space(text, pos):
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _command_argument(text, pos):
    pos = _skip_space(text, pos)
    if pos < len(text) and text[pos] == "[":
        end = text.find("]", pos + 1)
        if end >= 0:
            pos = _skip_space(text, end + 1)
    if pos < len(text) and text[pos] == "{":
        end = _balanced_group_end(text, pos)
        if end is not None:
            return text[pos + 1:end - 1], end
    return "", pos


def _environment_end(text, name, start):
    token = re.compile(
        r"\\(begin|end)\s*\{" + re.escape(name) + r"\}"
    )
    depth = 1
    for match in token.finditer(text, start):
        depth += 1 if match.group(1) == "begin" else -1
        if depth == 0:
            return match.start(), match.end()
    return len(text), len(text)


def _replace_math(text):
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: f"${match.group(1).strip()}$",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda match: f"\\begin{{equation}} {match.group(1).strip()} "
                      r"\end{equation}",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\$\$(.*?)\$\$",
        lambda match: f"\\begin{{equation}} {match.group(1).strip()} "
                      r"\end{equation}",
        text,
        flags=re.DOTALL,
    )
    return text


def clean_latex(text, preserve_math=False):
    text = strip_comments(text)
    if preserve_math:
        text = re.sub(r"\\label\s*\{[^{}]*\}", "", text)
        return re.sub(r"\s+", " ", text).strip()

    text = _replace_math(text)
    text = re.sub(r"\\par\b", "\n\n", text)
    math_blocks = []

    def hide_math(match):
        math_blocks.append(match.group(0))
        return f"ZZMATHBLOCK{len(math_blocks) - 1}ZZ"

    text = re.sub(
        r"\$(?:\\.|[^$])+\$|"
        r"\\begin\s*\{(?:equation|align|gather|multline|flalign|"
        r"alignat|eqnarray|displaymath)[^}]*\}"
        r".*?\\end\s*\{(?:equation|align|gather|multline|flalign|"
        r"alignat|eqnarray|displaymath)[^}]*\}",
        hide_math,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"\\(?:label|index|tag)\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:begin|end)\s*\{[^{}]+\}", "", text)
    text = re.sub(r"\\(?:centering|raggedright|raggedleft|noindent)\b", "", text)
    text = re.sub(r"\\(?:hline|toprule|midrule|bottomrule|item)\b", " ", text)
    text = re.sub(
        r"\\includegraphics\*?(?:\[[^\]]*\])?\s*\{[^{}]*\}",
        "",
        text,
    )
    text = re.sub(
        r"\\(?:textbf|textit|emph|textrm|textsf|texttt|textnormal)"
        r"\s*\{([^{}]*)\}",
        r"\1",
        text,
    )

    def replace_command(match):
        command = match.group(1)
        if command.startswith("cite") or command in {"nocite", "ref", "eqref"}:
            return match.group(0)
        return ""

    text = re.sub(
        r"\\([A-Za-z@]+)\*?",
        replace_command,
        text,
    )
    text = re.sub(r"\\([%#$&_{}])", r"\1", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    for index, math in enumerate(math_blocks):
        text = text.replace(f"ZZMATHBLOCK{index}ZZ", math)
    return text.strip()


def _split_sentences(text):
    """Split prose at sentence punctuation, protecting common non-final dots."""
    if not text:
        return []
    protected = []

    def hide(match):
        protected.append(match.group(0))
        return f"ZZPROTECTED{len(protected) - 1}ZZ"

    text = re.sub(
        r"\\(?:cite[a-zA-Z]*|nocite)\*?(?:\s*\[[^\]]*\]\s*)*"
        r"\{[^{}]*\}",
        hide,
        text,
    )
    text = re.sub(r"\$(?:\\.|[^$])+\$", hide, text)
    text = re.sub(r"https?://[^\s]*[A-Za-z0-9/#]", hide, text)
    boundaries = []
    for match in re.finditer(r"[.!?]+[\"')\]]*(?:\s+|$)", text):
        if text[match.start()] == ".":
            dotted_prefix = (
                text[:match.start()].rstrip() + "."
            ).lower()
            if any(dotted_prefix.endswith(item) for item in ABBREVIATIONS):
                continue
            if (
                match.start() > 0
                and match.end() < len(text)
                and text[match.start() - 1].isdigit()
                and text[match.end()].isdigit()
            ):
                continue
        if match.end() < len(text) and not text[match.end()].isupper() \
                and not text[match.end()].isdigit() \
                and text[match.end()] not in "\\\"'(":
            continue
        boundaries.append(match.end())

    sentences = []
    start = 0
    for end in boundaries:
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    for index, value in enumerate(protected):
        sentences = [
            sentence.replace(f"ZZPROTECTED{index}ZZ", value)
            for sentence in sentences
        ]
    return sentences


class _HierarchyParser:
    def __init__(self, source):
        self.source = source
        self.root = {"kind": "document", "content": "", "children": []}
        self.headings = []
        self.lists = []
        self.ignored_heading_level = None

    def _parent(self):
        if self.lists:
            return self.lists[-1].get("item") or self.lists[-1]["node"]
        return self.headings[-1][1] if self.headings else self.root

    @staticmethod
    def _add(parent, kind, content, children=None):
        node = {
            "kind": kind,
            "content": content,
            "children": [] if children is None else children,
        }
        parent["children"].append(node)
        return node

    def _paragraph(self, raw):
        if self.ignored_heading_level is not None:
            return
        cleaned = clean_latex(raw)
        if not cleaned:
            return
        parent = self._parent()
        for text in re.split(r"\n\s*\n", cleaned):
            text = text.strip()
            if not text:
                continue
            paragraph = self._add(parent, "paragraph", text)
            for sentence in _split_sentences(text):
                self._add(paragraph, "sentence", sentence)

    def parse(self):
        text = self.source
        cursor = 0
        prose_start = 0
        while cursor < len(text):
            heading = HEADING_COMMAND.search(text, cursor)
            environment = ENVIRONMENT.search(text, cursor)
            item_command = ITEM_COMMAND.search(text, cursor)
            display_math = DISPLAY_MATH.search(text, cursor)
            next_match = min(
                (
                    match for match in (
                        heading, environment, item_command, display_math
                    )
                    if match is not None
                ),
                key=lambda match: match.start(),
                default=None,
            )
            if next_match is None:
                self._paragraph(text[prose_start:])
                break

            self._paragraph(text[prose_start:next_match.start()])

            if next_match is item_command:
                cursor = item_command.end()
                prose_start = cursor
                if self.ignored_heading_level is None and self.lists:
                    label = ""
                    label_match = re.search(
                        r"\[([^\]]*)\]\s*$", item_command.group(0)
                    )
                    if label_match:
                        label = clean_latex(label_match.group(1))
                    item_node = self._add(
                        self.lists[-1]["node"],
                        "list_item",
                        label,
                    )
                    self.lists[-1]["item"] = item_node
                continue

            if next_match is display_math:
                opening = display_math.group(0)
                closing = "$$" if opening == "$$" else r"\]"
                content_start = display_math.end()
                content_end = text.find(closing, content_start)
                if content_end < 0:
                    content_end = len(text)
                    cursor = len(text)
                    prose_start = cursor
                else:
                    cursor = content_end + len(closing)
                    prose_start = cursor
                if self.ignored_heading_level is None:
                    content = clean_latex(
                        text[content_start:content_end],
                        preserve_math=True,
                    )
                    self._add(
                        self._parent(),
                        "formula",
                        f"\\begin{{equation}} {content} \\end{{equation}}",
                    )
                continue

            if next_match is heading:
                level = HEADING_LEVELS[heading.group(1)]
                title, end = _command_argument(text, heading.end())
                if end == heading.end():
                    cursor = heading.end()
                    prose_start = cursor
                    continue
                cursor = end
                prose_start = end
                while self.headings and self.headings[-1][0] >= level:
                    self.headings.pop()
                if (
                    self.ignored_heading_level is not None
                    and level <= self.ignored_heading_level
                ):
                    self.ignored_heading_level = None
                title = clean_latex(title)
                if REFERENCE_HEADING.match(title):
                    self.ignored_heading_level = level
                    continue
                if self.ignored_heading_level is not None:
                    continue
                node = self._add(self._parent(), "heading", title)
                self.headings.append((level, node))
                continue

            command, name = environment.group(1), environment.group(2).strip()
            if command == "end":
                if name in LIST_ENVS and self.lists:
                    self.lists.pop()
                cursor = environment.end()
                prose_start = cursor
                continue

            if name in REFERENCE_ENVS:
                _, cursor = _environment_end(text, name, environment.end())
                prose_start = cursor
                continue

            if name in PROTECTED_ENVIRONMENTS:
                content_start = environment.end()
                content_end, cursor = _environment_end(
                    text, name, content_start
                )
                raw_content = text[content_start:content_end]
                if self.ignored_heading_level is None:
                    if name in MATH_ENVS:
                        content = clean_latex(raw_content, preserve_math=True)
                        content = (
                            f"\\begin{{equation}} {content} "
                            r"\end{equation}"
                        )
                        self._add(self._parent(), "formula", content.strip())
                    else:
                        content = clean_latex(raw_content)
                        if content:
                            self._add(self._parent(), "figure", content)
                prose_start = cursor
                continue

            if name in LIST_ENVS:
                node = (
                    self._add(self._parent(), "list", "")
                    if self.ignored_heading_level is None else None
                )
                self.lists.append({
                    "node": node or {"children": []},
                    "item": None,
                })
                cursor = environment.end()
                prose_start = cursor
                continue

            if name in {"document"}:
                cursor = environment.end()
                prose_start = cursor
                continue
            cursor = environment.end()
            prose_start = cursor

        def set_container_text(node):
            for child in node["children"]:
                set_container_text(child)
            if node["kind"] in {"list", "list_item"}:
                child_content = " ".join(
                    child["content"] for child in node["children"]
                    if child["content"]
                )
                node["content"] = " ".join(
                    part for part in (node["content"], child_content) if part
                )

        set_container_text(self.root)
        return self.root


def parse_latex_source(source, version):
    source = strip_comments(source)
    source = re.sub(
        r"\\begin\s*\{abstract\}(.*?)\\end\s*\{abstract\}",
        r"\1",
        source,
        flags=re.DOTALL,
    )
    start = re.search(r"\\begin\s*\{document\}", source)
    if start:
        source = source[start.end():]
    end = re.search(r"\\end\s*\{document\}", source)
    if end:
        source = source[:end.start()]
    source = re.sub(
        r"\\(?:bibliography|bibliographystyle|addbibresource|"
        r"printbibliography)\s*"
        r"(?:\[[^\]]*\]\s*)?(?:\{[^{}]*\})+",
        "",
        source,
    )
    source = re.sub(
        r"\\printbibliography(?:\s*\[[^\]]*\])?",
        "",
        source,
    )
    source = re.sub(
        r"\\begin\s*\{(?:thebibliography|references)\}.*?"
        r"\\end\s*\{(?:thebibliography|references)\}",
        "",
        source,
        flags=re.DOTALL,
    )
    root = _HierarchyParser(source).parse()
    root["content"] = f"Document {version}"
    return root


def build_publication(arxiv_id, version_sources):
    """Build Milestone 2 tables for one arXiv publication.

    `version_sources` is an ordered list of (version string, source text or None).
    """
    per_version = []
    for version, source in version_sources:
        if source is None:
            per_version.append({
                "version": version,
                "tree": None,
                "status": "empty",
            })
            continue
        per_version.append({
            "version": version,
            "tree": parse_latex_source(source, version),
            "status": "parsed",
        })

    element_rows = []
    hierarchy_rows = []
    element_ids = {}
    id_counters = defaultdict(int)
    version_element_counts = defaultdict(int)

    def visit(node, version, parent_id=None):
        version_element_counts[version] += 1
        content = node["content"]
        identity = (node["kind"], content)
        if identity not in element_ids:
            id_counters[node["kind"]] += 1
            prefix = {
                "document": "root",
                "heading": "h",
                "paragraph": "p",
                "sentence": "s",
                "formula": "f",
                "figure": "g",
                "list": "l",
                "list_item": "i",
            }.get(node["kind"], "n")
            element_id = f"{arxiv_id.replace('/', '-')}#{prefix}{id_counters[node['kind']]}"
            element_ids[identity] = element_id
            element_rows.append({
                "element_id": element_id,
                "arxiv_id": arxiv_id,
                "kind": node["kind"],
                "content": content,
            })
        else:
            element_id = element_ids[identity]
        if parent_id is not None:
            edge = (version, element_id, parent_id)
            if edge not in seen_edges:
                seen_edges.add(edge)
                hierarchy_rows.append({
                    "arxiv_id": arxiv_id,
                    "version": int(version.lstrip("v")),
                    "element_id": element_id,
                    "parent_id": parent_id,
                })
        for child in node["children"]:
            visit(child, version, element_id)

    versions_rows = []
    seen_edges = set()
    for item in per_version:
        if item["tree"] is not None:
            visit(item["tree"], item["version"])
        versions_rows.append({
            "arxiv_id": arxiv_id,
            "version": int(item["version"].lstrip("v")),
            "status": item["status"],
            "element_count": version_element_counts[item["version"]],
        })

    return element_rows, hierarchy_rows, versions_rows


def parse_publication_sources(arxiv_id, version_directories):
    """Parse each version directory, retaining empty versions in coverage."""
    source_records = []
    for version, directory in version_directories:
        if directory is None or not Path(directory).is_dir():
            source_records.append((version, None))
            continue
        root = find_document_root(directory)
        if root is None:
            source_records.append((version, None))
            continue
        source = gather_compilation_source(root)
        source_records.append((version, source))
    return build_publication(arxiv_id, source_records)


__all__ = [
    "build_publication",
    "find_document_root",
    "gather_compilation_source",
    "parse_latex_source",
    "parse_publication_sources",
]
