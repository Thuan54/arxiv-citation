"""Turn raw bibliography entries into structured, standard BibTeX entries.

Input are the raw strings produced by Milestone 1 (``raw_references``): every
``\\bibitem`` found in the sources, or biblatex ``\\entry`` blocks taken from a
compiled ``.bbl`` file. The bibliography styles found on arXiv are very
different, so each entry is first routed to a parser for its style:

* ``biblatex``   - ``\\entry{key}{type}{}`` blocks written by biber
* ``structured`` - Springer-style ``\\begin{barticle}\\bauthor{...}`` macros
* ``bibinfo``    - ``\\bibinfo{title}{...}`` (revtex / Nature style)
* ``generic``    - plain text styles (plainnat, IEEE, APA/MLA, aastex, ...)

Every parser fills the same :class:`Entry` so the later deduplication step
does not depend on the original style.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from latex_text import (
    balanced_group, command_args, latex_to_text, skip_label,
)

MAX_YEAR = 2027
clean = latex_to_text

PARTICLES = {
    "van", "von", "der", "den", "de", "del", "della", "di", "da", "dos",
    "das", "du", "le", "la", "ten", "ter", "bin", "ibn", "al", "el", "dal",
}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Name:
    given: str
    family: str

    def display(self) -> str:
        return f"{self.given} {self.family}".strip()

    def bibtex(self) -> str:
        return f"{self.family}, {self.given}" if self.given else f"{{{self.family}}}"


@dataclass
class Entry:
    key: Optional[str]
    entry_type: str = "misc"
    fields: Dict[str, object] = field(default_factory=dict)
    authors: List[Name] = field(default_factory=list)
    editors: List[Name] = field(default_factory=list)
    et_al: bool = False
    raw: str = ""
    style: str = "generic"
    label: Optional[str] = None

    @property
    def year(self):
        return self.fields.get("year")

    def to_dict(self):
        return {
            "key": self.key,
            "entry_type": self.entry_type,
            "fields": dict(self.fields),
            "authors": [[a.given, a.family] for a in self.authors],
            "editors": [[a.given, a.family] for a in self.editors],
            "et_al": self.et_al,
            "style": self.style,
        }


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
_INITIAL = r"[A-ZÀ-ÖØ-Þ]{1,3}\.?"


def is_initials(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    parts = [p for p in re.split(r"[\s\-]+", token) if p]
    return all(
        re.fullmatch(_INITIAL, p) or re.fullmatch(r"[A-ZÀ-ÖØ-Þ]\.[A-ZÀ-ÖØ-Þ]\.?", p)
        for p in parts
    )


def name_from_natural(text: str) -> Name:
    tokens = text.split()
    if len(tokens) <= 1:
        return Name("", text.strip())
    i = len(tokens) - 1
    while i > 1 and tokens[i - 1].lower() in PARTICLES:
        i -= 1
    return Name(" ".join(tokens[:i]), " ".join(tokens[i:]))


def _strip_name(text: str) -> str:
    """Strip separators; keep the final period when it ends an initial."""
    text = text.strip(" ,;")
    if text.endswith(".") and not re.search(r"(?:^|[\s.\-])[A-ZÀ-ÖØ-Þ]{1,3}\.$", text) \
            and not re.search(r"\b(?:Jr|Sr)\.$", text):
        text = text.rstrip(".")
    return text.strip(" ,;")


def name_from_string(text: str) -> Optional[Name]:
    text = _strip_name(text)
    if not text:
        return None
    if "," in text:
        last, first = text.split(",", 1)
        first = first.strip()
        if re.fullmatch(r"(?:Jr|Sr|II|III|IV)\.?", first):   # "Smith, Jr."
            return Name("", f"{last.strip()} {first}")
        return Name(first, last.strip())
    return name_from_natural(text)


def _split_comma_part(part: str) -> List[str]:
    tokens = [t.strip() for t in part.split(",") if t.strip()]
    if len(tokens) <= 1:
        return [part.strip()]

    evens, odds = tokens[0::2], tokens[1::2]
    # "Last, F., Last2, F2."
    if len(tokens) % 2 == 0 and all(map(is_initials, odds)) and not any(map(is_initials, evens)):
        return [f"{tokens[i]}, {tokens[i + 1]}" for i in range(0, len(tokens), 2)]
    # "Last, First, First2 Last2, ..." (MLA)
    if len(tokens) >= 3 and " " not in tokens[0] and " " not in tokens[1] \
            and all(" " in t for t in tokens[2:]):
        return [f"{tokens[0]}, {tokens[1]}"] + tokens[2:]
    # "Last, First"
    if len(tokens) == 2 and " " not in tokens[1] and not is_initials(tokens[0]):
        return [f"{tokens[0]}, {tokens[1]}"]
    return tokens


def parse_authors(raw: str) -> Tuple[List[Name], bool]:
    """Parse an author list in any common style -> (names, truncated?)."""
    text = clean(raw)
    text = _strip_name(re.sub(r"^\s*(?:by|edited by)\s+", "", text, flags=re.I))

    et_al = False
    m = re.search(r"(?:[,;]?\s*(?:and\s+)?(?:et\.?\s*al\.?|others))\s*$", text, re.I)
    if m:
        et_al, text = True, _strip_name(text[:m.start()])

    if not text:
        return [], et_al

    if ";" in text:
        parts = [re.sub(r"^(?:and|&)\s+", "", p.strip()) for p in text.split(";")]
        pieces = [q for p in parts for q in ([p] if "," in p and len(p.split(",")) == 2 else _split_comma_part(p))]
    else:
        parts = re.split(r"\s*,?\s+(?:and|&)\s+", text)
        pieces = [q for p in parts for q in _split_comma_part(p)]

    names = [n for n in (name_from_string(p) for p in pieces) if n and n.family]
    return names, et_al


# --------------------------------------------------------------------------
# Identifiers, years, numbers
# --------------------------------------------------------------------------
_DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>\\{}]+')
_ARXIV_NEW = re.compile(
    r"(?i)(?:arxiv\s*[:/]?\s*|arxiv\.org/(?:abs|pdf)/|abs/)(\d{4}\.\d{4,5})(?:v\d+)?")
_ARXIV_OLD = re.compile(
    r"(?i)(?:arxiv\s*:?\s*|arxiv\.org/(?:abs|pdf)/|abs/)([a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?")
_URL_RE = re.compile(r"(https?://[^\s}\\]+)")


def find_doi(raw: str) -> Optional[str]:
    s = raw.replace("\\_", "_").replace("\\%", "%")
    for m in _DOI_RE.finditer(s):
        doi = m.group(0).rstrip(".,;:")
        while doi.endswith(")") and doi.count(")") > doi.count("("):
            doi = doi[:-1]
        return doi.lower()
    return None


def find_arxiv(raw: str) -> Optional[str]:
    for pattern in (_ARXIV_NEW, _ARXIV_OLD):
        m = pattern.search(raw)
        if m:
            return m.group(1).lower() if "/" in m.group(1) else m.group(1)
    return None


def find_url(raw: str) -> Optional[str]:
    for pattern in (r"\\b?url\s*\{([^{}]+)\}", r"\\href\s*\{([^{}]+)\}"):
        m = re.search(pattern, raw)
        if m:
            return m.group(1).replace("\\_", "_").replace("\\%", "%").strip().rstrip(".,;")
    m = _URL_RE.search(raw.replace("\\_", "_").replace("\\%", "%"))
    return m.group(1).rstrip(".,;)") if m else None


def _strip_ids(text: str, ranges: bool = True) -> str:
    text = _DOI_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _ARXIV_NEW.sub(" ", text)
    text = _ARXIV_OLD.sub(" ", text)
    text = re.sub(r"(?i)\[?\s*accessed[^\]\[]*?(?:\]|\.\s|\.$|$)", " ", text)
    return re.sub(r"\d+\s*[-–—]+\s*\d+", " ", text) if ranges else text


def find_year(text: str) -> Optional[int]:
    text = _strip_ids(text)
    paren = re.findall(r"\(\s*((?:1[5-9]|20)\d\d)[a-z]?\s*\)", text)
    candidates = paren or re.findall(r"(?<![\d.:/-])((?:1[5-9]|20)\d\d)(?!\d)", text)
    years = [int(y) for y in candidates if int(y) <= MAX_YEAR]
    return years[-1] if years else None


_PAGES = r"[A-Za-z]?\d+(?:\s*[-–—]+\s*[A-Za-z]?\d+)?"


def parse_numbers(text: str) -> Dict[str, str]:
    """volume / number / pages from the tail of a reference."""
    out: Dict[str, str] = {}
    m = re.search(r"\(\s*(?:1[5-9]|20)\d\d[a-z]?\s*\)\s*:\s*(" + _PAGES + ")", text)
    if m:
        out["pages"] = m.group(1)
    s = re.sub(r"\(\s*(?:1[5-9]|20)\d\d[a-z]?\s*\)", " ", _strip_ids(text, ranges=False))

    m = re.search(r"(\d+)\s*\(\s*(\d+[A-Za-z\-]*)\s*\)\s*[:,]?\s*(" + _PAGES + ")", s)
    if m:
        out["volume"], out["number"], out["pages"] = m.group(1), m.group(2), m.group(3)
    else:
        m = re.search(r"\bvol(?:ume|\.)?\s*\.?\s*(\d+)", s, re.I)
        if m:
            out["volume"] = m.group(1)
        m = re.search(r"\b(?:no|number|issue)\b\.?\s*(\d+)", s, re.I)
        if m:
            out["number"] = m.group(1)
        m = re.search(r"\bpp?\b\.?\s*(" + _PAGES + ")", s, re.I) or \
            re.search(r"\bpages?\s+(" + _PAGES + ")", s, re.I)
        if m and "pages" not in out:
            out["pages"] = m.group(1)
        if "volume" not in out and "pages" not in out:
            m = re.search(r"(?<![\w])(\d{1,4})\s*:\s*(" + _PAGES + ")", s)
            if m:
                out["volume"], out["pages"] = m.group(1), m.group(2)
        if "volume" not in out and "pages" not in out:
            m = re.search(r"(?:^|,\s*)(\d{1,4})\s*,\s*([A-Za-z]?\d+(?:\s*[-–—]+\s*\d+)?)\s*(?:$|\.|\()", s.strip())
            if m:
                out["volume"], out["pages"] = m.group(1), m.group(2)

    if "pages" in out:
        out["pages"] = re.sub(r"\s*[-–—]+\s*", "--", out["pages"])
    return out


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def tidy(text: Optional[str], title: bool = False) -> Optional[str]:
    if not text:
        return None
    text = text.strip(" \t\n,;:")
    if title:
        text = text.rstrip(".,; ")
    elif text.endswith(".") and not re.search(r"[A-Za-z]\.\s", text):
        # "Journal of the Franklin Institute." -> drop; "J. Funct. Anal." -> keep
        if len(text.rstrip(".").split()[-1]) >= 5:
            text = text[:-1]
    return text or None


_JOURNAL_WORDS = re.compile(
    r"\b(journal|j\.|transactions|trans\.|letters|lett\.|review|reviews|rev\.|annals|"
    r"magazine|physica|nature|science|bulletin|reports|acta|communications|comm\.|"
    r"plos|physics|chemistry|biology|mathematics)\b", re.I)
_CONF_WORDS = re.compile(
    r"\b(conference|proceedings|symposium|workshop|meeting|congress|colloquium|proc\.(?!\s*(?:natl|r\.|roy))|"
    r"neurips|nips|icml|iclr|cvpr|iccv|eccv|acl|emnlp|naacl|aaai|ijcai|kdd|sigir|icassp|"
    r"interspeech|neural information processing)\b", re.I)
_ARXIV_VENUE = re.compile(r"(?i)\b(arxiv|corr|preprint|biorxiv|medrxiv|ssrn)\b")


def infer_type(f: Dict[str, object], thesis_text: str = "") -> str:
    venue = f"{f.get('journal') or ''} {f.get('booktitle') or ''} {f.get('howpublished') or ''}"
    if re.search(r"(?i)ph\.?\s?d\.?\s+thesis|doctoral", thesis_text):
        return "phdthesis"
    if re.search(r"(?i)master'?s?\s+thesis", thesis_text):
        return "mastersthesis"
    if re.search(r"(?i)technical report|tech\.? rep", thesis_text):
        return "techreport"
    if f.get("eprint") and (not f.get("journal") or _ARXIV_VENUE.search(venue)) and not f.get("booktitle"):
        return "misc"
    if f.get("booktitle"):
        return "inproceedings"
    if f.get("journal"):
        return "article"
    if f.get("publisher") and f.get("title"):
        return "book"
    return "misc"


def _assign_venue(f: Dict[str, object], name: str, force: Optional[str] = None):
    name = tidy(name)
    if not name:
        return
    kind = force
    if kind is None:
        kind = "booktitle" if _CONF_WORDS.search(name) and not _JOURNAL_WORDS.search(name) else "journal"
    f[kind] = name


# --------------------------------------------------------------------------
# Header handling
# --------------------------------------------------------------------------
_NOISE = re.compile(
    r"\\endbibitem|\\BibitemOpen|\\BibitemShut\s*\{[^}]*\}|\\BIBentry\w+|"
    r"\\natexlab\{[^}]*\}|\\unskip|\\relax|\\penalty\s*\d+|\\protect|\\bibnamefont|"
    r"\\bibfnamefont|\\citenamefont|\\providecommand\{[^}]*\}\[\d\]\{[^}]*\}")


def split_header(raw: str):
    """-> (key, label, body) of a ``\\bibitem[label]{key}`` entry."""
    s = raw.strip()
    if not s.startswith("\\bibitem"):
        return None, None, s

    pos = len("\\bibitem")
    while pos < len(s) and s[pos].isspace():
        pos += 1

    label = None
    if pos < len(s) and s[pos] == "[":
        end = skip_label(s, pos)
        if end:
            label, pos = s[pos + 1:end - 1], end
            while pos < len(s) and s[pos].isspace():
                pos += 1

    key, body = None, s[pos:]
    if pos < len(s) and s[pos] == "{":
        end = balanced_group(s, pos)
        if end:
            key = re.sub(r"[\s%]+", "", s[pos + 1:end - 1]) or None
            body = s[end:]
    return key, label, body


def _label_year(label: Optional[str]) -> Optional[int]:
    if not label:
        return None
    m = re.search(r"\(\s*((?:1[5-9]|20)\d\d)[a-z]?\s*\)", label) or \
        re.search(r"\}\s*\{\s*((?:1[5-9]|20)\d\d)[a-z]?\s*\}\s*$", label)
    return int(m.group(1)) if m else None


def _label_surname(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    text = clean(re.sub(r"\\citeauthoryear\s*\{([^{}]*)\}.*", r"\1", label, flags=re.S))
    text = re.split(r"\(|\bet\.?\s*al|\band\b|&|,", text)[0].strip()
    return text or None


# --------------------------------------------------------------------------
# Style 1: biblatex  \entry{key}{type}{}
# --------------------------------------------------------------------------
_BIBLATEX_TYPES = {
    "article": "article", "inproceedings": "inproceedings", "book": "book",
    "mvbook": "book", "collection": "book", "incollection": "incollection",
    "inbook": "inbook", "proceedings": "proceedings", "report": "techreport",
    "techreport": "techreport", "thesis": "phdthesis", "phdthesis": "phdthesis",
    "mastersthesis": "mastersthesis", "unpublished": "unpublished",
    "online": "misc", "electronic": "misc", "misc": "misc", "manual": "manual",
}


def _biblatex_names(block: str) -> List[Name]:
    names = []
    starts = [m.start() for m in re.finditer(r"family=\{", block)]
    for i, start in enumerate(starts):
        seg = block[start:starts[i + 1] if i + 1 < len(starts) else len(block)]

        def val(label):
            m = re.search(label + r"=\{", seg)
            if not m:
                return ""
            end = balanced_group(seg, m.end() - 1)
            return clean(seg[m.end():end - 1]) if end else ""

        family = val("family")
        if family:
            suffix = val("suffix")
            names.append(Name(val("given"), f"{family} {suffix}".strip()))
    return names


def parse_biblatex(raw: str) -> Entry:
    m = re.match(r"\s*\\entry\s*\{([^}]*)\}\s*\{([^}]*)\}", raw)
    key, btype = (m.group(1).strip() or None, m.group(2).strip().lower()) if m else (None, "misc")
    entry = Entry(key=key, raw=raw, style="biblatex")
    vals: Dict[str, str] = {}

    for m in re.finditer(r"\\field\s*\{(\w+)\}\s*\{", raw):
        end = balanced_group(raw, m.end() - 1)
        if end:
            vals.setdefault(m.group(1), raw[m.end():end - 1])

    for m in re.finditer(r"\\verb\s*\{(\w+)\}\s*\\verb\s+(.*?)\s*\\endverb", raw, re.S):
        vals.setdefault(m.group(1), m.group(2).strip())

    for m in re.finditer(r"\\list\s*\{(\w+)\}\s*\{\d+\}\s*\{", raw):
        end = balanced_group(raw, m.end() - 1)
        if end:
            items = [clean(raw[a + 1:b - 1]) for a, b in _groups(raw[m.end():end - 1], m.end())]
            vals.setdefault(m.group(1), ", ".join(i for i in items if i))

    for role, target in (("author", "authors"), ("editor", "editors")):
        m = re.search(r"\\name\s*\{" + role + r"\}\s*\{\d+\}\s*\{[^{}]*\}\s*\{", raw)
        if m:
            end = balanced_group(raw, m.end() - 1)
            if end:
                setattr(entry, target, _biblatex_names(raw[m.end():end - 1]))

    f = entry.fields
    f["title"] = tidy(clean(vals.get("title", "")), title=True)
    f["journal"] = tidy(clean(vals.get("journaltitle") or vals.get("journal") or ""))
    f["booktitle"] = tidy(clean(vals.get("booktitle", "")))
    f["publisher"] = tidy(clean(vals.get("publisher", "")))
    f["address"] = tidy(clean(vals.get("location") or vals.get("address") or ""))
    f["volume"] = tidy(clean(vals.get("volume", "")))
    f["number"] = tidy(clean(vals.get("number", "")))
    f["series"] = tidy(clean(vals.get("series", "")))
    f["pages"] = tidy(clean(vals.get("pages", "").replace("\\bibrangedash", "--"))) 
    if f["pages"]:
        f["pages"] = re.sub(r"\s*[-–—]+\s*", "--", f["pages"])
    f["doi"] = find_doi(vals.get("doi", "")) or find_doi(raw)
    f["url"] = (vals.get("url") or vals.get("urlraw") or "").strip() or None
    f["eprint"] = find_arxiv(f"arXiv:{vals['eprint']}") if vals.get("eprint") and \
        "arxiv" in vals.get("eprinttype", "arxiv").lower() else None
    f["year"] = find_year(vals.get("year") or vals.get("date") or "") if (vals.get("year") or vals.get("date")) else None
    f["note"] = tidy(clean(vals.get("note", "")))

    entry.entry_type = _BIBLATEX_TYPES.get(btype, "misc")
    if btype == "thesis" and "master" in vals.get("type", "").lower():
        entry.entry_type = "mastersthesis"
    return entry


def _groups(text: str, offset: int = 0):
    """Spans of the top-level brace groups in `text`."""
    spans, i = [], 0
    while i < len(text):
        if text[i] == "{":
            end = balanced_group(text, i)
            if not end:
                break
            spans.append((offset + i, offset + end))
            i = end
        else:
            i += 1
    return spans


# --------------------------------------------------------------------------
# Style 2: Springer  \begin{barticle} \bauthor{\bsnm{..},\binits{..}} ...
# --------------------------------------------------------------------------
_B_TYPES = {
    "article": "article", "book": "book", "inproceedings": "inproceedings",
    "conference": "inproceedings", "incollection": "incollection",
    "chapter": "incollection", "inbook": "inbook", "phdthesis": "phdthesis",
    "mastersthesis": "mastersthesis", "techreport": "techreport",
    "misc": "misc", "other": "misc", "unpublished": "unpublished",
    "proceedings": "proceedings", "manual": "manual",
}


def _b_names(body: str, cmd: str) -> List[Name]:
    names = []
    for arg in command_args(body, cmd):
        snm = command_args(arg, "bsnm")
        if not snm:
            text = clean(arg)
            if text:
                names.append(Name("", text))
            continue
        given = (command_args(arg, "bfnm") or command_args(arg, "binits") or [""])[0]
        particle = (command_args(arg, "bparticle") or [""])[0]
        suffix = (command_args(arg, "bsuffix") or [""])[0]
        family = " ".join(x for x in (clean(particle), clean(snm[0]), clean(suffix)) if x)
        names.append(Name(clean(given), family))
    return names


def parse_structured(body: str, raw: str) -> Entry:
    env = re.search(r"\\begin\s*\{b(\w+)\}", body)
    btype = _B_TYPES.get(env.group(1).lower(), "misc") if env else "misc"
    entry = Entry(key=None, raw=raw, style="structured")
    entry.authors, entry.editors = _b_names(body, "bauthor"), _b_names(body, "beditor")

    def arg(name):
        found = command_args(body, name)
        return clean(found[0]) if found else ""

    f = entry.fields
    atitle, btitle, jtitle = arg("batitle"), arg("bbtitle"), arg("bjtitle")
    if btype in ("article", "misc", "unpublished"):
        f["title"] = tidy(atitle or btitle, title=True)
        f["journal"] = tidy(jtitle)
    elif btype in ("inproceedings", "incollection", "inbook"):
        f["title"] = tidy(atitle or arg("bctitle") or btitle, title=True)
        f["booktitle"] = tidy(btitle if atitle else "")
    else:
        f["title"] = tidy(btitle or atitle, title=True)
    f["volume"], f["number"] = tidy(arg("bvolume")), tidy(arg("bissue"))
    first, last = arg("bfpage"), arg("blpage")
    f["pages"] = f"{first}--{last}" if first and last else (first or None)
    f["year"] = find_year(arg("byear")) if arg("byear") else None
    f["publisher"], f["address"] = tidy(arg("bpublisher")), tidy(arg("blocation"))
    f["series"], f["note"] = tidy(arg("bsertitle")), tidy(arg("bcomment"))
    f["doi"], f["url"] = find_doi(raw), find_url(raw)
    f["eprint"] = find_arxiv(raw)
    entry.entry_type = btype
    return entry


# --------------------------------------------------------------------------
# Style 3: \bibinfo{title}{...}
# --------------------------------------------------------------------------
def parse_bibinfo(body: str, raw: str) -> Entry:
    entry = Entry(key=None, raw=raw, style="bibinfo")
    vals: Dict[str, List[str]] = {}

    for m in re.finditer(r"\\bib(?:info|field)\s*\{(\w+)\}\s*\{", body):
        end = balanced_group(body, m.end() - 1)
        if end:
            vals.setdefault(m.group(1), []).append(body[m.end():end - 1])

    def first(name):
        return vals.get(name, [""])[0]

    people = [v for v in vals.get("author", []) if v.strip()]
    # \bibfield{author}{\bibinfo{author}{..}} nests: keep the innermost text.
    people = [v for v in people if "\\bibinfo" not in v] or people
    if people:
        entry.authors, entry.et_al = parse_authors(" and ".join(people))
    eds = [v for v in vals.get("editor", []) if "\\bibinfo" not in v]
    if eds:
        entry.editors, _ = parse_authors(" and ".join(eds))

    f = entry.fields
    f["title"] = tidy(clean(first("title")), title=True)
    f["journal"] = tidy(clean(first("journal")))
    f["booktitle"] = tidy(clean(first("booktitle")))
    f["publisher"] = tidy(clean(first("publisher")))
    f["address"] = tidy(clean(first("address")))
    f["volume"], f["number"] = tidy(clean(first("volume"))), tidy(clean(first("number")))
    pages = clean(first("pages"))
    f["pages"] = re.sub(r"\s*[-–—]+\s*", "--", pages) if pages else None
    f["year"] = find_year(first("year")) if first("year") else None
    f["note"] = tidy(clean(first("note")))
    f["howpublished"] = tidy(clean(first("howpublished")))
    f["doi"], f["url"], f["eprint"] = find_doi(raw), find_url(raw), find_arxiv(raw)
    if not f["year"]:
        f["year"] = find_year(clean(body))
    entry.entry_type = infer_type(f, clean(body))
    return entry


# --------------------------------------------------------------------------
# Style 4: generic plain text
# --------------------------------------------------------------------------
_EMPH_START = re.compile(r"\\(?:emph|textit|textsl)\s*\{|\{\\(?:em|it|sl)\s")
_QUOTES = [
    re.compile(r"``(.+?)(?:\\?'\\?'|(?<!\\)\"|”)", re.S),
    re.compile(r"“(.+?)”", re.S),
    re.compile(r'(?<!\\)"(.+?)(?<!\\)"', re.S),
    re.compile(r"\\textquotedblleft\s*(.+?)\\textquotedblright", re.S),
]


def _first_emph(text: str):
    """(start, end, inner_raw) of the first emphasised group."""
    m = _EMPH_START.search(text)
    if not m:
        return None
    brace = m.end() - 1 if text[m.end() - 1] == "{" else m.start()
    end = balanced_group(text, brace)
    if not end:
        return None
    inner = text[brace + 1:end - 1] if text[m.end() - 1] == "{" else text[m.end():end - 1]
    return m.start(), end, inner


def _venue_from_text(entry: Entry, venue_raw: str, is_book: bool = False):
    f = entry.fields
    venue_raw = re.sub(r"\\(?:url|doi)\s*\{[^{}]*\}", " ", venue_raw)
    venue_raw = _DOI_RE.sub(" ", _URL_RE.sub(" ", venue_raw))
    venue_raw = re.sub(r"(?i)\b(?:doi|url)\s*:\s*|\[?\s*online\s*\]?\.?\s*(?:available\s*:)?", " ", venue_raw)
    venue_raw = venue_raw.strip()
    if not venue_raw:
        return

    plain = clean(venue_raw)
    emph = _first_emph(venue_raw)
    before = clean(venue_raw[:emph[0]]) if emph else ""
    in_prefix = bool(re.match(r"(?i)^in\b", before or plain))

    if emph:
        name = clean(emph[2]).strip(" .,")
        after = clean(venue_raw[emph[1]:])
        if in_prefix:
            _assign_venue(f, name, "booktitle")
        elif is_book:
            f["publisher"] = f.get("publisher") or tidy(re.split(r",\s*(?:\d|\()", after)[0].strip(" .,") or None)
        else:
            _assign_venue(f, name)
        numbers_src = after if not is_book else plain
    else:
        numbers_src = plain
        text = re.sub(r"(?i)^in\s+", "", plain)
        head = re.split(r",\s*(?:pp?\.|pages|vol|volume|no\.|number|\d)|\s*\(\s*\d{4}", text)[0].strip(" .,")
        m = re.match(r"^([A-Z][\w .\-]+):\s*(.+?)(?:,\s*(?:19|20)\d\d)?\.?$", head)
        if in_prefix:
            _assign_venue(f, head.split(". ")[0], "booktitle")
        elif _CONF_WORDS.search(head) or _JOURNAL_WORDS.search(head):
            _assign_venue(f, head.split(". ")[0])
        elif m:
            f["address"], f["publisher"] = tidy(m.group(1)), tidy(m.group(2))
        elif head and not re.search(r"\d", head) and len(head.split()) <= 6 and not is_book:
            # "Springer", "Wiley", ... or the start of "Journal Name, 12, 345"
            parts = [p.strip() for p in plain.split(",")]
            tail = re.sub(r"\(\s*\d{4}[a-z]?\s*\)", "", parts[1]).strip(" .") if len(parts) > 1 else ""
            abbreviations = re.findall(r"\b[A-Z][A-Za-z]{0,6}\.", parts[0])
            if re.fullmatch(_PAGES, tail) or len(abbreviations) >= 2:
                _assign_venue(f, parts[0], "journal")    # "Indiana Univ. Math. J., 1853--1886"
            else:
                f["publisher"] = tidy(head)
        elif is_book:
            f["publisher"] = tidy(head)

    if not f.get("publisher"):
        m = re.search(r"(?:^|\.\s+)([A-Z][\w .\-]{1,40}):\s*([^,:]+?)(?:,\s*(?:19|20)\d\d)?\.?\s*$", plain)
        if m:
            f["address"] = f.get("address") or tidy(m.group(1))
            f["publisher"] = tidy(m.group(2))

    for k, v in parse_numbers(numbers_src).items():
        f.setdefault(k, v)


def parse_generic(body: str, raw: str, label: Optional[str]) -> Entry:
    entry = Entry(key=None, raw=raw, style="generic")
    f = entry.fields
    text = _NOISE.sub(" ", body)
    text = re.sub(r"\{\s*\}", "", text).strip()           # braces left by \natexlab{a} etc.
    f["doi"], f["url"], f["eprint"] = find_doi(text), find_url(text), find_arxiv(text)

    blocks = [b.strip() for b in re.split(r"\\newblock\b", text) if b.strip()]
    year_hint = None

    if len(blocks) >= 2:                                   # plainnat-like
        authors_raw, title_raw, rest = blocks[0], blocks[1], blocks[2:]
        m = re.search(r"[\s,]+\(?((?:1[5-9]|20)\d\d)[a-z]?\)?\s*\.?\s*$", authors_raw)
        if m:
            year_hint, authors_raw = int(m.group(1)), authors_raw[:m.start()]
        entry.authors, entry.et_al = parse_authors(authors_raw)

        whole = re.fullmatch(
            r"\s*(?:\{\\(?:em|it|sl)\s(.*)\}|\\(?:emph|textit|textsl)\s*\{(.*)\})\s*[.,]?\s*", title_raw, re.S)
        is_book = False
        if whole:
            title_raw, is_book = whole.group(1) or whole.group(2), True
        else:
            e = _first_emph(title_raw)
            if e and e[0] > 10:
                rest, title_raw = [title_raw[e[0]:]] + rest, title_raw[:e[0]]
        f["title"] = tidy(clean(title_raw), title=True)
        _venue_from_text(entry, " ".join(rest), is_book)

    else:
        one = blocks[0] if blocks else ""
        quote = None
        for pattern in _QUOTES:                            # IEEE / MLA / Chicago
            m = pattern.search(one)
            if m and (quote is None or m.start() < quote.start()):
                quote = m
        apa = re.match(r"^(?P<a>[^()]+?)\s*\(\s*(?P<y>(?:1[5-9]|20)\d\d)[a-z]?\s*\)\s*[.,]\s*(?P<r>.+)$", one, re.S)
        astro = re.match(r"^(?P<a>[^()]*?[A-Za-z][^()]*?)[,\s]+(?P<y>(?:1[5-9]|20)\d\d)[a-z]?\s*[,.]\s*(?P<r>.*)$", one, re.S)

        if quote:
            entry.authors, entry.et_al = parse_authors(one[:quote.start()])
            f["title"] = tidy(clean(quote.group(1)), title=True)
            _venue_from_text(entry, one[quote.end():])
        elif apa and "," in apa.group("a"):
            year_hint = int(apa.group("y"))
            entry.authors, entry.et_al = parse_authors(apa.group("a"))
            parts = re.split(r"(?<![A-Z])\.\s+(?=[A-Z\\{])", apa.group("r"), maxsplit=1)
            f["title"] = tidy(clean(parts[0]), title=True)
            _venue_from_text(entry, parts[1] if len(parts) > 1 else "")
        elif astro and not re.search(r"(?<![A-Z])\.\s+[A-Z][a-z]", clean(astro.group("a"))):
            # aastex: "Authors, 2020, \apj, 1, 2"
            year_hint = int(astro.group("y"))
            entry.authors, entry.et_al = parse_authors(astro.group("a"))
            rest = clean(astro.group("r"))
            segs = [s.strip() for s in rest.split(",")]
            cut = next((i for i, s in enumerate(segs) if re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?(?:–\d+)?\.?", s)), len(segs))
            _assign_venue(f, ", ".join(segs[:cut]), "journal")
            tail = segs[cut:]
            if tail:
                f["volume"] = tail[0].rstrip(".")
            if len(tail) > 1:
                f["pages"] = tail[1].rstrip(".")
        elif re.match(r"^[^:]{3,200}?\b[A-Z]\.[^:]*:\s+\S", clean(one)) and not quote:
            plain = clean(one)                             # "Kim, J.U.: Title. Venue, 1--2 (2010)"
            authors_part, rest_part = plain.split(":", 1)
            entry.authors, entry.et_al = parse_authors(authors_part)
            parts = re.split(r"(?<![A-Z])\.\s+(?=[A-Z0-9\\{“\"`])", rest_part.strip(), maxsplit=1)
            f["title"] = tidy(parts[0], title=True)
            _venue_from_text(entry, parts[1] if len(parts) > 1 else "")
        else:                                              # "Authors. Title. Venue, year."
            parts = re.split(r"(?<![A-Z])(?<!et al)\.\s+(?=[A-Z0-9\\{“\"`])", one)
            if len(parts) >= 3:
                entry.authors, entry.et_al = parse_authors(parts[0])
                f["title"] = tidy(clean(parts[1]), title=True)
                _venue_from_text(entry, ". ".join(parts[2:]))
            else:
                f["note"] = tidy(clean(one))

    f["year"] = find_year(clean(text)) if year_hint is None else year_hint
    if f["year"] is None:
        f["year"] = _label_year(label)
    if not entry.authors:
        surname = _label_surname(label)
        if surname:
            entry.authors = [Name("", surname)]
    entry.entry_type = infer_type(f, clean(text))
    return entry


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def parse_reference(raw: str) -> Entry:
    """Parse one raw bibliography entry into an :class:`Entry`."""
    stripped = raw.strip()

    if re.match(r"\\entry\s*\{", stripped):
        entry = parse_biblatex(stripped)
    else:
        key, label, body = split_header(stripped)
        if re.search(r"\\begin\s*\{b(?:article|book|inproceedings|conference|incollection|"
                     r"chapter|inbook|phdthesis|mastersthesis|techreport|misc|other|"
                     r"unpublished|proceedings|manual)\}", body):
            entry = parse_structured(body, stripped)
        elif re.search(r"\\bibinfo\s*\{", body):
            entry = parse_bibinfo(body, stripped)
        else:
            entry = parse_generic(body, stripped, label)
        entry.key, entry.label = key, label
        if entry.fields.get("year") is None:
            entry.fields["year"] = _label_year(label)

    f = entry.fields
    if f.get("eprint") and not f.get("url"):
        f["url"] = f"https://arxiv.org/abs/{f['eprint']}"
    for k in list(f):
        if f[k] in ("", None):
            f[k] = None
    return entry


# --------------------------------------------------------------------------
# BibTeX rendering
# --------------------------------------------------------------------------
_BIBTEX_ORDER = ["author", "editor", "title", "journal", "booktitle", "series",
                 "volume", "number", "pages", "publisher", "address", "year",
                 "doi", "url", "eprint", "note"]


def _bibtex_escape(value: str) -> str:
    value = value.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    return value.replace("{", "").replace("}", "")


def to_bibtex(key: str, entry_type: str, fields: Dict[str, object],
              authors: List[Name], editors: List[Name], et_al: bool = False) -> str:
    """Render one standard BibTeX entry (UTF-8)."""
    lines = []

    def add(name, value):
        if value not in (None, "", []):
            lines.append(f"  {name} = {{{value}}},")

    for name in _BIBTEX_ORDER:
        if name == "author":
            if authors:
                add(name, " and ".join(a.bibtex() for a in authors) + (" and others" if et_al else ""))
        elif name == "editor":
            if editors:
                add(name, " and ".join(a.bibtex() for a in editors))
        elif name in ("doi", "url", "eprint"):
            add(name, fields.get(name))
        elif name == "title" and fields.get("title"):
            add(name, "{" + _bibtex_escape(str(fields["title"])) + "}")
        elif fields.get(name) is not None:
            add(name, _bibtex_escape(str(fields[name])))

    if fields.get("eprint"):
        lines.insert(-1 if fields.get("note") else len(lines), "  archiveprefix = {arXiv},")
    body = "\n".join(lines).rstrip(",")
    return f"@{entry_type}{{{key},\n{body}\n}}"
