"""Deduplicate the references of one publication across all its versions.

Pipeline for one arXiv paper:

1. every raw ``\\bibitem`` of every version is parsed (``bib_extract``);
2. entries that describe the *same underlying reference* are clustered, both
   inside a version and across versions, using identifiers (DOI / arXiv id),
   normalised titles, first author, year and, as a last resort, the text;
3. each cluster becomes one unified entry: one chosen citation key, and the
   fields of all duplicates are unioned instead of keeping a single entry;
4. for each version a ``{old_key: chosen_key}`` map says how its ``\\cite{}``
   commands must be renamed (see :func:`rename_cites`).
"""
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from bib_extract import (
    Entry, Name, infer_type, parse_reference, split_header, to_bibtex,
)
from latex_text  import latex_to_text

SCALAR_FIELDS = [
    "title", "journal", "booktitle", "series", "volume", "number", "pages",
    "publisher", "address", "year", "doi", "url", "eprint", "note",
    "howpublished",
]
_NUM_TOKENS_RE = re.compile(r"\d+")

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def norm_text(text: Optional[str]) -> str:
    """Lower-case ASCII words only: robust to formatting and spelling of marks."""
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", fold(text).lower()).strip()


def surname_key(name: Optional[Name]) -> str:
    if name is None or not name.family:
        return ""
    words = norm_text(name.family).split()
    return words[-1] if words else ""


def first_page(pages: Optional[str]) -> str:
    m = re.match(r"\s*([A-Za-z]?\d+)", pages or "")
    return m.group(1).lower() if m else ""


def ratio(a: str, b: str, min_ratio: float = 0.5) -> float:
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    if (2.0 * min(la, lb)) / (la + lb) < min_ratio:
        return 0.0
    sm = SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < min_ratio or sm.quick_ratio() < min_ratio:
        return 0.0
    return sm.ratio()

def _numbers_conflict(a: str, b: str) -> bool:
    """True when both strings contain numeric tokens that differ (e.g. 2022 vs 2023)."""
    na = _NUM_TOKENS_RE.findall(a)
    nb = _NUM_TOKENS_RE.findall(b)
    return bool(na and nb and na != nb)

def _block_tokens(text: str) -> List[str]:
    return [w for w in text.split() if len(w) >= 4][:3]
# --------------------------------------------------------------------------
# One parsed occurrence of a reference
# --------------------------------------------------------------------------
@dataclass
class Item:
    idx: int
    version: int
    position: int
    entry: Entry
    key: Optional[str]
    doi: str
    arxiv: str
    year: Optional[int]
    surname: str
    title: str
    journal: str
    page: str
    content: str


def _content_norm(entry: Entry) -> str:
    if entry.style == "biblatex":          # raw text is full of hashes
        parts = [entry.fields.get("title") or "",
                 " ".join(a.display() for a in entry.authors),
                 entry.fields.get("journal") or "", str(entry.year or "")]
        return norm_text(" ".join(parts))
    _, _, body = split_header(entry.raw)
    return norm_text(latex_to_text(body))


def make_item(idx: int, version: int, position: int, entry: Entry) -> Item:
    f = entry.fields
    return Item(
        idx=idx, version=version, position=position, entry=entry,
        key=entry.key,
        doi=(f.get("doi") or "").lower(),
        arxiv=(f.get("eprint") or "").lower(),
        year=f.get("year"),
        surname=surname_key(entry.authors[0]) if entry.authors else "",
        title=norm_text(f.get("title")),
        journal=norm_text(f.get("journal") or f.get("booktitle")),
        page=first_page(f.get("pages")),
        content=_content_norm(entry),
    )


# --------------------------------------------------------------------------
# "Is it the same reference?"
# --------------------------------------------------------------------------
def same_reference(a: Item, b: Item) -> bool:
    same_version = a.version == b.version

    if a.doi and b.doi:
        return a.doi == b.doi
    if a.arxiv and b.arxiv:
        return a.arxiv == b.arxiv

    author_eq = bool(a.surname and a.surname == b.surname)
    if a.surname and b.surname and not author_eq:
        return False

    short, long_ = sorted((a.title, b.title), key=len)
    equal_title = bool(a.title and a.title == b.title)
    prefix_title = bool(
        short and not equal_title and len(short.split()) >= 4
        and len(short) >= 20 and long_.startswith(short + " ")
    )

    gap = abs(a.year - b.year) if a.year and b.year else None
    limit = 10 if (not same_version and author_eq and (equal_title or prefix_title)) else 2
    if gap is not None and gap > limit:
        return False

    if equal_title and (len(a.title.split()) >= 3 or author_eq):
        return True
    if prefix_title and author_eq and not same_version:
        return True

    if not same_version:
        if (
            a.title and b.title and len(a.title) >= 20 and len(b.title) >= 20
            and author_eq and (gap is None or gap <= 1)
            and not _numbers_conflict(a.title, b.title)
            and ratio(a.title, b.title, 0.92) >= 0.92
        ):
            return True
        if a.key and a.key == b.key:
            if author_eq and (gap is None or gap <= 1):
                return True
            if a.title and b.title and ratio(a.title, b.title, 0.85) >= 0.85:
                return True
            if (not a.title or not b.title) and ratio(a.content, b.content, 0.6) >= 0.6:
                return True

    # Only match title-in-content when the other entry failed to isolate a title.
    for x, y in ((a, b), (b, a)):
        if (
            not y.title and len(x.title) >= 25
            and f" {x.title} " in f" {y.content} "
            and (author_eq or not (a.surname and b.surname))
        ):
            return True

    if (
        author_eq and a.year and a.year == b.year and a.page and a.page == b.page
        and (not a.journal or not b.journal or a.journal == b.journal)
    ):
        return True

    if a.content and a.content == b.content and len(a.content) >= 20:
        return True
    if (
        not same_version and len(a.content) >= 30
        and not _numbers_conflict(a.content, b.content)
        and ratio(a.content, b.content, 0.95) >= 0.95
    ):
        return True
    return False



class _Clusters:
    """Union-find that blocks conflicting DOIs/arXiv IDs and fuzzy same-version bridges."""

    def __init__(self, items: List[Item]):
        self.parent = list(range(len(items)))
        self.dois = [{i.doi} if i.doi else set() for i in items]
        self.arxivs = [{i.arxiv} if i.arxiv else set() for i in items]
        self.versions = [{i.version} for i in items]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int, allow_same_version_overlap: bool = True) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if len(self.dois[ra] | self.dois[rb]) > 1 or len(self.arxivs[ra] | self.arxivs[rb]) > 1:
            return False
        if not allow_same_version_overlap and (self.versions[ra] & self.versions[rb]):
            return False
        self.parent[rb] = ra
        self.dois[ra] |= self.dois[rb]
        self.arxivs[ra] |= self.arxivs[rb]
        self.versions[ra] |= self.versions[rb]
        return True


def cluster_items(items: List[Item], max_fuzzy_bucket: int = 80) -> List[int]:
    """Cluster id (root index) of every item using bounded token blocking."""
    clusters = _Clusters(items)
    exact_blocks: Dict[tuple, List[int]] = defaultdict(list)
    fuzzy_blocks: Dict[tuple, List[int]] = defaultdict(list)

    def exact_keys(it: Item):
        keys = []
        if it.doi:
            keys.append(("doi", it.doi))
        if it.arxiv:
            keys.append(("arxiv", it.arxiv))
        if it.title:
            keys.append(("title", it.title))
        if it.content:
            keys.append(("content", it.content))
        if it.key:
            keys.append(("key", it.key))
        if it.surname and it.year and it.page:
            keys.append(("sig", it.surname, it.year, it.page))
        return keys

    def fuzzy_keys(it: Item):
        keys = []
        tokens = _block_tokens(it.title) if it.title else _block_tokens(it.content)
        if it.surname:
            if tokens:
                for tok in tokens:
                    keys.append(("st", it.surname, tok))
            else:
                keys.append(("sy", it.surname, it.year))
        else:
            for tok in tokens:
                keys.append(("nt", tok, it.year))
        return keys

    for i, it in enumerate(items):
        candidates = set()
        for bk in exact_keys(it):
            candidates.update(exact_blocks[bk])
        for bk in fuzzy_keys(it):
            bucket = fuzzy_blocks[bk]
            if len(bucket) <= max_fuzzy_bucket:
                candidates.update(bucket)

        for c in sorted(candidates):
            if clusters.find(c) == clusters.find(i):
                continue
            other = items[c]
            if same_reference(other, it):
                strong = (
                    other.version == it.version
                    or (other.doi and other.doi == it.doi)
                    or (other.arxiv and other.arxiv == it.arxiv)
                    or (other.title and other.title == it.title)
                    or (other.content and other.content == it.content)
                )
                clusters.union(c, i, allow_same_version_overlap=strong)

        for bk in exact_keys(it):
            exact_blocks[bk].append(i)
        for bk in fuzzy_keys(it):
            fuzzy_blocks[bk].append(i)

    return [clusters.find(i) for i in range(len(items))]


# --------------------------------------------------------------------------
# Unification of a cluster
# --------------------------------------------------------------------------
def _completeness(it: Item) -> int:
    f = it.entry.fields
    return sum(1 for k in SCALAR_FIELDS if f.get(k)) + len(it.entry.authors)


def unify_cluster(members: List[Item]) -> dict:
    """Union of the fields of all duplicates (latest, most complete first)."""
    order = sorted(members, key=lambda m: (-m.version, -_completeness(m), m.position))

    fields = {}
    for name in SCALAR_FIELDS:
        for m in order:
            value = m.entry.fields.get(name)
            if value not in (None, ""):
                fields[name] = value
                break
        else:
            fields[name] = None

    def author_score(m: Item):
        names = m.entry.authors
        return (not m.entry.et_al, len(names), sum(len(n.display()) for n in names))

    with_authors = [m for m in order if m.entry.authors]
    best = max(with_authors, key=author_score) if with_authors else None
    authors = best.entry.authors if best else []
    et_al = best.entry.et_al if best else False
    editors = next((m.entry.editors for m in order if m.entry.editors), [])

    entry_type = next((m.entry.entry_type for m in order if m.entry.entry_type != "misc"), "misc")
    if entry_type == "misc":
        entry_type = infer_type(fields)

    return {"fields": fields, "authors": authors, "editors": editors,
            "et_al": et_al, "entry_type": entry_type, "order": order}


def synthetic_key(authors: List[Name], year, title: Optional[str], raw: str) -> str:
    surname = norm_text(authors[0].family).replace(" ", "") if authors else ""
    word = next((w for w in norm_text(title).split() if len(w) > 3), "") if title else ""
    key = f"{surname}{year or ''}{word}"
    return key or "ref" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------
# Publication level
# --------------------------------------------------------------------------
def deduplicate_publication(arxiv_id: str, versions: Dict[int, List[str]]) -> dict:
    """Parse + deduplicate the raw references of every version of a paper.

    `versions` maps version number -> list of raw ``\\bibitem`` strings (an empty
    list is a version without references: it is kept and reported as empty).
    """
    items: List[Item] = []
    style_counts: Counter = Counter()
    parse_errors = 0

    for version in sorted(versions):
        for position, raw in enumerate(versions[version]):
            try:
                entry = parse_reference(raw)
            except Exception:                      # never lose a reference
                parse_errors += 1
                key, _, body = split_header(raw)
                entry = Entry(key=key, raw=raw, style="unparsed",
                              fields={"note": latex_to_text(body)[:500] or None})
            style_counts[entry.style] += 1
            items.append(make_item(len(items), version, position, entry))

    roots = cluster_items(items) if items else []
    groups: Dict[int, List[Item]] = defaultdict(list)
    for it, root in zip(items, roots):
        groups[root].append(it)

    unified = {root: unify_cluster(members) for root, members in groups.items()}

    # ---- one citation key per cluster, unique inside the publication ------
    def latest_key(members):
        for m in sorted(members, key=lambda m: (-m.version, m.position)):
            if m.key:
                return m.key
        return None

    chosen: Dict[int, str] = {}
    used: set = set()
    for root in sorted(groups, key=lambda r: (-max(m.version for m in groups[r]),
                                              min(m.idx for m in groups[r]))):
        members, u = groups[root], unified[root]
        key = latest_key(members) or synthetic_key(
            u["authors"], u["fields"].get("year"), u["fields"].get("title"), members[0].entry.raw)
        base, n = key, 1
        while key in used:
            n += 1
            key = f"{base}_{n}"
        used.add(key)
        chosen[root] = key

    # ---- per-version \cite renaming ---------------------------------------
    cite_rename: Dict[int, Dict[str, str]] = {v: {} for v in versions}
    ambiguous_keys = 0
    seen: Dict[tuple, int] = {}
    for it, root in zip(items, roots):
        if not it.key:
            continue
        prior = seen.setdefault((it.version, it.key), root)
        if prior != root:                          # same key defined twice, different refs
            ambiguous_keys += 1
            continue
        if chosen[prior] != it.key:
            cite_rename[it.version][it.key] = chosen[prior]
    cite_rename = {v: m for v, m in cite_rename.items() if m}

    # ---- output rows -------------------------------------------------------
    entries = []
    for root in sorted(groups, key=lambda r: min(m.idx for m in groups[r])):
        members, u = groups[root], unified[root]
        key = chosen[root]
        entries.append({
            "citation_key": key,
            "entry_type": u["entry_type"],
            "fields": u["fields"],
            "authors": [[a.given, a.family] for a in u["authors"]],
            "editors": [[a.given, a.family] for a in u["editors"]],
            "et_al": u["et_al"],
            "versions": sorted({m.version for m in members}),
            "source_keys": {str(v): sorted({m.key for m in members if m.version == v and m.key})
                            for v in sorted({m.version for m in members})},
            "n_duplicates": len(members),
            "bibtex": to_bibtex(key, u["entry_type"], u["fields"], u["authors"], u["editors"], u["et_al"]),
        })

    per_version = {}
    for v in sorted(versions):
        in_v = [(it, r) for it, r in zip(items, roots) if it.version == v]
        per_version[str(v)] = {"raw": len(versions[v]), "distinct": len({r for _, r in in_v})}

    n_before = len(items)
    within = sum(c["raw"] - c["distinct"] for c in per_version.values())
    stats = {
        "versions": len(versions),
        "empty_versions": sum(1 for v in versions.values() if not v),
        "entries_before": n_before,
        "entries_after": len(entries),
        "within_version_duplicates": within,
        "cross_version_duplicates": n_before - len(entries) - within,
        "renamed_cite_keys": sum(len(m) for m in cite_rename.values()),
        "ambiguous_keys": ambiguous_keys,
        "parse_errors": parse_errors,
        "styles": dict(style_counts),
        "per_version": per_version,
    }
    return {"arxiv_id": arxiv_id, "entries": entries,
            "cite_rename": {str(v): m for v, m in cite_rename.items()}, "stats": stats}


# --------------------------------------------------------------------------
# Renaming \cite commands
# --------------------------------------------------------------------------
_CITE_RE = re.compile(
    r"(\\(?!citeauthoryear|citenamefont)[A-Za-z]*cite[A-Za-z]*\*?"   # \cite \citep \parencite \nocite ...
    r"(?:\s*\[[^\]]*\]){0,2}\s*\{)([^{}]*)(\})"
)


def rename_cites(text: str, mapping: Dict[str, str]) -> str:
    """Rename the keys inside every ``\\cite``-like command of `text`.

    `mapping` is the version's ``cite_rename`` map (old key -> chosen key).
    Keys that are not in the map are untouched, and a command that ends up
    citing the same key twice keeps it once.
    """
    if not mapping:
        return text

    def fix(match):
        keys = [k.strip() for k in match.group(2).split(",")]
        out, seen = [], set()
        for k in keys:
            new = mapping.get(k, k)
            if new not in seen:
                seen.add(new)
                out.append(new)
        return match.group(1) + ", ".join(out) + match.group(3)

    return _CITE_RE.sub(fix, text)
