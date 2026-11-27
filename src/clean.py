"""Phase 1: Canonical cleaning and normalization for Milestone 3 matching.

Shared by automatic labelling, manual labelling sheets, and feature engineering
so every stage sees identical normalized titles, authors, identifiers, years,
and venues.
"""
import re
import unicodedata
from typing import Iterable, List, Optional

MIN_YEAR = 1900
MAX_YEAR = 2027

_EXTRA_TRANSLIT = str.maketrans({
    "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE",
    "ø": "o",  "Ø": "O",
    "ł": "l",  "Ł": "L",
    "ß": "ss",
    "đ": "d",  "Đ": "D",
    "ð": "d",  "Ð": "D",
    "þ": "th", "Þ": "Th",
    "ı": "i",  "İ": "I",
    "’": "'",  "‘": "'",
    "“": '"',  "”": '"',
    "–": "-",  "—": "-", "−": "-",
})

STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "this", "to", "via", "with", "using", "based", "toward", "towards",
})

NAME_PARTICLES = frozenset({
    "van", "von", "der", "den", "de", "del", "della", "di", "da", "dos",
    "das", "du", "le", "la", "ten", "ter", "bin", "ibn", "al", "el", "dal",
    "st", "saint",
})

NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v", "phd", "md"})

_CORPORATE_RE = re.compile(
    r"(?i)\b(?:collaboration|consortium|working\s+group|experiment|team|"
    r"project|committee|institute|university|google|openai|meta|anthropic|"
    r"microsoft|deepmind|nvidia|huggingface)\b"
)
_ET_AL_RE = re.compile(r"(?i)\b(?:et\.?\s*al\.?|and\s+others|others)\b")

_LATEX_MATH_WORDS = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "varepsilon": "epsilon", "zeta": "zeta", "eta": "eta",
    "theta": "theta", "vartheta": "theta", "iota": "iota", "kappa": "kappa",
    "lambda": "lambda", "mu": "mu", "nu": "nu", "xi": "xi", "pi": "pi",
    "rho": "rho", "sigma": "sigma", "tau": "tau", "upsilon": "upsilon",
    "phi": "phi", "varphi": "phi", "chi": "chi", "psi": "psi", "omega": "omega",
    "Gamma": "gamma", "Delta": "delta", "Theta": "theta", "Lambda": "lambda",
    "Xi": "xi", "Pi": "pi", "Sigma": "sigma", "Phi": "phi", "Psi": "psi",
    "Omega": "omega", "infty": "infinity", "ell": "l", "times": "x",
    "log": "log", "ln": "ln", "exp": "exp", "sin": "sin", "cos": "cos",
    "tan": "tan", "min": "min", "max": "max", "sup": "sup", "inf": "inf",
    "lim": "lim", "arg": "arg", "det": "det", "dim": "dim", "deg": "deg",
    "gcd": "gcd", "ker": "ker", "mod": "mod", "Pr": "pr", "poly": "poly",
    "polylog": "polylog",
}

_LATEX_SYMBOL_ACCENTS = {
    '"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302",
    "~": "\u0303", "=": "\u0304", ".": "\u0307",
}
_LATEX_LETTER_ACCENTS = {
    "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327",
    "k": "\u0328", "r": "\u030a", "b": "\u0331", "d": "\u0323",
}

_SYM_ACC_RE = re.compile(
    r"""\\(["'`^~=.])\s*(?:\{\s*(\\[ij]|[A-Za-z])\s*\}|(\\[ij]|[A-Za-z]))"""
)
_LET_ACC_RE = re.compile(
    r"\\([uvHckrbd])(?![A-Za-z])\s*(?:\{\s*(\\[ij]|[A-Za-z])\s*\}|(\\[ij]|[A-Za-z]))"
)
_HREF_RE = re.compile(r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}")
_CMD_RE = re.compile(r"\\([A-Za-z@]+)\*?")
_MATH_DELIM_RE = re.compile(r"\$\$|\$|\\\(|\\\)|\\\[|\\\]")
_SUBTITLE_SPLIT_RE = re.compile(r"\s*(?::|\s+[-–—]\s+)\s*")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_NUM_RE = re.compile(r"\d+")

_DOI_PREFIX_RE = re.compile(r"(?i)^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)")
_DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>\\{}]+')
_ARXIV_NEW_RE = re.compile(
    r"(?i)(?:arxiv\s*[:/]?\s*|arxiv\.org/(?:abs|pdf|src)/|10\.48550/arxiv\.|abs/)(\d{4}\.\d{4,5})(?:v\d+)?"
)
_ARXIV_OLD_RE = re.compile(
    r"(?i)(?:arxiv\s*[:/]?\s*|arxiv\.org/(?:abs|pdf|src)/|10\.48550/arxiv\.|abs/)([a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?"
)
_BARE_ARXIV_NEW_RE = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$")
_BARE_ARXIV_OLD_RE = re.compile(r"(?i)^([a-z\-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?$")

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_ORDINAL_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I)
_VENUE_BOILERPLATE_RE = re.compile(
    r"\b(?:proceedings|proc|international|int|annual|conference|conf|"
    r"symposium|symp|workshop|meeting|on|of|the|in|for|and)\b"
)
_ARXIV_VENUE_RE = re.compile(r"(?i)\b(?:arxiv|corr|preprint|biorxiv|medrxiv|ssrn)\b")


def fold_ascii(text: Optional[str]) -> str:
    """Fold Unicode characters (including ø, æ, œ, ł, ß) to clean ASCII."""
    if not text:
        return ""
    s = text.translate(_EXTRA_TRANSLIT)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def strip_latex_and_math(text: Optional[str]) -> str:
    """Strip LaTeX commands and math delimiters while preserving meaningful tokens."""
    if not text:
        return ""
    s = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    s = s.replace("\\&", "&").replace("\\_", "_").replace("\\%", "%").replace("\\#", "#")
    s = _HREF_RE.sub(r"\1", s)
    s = _SYM_ACC_RE.sub(
        lambda m: unicodedata.normalize(
            "NFC",
            (m.group(2) or m.group(3)).lstrip("\\") + _LATEX_SYMBOL_ACCENTS[m.group(1)],
        ),
        s,
    )
    s = _LET_ACC_RE.sub(
        lambda m: unicodedata.normalize(
            "NFC",
            (m.group(2) or m.group(3)).lstrip("\\") + _LATEX_LETTER_ACCENTS[m.group(1)],
        ),
        s,
    )
    s = re.sub(r"\\([ij])(?![A-Za-z])\s*", r"\1", s)
    # Keep L^2 / x_1 compact as L2 / x1 before dropping braces.
    s = re.sub(r"([A-Za-z0-9])\s*[\^_]\s*\{?([A-Za-z0-9]+)\}?", r"\1\2", s)
    s = _MATH_DELIM_RE.sub(" ", s)
    s = _CMD_RE.sub(
        lambda m: f" {_LATEX_MATH_WORDS[m.group(1)]} " if m.group(1) in _LATEX_MATH_WORDS else " ",
        s,
    )
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def clean_title(title: Optional[str]) -> dict:
    """Normalize a title into full string, main title (pre-subtitle), tokens, and numbers."""
    if not title:
        return {"clean": "", "main": "", "tokens": [], "numbers": []}

    plain = fold_ascii(strip_latex_and_math(title))
    main_raw = _SUBTITLE_SPLIT_RE.split(plain, maxsplit=1)[0]
    clean = _NON_ALNUM_RE.sub(" ", plain.lower()).strip()
    main = _NON_ALNUM_RE.sub(" ", main_raw.lower()).strip()

    all_tokens = clean.split()
    tokens = [t for t in all_tokens if t not in STOP_WORDS and len(t) > 1]
    if not tokens and all_tokens:
        tokens = all_tokens

    return {
        "clean": clean,
        "main": main,
        "tokens": tokens,
        "numbers": _NUM_RE.findall(clean),
    }


def parse_author_name(raw_name: Optional[str]) -> Optional[dict]:
    """Parse one author string ('Given Family' or 'Family, Given') into surname & initials."""
    if not raw_name:
        return None
    s = fold_ascii(strip_latex_and_math(raw_name)).strip(" ,;.")
    if not s or _ET_AL_RE.fullmatch(s):
        return None
    if _CORPORATE_RE.search(s):
        return {
            "raw": s,
            "surname": "",
            "surname_full": "",
            "initials": [],
            "is_corporate": True,
        }

    if "," in s:
        family_raw, given_raw = (p.strip() for p in s.split(",", 1))
        if _NON_ALNUM_RE.sub("", given_raw.lower()) in NAME_SUFFIXES:
            tokens = family_raw.split()
            family_raw, given_raw = (
                (tokens[-1], " ".join(tokens[:-1])) if len(tokens) > 1 else (family_raw, "")
            )
    else:
        tokens = [t for t in s.split() if t]
        while len(tokens) > 1 and _NON_ALNUM_RE.sub("", tokens[-1].lower()) in NAME_SUFFIXES:
            tokens.pop()
        if not tokens:
            return None
        if len(tokens) == 1:
            family_raw, given_raw = tokens[0], ""
        else:
            idx = len(tokens) - 1
            while idx > 1 and _NON_ALNUM_RE.sub("", tokens[idx - 1].lower()) in NAME_PARTICLES:
                idx -= 1
            given_raw = " ".join(tokens[:idx])
            family_raw = " ".join(tokens[idx:])

    fam_words = _NON_ALNUM_RE.sub(" ", family_raw.lower()).split()
    while len(fam_words) > 1 and fam_words[-1] in NAME_SUFFIXES:
        fam_words.pop()
    if not fam_words:
        return None

    given_tokens = [t for t in re.split(r"[\s.\-]+", given_raw) if t]
    initials = []
    for tok in given_tokens:
        letters = re.sub(r"[^a-z]", "", tok.lower())
        if letters:
            initials.append(letters[0])

    return {
        "raw": s,
        "surname": fam_words[-1],
        "surname_full": " ".join(fam_words),
        "initials": initials,
        "is_corporate": False,
    }


def clean_authors(authors: Optional[Iterable[str]], et_al: bool = False) -> dict:
    """Normalize an ordered list of author strings, keeping order and particles."""
    parsed: List[dict] = []
    corporate: List[str] = []
    has_et_al = bool(et_al)

    for raw in (authors or []):
        if not raw:
            continue
        if _ET_AL_RE.search(raw):
            has_et_al = True
            raw = _ET_AL_RE.sub("", raw).strip(" ,;.")
            if not raw:
                continue
        info = parse_author_name(raw)
        if not info:
            continue
        if info["is_corporate"]:
            corporate.append(_NON_ALNUM_RE.sub(" ", info["raw"].lower()).strip())
        else:
            parsed.append(info)

    surnames = [p["surname"] for p in parsed]
    surnames_full = [p["surname_full"] for p in parsed]
    return {
        "first_surname": surnames[0] if surnames else "",
        "first_surname_full": surnames_full[0] if surnames_full else "",
        "first_initials": parsed[0]["initials"] if parsed else [],
        "surnames": surnames,
        "surnames_full": surnames_full,
        "initials": [p["initials"] for p in parsed],
        "count": len(parsed),
        "et_al": has_et_al,
        "corporate": corporate,
    }


def clean_doi(value: Optional[str]) -> Optional[str]:
    """Lowercase DOI with URL/prefix and trailing punctuation removed."""
    if not value:
        return None
    s = value.replace("\\_", "_").replace("\\%", "%").strip()
    s = _DOI_PREFIX_RE.sub("", s)
    m = _DOI_RE.search(s)
    if not m:
        return None
    doi = m.group(0).rstrip(".,;:")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1]
    return doi.lower() or None


def clean_arxiv_id(value: Optional[str]) -> Optional[str]:
    """Normalize a bare or embedded arXiv ID (version suffix removed; old-style supported)."""
    if not value:
        return None
    s = value.strip()
    m = _BARE_ARXIV_NEW_RE.match(s)
    if m:
        return m.group(1)
    m = _BARE_ARXIV_OLD_RE.match(s)
    if m:
        return m.group(1).lower()
    for pat in (_ARXIV_NEW_RE, _ARXIV_OLD_RE):
        m = pat.search(s)
        if m:
            aid = m.group(1)
            return aid.lower() if "/" in aid else aid
    return None


def extract_arxiv_ids(*texts: Optional[str]) -> List[str]:
    """Extract all unique normalized arXiv IDs across fields and raw fallback text."""
    found: List[str] = []
    seen = set()
    for text in texts:
        if not text:
            continue
        bare = clean_arxiv_id(text)
        if bare and bare not in seen:
            seen.add(bare)
            found.append(bare)
        for pat in (_ARXIV_NEW_RE, _ARXIV_OLD_RE):
            for m in pat.finditer(text):
                aid = m.group(1).lower() if "/" in m.group(1) else m.group(1)
                if aid not in seen:
                    seen.add(aid)
                    found.append(aid)
    return found


def year_from_arxiv_id(arxiv_id: Optional[str]) -> Optional[int]:
    """Derive 4-digit submission year from a normalized arXiv ID."""
    aid = clean_arxiv_id(arxiv_id)
    if not aid:
        return None
    m = re.match(r"^(?:[a-z\-]+(?:\.[a-z]{2})?/)?(\d{2})\d{2}", aid)
    if not m:
        return None
    yy = int(m.group(1))
    year = (1900 + yy) if yy >= 91 else (2000 + yy)
    return year if MIN_YEAR <= year <= MAX_YEAR else None


def clean_year(value) -> Optional[int]:
    """Parse a valid 4-digit year integer from int or date/year string."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value if MIN_YEAR <= value <= MAX_YEAR else None
    for y_str in _YEAR_RE.findall(str(value)):
        y = int(y_str)
        if MIN_YEAR <= y <= MAX_YEAR:
            return y
    return None


def clean_venue(venue: Optional[str]) -> dict:
    """Normalize venue name (lowercase, years/ordinals stripped, arXiv flag detected)."""
    if not venue:
        return {"clean": "", "compact": "", "is_arxiv": False}
    raw = fold_ascii(strip_latex_and_math(venue)).lower()
    is_arxiv = bool(_ARXIV_VENUE_RE.search(raw))
    s = _ORDINAL_RE.sub(" ", _YEAR_RE.sub(" ", raw))
    clean = _NON_ALNUM_RE.sub(" ", s).strip()
    compact_tokens = [t for t in clean.split() if not _VENUE_BOILERPLATE_RE.fullmatch(t)]
    compact = " ".join(compact_tokens) or clean
    return {"clean": clean, "compact": compact, "is_arxiv": is_arxiv}


def preprocess_bibtex_entry(row: dict, extra: Optional[dict] = None) -> dict:
    """Clean one row of `bibtex.jsonl` (optionally enriched from `bibtex_entries.jsonl`)."""
    extra = extra or {}
    extra_fields = extra.get("fields") or {}

    title_info = clean_title(row.get("title"))
    author_info = clean_authors(row.get("authors"), et_al=bool(extra.get("et_al", False)))

    doi = clean_doi(row.get("doi")) or clean_doi(extra_fields.get("doi"))
    raw_bibtex = extra.get("bibtex") or ""
    arxiv_ids = extract_arxiv_ids(
        extra_fields.get("eprint"),
        row.get("url"),
        row.get("journal"),
        doi,
        extra_fields.get("note"),
        raw_bibtex,
    )
    arxiv_id_ref = arxiv_ids[0] if arxiv_ids else None

    year = clean_year(row.get("year"))
    if year is None and arxiv_id_ref:
        year = year_from_arxiv_id(arxiv_id_ref)

    venue_raw = row.get("journal") or extra_fields.get("booktitle") or extra_fields.get("journal")
    venue_info = clean_venue(venue_raw)
    is_arxiv = venue_info["is_arxiv"] or bool(arxiv_id_ref and not venue_info["clean"])

    raw_src = raw_bibtex or " ".join(
        str(x) for x in (
            row.get("title") or "",
            " ".join(row.get("authors") or []),
            venue_raw or "",
            row.get("url") or "",
        )
        if x
    )
    raw_text_clean = _NON_ALNUM_RE.sub(" ", fold_ascii(strip_latex_and_math(raw_src)).lower()).strip()

    completeness = sum(bool(x) for x in (
        title_info["clean"],
        author_info["count"] > 0,
        year is not None,
        doi is not None,
        venue_info["clean"],
        row.get("url"),
    ))

    return {
        "arxiv_id": clean_arxiv_id(row["arxiv_id"]) or str(row["arxiv_id"]).strip(),
        "citation_key": row["citation_key"],
        "entry_type": (row.get("entry_type") or "misc").lower(),
        "title_clean": title_info["clean"],
        "title_main": title_info["main"],
        "title_tokens": title_info["tokens"],
        "title_numbers": title_info["numbers"],
        "first_surname": author_info["first_surname"],
        "first_surname_full": author_info["first_surname_full"],
        "first_initials": author_info["first_initials"],
        "author_surnames": author_info["surnames"],
        "author_initials": author_info["initials"],
        "author_count": author_info["count"],
        "et_al": author_info["et_al"],
        "corporate_authors": author_info["corporate"],
        "year": year,
        "doi": doi,
        "arxiv_id_extracted": arxiv_id_ref,
        "arxiv_ids_all": arxiv_ids,
        "venue_clean": venue_info["clean"],
        "venue_compact": venue_info["compact"],
        "is_arxiv_entry": is_arxiv,
        "raw_text_clean": raw_text_clean,
        "parse_completeness": completeness,
    }


def preprocess_reference(row: dict, extra_s2: Optional[dict] = None) -> dict:
    """Clean one row of `references_table.jsonl` (optionally enriched with S2 DOI/venue)."""
    extra_s2 = extra_s2 or {}
    ref_id = clean_arxiv_id(row["ref_arxiv_id"]) or str(row["ref_arxiv_id"]).strip()
    title_info = clean_title(row.get("ref_title"))
    author_info = clean_authors(row.get("ref_authors"))

    year = clean_year(row.get("ref_submission_date")) or clean_year(extra_s2.get("year"))
    arxiv_year = year_from_arxiv_id(ref_id)
    if year is None:
        year = arxiv_year

    ext_ids = extra_s2.get("externalIds") or {}
    doi = clean_doi(extra_s2.get("doi") or ext_ids.get("DOI"))
    venue_info = clean_venue(extra_s2.get("venue"))

    return {
        "arxiv_id": clean_arxiv_id(row["arxiv_id"]) or str(row["arxiv_id"]).strip(),
        "ref_arxiv_id": ref_id,
        "ref_s2_id": row.get("ref_s2_id"),
        "title_clean": title_info["clean"],
        "title_main": title_info["main"],
        "title_tokens": title_info["tokens"],
        "title_numbers": title_info["numbers"],
        "first_surname": author_info["first_surname"],
        "first_surname_full": author_info["first_surname_full"],
        "first_initials": author_info["first_initials"],
        "author_surnames": author_info["surnames"],
        "author_initials": author_info["initials"],
        "author_count": author_info["count"],
        "year": year,
        "arxiv_year": arxiv_year,
        "doi": doi,
        "venue_clean": venue_info["clean"],
        "venue_compact": venue_info["compact"],
        "is_arxiv_venue": venue_info["is_arxiv"],
    }