"""LaTeX helpers shared by the reference extractor.

Everything here is plain regular-expression based (no TeX engine): enough to
turn the bibliography markup found in arXiv sources into readable text.
"""
import re
import unicodedata

# --------------------------------------------------------------------------
# Accents and special letters
# --------------------------------------------------------------------------
_SYMBOL_ACCENTS = {
    '"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302",
    "~": "\u0303", "=": "\u0304", ".": "\u0307",
}
_LETTER_ACCENTS = {
    "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327",
    "k": "\u0328", "r": "\u030a", "b": "\u0331", "d": "\u0323",
}
_SPECIAL_LETTERS = {
    "ss": "ß", "o": "ø", "O": "Ø", "ae": "æ", "AE": "Æ", "oe": "œ",
    "OE": "Œ", "aa": "å", "AA": "Å", "l": "ł", "L": "Ł",
}

_SYMBOL_ACCENT_RE = re.compile(
    r"""\\(["'`^~=.])\s*(?:\{\s*(\\[ij]|[A-Za-z])\s*\}|(\\[ij]|[A-Za-z]))"""
)
_LETTER_ACCENT_RE = re.compile(
    r"\\([uvHckrbd])(?![A-Za-z])\s*(?:\{\s*(\\[ij]|[A-Za-z])\s*\}|(\\[ij]|[A-Za-z]))"
)
_SPECIAL_LETTER_RE = re.compile(
    r"\\(ss|ae|AE|oe|OE|aa|AA|o|O|l|L)(?![A-Za-z])\s*"
)

# Journal macros defined by aastex / the ADS ``aas_macros.sty`` file.
JOURNAL_MACROS = {
    "apj": "ApJ", "apjl": "ApJ", "apjs": "ApJS", "aj": "AJ", "aap": "A&A",
    "aapr": "A&A Rev.", "aaps": "A&AS", "mnras": "MNRAS", "pasp": "PASP",
    "pasj": "PASJ", "pasa": "PASA", "araa": "ARA&A", "apss": "Ap&SS",
    "solphys": "Sol. Phys.", "ssr": "Space Sci. Rev.", "icarus": "Icarus",
    "nat": "Nature", "prl": "Phys. Rev. Lett.", "prd": "Phys. Rev. D",
    "pra": "Phys. Rev. A", "prb": "Phys. Rev. B", "prc": "Phys. Rev. C",
    "pre": "Phys. Rev. E", "jcap": "JCAP", "physrep": "Phys. Rep.",
    "phr": "Phys. Rep.", "jgr": "J. Geophys. Res.", "grl": "Geophys. Res. Lett.",
    "qjras": "QJRAS", "memras": "MmRAS", "gca": "Geochim. Cosmochim. Acta",
    "ao": "Appl. Opt.", "bain": "Bull. Astron. Inst. Netherlands",
    "iaucirc": "IAU Circ.", "ibvs": "IBVS", "skytel": "S&T",
}

_MATH_RE = re.compile(
    r"(?<!\\)\$\$.+?(?<!\\)\$\$|(?<!\\)\$.+?(?<!\\)\$|\\\(.+?\\\)", re.S
)


def _compose(letter, mark):
    letter = letter.lstrip("\\")  # \i -> i, \j -> j
    return unicodedata.normalize("NFC", letter + mark)


def balanced_group(text, start, open_char="{", close_char="}", limit=None):
    """Index just after the group opening at text[start], or None."""
    stop = len(text) if limit is None else min(len(text), start + limit)
    depth = 0
    i = start

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


def skip_label(text, start, limit=2000):
    """End of an optional ``[label]`` (braces inside may contain ']')."""
    stop = min(len(text), start + limit)
    braces = 0
    i = start + 1

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


def command_args(text, name):
    """First brace argument of every ``\\name{...}`` occurrence."""
    found = []
    pattern = re.compile(r"\\" + name + r"(?![A-Za-z@])\s*\{")

    for match in pattern.finditer(text):
        start = match.end() - 1
        end = balanced_group(text, start)
        if end:
            found.append(text[start + 1:end - 1])

    return found


def latex_to_text(source, keep_math=True):
    """Convert a LaTeX fragment to readable unicode text."""
    if not source:
        return ""

    s = source.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(?<!\\)%.*", "", s)             # comments / BibitemOpen%
    s = s.replace("\\\\", " ")                   # forced line breaks

    s = s.replace("\\'\\'", "\u201d")                 # IEEE-style \'\' closing quotes

    # Protect escaped characters that would confuse later steps.
    s = s.replace("\\$", "\x02").replace("\\{", "\x03").replace("\\}", "\x04")

    math = []

    def stash(match):
        math.append(match.group(0))
        return f"\x05{len(math) - 1}\x05"

    s = _MATH_RE.sub(stash, s)

    s = (s.replace("\\&", "&").replace("\\%", "%")
          .replace("\\#", "#").replace("\\_", "_"))

    s = _SYMBOL_ACCENT_RE.sub(
        lambda m: _compose(m.group(2) or m.group(3), _SYMBOL_ACCENTS[m.group(1)]),
        s,
    )
    s = _LETTER_ACCENT_RE.sub(
        lambda m: _compose(m.group(2) or m.group(3), _LETTER_ACCENTS[m.group(1)]),
        s,
    )
    s = _SPECIAL_LETTER_RE.sub(lambda m: _SPECIAL_LETTERS[m.group(1)], s)
    s = re.sub(r"\\([ij])(?![A-Za-z])\s*", r"\1", s)   # dotless i / j

    # Links: keep the visible text (or the URL when there is none).
    for _ in range(3):
        s = re.sub(r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:url|doi)\s*\{([^{}]*)\}", r"\1", s)

    s = re.sub(
        r"\\([A-Za-z]+)(?![A-Za-z])",
        lambda m: JOURNAL_MACROS.get(m.group(1), m.group(0)),
        s,
    )

    s = s.replace("\\-", "")
    s = re.sub(r"\\[,;:!/> ]", " ", s)
    s = s.replace("~", " ")
    s = s.replace("---", "—").replace("--", "–")
    s = s.replace("``", "“").replace("''", "”")

    s = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", s)    # environment markers
    s = re.sub(r"\\(?:newblock|par)\b", " ", s)
    s = re.sub(r"\\[A-Za-z@]+\*?", "", s)        # drop remaining commands,
    s = s.replace("{", "").replace("}", "")      # keep their arguments

    s = s.replace("\x02", "$").replace("\x03", "{").replace("\x04", "}")

    def unstash(match):
        raw = math[int(match.group(1))]
        return raw if keep_math else raw.strip("$")

    s = re.sub(r"\x05(\d+)\x05", unstash, s)

    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return unicodedata.normalize("NFC", s)


def find_bibitems(tex):
    """Raw ``\\bibitem`` entries of a .tex file (comments are ignored)."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    starts = []

    for match in re.finditer(r"\\bibitem(?![A-Za-z@])", tex):
        before = tex[max(0, match.start() - 40):match.start()]
        if re.search(r"\\(?:re|provide)?newcommand\s*\{?\s*$|\\(?:[gex]?def|let)\s*$", before):
            continue
        pos = match.end()
        while pos < len(tex) and tex[pos].isspace():
            pos += 1
        if pos < len(tex) and tex[pos] == "[":
            pos = skip_label(tex, pos)
            if pos is None:
                continue
            while pos < len(tex) and tex[pos].isspace():
                pos += 1
        if pos < len(tex) and tex[pos] == "{":
            starts.append(match.start())

    entries = []
    end_marker = re.compile(r"\\end\s*\{(?:thebibliography|references)\}")

    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(tex)
        stop = end_marker.search(tex, start, end)
        if stop:
            end = stop.start()
        entry = tex[start:end].strip()
        if entry:
            entries.append(entry)

    return entries
