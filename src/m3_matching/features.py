"""Phase 4: Pairwise feature engineering for Milestone 3 reference matching."""
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from rapidfuzz.fuzz import ratio as _rf_ratio
except ImportError:
    _rf_ratio = None

FEATURE_COLS: List[str] = [
    # 1. Identifiers
    "arxiv_eq", "doi_eq", "arxiv_in_raw", "id_conflict",
    # 2. Title
    "title_exact", "title_main_exact", "title_prefix",
    "title_tok_jaccard", "title_tok_containment",
    "title_word_tfidf_cos", "title_char_ngram_cos",
    "title_lcs_ratio", "title_len_ratio", "title_num_overlap",
    "title_bm25", "title_bm25_rr",
    # 3. Raw-text fallback
    "raw_cand_title_cov", "raw_cand_surname_hit", "raw_cand_year_hit",
    # 4. Authors
    "first_surname_eq", "first_surname_jw",
    "surname_jaccard", "surname_containment",
    "first_initial_match", "author_count_ratio", "entry_et_al",
    # 5. Year
    "year_diff_abs", "year_eq", "year_within_2", "year_diff_arxiv", "year_missing",
    # 6. Venue & Meta
    "venue_tok_jaccard", "entry_is_arxiv", "cand_is_arxiv_venue",
    "parse_completeness", "n_candidates_log",
]

# Monotone constraints for LightGBM (+1 higher => more likely match, -1 lower => more likely, 0 free)
MONOTONE_CONSTRAINTS: List[int] = [
    # Identifiers
    1, 1, 1, -1,
    # Title
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    # Raw-text fallback
    1, 1, 1,
    # Authors
    1, 1, 1, 1, 1, 0, 0,
    # Year
    -1, 1, 1, -1, 0,
    # Venue & Meta
    1, 0, 0, 0, 0,
]


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    t = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1
    jaro = (matches / len1 + matches / len2 + (matches - t / 2.0) / matches) / 3.0
    prefix = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1.0 - jaro)


def _char_ngrams(text: str, ns: Tuple[int, ...] = (3, 4)) -> Counter:
    if not text:
        return Counter()
    padded = f" {text} "
    length = len(padded)
    counts: Counter = Counter()
    for n in ns:
        if length >= n:
            for i in range(length - n + 1):
                counts[padded[i:i + n]] += 1
    return counts


def _normalize_vec(counts: Counter, idf: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if not counts:
        return {}
    vec = {k: float(v) * (idf.get(k, 1.0) if idf else 1.0) for k, v in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0.0:
        return {}
    inv = 1.0 / norm
    return {k: v * inv for k, v in vec.items()}


def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b[k] for k, v in a.items() if k in b)


def _title_lcs_ratio(a: str, b: str, char_cos: float, tok_cont: float) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if _rf_ratio is not None:
        return _rf_ratio(a, b) / 100.0
    if char_cos < 0.15 and tok_cont < 0.15:
        return 0.0
    sm = SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < 0.25:
        return 0.0
    return sm.ratio()


def extract_publication_pairs(
    arxiv_id: str,
    partition: str,
    source_pool: str,
    entries: List[dict],
    cands: List[dict],
    labels: Dict[Tuple[str, str], Optional[str]],
) -> List[dict]:
    """Compute all m x n pairwise feature rows for one publication."""
    m, n = len(entries), len(cands)
    if m == 0 or n == 0:
        return []

    # In-publication word IDF (leak-free: uses only this paper's entries + candidates)
    df_word: Counter = Counter()
    for item in entries + cands:
        for tok in set(item.get("title_tokens") or []):
            df_word[tok] += 1
    n_docs = m + n
    idf_word = {t: math.log((n_docs + 1.0) / (df + 1.0)) + 1.0 for t, df in df_word.items()}

    # Candidate-pool BM25 statistics
    df_cand: Counter = Counter()
    cand_lens = []
    for c in cands:
        toks = c.get("title_tokens") or []
        cand_lens.append(len(toks))
        for tok in set(toks):
            df_cand[tok] += 1
    avgdl = (sum(cand_lens) / n) or 1.0
    bm25_idf = {t: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for t, df in df_cand.items()}

    # Precompute per-entry and per-candidate representations once: O(m + n)
    e_prep = []
    for e in entries:
        toks = e.get("title_tokens") or []
        e_arxivs = set(e.get("arxiv_ids_all") or [])
        if e.get("arxiv_id_extracted"):
            e_arxivs.add(e["arxiv_id_extracted"])
        e_prep.append({
            "tok_set": set(toks),
            "word_vec": _normalize_vec(Counter(toks), idf_word),
            "char_vec": _normalize_vec(_char_ngrams(e.get("title_clean") or "")),
            "surnames": set(e.get("author_surnames") or []),
            "venue_toks": set((e.get("venue_compact") or "").split()),
            "raw_words": set((e.get("raw_text_clean") or "").split()),
            "arxivs": e_arxivs,
            "nums": e.get("title_numbers") or [],
            "num_set": set(e.get("title_numbers") or []),
        })

    c_prep = []
    for c in cands:
        toks = c.get("title_tokens") or []
        c_aid = c["ref_arxiv_id"]
        c_prep.append({
            "tok_set": set(toks),
            "tok_counts": Counter(toks),
            "word_vec": _normalize_vec(Counter(toks), idf_word),
            "char_vec": _normalize_vec(_char_ngrams(c.get("title_clean") or "")),
            "surnames": set(c.get("author_surnames") or []),
            "venue_toks": set((c.get("venue_compact") or "").split()),
            "aid_norm": c_aid.replace(".", " ").replace("/", " ").lower(),
            "nums": c.get("title_numbers") or [],
            "num_set": set(c.get("title_numbers") or []),
        })

    n_cands_log = math.log1p(n)
    k1, b_param = 1.5, 0.75
    rows: List[dict] = []

    for i, e in enumerate(entries):
        ckey = e["citation_key"]
        has_label = (arxiv_id, ckey) in labels
        true_ref = labels.get((arxiv_id, ckey))

        ep = e_prep[i]
        e_title = e.get("title_clean") or ""
        e_main = e.get("title_main") or ""
        e_len = len(e_title)
        e_sur = e.get("first_surname") or ""
        e_init = (e.get("first_initials") or [None])[0]
        e_year = e.get("year")
        e_doi = e.get("doi")
        e_raw = e.get("raw_text_clean") or ""
        e_acount = e.get("author_count") or 0

        bm25_scores = []
        for j in range(n):
            dl = cand_lens[j]
            score = 0.0
            if ep["tok_set"] and dl > 0:
                cc = c_prep[j]["tok_counts"]
                denom_base = k1 * (1.0 - b_param + b_param * (dl / avgdl))
                for t in ep["tok_set"]:
                    tf = cc.get(t, 0)
                    if tf:
                        score += bm25_idf.get(t, 0.0) * (tf * (k1 + 1.0)) / (tf + denom_base)
            bm25_scores.append(score)

        order = sorted(range(n), key=lambda j: bm25_scores[j], reverse=True)
        bm25_ranks = [0] * n
        for r_idx, j in enumerate(order, 1):
            bm25_ranks[j] = r_idx if bm25_scores[j] > 0.0 else n + 1

        for j, c in enumerate(cands):
            cp = c_prep[j]
            c_aid = c["ref_arxiv_id"]
            c_title = c.get("title_clean") or ""
            c_main = c.get("title_main") or ""
            c_len = len(c_title)
            c_sur = c.get("first_surname") or ""
            c_init = (c.get("first_initials") or [None])[0]
            c_year = c.get("year")
            c_ayear = c.get("arxiv_year")
            c_doi = c.get("doi")
            c_acount = c.get("author_count") or 0

            # 1. Identifiers
            arxiv_eq = 1.0 if c_aid in ep["arxivs"] else 0.0
            doi_eq = 1.0 if (e_doi and c_doi and e_doi == c_doi) else 0.0
            arxiv_in_raw = 1.0 if (cp["aid_norm"] and cp["aid_norm"] in e_raw) else 0.0
            id_conflict = 1.0 if (
                (ep["arxivs"] and not arxiv_eq)
                or (e_doi and c_doi and not doi_eq)
            ) else 0.0

            # 2. Title
            title_exact = 1.0 if (e_title and e_title == c_title) else 0.0
            title_main_exact = 1.0 if (e_main and e_main == c_main) else 0.0
            if e_title and c_title and e_title != c_title:
                short, long_ = (e_title, c_title) if e_len <= c_len else (c_title, e_title)
                title_prefix = 1.0 if (len(short) >= 15 and long_.startswith(short + " ")) else 0.0
            else:
                title_prefix = 0.0

            et_set, ct_set = ep["tok_set"], cp["tok_set"]
            if et_set and ct_set:
                inter = len(et_set & ct_set)
                tok_jacc = inter / len(et_set | ct_set)
                tok_cont = inter / min(len(et_set), len(ct_set))
            else:
                tok_jacc = 0.0
                tok_cont = 0.0

            word_cos = _dot(ep["word_vec"], cp["word_vec"])
            char_cos = _dot(ep["char_vec"], cp["char_vec"])
            lcs_rat = _title_lcs_ratio(e_title, c_title, char_cos, tok_cont)
            len_rat = (min(e_len, c_len) / max(e_len, c_len)) if (e_len and c_len) else 0.0

            if ep["num_set"] and cp["num_set"]:
                if ep["nums"] == cp["nums"]:
                    num_score = 1.0
                elif ep["num_set"] & cp["num_set"]:
                    num_score = len(ep["num_set"] & cp["num_set"]) / len(ep["num_set"] | cp["num_set"])
                else:
                    num_score = -1.0
            else:
                num_score = 0.0

            # 3. Raw-text fallback
            raw_title_cov = (len(ct_set & ep["raw_words"]) / len(ct_set)) if (ct_set and ep["raw_words"]) else 0.0
            raw_sur_hit = 1.0 if (c_sur and c_sur in ep["raw_words"]) else 0.0
            raw_year_hit = 1.0 if (c_year is not None and str(c_year) in ep["raw_words"]) else 0.0

            # 4. Authors
            sur_eq = 1.0 if (e_sur and e_sur == c_sur) else 0.0
            sur_jw = 1.0 if sur_eq else jaro_winkler(e_sur, c_sur)
            es_set, cs_set = ep["surnames"], cp["surnames"]
            if es_set and cs_set:
                s_inter = len(es_set & cs_set)
                sur_jacc = s_inter / len(es_set | cs_set)
                sur_cont = s_inter / min(len(es_set), len(cs_set))
            else:
                sur_jacc = 0.0
                sur_cont = 0.0
            if e_init and c_init:
                init_match = 1.0 if e_init == c_init else -1.0
            else:
                init_match = 0.0
            acount_rat = (min(e_acount, c_acount) / max(e_acount, c_acount)) if (e_acount and c_acount) else 0.0

            # 5. Year
            if e_year is not None and c_year is not None:
                ydiff = float(abs(e_year - c_year))
                yeq = 1.0 if ydiff == 0.0 else 0.0
                yw2 = 1.0 if ydiff <= 2.0 else 0.0
                ymiss = 0.0
            else:
                ydiff = float("nan")
                yeq = 0.0
                yw2 = 0.0
                ymiss = 1.0
            ydiff_arxiv = (
                float(abs(e_year - c_ayear))
                if (e_year is not None and c_ayear is not None)
                else float("nan")
            )

            # 6. Venue & Meta
            ev_set, cv_set = ep["venue_toks"], cp["venue_toks"]
            venue_jacc = (len(ev_set & cv_set) / len(ev_set | cv_set)) if (ev_set and cv_set) else 0.0

            rows.append({
                "arxiv_id": arxiv_id,
                "partition": partition,
                "source_pool": source_pool,
                "citation_key": ckey,
                "ref_arxiv_id": c_aid,
                "has_label": has_label,
                "groundtruth_arxiv_id": true_ref,
                "label": int(true_ref == c_aid) if has_label else -1,
                "arxiv_eq": arxiv_eq,
                "doi_eq": doi_eq,
                "arxiv_in_raw": arxiv_in_raw,
                "id_conflict": id_conflict,
                "title_exact": title_exact,
                "title_main_exact": title_main_exact,
                "title_prefix": title_prefix,
                "title_tok_jaccard": tok_jacc,
                "title_tok_containment": tok_cont,
                "title_word_tfidf_cos": word_cos,
                "title_char_ngram_cos": char_cos,
                "title_lcs_ratio": lcs_rat,
                "title_len_ratio": len_rat,
                "title_num_overlap": num_score,
                "title_bm25": bm25_scores[j],
                "title_bm25_rr": 1.0 / bm25_ranks[j],
                "raw_cand_title_cov": raw_title_cov,
                "raw_cand_surname_hit": raw_sur_hit,
                "raw_cand_year_hit": raw_year_hit,
                "first_surname_eq": sur_eq,
                "first_surname_jw": sur_jw,
                "surname_jaccard": sur_jacc,
                "surname_containment": sur_cont,
                "first_initial_match": init_match,
                "author_count_ratio": acount_rat,
                "entry_et_al": 1.0 if e.get("et_al") else 0.0,
                "year_diff_abs": ydiff,
                "year_eq": yeq,
                "year_within_2": yw2,
                "year_diff_arxiv": ydiff_arxiv,
                "year_missing": ymiss,
                "venue_tok_jaccard": venue_jacc,
                "entry_is_arxiv": 1.0 if e.get("is_arxiv_entry") else 0.0,
                "cand_is_arxiv_venue": 1.0 if c.get("is_arxiv_venue") else 0.0,
                "parse_completeness": (e.get("parse_completeness") or 0) / 6.0,
                "n_candidates_log": n_cands_log,
            })

    return rows


def _load_by_pub(path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                grouped[r["arxiv_id"]].append(r)
    return grouped


def _load_labels(path: Path) -> Dict[Tuple[str, str], Optional[str]]:
    labels: Dict[Tuple[str, str], Optional[str]] = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                labels[(r["arxiv_id"], r["citation_key"])] = r.get("true_ref_arxiv_id")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Extract Phase 4 pairwise matching features.")
    parser.add_argument("--splits", default="data/matching/splits.json")
    parser.add_argument("--bibtex", default="data/matching/preprocessed_bibtex.jsonl")
    parser.add_argument("--references", default="data/matching/preprocessed_references.jsonl")
    parser.add_argument(
        "--labels",
        default="data/matching/labels.jsonl",
        help="Merged labels.jsonl (or pass data/matching/auto_labels.jsonl before manual merge)",
    )
    parser.add_argument("--out", default="data/matching/pair_features.parquet")
    args = parser.parse_args()

    t0 = time.perf_counter()
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    bib_by_pub = _load_by_pub(Path(args.bibtex))
    refs_by_pub = _load_by_pub(Path(args.references))
    labels = _load_labels(Path(args.labels))

    pub_meta: List[Tuple[str, str, str]] = []
    if isinstance(splits.get("partitions"), dict):
        for part in ("train", "valid", "test"):
            for src in ("manual", "auto"):
                for aid in splits["partitions"][part][src]:
                    pub_meta.append((aid, part, src))
    else:
        split_dict = splits.get("splits", splits)
        for part in ("train", "valid", "test"):
            for aid in split_dict.get(part, []):
                pub_meta.append((aid, part, "auto"))

    all_rows: List[dict] = []
    for aid, part, src in pub_meta:
        all_rows.extend(
            extract_publication_pairs(
                arxiv_id=aid,
                partition=part,
                source_pool=src,
                entries=bib_by_pub.get(aid, []),
                cands=refs_by_pub.get(aid, []),
                labels=labels,
            )
        )

    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    summary = {
        "publications": len(pub_meta),
        "total_pairs": len(df),
        "labelled_pairs": int(df["has_label"].sum()),
        "positive_pairs": int((df["label"] == 1).sum()),
        "features": len(FEATURE_COLS),
        "elapsed_seconds": round(elapsed, 2),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()