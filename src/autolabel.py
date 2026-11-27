"""Phase 3B: High-precision rule-based automatic matcher -> auto_labels.jsonl."""
import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List


def _load_by_pub(path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[row["arxiv_id"]].append(row)
    return grouped


def _strict_token_set_sim(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Sorted unique-token string ratio (penalizes both missing and extra tokens)."""
    if not tokens_a or not tokens_b:
        return 0.0
    sa, sb = sorted(set(tokens_a)), sorted(set(tokens_b))
    s_a, s_b = " ".join(sa), " ".join(sb)
    sm = SequenceMatcher(None, s_a, s_b, autojunk=False)
    if sm.real_quick_ratio() < 0.90 or sm.quick_ratio() < 0.90:
        return 0.0
    return sm.ratio()


def _surname_compatible(entry: dict, ref: dict) -> bool:
    ea, ra = entry.get("first_surname", ""), ref.get("first_surname", "")
    if not ea or not ra:
        return False
    if ea == ra:
        return True
    return ea in set(ref.get("author_surnames") or []) and ra in set(entry.get("author_surnames") or [])


def _numbers_conflict(entry: dict, ref: dict) -> bool:
    na, nb = entry.get("title_numbers") or [], ref.get("title_numbers") or []
    return bool(na and nb and na != nb)


def match_entry_auto(entry: dict, candidates: List[dict]) -> dict:
    """Cascade matcher: first strict rule that fires wins."""
    # Rule 1: exact arXiv ID match
    entry_arxivs = set(entry.get("arxiv_ids_all") or [])
    if entry.get("arxiv_id_extracted"):
        entry_arxivs.add(entry["arxiv_id_extracted"])
    if entry_arxivs:
        hits = [c for c in candidates if c["ref_arxiv_id"] in entry_arxivs]
        if len(hits) == 1:
            return {"true_ref_arxiv_id": hits[0]["ref_arxiv_id"], "rule": "rule_1_arxiv_id", "confidence": 1.0, "unsure": False}
        if len(hits) > 1:
            return {"true_ref_arxiv_id": hits[0]["ref_arxiv_id"], "rule": "rule_1_arxiv_id", "confidence": 1.0, "unsure": True}

    # Rule 2: exact DOI match
    if entry.get("doi"):
        hits = [c for c in candidates if c.get("doi") and c["doi"] == entry["doi"]]
        if len(hits) == 1:
            return {"true_ref_arxiv_id": hits[0]["ref_arxiv_id"], "rule": "rule_2_doi", "confidence": 1.0, "unsure": False}
        if len(hits) > 1:
            return {"true_ref_arxiv_id": hits[0]["ref_arxiv_id"], "rule": "rule_2_doi", "confidence": 1.0, "unsure": True}

    # Rule 3: exact normalized title (>= 3 words) + (author compatible OR year within 2)
    et = entry.get("title_clean") or ""
    if et and len(et.split()) >= 3:
        ey = entry.get("year")
        hits = []
        for c in candidates:
            if c.get("title_clean") != et:
                continue
            cy = c.get("year")
            year_ok = ey is not None and cy is not None and abs(ey - cy) <= 2
            if _surname_compatible(entry, c) or year_ok:
                hits.append(c)
        if len(hits) == 1:
            return {"true_ref_arxiv_id": hits[0]["ref_arxiv_id"], "rule": "rule_3_exact_title", "confidence": 0.95, "unsure": False}
        if len(hits) > 1:
            return {"true_ref_arxiv_id": hits[0]["ref_arxiv_id"], "rule": "rule_3_exact_title", "confidence": 0.90, "unsure": True}

    # Rule 4: strict token-set similarity >= 0.92 + same first surname + year within 1 + margin
    e_toks = entry.get("title_tokens") or []
    e_sur = entry.get("first_surname") or ""
    ey = entry.get("year")
    if len(e_toks) >= 3 and e_sur and ey is not None:
        scored = []
        for c in candidates:
            if c.get("first_surname") != e_sur:
                continue
            cy = c.get("year")
            if cy is None or abs(ey - cy) > 1:
                continue
            if _numbers_conflict(entry, c):
                continue
            sim = _strict_token_set_sim(e_toks, c.get("title_tokens") or [])
            if sim >= 0.92:
                scored.append((sim, c))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best_sim, best_c = scored[0]
            runner_sim = scored[1][0] if len(scored) > 1 else 0.0
            if len(scored) == 1 or (best_sim - runner_sim) >= 0.05:
                return {
                    "true_ref_arxiv_id": best_c["ref_arxiv_id"],
                    "rule": "rule_4_fuzzy_title",
                    "confidence": round(0.85 * best_sim, 4),
                    "unsure": False,
                }
            return {"true_ref_arxiv_id": None, "rule": "rule_none_ambiguous", "confidence": 0.0, "unsure": True}

    return {"true_ref_arxiv_id": None, "rule": "rule_none", "confidence": 0.0, "unsure": False}


def main():
    parser = argparse.ArgumentParser(description="Auto-label the automatic pool from splits.json.")
    parser.add_argument("--splits", default="data/matching/splits.json")
    parser.add_argument("--bibtex", default="data/matching/preprocessed_bibtex.jsonl")
    parser.add_argument("--references", default="data/matching/preprocessed_references.jsonl")
    parser.add_argument("--out", default="data/matching/auto_labels.jsonl")
    args = parser.parse_args()

    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    auto_pubs = sorted(
        splits["partitions"]["train"]["auto"]
        + splits["partitions"]["valid"]["auto"]
        + splits["partitions"]["test"]["auto"]
    )

    bib_by_pub = _load_by_pub(Path(args.bibtex))
    refs_by_pub = _load_by_pub(Path(args.references))

    rule_counts: Counter = Counter()
    matched = 0
    total = 0
    unsure_count = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_candidates = 0
    unique_matched_refs = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for aid in auto_pubs:
            entries = bib_by_pub.get(aid, [])
            candidates = refs_by_pub.get(aid, [])
            total_candidates += len(candidates)
            matched_in_pub = set()
            for entry in entries:
                res = match_entry_auto(entry, candidates)
                total += 1
                rule_counts[res["rule"]] += 1
                if res["true_ref_arxiv_id"] is not None:
                    matched += 1
                    matched_in_pub.add(res["true_ref_arxiv_id"])
                if res["unsure"]:
                    unsure_count += 1
                record = {
                    "arxiv_id": aid,
                    "citation_key": entry["citation_key"],
                    "true_ref_arxiv_id": res["true_ref_arxiv_id"],
                    "source": "auto",
                    "annotator": res["rule"],
                    "confidence": res["confidence"],
                    "unsure": res["unsure"],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            unique_matched_refs += len(matched_in_pub)

    print(json.dumps({
        "auto_publications": len(auto_pubs),
        "total_entries_labelled": total,
        "total_candidates": total_candidates,
        "matched_entries": matched,
        "unique_candidates_matched": unique_matched_refs,
        "match_rate_pct": round(100 * matched / total, 2) if total else 0.0,
        "candidate_coverage_pct": round(100 * unique_matched_refs / total_candidates, 2) if total_candidates else 0.0,
        "unsure_entries": unsure_count,
        "by_rule": dict(rule_counts.most_common()),
    }, indent=2))


if __name__ == "__main__":
    main()
    