"""Phase 0 + Phase 1 runner: join M1 references + M2 bibtex, clean, and audit eligibility."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from clean import clean_arxiv_id, preprocess_bibtex_entry, preprocess_reference


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc


def _load_bibtex_extras(path: Optional[Path]) -> Dict[Tuple[str, str], dict]:
    """Optional index from `bibtex_entries.jsonl` -> (arxiv_id, citation_key) -> extra info."""
    extras: Dict[Tuple[str, str], dict] = {}
    if not path or not path.exists():
        return extras
    for rec in _iter_jsonl(path):
        aid = clean_arxiv_id(rec.get("arxiv_id")) or rec.get("arxiv_id")
        if not aid:
            continue
        for entry in rec.get("entries", []):
            key = entry.get("citation_key")
            if key:
                extras[(aid, key)] = {
                    "fields": entry.get("fields") or {},
                    "et_al": entry.get("et_al", False),
                    "bibtex": entry.get("bibtex") or "",
                }
    return extras


def _load_s2_extras(path: Optional[Path]) -> Dict[Tuple[str, str], dict]:
    """Optional index from `s2_references.jsonl` -> (arxiv_id, ref_arxiv_id) -> DOI/venue."""
    extras: Dict[Tuple[str, str], dict] = {}
    if not path or not path.exists():
        return extras
    for rec in _iter_jsonl(path):
        aid = clean_arxiv_id(rec.get("arxiv_id")) or rec.get("arxiv_id")
        if not aid:
            continue
        for ref in rec.get("s2_references") or []:
            cited = ref.get("citedPaper") if isinstance(ref, dict) and "citedPaper" in ref else ref
            if not isinstance(cited, dict):
                continue
            ext_ids = cited.get("external_ids") or cited.get("externalIds") or {}
            ref_aid = clean_arxiv_id(cited.get("arxiv_id") or ext_ids.get("ArXiv"))
            if ref_aid:
                extras.setdefault((aid, ref_aid), {
                    "external_ids": ext_ids,
                    "doi": ext_ids.get("DOI") or ext_ids.get("doi"),
                    "venue": cited.get("venue"),
                    "year": cited.get("year"),
                })
    return extras

def load_and_preprocess(
    bibtex_path: Path,
    references_path: Path,
    bibtex_entries_path: Optional[Path] = None,
    s2_references_path: Optional[Path] = None,
) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]], dict]:
    """Load, clean, deduplicate by key, and compute Phase 0 eligibility metrics."""
    bib_extras = _load_bibtex_extras(bibtex_entries_path)
    s2_extras = _load_s2_extras(s2_references_path)

    bib_by_pub: Dict[str, List[dict]] = defaultdict(list)
    seen_bib_keys = set()
    dup_bib_keys = 0

    for row in _iter_jsonl(bibtex_path):
        aid = clean_arxiv_id(row["arxiv_id"]) or str(row["arxiv_id"]).strip()
        key = row["citation_key"]
        pair = (aid, key)
        if pair in seen_bib_keys:
            dup_bib_keys += 1
            continue
        seen_bib_keys.add(pair)
        cleaned = preprocess_bibtex_entry(row, bib_extras.get(pair))
        bib_by_pub[aid].append(cleaned)

    refs_by_pub: Dict[str, List[dict]] = defaultdict(list)
    seen_ref_keys = set()
    dup_ref_keys = 0

    for row in _iter_jsonl(references_path):
        aid = clean_arxiv_id(row["arxiv_id"]) or str(row["arxiv_id"]).strip()
        ref_aid = clean_arxiv_id(row["ref_arxiv_id"]) or str(row["ref_arxiv_id"]).strip()
        pair = (aid, ref_aid)
        if pair in seen_ref_keys:
            dup_ref_keys += 1
            continue
        seen_ref_keys.add(pair)
        cleaned = preprocess_reference(row, s2_extras.get(pair))
        refs_by_pub[aid].append(cleaned)

    all_pubs = sorted(set(bib_by_pub) | set(refs_by_pub))
    per_pub = []
    manual_eligible = 0
    auto_eligible = 0
    zero_bib = []
    zero_ref = []

    for aid in all_pubs:
        entries = bib_by_pub.get(aid, [])
        refs = refs_by_pub.get(aid, [])
        m, n = len(entries), len(refs)
        if m == 0:
            zero_bib.append(aid)
        if n == 0:
            zero_ref.append(aid)

        is_manual = (m >= 20 and n >= 5)
        is_auto = (m >= 1 and n >= 1)
        manual_eligible += int(is_manual)
        auto_eligible += int(is_auto)

        per_pub.append({
            "arxiv_id": aid,
            "m_entries": m,
            "n_references": n,
            "ratio_m_to_n": round(m / n, 3) if n else None,
            "manual_eligible": is_manual,
            "auto_eligible": is_auto,
            "pct_title": round(100 * sum(bool(e["title_clean"]) for e in entries) / m, 1) if m else 0.0,
            "pct_authors": round(100 * sum(e["author_count"] > 0 for e in entries) / m, 1) if m else 0.0,
            "pct_year": round(100 * sum(e["year"] is not None for e in entries) / m, 1) if m else 0.0,
            "pct_doi": round(100 * sum(e["doi"] is not None for e in entries) / m, 1) if m else 0.0,
            "pct_arxiv_id": round(100 * sum(e["arxiv_id_extracted"] is not None for e in entries) / m, 1) if m else 0.0,
        })

    audit = {
        "total_publications_seen": len(all_pubs),
        "total_bibtex_entries": sum(len(v) for v in bib_by_pub.values()),
        "total_reference_candidates": sum(len(v) for v in refs_by_pub.values()),
        "duplicate_bibtex_keys_dropped": dup_bib_keys,
        "duplicate_reference_ids_dropped": dup_ref_keys,
        "manual_eligible_publications": manual_eligible,
        "auto_eligible_publications": auto_eligible,
        "publications_with_zero_bibtex": len(zero_bib),
        "publications_with_zero_references": len(zero_ref),
        "per_publication": per_pub,
    }
    return bib_by_pub, refs_by_pub, audit


def main():
    parser = argparse.ArgumentParser(description="Preprocess M1+M2 tables for Milestone 3 matching.")
    parser.add_argument("--bibtex", default="data/parsed/bibtex.jsonl")
    parser.add_argument("--bibtex-entries", default="data/parsed/bibtex_entries.jsonl")
    parser.add_argument("--references", default="data/metadata/references_table.jsonl")
    parser.add_argument("--s2-references", default="data/metadata/s2_references.jsonl")
    parser.add_argument("--out-dir", default="data/matching")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bib_by_pub, refs_by_pub, audit = load_and_preprocess(
        bibtex_path=Path(args.bibtex),
        references_path=Path(args.references),
        bibtex_entries_path=Path(args.bibtex_entries) if args.bibtex_entries else None,
        s2_references_path=Path(args.s2_references) if args.s2_references else None,
    )

    with (out_dir / "preprocessed_bibtex.jsonl").open("w", encoding="utf-8") as fh:
        for aid in sorted(bib_by_pub):
            for row in bib_by_pub[aid]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "preprocessed_references.jsonl").open("w", encoding="utf-8") as fh:
        for aid in sorted(refs_by_pub):
            for row in refs_by_pub[aid]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "eligibility.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    summary = {k: v for k, v in audit.items() if k != "per_publication"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()