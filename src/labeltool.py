"""Phase 3C: Manual labelling sheets, auto-label audit, Cohen's kappa, and labels.jsonl merge."""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_jsonl_by_pub(path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[row["arxiv_id"]].append(row)
    return grouped


def _load_raw_bibtex_map(path: Path) -> Dict[Tuple[str, str], dict]:
    index: Dict[Tuple[str, str], dict] = {}
    if not path.exists():
        return index
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            index[(row["arxiv_id"], row["citation_key"])] = row
    return index


def _load_citing_sentences(elements_path: Optional[Path], target_pubs: set) -> Dict[Tuple[str, str], str]:
    """Find first sentence mentioning \\cite*{...key...} per (arxiv_id, citation_key)."""
    found: Dict[Tuple[str, str], str] = {}
    if not elements_path or not elements_path.exists():
        return found
    cite_re = re.compile(r"\\[A-Za-z]*[Cc]ite[A-Za-z]*\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^{}]*)\}")
    with elements_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            aid = row.get("arxiv_id")
            if aid not in target_pubs:
                continue
            content = row.get("content") or ""
            if "cite" not in content.lower():
                continue
            for m in cite_re.finditer(content):
                for key in m.group(1).split(","):
                    k = key.strip()
                    if k and (aid, k) not in found:
                        found[(aid, k)] = content[:400]
    return found


def _quick_score(entry: dict, cand: dict) -> float:
    """Fast lexical score solely to order the top-8 UI hints (free search always available)."""
    score = 0.0
    if entry.get("arxiv_id_extracted") and entry["arxiv_id_extracted"] == cand["ref_arxiv_id"]:
        score += 5.0
    if entry.get("doi") and cand.get("doi") and entry["doi"] == cand["doi"]:
        score += 5.0
    et, ct = set(entry.get("title_tokens") or []), set(cand.get("title_tokens") or [])
    if et and ct:
        score += 2.0 * len(et & ct) / len(et | ct)
    elif entry.get("title_clean") and cand.get("title_clean"):
        score += SequenceMatcher(None, entry["title_clean"], cand["title_clean"]).ratio()
    if entry.get("first_surname") and entry["first_surname"] == cand.get("first_surname"):
        score += 1.0
    ey, cy = entry.get("year"), cand.get("year")
    if ey and cy and abs(ey - cy) <= 1:
        score += 0.5
    return score


def _format_cand(cand: dict, raw_ref: Optional[dict]) -> str:
    title = (raw_ref or {}).get("ref_title") or cand.get("title_clean") or ""
    authors = (raw_ref or {}).get("ref_authors") or []
    first_auth = authors[0] if authors else (cand.get("first_surname") or "?")
    year = cand.get("year") or "?"
    return f"{cand['ref_arxiv_id']} | {year} | {first_auth} | {title[:110]}"


def cmd_sheets(args):
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    manual_pubs = sorted(
        splits["partitions"]["train"]["manual"]
        + splits["partitions"]["valid"]["manual"]
        + splits["partitions"]["test"]["manual"]
    )
    bib_by_pub = _load_jsonl_by_pub(Path(args.bibtex))
    refs_by_pub = _load_jsonl_by_pub(Path(args.references))
    raw_bib = _load_raw_bibtex_map(Path(args.raw_bibtex))
    raw_refs = {}
    if Path(args.raw_references).exists():
        with Path(args.raw_references).open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    raw_refs[(r["arxiv_id"], r["ref_arxiv_id"])] = r

    citing_sents = _load_citing_sentences(
        Path(args.elements) if args.elements else None, set(manual_pubs)
    )

    out_dir = Path(args.out_dir)
    sheets_dir = out_dir / "manual_sheets"
    cands_dir = out_dir / "candidate_lists"
    double_dir = out_dir / "double_label"
    for d in (sheets_dir, cands_dir, double_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    fieldnames = [
        "arxiv_id", "citation_key", "entry_type", "title", "authors", "year",
        "doi", "journal", "url", "citing_sentence",
        *[f"hint_{i}" for i in range(1, 9)],
        "true_ref_arxiv_id", "unsure", "annotator", "note",
    ]

    total_entries = 0
    double_rows = []
    for idx, aid in enumerate(manual_pubs):
        entries = bib_by_pub.get(aid, [])
        cands = refs_by_pub.get(aid, [])
        member_tag = f"member_{(idx % splits['members']) + 1}"

        # Write full candidate table for unbiased free search
        with (cands_dir / f"{aid.replace('/', '_')}_candidates.csv").open("w", encoding="utf-8", newline="") as cfh:
            w = csv.DictWriter(cfh, fieldnames=["ref_arxiv_id", "year", "first_surname", "ref_title", "ref_authors", "doi", "venue"])
            w.writeheader()
            for c in sorted(cands, key=lambda x: (x.get("year") or 0, x["ref_arxiv_id"])):
                rr = raw_refs.get((aid, c["ref_arxiv_id"])) or {}
                w.writerow({
                    "ref_arxiv_id": c["ref_arxiv_id"],
                    "year": c.get("year") or "",
                    "first_surname": c.get("first_surname") or "",
                    "ref_title": rr.get("ref_title") or c.get("title_clean") or "",
                    "ref_authors": "; ".join(rr.get("ref_authors") or []),
                    "doi": c.get("doi") or "",
                    "venue": c.get("venue_clean") or "",
                })

        pub_rows = []
        for e in entries:
            rb = raw_bib.get((aid, e["citation_key"])) or {}
            ranked = sorted(cands, key=lambda c: _quick_score(e, c), reverse=True)[:8]
            hints = [_format_cand(c, raw_refs.get((aid, c["ref_arxiv_id"]))) for c in ranked]
            hints += [""] * (8 - len(hints))
            row = {
                "arxiv_id": aid,
                "citation_key": e["citation_key"],
                "entry_type": rb.get("entry_type") or e.get("entry_type") or "",
                "title": rb.get("title") or e.get("title_clean") or "",
                "authors": "; ".join(rb.get("authors") or []),
                "year": rb.get("year") if rb.get("year") is not None else (e.get("year") or ""),
                "doi": rb.get("doi") or e.get("doi") or "",
                "journal": rb.get("journal") or e.get("venue_clean") or "",
                "url": rb.get("url") or "",
                "citing_sentence": citing_sents.get((aid, e["citation_key"]), ""),
                **{f"hint_{i}": hints[i - 1] for i in range(1, 9)},
                "true_ref_arxiv_id": "",
                "unsure": "false",
                "annotator": member_tag,
                "note": "",
            }
            pub_rows.append(row)
            total_entries += 1

        with (sheets_dir / f"{member_tag}_{aid.replace('/', '_')}.csv").open("w", encoding="utf-8", newline="") as sfh:
            w = csv.DictWriter(sfh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(pub_rows)

        # Sample 20% for double-labelling by a second member
        k_double = max(1, round(0.20 * len(pub_rows)))
        second_member = f"member_{((idx + 1) % splits['members']) + 1}"
        for r in rng.sample(pub_rows, min(k_double, len(pub_rows))):
            dup = dict(r)
            dup["annotator"] = second_member
            double_rows.append(dup)

    with (double_dir / "double_label_20pct.csv").open("w", encoding="utf-8", newline="") as dfh:
        w = csv.DictWriter(dfh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(double_rows)

    print(json.dumps({
        "manual_publications": len(manual_pubs),
        "total_manual_entries": total_entries,
        "double_label_entries": len(double_rows),
        "sheets_dir": str(sheets_dir),
        "candidates_dir": str(cands_dir),
    }, indent=2))


def cmd_audit_sample(args):
    """Export stratified 100-entry audit sample from auto_labels.jsonl for precision reporting."""
    rng = random.Random(args.seed)
    by_rule: Dict[str, List[dict]] = defaultdict(list)
    with Path(args.auto_labels).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                by_rule[r["annotator"]].append(r)

    # Oversample fuzzy + title rules where errors are most likely
    quotas = {
        "rule_4_fuzzy_title": 35,
        "rule_3_exact_title": 25,
        "rule_2_doi": 15,
        "rule_1_arxiv_id": 10,
        "rule_none": 15,
    }
    sample = []
    for rule, q in quotas.items():
        pool = by_rule.get(rule, [])
        sample.extend(rng.sample(pool, min(q, len(pool))))

    raw_bib = _load_raw_bibtex_map(Path(args.raw_bibtex))
    raw_refs = {}
    if Path(args.raw_references).exists():
        with Path(args.raw_references).open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    raw_refs[(r["arxiv_id"], r["ref_arxiv_id"])] = r

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arxiv_id", "citation_key", "auto_rule", "auto_ref_arxiv_id",
        "bib_title", "bib_authors", "bib_year",
        "ref_title", "ref_authors", "ref_date",
        "is_correct", "corrected_ref_arxiv_id", "note",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sample:
            rb = raw_bib.get((r["arxiv_id"], r["citation_key"])) or {}
            rr = raw_refs.get((r["arxiv_id"], r["true_ref_arxiv_id"])) if r["true_ref_arxiv_id"] else {}
            w.writerow({
                "arxiv_id": r["arxiv_id"],
                "citation_key": r["citation_key"],
                "auto_rule": r["annotator"],
                "auto_ref_arxiv_id": r["true_ref_arxiv_id"] or "NONE",
                "bib_title": rb.get("title") or "",
                "bib_authors": "; ".join(rb.get("authors") or []),
                "bib_year": rb.get("year") or "",
                "ref_title": (rr or {}).get("ref_title") or "",
                "ref_authors": "; ".join((rr or {}).get("ref_authors") or []),
                "ref_date": (rr or {}).get("ref_submission_date") or "",
                "is_correct": "",
                "corrected_ref_arxiv_id": "",
                "note": "",
            })
    print(f"Wrote {len(sample)} auto-audit rows to {out_path}")


def _norm_label(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = val.strip()
    if not s or s.upper() in {"NONE", "NULL", "N/A", "-"}:
        return None
    return s


def cohens_kappa(labels_a: List[Optional[str]], labels_b: List[Optional[str]]) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in (set(ca) | set(cb)))
    return 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)


def cmd_kappa(args):
    primary: Dict[Tuple[str, str], Optional[str]] = {}
    for p in sorted(Path(args.sheets_dir).glob("*.csv")):
        with p.open("r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                primary[(r["arxiv_id"].strip(), r["citation_key"].strip())] = _norm_label(r["true_ref_arxiv_id"])

    la, lb, disagreements = [], [], []
    with Path(args.double_csv).open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["arxiv_id"].strip(), r["citation_key"].strip())
            if key not in primary:
                continue
            a = primary[key]
            b = _norm_label(r["true_ref_arxiv_id"])
            la.append(a)
            lb.append(b)
            if a != b:
                disagreements.append({"arxiv_id": key[0], "citation_key": key[1], "primary": a, "secondary": b})

    exact = sum(a == b for a, b in zip(la, lb))
    print(json.dumps({
        "pairs_compared": len(la),
        "exact_agreement_pct": round(100 * exact / len(la), 2) if la else 0.0,
        "cohens_kappa": round(cohens_kappa(la, lb), 4),
        "disagreements_count": len(disagreements),
        "disagreements": disagreements[:25],
    }, indent=2))


def cmd_merge(args):
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    manual_pubs = set(
        splits["partitions"]["train"]["manual"]
        + splits["partitions"]["valid"]["manual"]
        + splits["partitions"]["test"]["manual"]
    )
    bib_by_pub = _load_jsonl_by_pub(Path(args.bibtex))
    refs_by_pub = _load_jsonl_by_pub(Path(args.references))
    valid_cands = {aid: {c["ref_arxiv_id"] for c in cs} for aid, cs in refs_by_pub.items()}

    manual_records: Dict[Tuple[str, str], dict] = {}
    errors = []
    for p in sorted(Path(args.sheets_dir).glob("*.csv")):
        with p.open("r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                aid = r["arxiv_id"].strip()
                ckey = r["citation_key"].strip()
                if aid not in manual_pubs:
                    continue
                target = _norm_label(r["true_ref_arxiv_id"])
                unsure = r.get("unsure", "false").strip().lower() in {"true", "1", "yes"}
                if target is not None and target not in valid_cands.get(aid, set()):
                    errors.append(f"Invalid candidate {target} for ({aid}, {ckey}) in {p.name}")
                if unsure and not args.allow_unsure:
                    errors.append(f"Unresolved unsure=true for ({aid}, {ckey}) in {p.name}")
                manual_records[(aid, ckey)] = {
                    "arxiv_id": aid,
                    "citation_key": ckey,
                    "true_ref_arxiv_id": target,
                    "source": "manual",
                    "annotator": (r.get("annotator") or "manual").strip(),
                    "confidence": 1.0,
                    "unsure": unsure,
                }

    # Verify every entry of every manual publication is labelled
    for aid in sorted(manual_pubs):
        for e in bib_by_pub.get(aid, []):
            if (aid, e["citation_key"]) not in manual_records:
                errors.append(f"Missing manual label for ({aid}, {e['citation_key']})")

    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):", *errors[:20], sep="\n  ")
        raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for key in sorted(manual_records):
            out_fh.write(json.dumps(manual_records[key], ensure_ascii=False) + "\n")
            written += 1
        with Path(args.auto_labels).open("r", encoding="utf-8") as auto_fh:
            for line in auto_fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["arxiv_id"] in manual_pubs:
                    continue
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(json.dumps({
        "status": "ok",
        "manual_entries": len(manual_records),
        "total_labels_written": written,
        "output": str(out_path),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Milestone 3 manual labelling & validation tool.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("sheets")
    p1.add_argument("--splits", default="data/matching/splits.json")
    p1.add_argument("--bibtex", default="data/matching/preprocessed_bibtex.jsonl")
    p1.add_argument("--references", default="data/matching/preprocessed_references.jsonl")
    p1.add_argument("--raw-bibtex", default="data/parsed/bibtex.jsonl")
    p1.add_argument("--raw-references", default="data/metadata/references_table.jsonl")
    p1.add_argument("--elements", default="data/processed/elements.jsonl")
    p1.add_argument("--out-dir", default="data/matching/labelling")
    p1.add_argument("--seed", type=int, default=42)
    p1.set_defaults(func=cmd_sheets)

    p2 = sub.add_parser("audit-sample")
    p2.add_argument("--auto-labels", default="data/matching/auto_labels.jsonl")
    p2.add_argument("--raw-bibtex", default="data/parsed/bibtex.jsonl")
    p2.add_argument("--raw-references", default="data/metadata/references_table.jsonl")
    p2.add_argument("--out", default="data/matching/labelling/auto_audit_100.csv")
    p2.add_argument("--seed", type=int, default=42)
    p2.set_defaults(func=cmd_audit_sample)

    p3 = sub.add_parser("kappa")
    p3.add_argument("--sheets-dir", default="data/matching/labelling/manual_sheets")
    p3.add_argument("--double-csv", default="data/matching/labelling/double_label/double_label_20pct.csv")
    p3.set_defaults(func=cmd_kappa)

    p4 = sub.add_parser("merge")
    p4.add_argument("--splits", default="data/matching/splits.json")
    p4.add_argument("--bibtex", default="data/matching/preprocessed_bibtex.jsonl")
    p4.add_argument("--references", default="data/matching/preprocessed_references.jsonl")
    p4.add_argument("--sheets-dir", default="data/matching/labelling/manual_sheets")
    p4.add_argument("--auto-labels", default="data/matching/auto_labels.jsonl")
    p4.add_argument("--out", default="data/matching/labels.jsonl")
    p4.add_argument("--allow-unsure", action="store_true")
    p4.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()