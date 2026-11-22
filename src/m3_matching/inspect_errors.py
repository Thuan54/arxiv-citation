"""Phase 6A: Error Inspection, Collision Case Studies, Missed-Match Cards & 100-Entry Audit Tool.

Outputs (default mode)
----------------------
1. data/matching/analysis/m4_m5_errors_and_collisions.md
   Human-readable cards for every M4/M5 miss on valid plus every resolved
   top-1 candidate collision group (M4 vs M5 side-by-side).
2. data/matching/analysis/suspected_missed_matches_readable.md
   Side-by-side BibTeX vs predicted arXiv reference cards for rule_none entries
   where Stage-1.5 Has-Match predicted high match probability.
3. data/matching/analysis/audit_sample_100.csv
   Seeded stratified 100-entry valid sample (50 auto-matched + 50 rule_none).

Scoring mode (--score-audit)
----------------------------
Computes auto-label precision, rule_none false-negative rate, stratified true
match rate, audited MRR@5 (all & matchable), Hit@1, and audited Has-Match F1.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def _fmt_authors(authors: Any, max_n: int = 3) -> str:
    if not authors:
        return "—"
    if isinstance(authors, str):
        return authors
    if isinstance(authors, (list, tuple)):
        items = [str(a) for a in authors if a]
        if len(items) > max_n:
            return ", ".join(items[:max_n]) + f" et al. ({len(items)})"
        return ", ".join(items) if items else "—"
    return str(authors)


def _fmt_ref_line(ref_id: Optional[str], ref_meta: Optional[Dict[str, Any]]) -> str:
    if not ref_id or pd.isna(ref_id):
        return "`None` (unmatched)"
    if not ref_meta:
        return f"`{ref_id}` (metadata unavailable)"
    title = ref_meta.get("ref_title") or "—"
    authors = _fmt_authors(ref_meta.get("ref_authors"))
    date = ref_meta.get("ref_submission_date") or "—"
    return f"`{ref_id}` | **{title}** | {authors} | {date}"


def load_bibtex_lookup(
    bibtex_path: Path,
    target_pubs: Set[str],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Stream bibtex.jsonl keeping only publications in target_pubs."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not bibtex_path.exists():
        return out
    with bibtex_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pub = str(row.get("arxiv_id", ""))
            if pub in target_pubs:
                out[(pub, str(row.get("citation_key", "")))] = row
    return out


def load_references_lookup(
    refs_path: Path,
    target_pubs: Set[str],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Stream references_table.jsonl keeping only publications in target_pubs."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not refs_path.exists():
        return out
    with refs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pub = str(row.get("arxiv_id", ""))
            if pub in target_pubs:
                out[(pub, str(row.get("ref_arxiv_id", "")))] = row
    return out


def _top_k_per_entry(
    scored_df: pd.DataFrame,
    score_col: str,
    k: int = 3,
) -> Dict[Tuple[str, str], List[Tuple[str, float, int]]]:
    df = scored_df.sort_values(
        by=["arxiv_id", "citation_key", score_col, "ref_arxiv_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    out: Dict[Tuple[str, str], List[Tuple[str, float, int]]] = {}
    for (pub, ckey), grp in df.groupby(["arxiv_id", "citation_key"], sort=False):
        top = grp.head(k)
        out[(str(pub), str(ckey))] = [
            (str(r), float(s), int(lbl))
            for r, s, lbl in zip(top["ref_arxiv_id"], top[score_col], top["label"])
        ]
    return out


def write_errors_and_collisions_report(
    scored_df: pd.DataFrame,
    bib_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    ref_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    out_md: Path,
    out_jsonl: Path,
) -> Dict[str, int]:
    """Generate readable Markdown + JSONL of M4/M5 misses and resolved collisions."""
    m4_top = _top_k_per_entry(scored_df, "m4_score", k=3)
    m5_top = _top_k_per_entry(scored_df, "m5_score", k=3)

    entry_keys = ["arxiv_id", "citation_key"]
    pos_rows = scored_df[scored_df["label"] > 0].drop_duplicates(subset=entry_keys)
    gt_map = {
        (str(a), str(c)): str(r)
        for a, c, r in zip(pos_rows["arxiv_id"], pos_rows["citation_key"], pos_rows["ref_arxiv_id"])
    }
    p_has_map = (
        scored_df.drop_duplicates(subset=entry_keys)
        .set_index(entry_keys)["p_has_match"]
        .to_dict()
        if "p_has_match" in scored_df.columns
        else {}
    )
    m5_assign_map = (
        scored_df.drop_duplicates(subset=entry_keys)
        .set_index(entry_keys)["m5_assigned_ref"]
        .to_dict()
        if "m5_assigned_ref" in scored_df.columns
        else {}
    )

    m4_misses: List[Dict[str, Any]] = []
    m5_misses: List[Dict[str, Any]] = []
    for key, gt_ref in sorted(gt_map.items()):
        m4_cands = m4_top.get(key, [])
        m5_cands = m5_top.get(key, [])
        m4_c1 = m4_cands[0][0] if m4_cands else None
        m5_c1 = m5_cands[0][0] if m5_cands else None
        if m4_c1 != gt_ref or m5_c1 != gt_ref:
            rec = {
                "arxiv_id": key[0],
                "citation_key": key[1],
                "gt_ref": gt_ref,
                "m4_top3": m4_cands,
                "m5_top3": m5_cands,
                "p_has_match": round(float(p_has_map.get(key, 0.5)), 4),
                "m5_assigned_ref": m5_assign_map.get(key),
                "m4_missed": m4_c1 != gt_ref,
                "m5_missed": m5_c1 != gt_ref,
            }
            if m4_c1 != gt_ref:
                m4_misses.append(rec)
            if m5_c1 != gt_ref:
                m5_misses.append(rec)

    m4_top1_rows = (
        scored_df.sort_values(
            by=["arxiv_id", "citation_key", "m4_score", "ref_arxiv_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .groupby(entry_keys, sort=False)
        .first()
        .reset_index()
    )
    if "p_has_match" in m4_top1_rows.columns:
        plausible = m4_top1_rows[m4_top1_rows["p_has_match"] >= 0.25]
    else:
        plausible = m4_top1_rows
    cc = plausible.groupby(["arxiv_id", "ref_arxiv_id"])["citation_key"].transform("count")
    coll_df = plausible[cc >= 2].copy()

    lines: List[str] = [
        "# M4 / M5 Valid Errors & Collision Resolution Case Studies\n",
        f"- **M4 Top-1 Misses on Matchable Valid Entries:** {len(m4_misses)}",
        f"- **M5 Top-1 Misses on Matchable Valid Entries:** {len(m5_misses)}",
        f"- **M4 Top-1 Collision Groups (`p_has >= 0.25`):** "
        f"{coll_df.groupby(['arxiv_id', 'ref_arxiv_id']).ngroups if not coll_df.empty else 0}\n",
        "---\n",
        "## 1. M4 & M5 Top-1 Misses\n",
    ]

    seen_miss_keys: Set[Tuple[str, str]] = set()
    all_misses = []
    for rec in m4_misses + m5_misses:
        k = (rec["arxiv_id"], rec["citation_key"])
        if k not in seen_miss_keys:
            seen_miss_keys.add(k)
            all_misses.append(rec)

    if not all_misses:
        lines.append("_No M4 or M5 top-1 misses on valid._\n")
    else:
        for i, rec in enumerate(all_misses, 1):
            pub, ckey = rec["arxiv_id"], rec["citation_key"]
            bib = bib_lookup.get((pub, ckey), {})
            status = (
                "FIXED BY M5" if rec["m4_missed"] and not rec["m5_missed"]
                else ("M5 REGRESSION" if not rec["m4_missed"] and rec["m5_missed"] else "BOTH MISSED")
            )
            lines.append(f"### Case {i}: `{pub}` / `{ckey}` — **{status}**")
            lines.append(
                f"- **BibTeX:** **{bib.get('title') or '—'}** | "
                f"{_fmt_authors(bib.get('authors'))} | Year: `{bib.get('year')}` | "
                f"Venue: `{bib.get('journal') or '—'}` | DOI: `{bib.get('doi') or '—'}`"
            )
            lines.append(
                f"- **Ground Truth:** {_fmt_ref_line(rec['gt_ref'], ref_lookup.get((pub, rec['gt_ref'])))}"
            )
            m4_c1 = rec["m4_top3"][0][0] if rec["m4_top3"] else None
            m5_c1 = rec["m5_top3"][0][0] if rec["m5_top3"] else None
            lines.append(
                f"- **M4 Top-1:** {_fmt_ref_line(m4_c1, ref_lookup.get((pub, str(m4_c1))))}"
            )
            lines.append(
                f"- **M5 Top-1:** {_fmt_ref_line(m5_c1, ref_lookup.get((pub, str(m5_c1))))} "
                f"(assigned=`{rec['m5_assigned_ref']}`, `p_has={rec['p_has_match']}`)\n"
            )

    lines.append("---\n")
    lines.append("## 2. M4 Top-1 Collision Groups & M5 Resolution\n")
    if coll_df.empty:
        lines.append("_No plausible top-1 collision groups on valid._\n")
    else:
        for g_idx, ((pub, cand_ref), grp) in enumerate(
            coll_df.groupby(["arxiv_id", "ref_arxiv_id"]), 1
        ):
            lines.append(
                f"### Collision Group {g_idx}: Publication `{pub}` contesting "
                f"{_fmt_ref_line(cand_ref, ref_lookup.get((str(pub), str(cand_ref))))}"
            )
            for _, row in grp.iterrows():
                ckey = str(row["citation_key"])
                key = (str(pub), ckey)
                bib = bib_lookup.get(key, {})
                gt_ref = gt_map.get(key)
                m5_c1 = m5_top.get(key, [(None, 0.0, 0)])[0][0]
                m5_ass = m5_assign_map.get(key)
                lines.append(
                    f"- Entry `{ckey}` (`p_has={row.get('p_has_match', 0.0):.3f}`, `m4={row['m4_score']:.2f}`): "
                    f"**{bib.get('title') or '—'}** ({_fmt_authors(bib.get('authors'))}, {bib.get('year')})"
                )
                lines.append(
                    f"  - GT: `{gt_ref}` | M4 Top-1: `{cand_ref}` | M5 Top-1: `{m5_c1}` (assigned: `{m5_ass}`)"
                )
            lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in all_misses:
            f.write(json.dumps(rec) + "\n")

    return {
        "n_m4_misses": len(m4_misses),
        "n_m5_misses": len(m5_misses),
        "n_collision_groups": int(
            coll_df.groupby(["arxiv_id", "ref_arxiv_id"]).ngroups
        ) if not coll_df.empty else 0,
    }


def write_suspected_missed_matches_report(
    missed_files: List[Tuple[str, Path]],
    bib_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    ref_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    out_md: Path,
) -> int:
    """Render human-readable side-by-side cards for suspected rule_none missed matches."""
    lines = [
        "# Suspected Missed Matches Discovered in `rule_none`\n",
        "These BibTeX entries were labelled `null` (`rule_none`) by strict automatic rules, "
        "but Stage-1.5 Has-Match predicted high match probability.\n",
    ]
    total = 0
    for section_name, path in missed_files:
        lines.append(f"## {section_name} (`{path}`)\n")
        if not path.exists():
            lines.append("_File not found._\n")
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            lines.append("_No entries flagged above threshold._\n")
            continue
        for idx, rec in enumerate(rows, 1):
            total += 1
            pub = str(rec["arxiv_id"])
            ckey = str(rec["citation_key"])
            pred_ref = str(rec["top1_ref_arxiv_id"])
            bib = bib_lookup.get((pub, ckey), {})
            ref = ref_lookup.get((pub, pred_ref), {})
            lines.append(
                f"### {idx}. `{pub}` / `{ckey}` — `P(has_match) = {rec.get('p_has_match', 0.0):.4f}`"
            )
            lines.append(
                f"- **BibTeX:** **{bib.get('title') or '—'}** | "
                f"{_fmt_authors(bib.get('authors'))} | Year: `{bib.get('year')}` | "
                f"Journal: `{bib.get('journal') or '—'}` | DOI: `{bib.get('doi') or '—'}`"
            )
            lines.append(f"- **Predicted Ref:** {_fmt_ref_line(pred_ref, ref)}")
            lines.append(
                f"- **Signals:** `m4_score={rec.get('top1_score')}`, "
                f"`margin_1_2={rec.get('score_margin_1_2')}`, "
                f"`title_cos={rec.get('top1_title_word_tfidf_cos')}`, "
                f"`raw_cov={rec.get('top1_raw_cand_title_cov')}`, "
                f"`first_surname_jw={rec.get('top1_first_surname_jw')}`, "
                f"`claim_cnt={rec.get('top1_cand_claim_count')}`\n"
            )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return total


def build_stratified_audit_csv(
    scored_df: pd.DataFrame,
    bib_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    ref_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    out_csv: Path,
    n_matched: int = 50,
    n_unmatched: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a seeded stratified 100-entry valid audit CSV for human verification."""
    rng = np.random.default_rng(seed)
    entry_keys = ["arxiv_id", "citation_key"]
    m5_top = _top_k_per_entry(scored_df, "m5_score", k=5)

    ent_summary = (
        scored_df.groupby(entry_keys, sort=False)
        .agg(
            has_auto_match=("label", lambda s: int(s.max() > 0)),
            p_has_match=("p_has_match", "first"),
            m5_assigned_ref=("m5_assigned_ref", "first"),
        )
        .reset_index()
    )
    pos_rows = scored_df[scored_df["label"] > 0].drop_duplicates(subset=entry_keys)
    gt_map = {
        (str(a), str(c)): str(r)
        for a, c, r in zip(pos_rows["arxiv_id"], pos_rows["citation_key"], pos_rows["ref_arxiv_id"])
    }

    matched_pool = ent_summary[ent_summary["has_auto_match"] == 1].reset_index(drop=True)
    unmatched_pool = ent_summary[ent_summary["has_auto_match"] == 0].reset_index(drop=True)

    k_m = min(n_matched, len(matched_pool))
    k_u = min(n_unmatched, len(unmatched_pool))
    idx_m = sorted(rng.choice(len(matched_pool), size=k_m, replace=False).tolist())
    idx_u = sorted(rng.choice(len(unmatched_pool), size=k_u, replace=False).tolist())

    sampled = pd.concat(
        [
            matched_pool.iloc[idx_m].assign(strata="auto_matched"),
            unmatched_pool.iloc[idx_u].assign(strata="rule_none"),
        ],
        ignore_index=True,
    )

    pop_counts = {
        "auto_matched": int(len(matched_pool)),
        "rule_none": int(len(unmatched_pool)),
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "strata",
        "stratum_pop_n",
        "arxiv_id",
        "citation_key",
        "bib_title",
        "bib_authors",
        "bib_year",
        "bib_journal",
        "auto_label_ref",
        "p_has_match",
        "m5_assigned_ref",
        "m5_cand_1",
        "m5_cand_1_title",
        "m5_cand_1_authors",
        "m5_cand_2",
        "m5_cand_2_title",
        "m5_cand_3",
        "m5_cand_3_title",
        "m5_cand_4",
        "m5_cand_5",
        "human_verdict",
        "human_true_ref",
        "notes",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in sampled.iterrows():
            pub = str(row["arxiv_id"])
            ckey = str(row["citation_key"])
            bib = bib_lookup.get((pub, ckey), {})
            cands = [c[0] for c in m5_top.get((pub, ckey), [])]
            cands += [""] * (5 - len(cands))
            c1_meta = ref_lookup.get((pub, cands[0]), {}) if cands[0] else {}
            c2_meta = ref_lookup.get((pub, cands[1]), {}) if cands[1] else {}
            c3_meta = ref_lookup.get((pub, cands[2]), {}) if cands[2] else {}
            strata = str(row["strata"])
            writer.writerow({
                "sample_id": i + 1,
                "strata": strata,
                "stratum_pop_n": pop_counts[strata],
                "arxiv_id": pub,
                "citation_key": ckey,
                "bib_title": bib.get("title") or "",
                "bib_authors": _fmt_authors(bib.get("authors"), max_n=4),
                "bib_year": bib.get("year") or "",
                "bib_journal": bib.get("journal") or "",
                "auto_label_ref": gt_map.get((pub, ckey), ""),
                "p_has_match": round(float(row["p_has_match"]), 4),
                "m5_assigned_ref": "" if pd.isna(row["m5_assigned_ref"]) else str(row["m5_assigned_ref"]),
                "m5_cand_1": cands[0],
                "m5_cand_1_title": c1_meta.get("ref_title") or "",
                "m5_cand_1_authors": _fmt_authors(c1_meta.get("ref_authors"), max_n=3),
                "m5_cand_2": cands[1],
                "m5_cand_2_title": c2_meta.get("ref_title") or "",
                "m5_cand_3": cands[2],
                "m5_cand_3_title": c3_meta.get("ref_title") or "",
                "m5_cand_4": cands[3],
                "m5_cand_5": cands[4],
                "human_verdict": "",
                "human_true_ref": "",
                "notes": "",
            })

    return {
        "out_csv": str(out_csv),
        "n_sampled_matched": k_m,
        "n_sampled_rule_none": k_u,
        "population_counts": pop_counts,
    }


def score_audit_csv(audit_csv: Path, has_match_threshold: float = 0.45) -> Dict[str, Any]:
    """Compute unbiased human-audited precision, FN rate, and Horvitz-Thompson MRR@5."""
    rows: List[Dict[str, str]] = []
    with audit_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            verdict = (r.get("human_verdict") or "").strip().lower()
            if verdict in {"match", "no_match"}:
                rows.append(r)

    if not rows:
        raise ValueError(
            "No completed rows found. Fill 'human_verdict' with 'match' or 'no_match' "
            "(and 'human_true_ref' when 'match')."
        )

    matched_stratum = [r for r in rows if r["strata"] == "auto_matched"]
    none_stratum = [r for r in rows if r["strata"] == "rule_none"]

    auto_tp = sum(
        1
        for r in matched_stratum
        if r["human_verdict"].strip().lower() == "match"
        and (r["human_true_ref"].strip() == r["auto_label_ref"].strip())
    )
    auto_prec = auto_tp / max(1, len(matched_stratum))

    none_fn = sum(
        1 for r in none_stratum if r["human_verdict"].strip().lower() == "match"
    )
    none_fnr = none_fn / max(1, len(none_stratum))

    pop_m = float(matched_stratum[0]["stratum_pop_n"]) if matched_stratum else 0.0
    pop_u = float(none_stratum[0]["stratum_pop_n"]) if none_stratum else 0.0
    w_m = pop_m / max(1, len(matched_stratum)) if matched_stratum else 1.0
    w_u = pop_u / max(1, len(none_stratum)) if none_stratum else 1.0

    rr_w_sum = 0.0
    hit1_w_sum = 0.0
    matchable_w_sum = 0.0
    total_w_sum = 0.0
    hm_tp = hm_fp = hm_fn = 0.0

    for r in rows:
        w = w_m if r["strata"] == "auto_matched" else w_u
        total_w_sum += w
        is_match = r["human_verdict"].strip().lower() == "match"
        true_ref = r["human_true_ref"].strip()
        cands = [r.get(f"m5_cand_{k}", "").strip() for k in range(1, 6)]
        pred_pos = float(r.get("p_has_match") or 0.0) >= has_match_threshold

        if is_match:
            matchable_w_sum += w
            if true_ref and true_ref in cands:
                rank = cands.index(true_ref) + 1
                rr_w_sum += w * (1.0 / rank)
                if rank == 1:
                    hit1_w_sum += w
            if pred_pos:
                hm_tp += w
            else:
                hm_fn += w
        else:
            if pred_pos:
                hm_fp += w

    hm_prec = hm_tp / max(1e-12, hm_tp + hm_fp)
    hm_rec = hm_tp / max(1e-12, hm_tp + hm_fn)
    hm_f1 = (2.0 * hm_prec * hm_rec) / max(1e-12, hm_prec + hm_rec)

    return {
        "n_audited_total": len(rows),
        "n_audited_auto_matched": len(matched_stratum),
        "n_audited_rule_none": len(none_stratum),
        "auto_label_precision": round(auto_prec, 4),
        "rule_none_false_negative_rate": round(none_fnr, 4),
        "estimated_true_match_rate": round(matchable_w_sum / max(1e-12, total_w_sum), 4),
        "audited_mrr5_all_weighted": round(rr_w_sum / max(1e-12, total_w_sum), 4),
        "audited_mrr5_matchable_weighted": round(rr_w_sum / max(1e-12, matchable_w_sum), 4),
        "audited_hit1_matchable_weighted": round(hit1_w_sum / max(1e-12, matchable_w_sum), 4),
        "audited_has_match_precision": round(hm_prec, 4),
        "audited_has_match_recall": round(hm_rec, 4),
        "audited_has_match_f1": round(hm_f1, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect M4/M5 valid errors, collisions, missed matches & 100-entry audit."
    )
    parser.add_argument("--models-dir", type=Path, default=Path("data/matching/models"))
    parser.add_argument("--bibtex", type=Path, default=Path("data/processed/bibtex.jsonl"))
    parser.add_argument("--references", type=Path, default=Path("data/metadata/references_table.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/matching/analysis"))
    parser.add_argument(
        "--score-audit",
        type=Path,
        default=None,
        help="Score a completed human-audited CSV instead of generating reports.",
    )
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.score_audit is not None:
        result = score_audit_csv(args.score_audit, has_match_threshold=args.threshold)
        print(json.dumps(result, indent=2))
        return

    scored_path = args.models_dir / "valid_scored_pairs.parquet"
    if not scored_path.exists():
        raise SystemExit(
            f"Missing {scored_path}. Run updated src/03_matching/models.py first."
        )

    scored_df = pd.read_parquet(scored_path)
    target_pubs: Set[str] = set(scored_df["arxiv_id"].astype(str).unique())

    missed_files = [
        ("Valid Split", args.models_dir / "suspected_missed_matches_valid.jsonl"),
        ("Train OOF Split", args.models_dir / "suspected_missed_matches_train_oof.jsonl"),
    ]
    for _, mp in missed_files:
        if mp.exists():
            for line in mp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    target_pubs.add(str(json.loads(line)["arxiv_id"]))

    bib_lookup = load_bibtex_lookup(args.bibtex, target_pubs)
    ref_lookup = load_references_lookup(args.references, target_pubs)

    err_summary = write_errors_and_collisions_report(
        scored_df,
        bib_lookup,
        ref_lookup,
        args.out_dir / "m4_m5_errors_and_collisions.md",
        args.out_dir / "m4_m5_errors.jsonl",
    )
    n_missed = write_suspected_missed_matches_report(
        missed_files,
        bib_lookup,
        ref_lookup,
        args.out_dir / "suspected_missed_matches_readable.md",
    )
    audit_info = build_stratified_audit_csv(
        scored_df,
        bib_lookup,
        ref_lookup,
        args.out_dir / "audit_sample_100.csv",
        n_matched=50,
        n_unmatched=50,
        seed=args.seed,
    )

    print(
        json.dumps(
            {
                "errors_and_collisions": err_summary,
                "suspected_missed_cards_written": n_missed,
                "audit_sample": audit_info,
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()