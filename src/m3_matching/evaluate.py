"""Phase 6: Statistical Evaluation, Feature-Group Ablations & Poster Plots.

Produces
--------
1. data/matching/analysis/phase6_evaluation.json
   - Publication-clustered 95% bootstrap CIs (1,000 resamples by arxiv_id)
   - Paired publication-clustered bootstrap significance tests (M5 - M4, M4 - B1)
   - Drop-one-feature-group ablation study (Clean Valid & Asymmetric Hard Stress)
2. data/matching/plots/ (300 DPI PNG + vector SVG, Portrait A2 poster ready):
   - 01_model_ladder_clean_vs_stress.{png,svg}
   - 02_has_match_pr_and_calibration.{png,svg}
   - 03_feature_importance_and_ablation.{png,svg}
   - 04_collision_and_score_separation.{png,svg}

Classical ML only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, precision_recall_curve

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from has_match import expected_calibration_error, train_and_evaluate_has_match
from models import (
    apply_asymmetric_hard_stress,
    discover_feature_cols,
    evaluate_ranking_dual,
    fit_stage1_models,
)
from sampling import load_pub_to_split


PALETTE = {
    "B1": "#7F8C8D",
    "M3": "#3498DB",
    "M4": "#E67E22",
    "M5": "#1B6CA8",
    "match": "#1B9E77",
    "nomatch": "#D95F02",
    "accent": "#C0392B",
    "dark": "#1F2937",
    "grid": "#E5E7EB",
}


def _configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#E2E8F0",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _per_pub_ranking_stats(
    pair_df: pd.DataFrame,
    score_col: str,
    k: int = 5,
) -> pd.DataFrame:
    """Compute exact per-publication (n_entries, n_matchable, rr_sum, hit1_sum)."""
    df = pair_df.sort_values(
        by=["arxiv_id", "citation_key", score_col, "ref_arxiv_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    entry_keys = ["arxiv_id", "citation_key"]
    df["_rank"] = df.groupby(entry_keys, sort=False).cumcount() + 1

    ent = df.groupby(entry_keys, sort=False).agg(
        has_match=("label", lambda s: int(s.max() > 0)),
    ).reset_index()

    pos = df[df["label"] > 0].groupby(entry_keys, sort=False)["_rank"].min()
    ent["_pos_rank"] = ent.set_index(entry_keys).index.map(pos)
    ent["rr"] = np.where(
        ent["_pos_rank"].notna() & (ent["_pos_rank"] <= k),
        1.0 / ent["_pos_rank"].fillna(1.0),
        0.0,
    )
    ent["hit1"] = np.where(ent["_pos_rank"] == 1, 1.0, 0.0)

    return (
        ent.groupby("arxiv_id", sort=True)
        .agg(
            n_entries=("has_match", "count"),
            n_matchable=("has_match", "sum"),
            rr_sum=("rr", "sum"),
            hit1_sum=("hit1", "sum"),
        )
        .reset_index()
    )


def publication_bootstrap_ci(
    pair_df: pd.DataFrame,
    score_col: str,
    n_boot: int = 1000,
    seed: int = 42,
    k: int = 5,
) -> Dict[str, Any]:
    """Compute 95% publication-clustered bootstrap CIs for dual MRR@5 & Hit@1."""
    pub = _per_pub_ranking_stats(pair_df, score_col=score_col, k=k)
    n_pubs = len(pub)
    if n_pubs == 0:
        return {}

    n_ent = pub["n_entries"].to_numpy(dtype=float)
    n_mat = pub["n_matchable"].to_numpy(dtype=float)
    rr_s = pub["rr_sum"].to_numpy(dtype=float)
    h1_s = pub["hit1_sum"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pubs, size=(n_boot, n_pubs))

    b_ent = np.maximum(n_ent[idx].sum(axis=1), 1.0)
    b_mat = np.maximum(n_mat[idx].sum(axis=1), 1.0)
    b_rr = rr_s[idx].sum(axis=1)
    b_h1 = h1_s[idx].sum(axis=1)

    def _pack(point: float, dist: np.ndarray) -> Dict[str, float]:
        lo, hi = np.percentile(dist, [2.5, 97.5])
        return {
            "point": round(float(point), 4),
            "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4),
        }

    return {
        "n_pubs": int(n_pubs),
        "n_boot": int(n_boot),
        "mrr5_all": _pack(rr_s.sum() / max(1.0, n_ent.sum()), b_rr / b_ent),
        "mrr5_matchable": _pack(rr_s.sum() / max(1.0, n_mat.sum()), b_rr / b_mat),
        "hit1_matchable": _pack(h1_s.sum() / max(1.0, n_mat.sum()), b_h1 / b_mat),
    }


def paired_publication_bootstrap_test(
    pair_df: pd.DataFrame,
    score_col_a: str,
    score_col_b: str,
    n_boot: int = 1000,
    seed: int = 42,
    k: int = 5,
) -> Dict[str, float]:
    """Paired publication-clustered bootstrap test for delta = metric(A) - metric(B)."""
    pub_a = _per_pub_ranking_stats(pair_df, score_col=score_col_a, k=k)
    pub_b = _per_pub_ranking_stats(pair_df, score_col=score_col_b, k=k)
    merged = pub_a.merge(pub_b, on="arxiv_id", suffixes=("_a", "_b"))
    n_pubs = len(merged)
    if n_pubs == 0:
        return {}

    n_mat = merged["n_matchable_a"].to_numpy(dtype=float)
    rr_a = merged["rr_sum_a"].to_numpy(dtype=float)
    rr_b = merged["rr_sum_b"].to_numpy(dtype=float)
    h1_a = merged["hit1_sum_a"].to_numpy(dtype=float)
    h1_b = merged["hit1_sum_b"].to_numpy(dtype=float)

    point_mrr_diff = (rr_a.sum() - rr_b.sum()) / max(1.0, n_mat.sum())
    point_h1_diff = (h1_a.sum() - h1_b.sum()) / max(1.0, n_mat.sum())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pubs, size=(n_boot, n_pubs))
    b_mat = np.maximum(n_mat[idx].sum(axis=1), 1.0)
    mrr_diff = (rr_a[idx].sum(axis=1) - rr_b[idx].sum(axis=1)) / b_mat

    p_mrr = 2.0 * min(float((mrr_diff <= 0).mean()), float((mrr_diff >= 0).mean()))
    lo, hi = np.percentile(mrr_diff, [2.5, 97.5])
    return {
        "delta_mrr5_matchable": round(float(point_mrr_diff), 5),
        "ci_low": round(float(lo), 5),
        "ci_high": round(float(hi), 5),
        "delta_hit1_matchable": round(float(point_h1_diff), 5),
        "p_value_two_sided": round(min(1.0, p_mrr), 4),
    }


def partition_feature_groups(feature_cols: Sequence[str]) -> Dict[str, List[str]]:
    """Map numeric feature columns into 5 interpretable Milestone 3 groups."""
    groups: Dict[str, List[str]] = {
        "Identifiers": [],
        "Title": [],
        "Authors": [],
        "Year & Venue": [],
        "Raw-Text Fallbacks": [],
    }
    for c in feature_cols:
        cl = c.lower()
        if cl.startswith("raw_"):
            groups["Raw-Text Fallbacks"].append(c)
        elif any(k in cl for k in ("arxiv", "doi", "id_conflict")):
            groups["Identifiers"].append(c)
        elif "title" in cl:
            groups["Title"].append(c)
        elif any(k in cl for k in ("surname", "author")):
            groups["Authors"].append(c)
        elif any(k in cl for k in ("year", "venue", "journal")):
            groups["Year & Venue"].append(c)
    return {k: v for k, v in groups.items() if v}


def run_feature_group_ablations(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Drop-One-Group-Out ablations on Clean Valid and Asymmetric Hard Stress."""
    all_cols = discover_feature_cols(train_raw)
    groups = partition_feature_groups(all_cols)
    va_stress_raw = apply_asymmetric_hard_stress(valid_raw, seed=seed)

    def _eval_subset(cols: List[str], exclude_ids: bool = False) -> Dict[str, float]:
        tr_s, va_s, st_s, _ = fit_stage1_models(
            train_raw,
            valid_raw,
            va_stress_raw,
            feature_cols=cols,
            n_splits=5,
            seed=seed,
            exclude_ids=exclude_ids,
        )
        hm = train_and_evaluate_has_match(
            tr_s,
            va_s,
            st_s,
            score_col="m4_score",
            n_splits=5,
            seed=seed,
            exclude_id_features=exclude_ids,
        )
        va_ev = evaluate_ranking_dual(va_s, "m4_score")
        st_ev = evaluate_ranking_dual(st_s, "m4_score") if st_s is not None else {}
        hm_va = hm.metrics["valid"][hm.best_model_name]
        hm_st = hm.metrics.get("test", {}).get(hm.best_model_name, {})
        return {
            "n_features": len(cols),
            "clean_mrr5_matchable": va_ev["mrr5_matchable"],
            "clean_hit1_matchable": va_ev["hit1_matchable"],
            "clean_has_match_f1": hm_va["f1"],
            "stress_mrr5_matchable": st_ev.get("mrr5_matchable", 0.0),
            "stress_hit1_matchable": st_ev.get("hit1_matchable", 0.0),
            "stress_has_match_f1": hm_st.get("f1", 0.0),
        }

    full_metrics = _eval_subset(all_cols, exclude_ids=False)
    results: Dict[str, Any] = {"Full_Model": full_metrics, "drop_one_group": {}}

    for g_name, g_cols in groups.items():
        kept = [c for c in all_cols if c not in set(g_cols)]
        if not kept:
            continue
        sub = _eval_subset(kept, exclude_ids=(g_name == "Identifiers"))
        sub["dropped_cols"] = g_cols
        sub["delta_clean_mrr5"] = round(
            sub["clean_mrr5_matchable"] - full_metrics["clean_mrr5_matchable"], 4
        )
        sub["delta_clean_has_match_f1"] = round(
            sub["clean_has_match_f1"] - full_metrics["clean_has_match_f1"], 4
        )
        sub["delta_stress_mrr5"] = round(
            sub["stress_mrr5_matchable"] - full_metrics["stress_mrr5_matchable"], 4
        )
        sub["delta_stress_has_match_f1"] = round(
            sub["stress_has_match_f1"] - full_metrics["stress_has_match_f1"], 4
        )
        results["drop_one_group"][g_name] = sub

    return results


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_01_model_ladder(
    eval_report: Dict[str, Any],
    boot_clean: Dict[str, Any],
    boot_stress: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Plot 1: Dual MRR@5 Model Ladder (Clean Valid vs Asymmetric Hard Stress)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)
    models = [
        ("B1_lexical", "B1_no_id", "B1\nLexical"),
        ("M3_logreg", "M3_no_id", "M3\nLogReg"),
        ("M4_lgbm_rank", "M4_no_id", "M4\nLGBM Rank"),
        ("M5_collective", "M5_no_id", "M5\nCollective"),
    ]

    clean_src = eval_report.get("valid_ladder", {})
    stress_src = eval_report.get("stress_benchmark", {}).get("asymmetric_hard_stress", {})

    panels = [
        (axes[0], "Clean Valid (112 Unseen Papers)", clean_src, boot_clean, False),
        (axes[1], "Asymmetric Hard-Citation Stress (IDs Masked)", stress_src, boot_stress, True),
    ]

    for ax, title, src, boot, is_stress in panels:
        x = np.arange(len(models))
        w = 0.36
        mat_vals, mat_err_lo, mat_err_hi, all_vals = [], [], [], []
        oracle = 0.0

        for k_clean, k_stress, _ in models:
            key = k_stress if is_stress else k_clean
            row = src.get(key, {})
            mv = float(row.get("mrr5_matchable", 0.0))
            av = float(row.get("mrr5_all", 0.0))
            oracle = max(oracle, float(row.get("oracle_ceiling", 0.0)))
            mat_vals.append(mv)
            all_vals.append(av)

            b_key = key.split("_")[0]
            b_row = boot.get(b_key, {}).get("mrr5_matchable", {})
            mat_err_lo.append(max(0.0, mv - float(b_row.get("ci_low", mv))))
            mat_err_hi.append(max(0.0, float(b_row.get("ci_high", mv)) - mv))

        bars1 = ax.bar(
            x - w / 2,
            mat_vals,
            width=w,
            color=PALETTE["M5"],
            label="MRR@5 (Matchable Only)",
            yerr=[mat_err_lo, mat_err_hi],
            capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": "#334155"},
        )
        bars2 = ax.bar(
            x + w / 2,
            all_vals,
            width=w,
            color=PALETTE["M4"],
            alpha=0.88,
            label="MRR@5 (All Labelled)",
        )
        if oracle > 0:
            ax.axhline(
                oracle,
                color=PALETTE["accent"],
                linestyle="--",
                linewidth=1.3,
                label=f"All-Labelled Oracle Ceiling ({oracle:.3f})",
            )

        for b in bars1:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.018,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=PALETTE["dark"],
            )
        for b in bars2:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.014,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#7C2D12",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([m[2] for m in models])
        ax.set_ylim(0.0, 1.12)
        ax.set_ylabel("MRR@5 Score")
        ax.set_title(title)
        ax.grid(axis="y", color=PALETTE["grid"], linestyle="-", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Reference Matching Model Ladder: Dual MRR@5 Evaluation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, out_dir, "01_model_ladder_clean_vs_stress")


def plot_02_has_match_pr_and_calibration(
    hm_entries_df: pd.DataFrame,
    eval_report: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Plot 2: Stage-1.5 Has-Match Precision-Recall Curve & 10-Bin Reliability Diagram."""
    fig, (ax_pr, ax_cal) = plt.subplots(1, 2, figsize=(12.2, 4.8))

    y = hm_entries_df["y_has_match"].to_numpy(dtype=int)
    p_h2 = np.clip(hm_entries_df["p_has_match"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    p_h0 = expit(hm_entries_df["top1_score"].to_numpy(dtype=float))

    for label, probs, color, ls in [
        ("H2 Calibrated LGBM (Stage-1.5)", p_h2, PALETTE["M5"], "-"),
        ("H0 Raw M4 Score Baseline", p_h0, PALETTE["B1"], "--"),
    ]:
        prec, rec, _ = precision_recall_curve(y, probs)
        ap = average_precision_score(y, probs)
        ax_pr.plot(rec, prec, label=f"{label} (PR-AUC={ap:.4f})", color=color, linestyle=ls, linewidth=2.2)

    hm_best = eval_report.get("has_match", {}).get("best_model", "H2_lgbm_calibrated")
    hm_m = eval_report.get("has_match", {}).get("metrics", {}).get("valid", {}).get(hm_best, {})
    if hm_m:
        ax_pr.scatter(
            [hm_m.get("recall", 0)],
            [hm_m.get("precision", 0)],
            color=PALETTE["accent"],
            s=65,
            zorder=5,
            label=f"Tuned t*={hm_m.get('threshold', 0.5):.3f} (F1={hm_m.get('f1', 0):.4f})",
        )

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Stage-1.5 Has-Match Precision-Recall (Valid)")
    ax_pr.set_xlim(0.0, 1.02)
    ax_pr.set_ylim(0.0, 1.04)
    ax_pr.grid(color=PALETTE["grid"], linewidth=0.7)
    ax_pr.legend(loc="lower left", fontsize=8.5)

    bins = np.linspace(0.0, 1.0, 11)
    bin_conf, bin_acc = [], []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_h2 >= lo) & (p_h2 <= hi if i == 9 else p_h2 < hi)
        if mask.sum() > 0:
            bin_conf.append(float(p_h2[mask].mean()))
            bin_acc.append(float(y[mask].mean()))

    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1.4, label="Perfect Calibration")
    ax_cal.plot(bin_conf, bin_acc, marker="o", markersize=6.5, linewidth=2.2, color=PALETTE["M5"], label="H2 Calibrated LGBM")
    ece = expected_calibration_error(y, p_h2, n_bins=10)
    brier = float(np.mean((y - p_h2) ** 2))
    ax_cal.text(
        0.05,
        0.84,
        f"10-bin ECE = {ece:.4f}\nBrier Score = {brier:.4f}\nValid N = {len(y):,}",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
        fontsize=9.5,
    )
    ax_cal.set_xlabel("Mean Predicted Probability P(has_match)")
    ax_cal.set_ylabel("Empirical Match Fraction")
    ax_cal.set_title("Has-Match Probability Calibration (Reliability)")
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.04)
    ax_cal.grid(color=PALETTE["grid"], linewidth=0.7)
    ax_cal.legend(loc="lower right", fontsize=8.8)

    fig.tight_layout()
    _save_fig(fig, out_dir, "02_has_match_pr_and_calibration")


def plot_03_feature_importance_and_ablation(
    eval_report: Dict[str, Any],
    ablations: Optional[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Plot 3: Top-8 Stage-1 Feature Gain + Feature-Group Ablation Delta."""
    has_abl = bool(ablations and ablations.get("drop_one_group"))
    n_cols = 2 if has_abl else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(13.0 if has_abl else 7.2, 5.2))
    ax_imp = axes[0] if has_abl else axes

    gain_dict = eval_report.get("top_10_stage1_features_gain", {})
    items = list(gain_dict.items())[:8]
    if items:
        names = [k for k, _ in items][::-1]
        vals = np.array([float(v) for _, v in items][::-1])
        pct = 100.0 * vals / max(vals.sum(), 1e-9)
        bars = ax_imp.barh(names, pct, color=PALETTE["M5"], height=0.62)
        for b, v in zip(bars, pct):
            ax_imp.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.8)
        ax_imp.set_xlim(0, max(pct) * 1.18)
    ax_imp.set_xlabel("Share of Top-8 Total Split Gain (%)")
    ax_imp.set_title("Stage-1 LightGBM Ranker: Top Feature Importance")
    ax_imp.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)
    ax_imp.set_axisbelow(True)

    if has_abl and ablations is not None:
        ax_ab = axes[1]
        drop_map = ablations["drop_one_group"]
        g_names = list(drop_map.keys())
        d_mrr = [float(drop_map[g]["delta_stress_mrr5"]) for g in g_names]
        d_f1 = [float(drop_map[g]["delta_stress_has_match_f1"]) for g in g_names]

        y_pos = np.arange(len(g_names))
        h = 0.36
        ax_ab.barh(y_pos - h / 2, d_mrr, height=h, color=PALETTE["M4"], label="Δ MRR@5 (Hard Stress)")
        ax_ab.barh(y_pos + h / 2, d_f1, height=h, color=PALETTE["accent"], alpha=0.85, label="Δ Has-Match F1 (Hard Stress)")
        ax_ab.axvline(0.0, color=PALETTE["dark"], linewidth=1.0)
        ax_ab.set_yticks(y_pos)
        ax_ab.set_yticklabels([f"Drop {g}" for g in g_names])
        ax_ab.set_xlabel("Metric Change vs. Full Model (Negative = Group Helps)")
        ax_ab.set_title("Drop-One-Feature-Group Ablation Study")
        ax_ab.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)
        ax_ab.set_axisbelow(True)
        ax_ab.legend(loc="lower left", fontsize=8.5)

    fig.tight_layout()
    _save_fig(fig, out_dir, "03_feature_importance_and_ablation")


def plot_04_collision_and_score_separation(
    hm_entries_df: pd.DataFrame,
    eval_report: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Plot 4: M4 vs M5 Collision Resolution + Calibrated P(has_match) Separation."""
    fig, (ax_col, ax_sep) = plt.subplots(1, 2, figsize=(12.2, 4.6))

    c_val = eval_report.get("m5_collisions_valid", {})
    c_str = eval_report.get("stress_benchmark", {}).get("asymmetric_hard_stress", {}).get("m5_collisions", {})

    metrics_labels = ["Valid Coll.\nMRR@5", "Valid Coll.\nHit@1", "Stress Coll.\nMRR@5", "Stress Coll.\nHit@1"]
    m4_vals = [
        float(c_val.get("m4_collision_mrr5", 0.0)),
        float(c_val.get("m4_collision_hit1", 0.0)),
        float(c_str.get("m4_collision_mrr5", 0.0)),
        float(c_str.get("m4_collision_hit1", 0.0)),
    ]
    m5_vals = [
        float(c_val.get("m5_collision_mrr5", 0.0)),
        float(c_val.get("m5_collision_hit1", 0.0)),
        float(c_str.get("m5_collision_mrr5", 0.0)),
        float(c_str.get("m5_collision_hit1", 0.0)),
    ]

    x = np.arange(len(metrics_labels))
    w = 0.35
    b4 = ax_col.bar(x - w / 2, m4_vals, width=w, color=PALETTE["M4"], label="M4 Pairwise Ranker")
    b5 = ax_col.bar(x + w / 2, m5_vals, width=w, color=PALETTE["M5"], label="M5 Calibrated Hungarian")
    for bars in (b4, b5):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax_col.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.3)
    ax_col.set_xticks(x)
    ax_col.set_xticklabels(metrics_labels)
    ax_col.set_ylim(0.0, 1.12)
    ax_col.set_ylabel("Score on Colliding Entries")
    ax_col.set_title(
        f"Stage-2 M5 Collision Resolution (Valid={c_val.get('n_collision_groups', 0)}, "
        f"Stress={c_str.get('n_collision_groups', 0)} groups)"
    )
    ax_col.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax_col.set_axisbelow(True)
    ax_col.legend(loc="lower right", fontsize=8.8)

    y = hm_entries_df["y_has_match"].to_numpy(dtype=int)
    probs = np.clip(hm_entries_df["p_has_match"].to_numpy(dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, 26)
    ax_sep.hist(probs[y == 0], bins=bins, alpha=0.75, color=PALETTE["nomatch"], label=f"No-Match / rule_none (n={(y == 0).sum():,})")
    ax_sep.hist(probs[y == 1], bins=bins, alpha=0.75, color=PALETTE["match"], label=f"Matchable Entries (n={(y == 1).sum():,})")
    t_star = float(eval_report.get("has_match", {}).get("threshold_t", 0.5))
    ax_sep.axvline(t_star, color=PALETTE["dark"], linestyle="--", linewidth=1.6, label=f"Decision Threshold t*={t_star:.3f}")
    ax_sep.set_yscale("log")
    ax_sep.set_xlabel("Calibrated P(has_match)")
    ax_sep.set_ylabel("Entry Count (Log Scale)")
    ax_sep.set_title("Entry-Level Matchability Separation (Valid Split)")
    ax_sep.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax_sep.set_axisbelow(True)
    ax_sep.legend(loc="upper center", fontsize=8.3)

    fig.tight_layout()
    _save_fig(fig, out_dir, "04_collision_and_score_separation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: Bootstrap CIs, feature-group ablations & 4 poster plots."
    )
    parser.add_argument("--models-dir", type=Path, default=Path("data/matching/models"))
    parser.add_argument("--features", type=Path, default=Path("data/matching/pair_features.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/matching/splits_auto.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("data/matching/plots"))
    parser.add_argument("--out-json", type=Path, default=Path("data/matching/analysis/phase6_evaluation.json"))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _configure_plot_style()

    report_path = args.models_dir / "eval_report.json"
    va_scored_path = args.models_dir / "valid_scored_pairs.parquet"
    st_scored_path = args.models_dir / "stress_scored_pairs.parquet"
    hm_ent_path = args.models_dir / "valid_has_match_entries.parquet"

    for req in (report_path, va_scored_path, hm_ent_path):
        if not req.exists():
            raise SystemExit(f"Missing {req}. Run updated src/03_matching/models.py first.")

    eval_report = json.loads(report_path.read_text(encoding="utf-8"))
    va_scored = pd.read_parquet(va_scored_path)
    hm_entries = pd.read_parquet(hm_ent_path)
    st_scored = pd.read_parquet(st_scored_path) if st_scored_path.exists() else None

    boot_clean = {
        name: publication_bootstrap_ci(va_scored, col, n_boot=args.n_boot, seed=args.seed)
        for name, col in [
            ("B1", "b1_score"),
            ("M3", "m3_score"),
            ("M4", "m4_score"),
            ("M5", "m5_score"),
        ]
    }
    paired_clean = {
        "M5_minus_M4": paired_publication_bootstrap_test(
            va_scored, "m5_score", "m4_score", n_boot=args.n_boot, seed=args.seed
        ),
        "M5_minus_B1": paired_publication_bootstrap_test(
            va_scored, "m5_score", "b1_score", n_boot=args.n_boot, seed=args.seed
        ),
    }

    boot_stress: Dict[str, Any] = {}
    paired_stress: Dict[str, Any] = {}
    if st_scored is not None and not st_scored.empty:
        boot_stress = {
            name: publication_bootstrap_ci(st_scored, col, n_boot=args.n_boot, seed=args.seed)
            for name, col in [
                ("B1", "b1_score"),
                ("M3", "m3_score"),
                ("M4", "m4_score"),
                ("M5", "m5_score"),
            ]
        }
        paired_stress = {
            "M5_minus_M4": paired_publication_bootstrap_test(
                st_scored, "m5_score", "m4_score", n_boot=args.n_boot, seed=args.seed
            ),
            "M4_minus_B1": paired_publication_bootstrap_test(
                st_scored, "m4_score", "b1_score", n_boot=args.n_boot, seed=args.seed
            ),
            "M5_minus_B1": paired_publication_bootstrap_test(
                st_scored, "m5_score", "b1_score", n_boot=args.n_boot, seed=args.seed
            ),
        }

    ablations: Optional[Dict[str, Any]] = None
    if not args.skip_ablations and args.features.exists():
        pair_df = pd.read_parquet(args.features)
        pair_df["arxiv_id"] = pair_df["arxiv_id"].astype(str)
        pair_df["citation_key"] = pair_df["citation_key"].astype(str)
        pair_df["ref_arxiv_id"] = pair_df["ref_arxiv_id"].astype(str)
        if args.splits is not None and args.splits.exists():
            pub_to_split = load_pub_to_split(args.splits)
            mapped = pair_df["arxiv_id"].map(pub_to_split)
            pair_df = pair_df.loc[mapped.notna()].copy()
            pair_df["partition"] = mapped.loc[mapped.notna()].astype(str)
        tr_raw = pair_df[pair_df["partition"] == "train"].copy()
        va_raw = pair_df[pair_df["partition"] == "valid"].copy()
        ablations = run_feature_group_ablations(tr_raw, va_raw, seed=args.seed)

    plot_01_model_ladder(eval_report, boot_clean, boot_stress, args.plots_dir)
    plot_02_has_match_pr_and_calibration(hm_entries, eval_report, args.plots_dir)
    plot_03_feature_importance_and_ablation(eval_report, ablations, args.plots_dir)
    plot_04_collision_and_score_separation(hm_entries, eval_report, args.plots_dir)

    phase6_out: Dict[str, Any] = {
        "bootstrap_95ci_clean_valid": boot_clean,
        "paired_bootstrap_clean_valid": paired_clean,
        "bootstrap_95ci_hard_stress": boot_stress,
        "paired_bootstrap_hard_stress": paired_stress,
        "feature_group_ablations": ablations,
        "plots_dir": str(args.plots_dir),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(phase6_out, indent=2), encoding="utf-8")
    print(json.dumps(phase6_out, indent=2))


if __name__ == "__main__":
    main()