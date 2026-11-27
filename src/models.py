"""Phase 5: Reference Matching Model Ladder, Has-Match Integration & Calibrated M5.

Upgrades
--------
1. Fold-consistent z-score standardization of Stage-1 M3/M4 scores inside each
   GroupKFold split (fit strictly on fold training rows) so train_oof, valid,
   and test live on the exact same score scale.
2. Recall-safe decoupled M5 tuning: Step 1 selects (T*, tau*) maximizing
   gated_f1_at_1 without sacrificing ranking; Step 2 selects alpha* maximizing
   (mrr5_matchable, collision_mrr5, gated_f1_at_1).
3. Asymmetric Hard-Citation Stress Benchmark: non-rank-preserving realistic
   corruptions (title-missing physics styles, retitled preprints, same-author
   distractor collisions, and masked IDs) to expose real B1 < M3 < M4 < M5 separation.
4. Saves valid_scored_pairs.parquet and stress_scored_pairs.parquet for Phase 6.

Classical ML only. Test split is untouched unless --eval-test is explicitly set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from has_match import (
    ID_PAIR_FEATURES,
    export_suspected_missed_matches,
    train_and_evaluate_has_match,
)
from sampling import load_pub_to_split


META_COLS = {
    "arxiv_id",
    "citation_key",
    "ref_arxiv_id",
    "partition",
    "split",
    "label",
    "has_match",
    "source",
    "rule",
    "true_ref_arxiv_id",
    "annotator",
    "confidence",
    "unsure",
    "b1_score",
    "m3_score",
    "m4_score",
    "m5_score",
    "p_has_match",
    "m5_assigned_ref",
}

NO_ID_DROP_COLS = set(ID_PAIR_FEATURES) | {"title_exact"}

POSITIVE_MONOTONE_PREFIXES = (
    "arxiv_eq",
    "doi_eq",
    "arxiv_in_raw",
    "doi_in_raw",
    "title_",
    "raw_cand_title",
    "raw_cand_surname",
    "first_surname",
    "surname_",
    "author_",
    "year_exact",
    "venue_",
)

NEGATIVE_MONOTONE_COLS = {
    "id_conflict",
    "year_diff_abs",
    "year_diff_arxiv",
}


def discover_feature_cols(
    df: pd.DataFrame,
    drop_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return numeric feature columns, excluding all ID/meta/target columns."""
    exclude = set(META_COLS) | set(drop_cols or ())
    return [
        col
        for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]


def build_monotone_constraints(feature_cols: Sequence[str]) -> List[int]:
    """Assign +1/-1/0 monotone constraints for LightGBM LambdaRank."""
    constraints: List[int] = []
    for col in feature_cols:
        if col in NEGATIVE_MONOTONE_COLS:
            constraints.append(-1)
        elif col.endswith("_len_ratio") or col.endswith("_diff"):
            constraints.append(0)
        elif any(col.startswith(pref) for pref in POSITIVE_MONOTONE_PREFIXES):
            constraints.append(1)
        else:
            constraints.append(0)
    return constraints


def compute_b1_lexical_score(
    df: pd.DataFrame,
    exclude_ids: bool = False,
) -> np.ndarray:
    """Unsupervised lexical baseline combining title, author, year, and optional IDs."""
    score = np.zeros(len(df), dtype=float)
    weights: List[Tuple[str, float]] = [
        ("title_word_tfidf_cos", 2.5),
        ("title_char_tfidf_cos", 1.5),
        ("title_char_ngram_cos", 1.5),
        ("title_jw", 1.5),
        ("raw_cand_title_cov", 2.0),
        ("title_bm25_rr", 1.0),
        ("first_surname_jw", 1.5),
        ("surname_containment", 1.2),
        ("raw_cand_surname_hit", 1.0),
        ("author_jaccard", 0.8),
        ("year_exact", 0.4),
        ("id_conflict", -2.5),
    ]
    if not exclude_ids:
        weights.extend([
            ("arxiv_eq", 5.0),
            ("doi_eq", 4.5),
            ("arxiv_in_raw", 2.5),
            ("doi_in_raw", 2.0),
        ])
    for col, w in weights:
        if col in df.columns:
            vals = np.nan_to_num(df[col].to_numpy(dtype=float), nan=0.0)
            score += w * vals
    return score


def standardize_fold_scores(
    train_scores: np.ndarray,
    *target_arrays: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], ...]:
    """Standardize target score arrays using (mean, std) fit strictly on train_scores."""
    tr = np.asarray(train_scores, dtype=float)
    mu = float(np.mean(tr)) if len(tr) else 0.0
    sd = float(np.std(tr)) if len(tr) else 1.0
    if sd < 1e-6:
        sd = 1.0
    out: List[Optional[np.ndarray]] = []
    for arr in target_arrays:
        if arr is None:
            out.append(None)
        else:
            out.append((np.asarray(arr, dtype=float) - mu) / sd)
    return tuple(out)


def _make_m3_logreg() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])


def _make_m4_ranker(feature_cols: Sequence[str], seed: int = 42) -> Any:
    if lgb is None:
        raise RuntimeError("lightgbm is required for M4_lgbm_rank")
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 5],
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        monotone_constraints=build_monotone_constraints(feature_cols),
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _group_sizes(df: pd.DataFrame) -> np.ndarray:
    return (
        df.groupby(["arxiv_id", "citation_key"], sort=False)
        .size()
        .to_numpy(dtype=int)
    )


def _filter_matchable_queries(df: pd.DataFrame) -> pd.DataFrame:
    has_pos = df.groupby(["arxiv_id", "citation_key"], sort=False)["label"].transform("max")
    return df[has_pos > 0].copy()


def fit_stage1_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feature_cols: Sequence[str],
    n_splits: int = 5,
    seed: int = 42,
    exclude_ids: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, float]]:
    """Fit B1, M3 (LogReg), and M4 (LightGBM LambdaRank) with fold-standardized OOF."""
    tr = train_df.sort_values(
        ["arxiv_id", "citation_key", "ref_arxiv_id"], kind="mergesort"
    ).reset_index(drop=True)
    va = valid_df.sort_values(
        ["arxiv_id", "citation_key", "ref_arxiv_id"], kind="mergesort"
    ).reset_index(drop=True)
    te = (
        test_df.sort_values(
            ["arxiv_id", "citation_key", "ref_arxiv_id"], kind="mergesort"
        ).reset_index(drop=True)
        if test_df is not None and not test_df.empty
        else None
    )

    tr["b1_score"] = compute_b1_lexical_score(tr, exclude_ids=exclude_ids)
    va["b1_score"] = compute_b1_lexical_score(va, exclude_ids=exclude_ids)
    if te is not None:
        te["b1_score"] = compute_b1_lexical_score(te, exclude_ids=exclude_ids)

    X_tr = tr[list(feature_cols)].to_numpy(dtype=float)
    y_tr = tr["label"].to_numpy(dtype=int)
    g_tr = tr["arxiv_id"].to_numpy()

    X_va = va[list(feature_cols)].to_numpy(dtype=float)
    X_te = te[list(feature_cols)].to_numpy(dtype=float) if te is not None else None

    unique_pubs = len(np.unique(g_tr))
    k_folds = min(n_splits, max(2, unique_pubs))
    gkf = GroupKFold(n_splits=k_folds)

    # M3: Pairwise L2 Logistic Regression (fold-standardized)
    m3_oof = np.zeros(len(tr), dtype=float)
    m3_va = np.zeros(len(va), dtype=float)
    m3_te = np.zeros(len(te), dtype=float) if te is not None else None

    for tr_idx, oof_idx in gkf.split(X_tr, y_tr, g_tr):
        clf = _make_m3_logreg()
        clf.fit(X_tr[tr_idx], y_tr[tr_idx])
        s_tr = clf.decision_function(X_tr[tr_idx])
        s_oof, s_va, s_te = standardize_fold_scores(
            s_tr,
            clf.decision_function(X_tr[oof_idx]),
            clf.decision_function(X_va),
            clf.decision_function(X_te) if X_te is not None else None,
        )
        assert s_oof is not None and s_va is not None
        m3_oof[oof_idx] = s_oof
        m3_va += s_va / k_folds
        if m3_te is not None and s_te is not None:
            m3_te += s_te / k_folds

    tr["m3_score"] = m3_oof
    va["m3_score"] = m3_va
    if te is not None and m3_te is not None:
        te["m3_score"] = m3_te

    # M4: LightGBM LambdaRank (fold-standardized OOF -> fixes early-stopping scale drift)
    m4_oof = np.zeros(len(tr), dtype=float)
    m4_va = np.zeros(len(va), dtype=float)
    m4_te = np.zeros(len(te), dtype=float) if te is not None else None
    gain_accum = np.zeros(len(feature_cols), dtype=float)

    for fold_i, (tr_idx, oof_idx) in enumerate(gkf.split(X_tr, y_tr, g_tr)):
        fold_tr_df = _filter_matchable_queries(tr.iloc[tr_idx])
        fold_va_df = _filter_matchable_queries(tr.iloc[oof_idx])

        ranker = _make_m4_ranker(feature_cols, seed=seed + fold_i)
        fit_kwargs: Dict[str, Any] = {
            "X": fold_tr_df[list(feature_cols)].to_numpy(dtype=float),
            "y": fold_tr_df["label"].to_numpy(dtype=int),
            "group": _group_sizes(fold_tr_df),
        }
        if not fold_va_df.empty:
            fit_kwargs["eval_set"] = [
                (
                    fold_va_df[list(feature_cols)].to_numpy(dtype=float),
                    fold_va_df["label"].to_numpy(dtype=int),
                )
            ]
            fit_kwargs["eval_group"] = [_group_sizes(fold_va_df)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=40, verbose=False)]

        ranker.fit(**fit_kwargs)
        raw_tr_scores = ranker.predict(X_tr[tr_idx])
        s_oof, s_va, s_te = standardize_fold_scores(
            raw_tr_scores,
            ranker.predict(X_tr[oof_idx]),
            ranker.predict(X_va),
            ranker.predict(X_te) if X_te is not None else None,
        )
        assert s_oof is not None and s_va is not None
        m4_oof[oof_idx] = s_oof
        m4_va += s_va / k_folds
        if m4_te is not None and s_te is not None:
            m4_te += s_te / k_folds
        gain_accum += ranker.booster_.feature_importance(importance_type="gain") / k_folds

    tr["m4_score"] = m4_oof
    va["m4_score"] = m4_va
    if te is not None and m4_te is not None:
        te["m4_score"] = m4_te

    top_gain = {
        feature_cols[i]: round(float(gain_accum[i]), 2)
        for i in np.argsort(gain_accum)[::-1][:10]
    }
    return tr, va, te, top_gain


# ---------------------------------------------------------------------------
# Stage-2 Calibrated, Collision-Aware M5 Collective Assignment
# ---------------------------------------------------------------------------

def solve_pub_hungarian(prob_matrix: np.ndarray, tau: float) -> np.ndarray:
    """Solve maximum-weight 1-to-1 matching with dummy 'null' columns of weight tau."""
    m, n = prob_matrix.shape
    if m == 0:
        return np.array([], dtype=int)
    if n == 0:
        return np.full(m, -1, dtype=int)

    dummy = np.full((m, m), float(tau), dtype=float)
    aug = np.hstack([prob_matrix, dummy])
    row_ind, col_ind = linear_sum_assignment(-aug)

    assigned = np.full(m, -1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        if c < n and prob_matrix[r, c] >= tau:
            assigned[r] = int(c)
    return assigned


def _compute_calibrated_pair_probs(
    df: pd.DataFrame,
    score_col: str,
    p_has_col: str,
    temperature: float,
) -> np.ndarray:
    """Compute P_pair(e, c) = p_has_match(e) * softmax_c(score(e, c) / T)."""
    temp = max(float(temperature), 1e-3)
    entry_keys = ["arxiv_id", "citation_key"]
    tmp = pd.DataFrame({
        "arxiv_id": df["arxiv_id"].to_numpy(),
        "citation_key": df["citation_key"].to_numpy(),
        "_scaled": df[score_col].to_numpy(dtype=float) / temp,
    })
    s_max = tmp.groupby(entry_keys, sort=False)["_scaled"].transform("max").to_numpy(dtype=float)
    exp_s = np.exp(np.clip(tmp["_scaled"].to_numpy(dtype=float) - s_max, -40.0, 0.0))
    tmp["_exp_s"] = exp_s
    sum_exp = tmp.groupby(entry_keys, sort=False)["_exp_s"].transform("sum").to_numpy(dtype=float)
    q_cond = exp_s / np.maximum(sum_exp, 1e-12)

    if p_has_col in df.columns:
        p_has = np.clip(df[p_has_col].to_numpy(dtype=float), 0.0, 1.0)
    else:
        p_has = np.ones(len(df), dtype=float)
    return p_has * q_cond


def _solve_all_pubs_assignment(
    df: pd.DataFrame,
    p_pair: np.ndarray,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Hungarian matching per publication; return (is_own, is_other_claim, assigned_ref)."""
    n_rows = len(df)
    is_own = np.zeros(n_rows, dtype=bool)
    is_other_claim = np.zeros(n_rows, dtype=bool)
    assigned_ref_col = np.empty(n_rows, dtype=object)
    assigned_ref_col[:] = None

    keys_arr = df["citation_key"].to_numpy()
    refs_arr = df["ref_arxiv_id"].to_numpy()

    for _, pub_idx in df.groupby("arxiv_id", sort=False).groups.items():
        pos = pub_idx.to_numpy()
        sub_keys = keys_arr[pos]
        sub_refs = refs_arr[pos]
        sub_probs = p_pair[pos]

        entries, e_inv = np.unique(sub_keys, return_inverse=True)
        cands, c_inv = np.unique(sub_refs, return_inverse=True)
        m, n = len(entries), len(cands)

        mat = np.zeros((m, n), dtype=float)
        mat[e_inv, c_inv] = sub_probs

        assigned_c = solve_pub_hungarian(mat, tau=tau)
        cand_owner = np.full(n, -1, dtype=int)
        for e_i, c_j in enumerate(assigned_c):
            if c_j >= 0:
                cand_owner[c_j] = e_i

        entry_assigned_c = assigned_c[e_inv]
        cand_claimed_by = cand_owner[c_inv]

        is_own[pos] = entry_assigned_c == c_inv
        is_other_claim[pos] = (cand_claimed_by >= 0) & (cand_claimed_by != e_inv)

        entry_ref_lookup = np.array(
            [str(cands[c]) if c >= 0 else None for c in assigned_c],
            dtype=object,
        )
        assigned_ref_col[pos] = entry_ref_lookup[e_inv]

    return is_own, is_other_claim, assigned_ref_col


def apply_m5_collective(
    pair_df: pd.DataFrame,
    score_col: str = "m4_score",
    p_has_col: str = "p_has_match",
    tau: float = 0.35,
    alpha: float = 0.25,
    temperature: float = 0.5,
) -> pd.DataFrame:
    """Apply calibrated Stage-2 Hungarian collective assignment per publication."""
    df = pair_df.reset_index(drop=True).copy()
    if df.empty:
        df["m5_score"] = np.array([], dtype=float)
        df["m5_assigned_ref"] = pd.Series(dtype=object)
        return df

    p_pair = _compute_calibrated_pair_probs(df, score_col, p_has_col, temperature)
    is_own, is_other_claim, assigned_ref_col = _solve_all_pubs_assignment(df, p_pair, tau)

    df["m5_score"] = (
        p_pair
        + float(alpha) * is_own.astype(float)
        - 0.5 * float(alpha) * is_other_claim.astype(float)
    )
    df["m5_assigned_ref"] = assigned_ref_col
    return df


# ---------------------------------------------------------------------------
# Dual MRR@5 Evaluation & Collision Diagnostics
# ---------------------------------------------------------------------------

def evaluate_ranking_dual(
    pair_df: pd.DataFrame,
    score_col: str,
    p_has_col: Optional[str] = None,
    has_match_threshold: Optional[float] = None,
    assigned_ref_col: Optional[str] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """Compute BOTH all-labelled MRR@5 and matchable-only MRR@5 (+ gated F1@1)."""
    if pair_df.empty:
        return {
            "n_entries": 0,
            "n_matchable": 0,
            "n_no_match": 0,
            "oracle_ceiling": 0.0,
            "mrr5_all": 0.0,
            "mrr5_matchable": 0.0,
            "hit1_matchable": 0.0,
            "hit5_matchable": 0.0,
        }

    df = pair_df.sort_values(
        by=["arxiv_id", "citation_key", score_col, "ref_arxiv_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    entry_keys = ["arxiv_id", "citation_key"]
    ranks = df.groupby(entry_keys, sort=False).cumcount().to_numpy() + 1

    entry_has_pos = df.groupby(entry_keys, sort=False)["label"].max() > 0
    n_entries = int(len(entry_has_pos))
    n_matchable = int(entry_has_pos.sum())
    n_no_match = n_entries - n_matchable

    pos_mask = df["label"].to_numpy() > 0
    pos_ranks = ranks[pos_mask]
    if len(pos_ranks) != n_matchable:
        pos_df = df.loc[pos_mask, entry_keys].copy()
        pos_df["_r"] = pos_ranks
        pos_ranks = pos_df.groupby(entry_keys, sort=False)["_r"].min().to_numpy()

    rr = np.where(pos_ranks <= k, 1.0 / pos_ranks, 0.0)
    rr_sum = float(rr.sum())
    hit1_sum = float((pos_ranks == 1).sum())
    hitk_sum = float((pos_ranks <= k).sum())

    res: Dict[str, Any] = {
        "n_entries": n_entries,
        "n_matchable": n_matchable,
        "n_no_match": n_no_match,
        "oracle_ceiling": round(n_matchable / max(1, n_entries), 4),
        "mrr5_all": round(rr_sum / max(1, n_entries), 4),
        "mrr5_matchable": round(rr_sum / max(1, n_matchable), 4),
        "hit1_matchable": round(hit1_sum / max(1, n_matchable), 4),
        "hit5_matchable": round(hitk_sum / max(1, n_matchable), 4),
    }

    top1 = df.loc[ranks == 1].copy()
    top1_is_true_pos = top1["label"].to_numpy() > 0
    top1_has_gt = (
        top1.set_index(entry_keys).index.map(entry_has_pos).to_numpy(dtype=bool)
    )

    pred_has_match: Optional[np.ndarray] = None
    pred_correct_top1: Optional[np.ndarray] = None

    if assigned_ref_col and assigned_ref_col in top1.columns:
        assigned_refs = top1[assigned_ref_col].to_numpy()
        pred_has_match = np.array(
            [r is not None and pd.notna(r) for r in assigned_refs], dtype=bool
        )
        pos_rows = df.loc[pos_mask, entry_keys + ["ref_arxiv_id"]].drop_duplicates(
            subset=entry_keys
        )
        gt_map = pos_rows.set_index(entry_keys)["ref_arxiv_id"]
        gt_refs = top1.set_index(entry_keys).index.map(gt_map).to_numpy()
        pred_correct_top1 = pred_has_match & top1_has_gt & (assigned_refs == gt_refs)
    elif p_has_col and p_has_col in top1.columns and has_match_threshold is not None:
        pred_has_match = top1[p_has_col].to_numpy(dtype=float) >= float(has_match_threshold)
        pred_correct_top1 = pred_has_match & top1_is_true_pos

    if pred_has_match is not None and pred_correct_top1 is not None:
        n_pred_pos = int(pred_has_match.sum())
        tp_correct = int(pred_correct_top1.sum())
        tn_null = int((~pred_has_match & ~top1_has_gt).sum())
        prec = tp_correct / max(1, n_pred_pos)
        rec = tp_correct / max(1, n_matchable)
        f1 = (2.0 * prec * rec) / max(prec + rec, 1e-12)
        res.update({
            "gated_precision_at_1": round(prec, 4),
            "gated_recall_at_1": round(rec, 4),
            "gated_f1_at_1": round(f1, 4),
            "null_detection_acc": round(tn_null / max(1, n_no_match), 4),
            "overall_decision_acc": round((tp_correct + tn_null) / max(1, n_entries), 4),
        })
    return res


def evaluate_collisions(
    pair_df: pd.DataFrame,
    base_score_col: str = "m4_score",
    m5_score_col: str = "m5_score",
    p_has_col: str = "p_has_match",
    min_prob_for_collision: float = 0.25,
) -> Dict[str, Any]:
    """Measure M4 vs M5 specifically on entries involved in top-1 candidate collisions."""
    if pair_df.empty:
        return {"n_collision_groups": 0, "n_collision_entries": 0}

    entry_keys = ["arxiv_id", "citation_key"]
    df_base = pair_df.sort_values(
        by=["arxiv_id", "citation_key", base_score_col, "ref_arxiv_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top1_base = df_base.groupby(entry_keys, sort=False).first().reset_index()

    if p_has_col in top1_base.columns:
        plausible = top1_base[top1_base[p_has_col] >= min_prob_for_collision].copy()
    else:
        plausible = top1_base.copy()

    claim_counts = plausible.groupby(["arxiv_id", "ref_arxiv_id"])["citation_key"].transform("count")
    colliding = plausible[claim_counts >= 2]
    n_groups = (
        int(colliding.groupby(["arxiv_id", "ref_arxiv_id"]).ngroups)
        if not colliding.empty
        else 0
    )

    m4_dup_rate = round(float((claim_counts >= 2).mean()), 4) if len(plausible) else 0.0
    m5_dup_rate = 0.0
    if "m5_assigned_ref" in top1_base.columns:
        m5_assigned = top1_base[top1_base["m5_assigned_ref"].notna()]
        if not m5_assigned.empty:
            m5_cc = m5_assigned.groupby(["arxiv_id", "m5_assigned_ref"])["citation_key"].transform("count")
            m5_dup_rate = round(float((m5_cc >= 2).mean()), 4)

    if colliding.empty:
        return {
            "n_collision_groups": 0,
            "n_collision_entries": 0,
            "n_matchable_collision_entries": 0,
            "m4_collision_mrr5": 0.0,
            "m5_collision_mrr5": 0.0,
            "m4_collision_hit1": 0.0,
            "m5_collision_hit1": 0.0,
            "m4_duplicate_top1_rate": m4_dup_rate,
            "m5_duplicate_assigned_rate": m5_dup_rate,
        }

    coll_keys = set(zip(colliding["arxiv_id"], colliding["citation_key"]))
    mask = [
        (a, c) in coll_keys
        for a, c in zip(pair_df["arxiv_id"], pair_df["citation_key"])
    ]
    sub_df = pair_df.loc[mask]

    m4_eval = evaluate_ranking_dual(sub_df, base_score_col)
    m5_eval = evaluate_ranking_dual(sub_df, m5_score_col)

    return {
        "n_collision_groups": n_groups,
        "n_collision_entries": int(m4_eval["n_entries"]),
        "n_matchable_collision_entries": int(m4_eval["n_matchable"]),
        "m4_collision_mrr5": m4_eval["mrr5_matchable"],
        "m5_collision_mrr5": m5_eval["mrr5_matchable"],
        "m4_collision_hit1": m4_eval["hit1_matchable"],
        "m5_collision_hit1": m5_eval["hit1_matchable"],
        "m4_duplicate_top1_rate": m4_dup_rate,
        "m5_duplicate_assigned_rate": m5_dup_rate,
    }


def tune_m5_hyperparams(
    valid_df: pd.DataFrame,
    score_col: str = "m4_score",
    p_has_col: str = "p_has_match",
    temperatures: Sequence[float] = (0.25, 0.5, 1.0),
    taus: Sequence[float] = (0.10, 0.15, 0.25, 0.35, 0.45, 0.55),
    alphas: Sequence[float] = (0.0, 0.15, 0.30, 0.50, 0.80),
) -> Dict[str, Any]:
    """Recall-safe, decoupled 2-step M5 tuning on valid."""
    df = valid_df.reset_index(drop=True).copy()
    m4_base = evaluate_ranking_dual(df, score_col)
    mrr_floor = m4_base["mrr5_matchable"] - 0.0015

    # Step 1: tune (temperature, tau) with neutral alpha=0.30
    probe_alpha = 0.30
    best_step1_key: Optional[Tuple[float, ...]] = None
    best_t_tau: Tuple[float, float] = (float(temperatures[0]), float(taus[0]))
    cached_best_solve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    for temp in temperatures:
        p_pair = _compute_calibrated_pair_probs(df, score_col, p_has_col, temp)
        for tau in taus:
            is_own, is_other_claim, assigned_ref_col = _solve_all_pubs_assignment(
                df, p_pair, tau
            )
            df["m5_assigned_ref"] = assigned_ref_col
            df["m5_score"] = (
                p_pair
                + probe_alpha * is_own.astype(float)
                - 0.5 * probe_alpha * is_other_claim.astype(float)
            )
            ev = evaluate_ranking_dual(
                df, score_col="m5_score", assigned_ref_col="m5_assigned_ref"
            )
            meets_floor = 1.0 if ev["mrr5_matchable"] >= mrr_floor else 0.0
            key = (
                meets_floor,
                round(ev.get("gated_f1_at_1", 0.0), 4),
                round(ev["mrr5_matchable"], 6),
                round(ev.get("overall_decision_acc", 0.0), 4),
                -abs(tau - 0.30),
            )
            if best_step1_key is None or key > best_step1_key:
                best_step1_key = key
                best_t_tau = (float(temp), float(tau))
                cached_best_solve = (p_pair, is_own, is_other_claim, assigned_ref_col)

    assert cached_best_solve is not None
    best_temp, best_tau = best_t_tau
    p_pair, is_own, is_other_claim, assigned_ref_col = cached_best_solve
    df["m5_assigned_ref"] = assigned_ref_col

    # Step 2: sweep alpha cleanly at fixed (T*, tau*)
    best_step2_key: Optional[Tuple[float, ...]] = None
    best_params: Dict[str, Any] = {}
    for alpha in alphas:
        df["m5_score"] = (
            p_pair
            + float(alpha) * is_own.astype(float)
            - 0.5 * float(alpha) * is_other_claim.astype(float)
        )
        ev = evaluate_ranking_dual(
            df, score_col="m5_score", assigned_ref_col="m5_assigned_ref"
        )
        coll = evaluate_collisions(
            df,
            base_score_col=score_col,
            m5_score_col="m5_score",
            p_has_col=p_has_col,
        )
        key = (
            round(ev["mrr5_matchable"], 6),
            round(coll.get("m5_collision_mrr5", 0.0), 6),
            round(ev.get("gated_f1_at_1", 0.0), 4),
            round(ev["hit1_matchable"], 6),
            -abs(alpha - 0.30),
        )
        if best_step2_key is None or key > best_step2_key:
            best_step2_key = key
            best_params = {
                "temperature": best_temp,
                "tau": best_tau,
                "alpha": float(alpha),
                "valid_mrr5_matchable": ev["mrr5_matchable"],
                "valid_mrr5_all": ev["mrr5_all"],
                "valid_gated_f1_at_1": ev.get("gated_f1_at_1", 0.0),
                "valid_gated_recall_at_1": ev.get("gated_recall_at_1", 0.0),
                "valid_collision_mrr5": coll.get("m5_collision_mrr5", 0.0),
            }
    return best_params


# ---------------------------------------------------------------------------
# Asymmetric Hard-Citation Stress Benchmark (Non-Rank-Preserving)
# ---------------------------------------------------------------------------

def apply_asymmetric_hard_stress(
    valid_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a realistic, non-rank-preserving hard-citation stress benchmark.

    Masks all exact ID features, then applies three asymmetric real-world
    citation failure modes to matchable entries in valid_df:
      - 20% Title-Missing (physics/astronomy style) + competing same-first-author distractor.
      - 20% Retitled Preprint / Same-Author Series: true candidate's title drops sharply
            while a competing candidate (preferring another entry's true ref in the same
            publication) gets strong title + author overlap, creating real M5 collisions.
      - 15% Author Parse Failure: true candidate's parsed author fields degrade while
            raw-text coverage and year remain intact.
    """
    rng = np.random.default_rng(seed)
    noisy = valid_df.reset_index(drop=True).copy()

    for col in NO_ID_DROP_COLS:
        if col in noisy.columns:
            noisy[col] = 0.0

    parsed_title_cols = [
        c for c in noisy.columns
        if any(
            k in c
            for k in (
                "title_word",
                "title_char",
                "title_jw",
                "title_token",
                "title_jaccard",
                "title_bm25",
            )
        )
        and pd.api.types.is_numeric_dtype(noisy[c])
    ]
    raw_title_cols = [
        c for c in noisy.columns
        if "raw_cand_title" in c and pd.api.types.is_numeric_dtype(noisy[c])
    ]
    all_title_cols = parsed_title_cols + raw_title_cols
    author_cols = [
        c for c in noisy.columns
        if any(k in c for k in ("surname", "author_"))
        and pd.api.types.is_numeric_dtype(noisy[c])
    ]

    label_arr = noisy["label"].to_numpy(dtype=int)
    ref_arr = noisy["ref_arxiv_id"].astype(str).to_numpy()

    # Track which candidates are true matches for *some* entry in each publication
    # so distractors mirror realistic same-author collisions inside a paper.
    pub_claimed_refs: Dict[str, set] = {}
    pos_sub = noisy.loc[label_arr > 0, ["arxiv_id", "ref_arxiv_id"]]
    for pub_id, ref_id in zip(pos_sub["arxiv_id"].astype(str), pos_sub["ref_arxiv_id"].astype(str)):
        pub_claimed_refs.setdefault(pub_id, set()).add(ref_id)

    col_arrays: Dict[str, np.ndarray] = {
        c: noisy[c].to_numpy(dtype=float).copy()
        for c in set(all_title_cols + author_cols + ["year_diff_abs", "year_diff_arxiv", "year_exact"])
        if c in noisy.columns
    }

    for (pub_id, _), idx_obj in noisy.groupby(["arxiv_id", "citation_key"], sort=False).groups.items():
        pos = idx_obj.to_numpy()
        pos_labels = label_arr[pos]
        if pos_labels.max() <= 0 or len(pos) < 2:
            continue

        u = float(rng.random())
        true_rows = pos[pos_labels > 0]
        neg_rows = pos[pos_labels == 0]
        true_refs = set(ref_arr[true_rows])

        # Prefer a distractor that is another entry's true ref in this publication
        claimed_negs = [
            int(r) for r in neg_rows
            if ref_arr[r] in pub_claimed_refs.get(str(pub_id), set()) and ref_arr[r] not in true_refs
        ]
        distractor = int(rng.choice(claimed_negs if claimed_negs else neg_rows))

        if u < 0.20:
            # Mode 1: Title-missing citation (physics/astronomy style) + shared first author
            for c in all_title_cols:
                col_arrays[c][pos] = 0.0
            for c in author_cols:
                col_arrays[c][true_rows] *= float(rng.uniform(0.55, 0.80))
                col_arrays[c][distractor] = min(
                    0.99, max(col_arrays[c][distractor], float(rng.uniform(0.85, 0.98)))
                )
        elif u < 0.40:
            # Mode 2: Retitled preprint / same-author series distractor collision
            p_scale = float(rng.uniform(0.08, 0.28))
            r_scale = float(rng.uniform(0.30, 0.55))
            for c in parsed_title_cols:
                col_arrays[c][true_rows] *= p_scale
                col_arrays[c][distractor] = min(
                    0.96, col_arrays[c][distractor] + float(rng.uniform(0.55, 0.80))
                )
            for c in raw_title_cols:
                col_arrays[c][true_rows] *= r_scale
                col_arrays[c][distractor] = min(
                    0.96, col_arrays[c][distractor] + float(rng.uniform(0.45, 0.70))
                )
            for c in author_cols:
                col_arrays[c][distractor] = min(
                    0.98, max(col_arrays[c][distractor], float(rng.uniform(0.75, 0.95)))
                )
            for yc in ("year_diff_abs", "year_diff_arxiv"):
                if yc in col_arrays:
                    col_arrays[yc][true_rows] += float(rng.choice([1.0, 2.0]))
            if "year_exact" in col_arrays:
                col_arrays["year_exact"][true_rows] = 0.0
        elif u < 0.55:
            # Mode 3: Author parse failure + partial title truncation (raw-text fallback survives)
            for c in author_cols:
                col_arrays[c][true_rows] *= float(rng.uniform(0.05, 0.25))
                col_arrays[c][distractor] = min(
                    0.95, max(col_arrays[c][distractor], float(rng.uniform(0.72, 0.92)))
                )
            for c in parsed_title_cols:
                col_arrays[c][true_rows] *= float(rng.uniform(0.25, 0.50))
                col_arrays[c][distractor] = min(
                    0.92, col_arrays[c][distractor] + float(rng.uniform(0.38, 0.62))
                )

    for c, arr in col_arrays.items():
        noisy[c] = arr
    return noisy


def _attach_p_has_match(
    pair_df: pd.DataFrame,
    entry_df: pd.DataFrame,
) -> pd.DataFrame:
    lookup = entry_df[["arxiv_id", "citation_key", "p_has_match"]].drop_duplicates(
        subset=["arxiv_id", "citation_key"]
    )
    out = pair_df.drop(columns=["p_has_match"], errors="ignore").merge(
        lookup, on=["arxiv_id", "citation_key"], how="left"
    )
    out["p_has_match"] = out["p_has_match"].fillna(0.5)
    return out


def build_predictions_table(
    pair_df: pd.DataFrame,
    score_col: str = "m5_score",
    partition_name: Optional[str] = None,
) -> pd.DataFrame:
    """Build top-5 candidate predictions table matching Milestone 3 schema."""
    if pair_df.empty:
        return pd.DataFrame(
            columns=[
                "arxiv_id",
                "partition",
                "citation_key",
                "groundtruth_arxiv_id",
                "cand_1",
                "cand_2",
                "cand_3",
                "cand_4",
                "cand_5",
                "p_has_match",
                "m5_assigned_ref",
            ]
        )

    df = pair_df.sort_values(
        by=["arxiv_id", "citation_key", score_col, "ref_arxiv_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    rows: List[Dict[str, Any]] = []
    for (pub_id, ckey), grp in df.groupby(["arxiv_id", "citation_key"], sort=False):
        cands = list(dict.fromkeys(grp["ref_arxiv_id"].astype(str).tolist()))[:5]
        cands += [None] * (5 - len(cands))
        pos = grp[grp["label"] > 0]
        gt = str(pos["ref_arxiv_id"].iloc[0]) if not pos.empty else None
        part = partition_name or str(grp["partition"].iloc[0])
        p_has = float(grp["p_has_match"].iloc[0]) if "p_has_match" in grp.columns else None
        m5_ref = grp["m5_assigned_ref"].iloc[0] if "m5_assigned_ref" in grp.columns else None
        rows.append({
            "arxiv_id": str(pub_id),
            "partition": part,
            "citation_key": str(ckey),
            "groundtruth_arxiv_id": gt,
            "cand_1": cands[0],
            "cand_2": cands[1],
            "cand_3": cands[2],
            "cand_4": cands[3],
            "cand_5": cands[4],
            "p_has_match": round(p_has, 5) if p_has is not None else None,
            "m5_assigned_ref": str(m5_ref) if m5_ref is not None and pd.notna(m5_ref) else None,
        })
    return pd.DataFrame(rows)


def run_hard_stress_benchmark(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """Run real no-ID feature ablation + asymmetric hard-citation stress benchmark."""
    no_id_cols = discover_feature_cols(train_raw, drop_cols=NO_ID_DROP_COLS)
    if not no_id_cols:
        return {"skipped": "No non-ID numeric features found"}, None

    va_stress_raw = apply_asymmetric_hard_stress(valid_raw, seed=seed)

    tr_noid, va_noid_clean, va_stress, noid_top_gain = fit_stage1_models(
        train_raw,
        valid_raw,
        va_stress_raw,
        feature_cols=no_id_cols,
        n_splits=5,
        seed=seed,
        exclude_ids=True,
    )
    hm_noid = train_and_evaluate_has_match(
        tr_noid,
        va_noid_clean,
        va_stress,
        score_col="m4_score",
        n_splits=5,
        seed=seed,
        exclude_id_features=True,
    )
    va_noid_clean = _attach_p_has_match(va_noid_clean, hm_noid.valid_entry_df)
    assert va_stress is not None and hm_noid.test_entry_df is not None
    va_stress = _attach_p_has_match(va_stress, hm_noid.test_entry_df)

    m5_params_clean = tune_m5_hyperparams(va_noid_clean)
    va_noid_clean = apply_m5_collective(
        va_noid_clean,
        tau=m5_params_clean["tau"],
        alpha=m5_params_clean["alpha"],
        temperature=m5_params_clean["temperature"],
    )

    m5_params_stress = tune_m5_hyperparams(va_stress)
    va_stress = apply_m5_collective(
        va_stress,
        tau=m5_params_stress["tau"],
        alpha=m5_params_stress["alpha"],
        temperature=m5_params_stress["temperature"],
    )

    summary = {
        "real_no_id_ablation": {
            "n_features": len(no_id_cols),
            "dropped_cols": sorted(NO_ID_DROP_COLS),
            "B1_no_id": evaluate_ranking_dual(va_noid_clean, "b1_score"),
            "M3_no_id": evaluate_ranking_dual(va_noid_clean, "m3_score"),
            "M4_no_id": evaluate_ranking_dual(
                va_noid_clean,
                "m4_score",
                p_has_col="p_has_match",
                has_match_threshold=hm_noid.threshold_t,
            ),
            "M5_no_id": evaluate_ranking_dual(
                va_noid_clean,
                "m5_score",
                assigned_ref_col="m5_assigned_ref",
            ),
            "m5_tuned_params": m5_params_clean,
            "m5_collisions": evaluate_collisions(va_noid_clean),
            "has_match_valid": hm_noid.metrics["valid"][hm_noid.best_model_name],
            "top_10_features_gain": noid_top_gain,
        },
        "asymmetric_hard_stress": {
            "note": (
                "Controlled non-rank-preserving hard-citation stress test: "
                "IDs masked + 20% title-missing + 25% retitled preprint + 15% hard distractor"
            ),
            "B1_no_id": evaluate_ranking_dual(va_stress, "b1_score"),
            "M3_no_id": evaluate_ranking_dual(va_stress, "m3_score"),
            "M4_no_id": evaluate_ranking_dual(
                va_stress,
                "m4_score",
                p_has_col="p_has_match",
                has_match_threshold=hm_noid.threshold_t,
            ),
            "M5_no_id": evaluate_ranking_dual(
                va_stress,
                "m5_score",
                assigned_ref_col="m5_assigned_ref",
            ),
            "m5_tuned_params": m5_params_stress,
            "m5_collisions": evaluate_collisions(va_stress),
            "has_match_stress": hm_noid.metrics.get("test", {}).get(hm_noid.best_model_name, {}),
        },
    }
    return summary, va_stress


def _attach_rule_metadata(pair_df: pd.DataFrame, labels_path: Optional[Path]) -> pd.DataFrame:
    if labels_path is None or not labels_path.exists():
        return pair_df
    rule_map: Dict[Tuple[str, str], str] = {}
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rule = row.get("rule") or row.get("annotator")
            if rule:
                rule_map[(str(row["arxiv_id"]), str(row["citation_key"]))] = str(rule)
    if not rule_map:
        return pair_df
    df = pair_df.copy()
    df["rule"] = [
        rule_map.get((str(a), str(c)), "unknown")
        for a, c in zip(df["arxiv_id"], df["citation_key"])
    ]
    return df


def _save_scored_pairs(df: pd.DataFrame, out_path: Path) -> None:
    keep = [
        c
        for c in (
            "arxiv_id",
            "citation_key",
            "ref_arxiv_id",
            "partition",
            "label",
            "rule",
            "b1_score",
            "m3_score",
            "m4_score",
            "m5_score",
            "p_has_match",
            "m5_assigned_ref",
        )
        if c in df.columns
    ]
    out = df[keep].copy()
    if "m5_assigned_ref" in out.columns:
        out["m5_assigned_ref"] = out["m5_assigned_ref"].where(
            out["m5_assigned_ref"].notna(), None
        )
    out.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Train B1/M3/M4, Has-Match Classifier, and calibrated M5."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/matching/pair_features.parquet"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=None,
        help="Optional splits manifest to override partition by arxiv_id (e.g. splits_auto.json).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/matching/auto_labels.jsonl"),
        help="Optional labels JSONL to attach auto-label rule metadata.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/matching/models"),
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate frozen model on test split (use ONLY once for final evaluation).",
    )
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="Skip the no-ID ablation and asymmetric hard-citation stress benchmark.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.features.exists():
        raise SystemExit(f"Features file not found: {args.features}")

    pair_df = pd.read_parquet(args.features)
    pair_df["arxiv_id"] = pair_df["arxiv_id"].astype(str)
    pair_df["citation_key"] = pair_df["citation_key"].astype(str)
    pair_df["ref_arxiv_id"] = pair_df["ref_arxiv_id"].astype(str)

    if args.splits is not None:
        pub_to_split = load_pub_to_split(args.splits)
        mapped = pair_df["arxiv_id"].map(pub_to_split)
        if mapped.isna().any():
            pair_df = pair_df.loc[mapped.notna()].copy()
            mapped = mapped.loc[mapped.notna()]
        pair_df["partition"] = mapped.astype(str)

    pair_df = _attach_rule_metadata(pair_df, args.labels)

    part_pubs = {
        p: set(pair_df.loc[pair_df["partition"] == p, "arxiv_id"].unique())
        for p in ("train", "valid", "test")
    }
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = part_pubs[a] & part_pubs[b]
        if overlap:
            raise RuntimeError(f"Publication leakage between {a} and {b}: {len(overlap)} pubs")

    train_raw = pair_df[pair_df["partition"] == "train"].copy()
    valid_raw = pair_df[pair_df["partition"] == "valid"].copy()
    test_raw = (
        pair_df[pair_df["partition"] == "test"].copy()
        if args.eval_test
        else None
    )

    if train_raw.empty or valid_raw.empty:
        raise SystemExit(
            f"Empty split detected (train={len(train_raw)}, valid={len(valid_raw)})."
        )

    feature_cols = discover_feature_cols(pair_df)

    # 1. Stage-1 Models (fold-standardized B1, M3, M4)
    tr_df, va_df, te_df, top_gain = fit_stage1_models(
        train_raw,
        valid_raw,
        test_raw,
        feature_cols=feature_cols,
        n_splits=5,
        seed=args.seed,
        exclude_ids=False,
    )

    # 2. Stage-1.5 Calibrated Has-Match Classifier
    hm_res = train_and_evaluate_has_match(
        tr_df,
        va_df,
        te_df,
        score_col="m4_score",
        n_splits=5,
        seed=args.seed,
        exclude_id_features=False,
    )
    tr_df = _attach_p_has_match(tr_df, hm_res.train_entry_df)
    va_df = _attach_p_has_match(va_df, hm_res.valid_entry_df)
    if te_df is not None and hm_res.test_entry_df is not None:
        te_df = _attach_p_has_match(te_df, hm_res.test_entry_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    missed_valid_info = export_suspected_missed_matches(
        hm_res.valid_entry_df,
        args.out_dir / "suspected_missed_matches_valid.jsonl",
        min_prob=0.85,
        top_k=50,
    )
    missed_train_info = export_suspected_missed_matches(
        hm_res.train_entry_df,
        args.out_dir / "suspected_missed_matches_train_oof.jsonl",
        min_prob=0.90,
        top_k=100,
    )

    # 3. Stage-2 Calibrated, Recall-Safe Decoupled M5 Collective Tuning on valid
    m5_params = tune_m5_hyperparams(va_df)
    tr_df = apply_m5_collective(
        tr_df,
        tau=m5_params["tau"],
        alpha=m5_params["alpha"],
        temperature=m5_params["temperature"],
    )
    va_df = apply_m5_collective(
        va_df,
        tau=m5_params["tau"],
        alpha=m5_params["alpha"],
        temperature=m5_params["temperature"],
    )
    if te_df is not None:
        te_df = apply_m5_collective(
            te_df,
            tau=m5_params["tau"],
            alpha=m5_params["alpha"],
            temperature=m5_params["temperature"],
        )

    # 4. Assemble Dual MRR@5 Report
    report: Dict[str, Any] = {
        "split_counts": {
            "train_pubs": len(part_pubs["train"]),
            "valid_pubs": len(part_pubs["valid"]),
            "test_pubs": len(part_pubs["test"]),
            "n_pair_features": len(feature_cols),
            "n_has_match_features": len(hm_res.meta_feature_cols),
        },
        "valid_ladder": {
            "B1_lexical": evaluate_ranking_dual(va_df, "b1_score"),
            "M3_logreg": evaluate_ranking_dual(va_df, "m3_score"),
            "M4_lgbm_rank": evaluate_ranking_dual(
                va_df,
                "m4_score",
                p_has_col="p_has_match",
                has_match_threshold=hm_res.threshold_t,
            ),
            "M5_collective": evaluate_ranking_dual(
                va_df,
                "m5_score",
                assigned_ref_col="m5_assigned_ref",
            ),
        },
        "has_match": {
            "best_model": hm_res.best_model_name,
            "threshold_t": round(hm_res.threshold_t, 4),
            "metrics": hm_res.metrics,
            "top_10_features_gain": hm_res.top_features_gain,
            "suspected_missed_matches_valid": missed_valid_info,
            "suspected_missed_matches_train_oof": missed_train_info,
        },
        "m5_tuned_params": m5_params,
        "m5_collisions_valid": evaluate_collisions(va_df),
        "m5_collisions_train_oof": evaluate_collisions(tr_df),
        "train_oof_M5": evaluate_ranking_dual(
            tr_df,
            "m5_score",
            assigned_ref_col="m5_assigned_ref",
        ),
        "top_10_stage1_features_gain": top_gain,
    }

    if "rule" in va_df.columns:
        r4 = va_df[va_df["rule"] == "rule_4_fuzzy_title"]
        if not r4.empty:
            report["valid_rule4_fuzzy_slice"] = {
                "B1_lexical": evaluate_ranking_dual(r4, "b1_score"),
                "M3_logreg": evaluate_ranking_dual(r4, "m3_score"),
                "M4_lgbm_rank": evaluate_ranking_dual(r4, "m4_score"),
                "M5_collective": evaluate_ranking_dual(r4, "m5_score"),
            }

    if te_df is not None:
        report["test_ladder"] = {
            "B1_lexical": evaluate_ranking_dual(te_df, "b1_score"),
            "M3_logreg": evaluate_ranking_dual(te_df, "m3_score"),
            "M4_lgbm_rank": evaluate_ranking_dual(
                te_df,
                "m4_score",
                p_has_col="p_has_match",
                has_match_threshold=hm_res.threshold_t,
            ),
            "M5_collective": evaluate_ranking_dual(
                te_df,
                "m5_score",
                assigned_ref_col="m5_assigned_ref",
            ),
        }
        report["m5_collisions_test"] = evaluate_collisions(te_df)

    va_stress_scored: Optional[pd.DataFrame] = None
    if not args.skip_stress:
        stress_summary, va_stress_scored = run_hard_stress_benchmark(
            train_raw, valid_raw, seed=args.seed
        )
        report["stress_benchmark"] = stress_summary

    (args.out_dir / "eval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _save_scored_pairs(va_df, args.out_dir / "valid_scored_pairs.parquet")
    hm_res.valid_entry_df.to_parquet(
        args.out_dir / "valid_has_match_entries.parquet", index=False
    )
    if va_stress_scored is not None:
        _save_scored_pairs(va_stress_scored, args.out_dir / "stress_scored_pairs.parquet")

    build_predictions_table(tr_df, "m5_score", "train").to_parquet(
        args.out_dir / "train_oof_predictions.parquet", index=False
    )
    build_predictions_table(va_df, "m5_score", "valid").to_parquet(
        args.out_dir / "valid_predictions.parquet", index=False
    )
    if te_df is not None:
        build_predictions_table(te_df, "m5_score", "test").to_parquet(
            args.out_dir / "test_predictions.parquet", index=False
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()