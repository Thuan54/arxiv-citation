"""Stage-1.5 Calibrated Entry-Level Has-Match Classifier.

Predicts P(has_match | entry, candidate_list) for each BibTeX entry using
strictly leak-free Stage-1 out-of-fold (OOF) ranker scores, top-1/top-2
evidence margins, and publication-level candidate contention features.

Classical ML only (LogisticRegression, LightGBM, Platt calibration).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


ID_PAIR_FEATURES = (
    "arxiv_eq",
    "doi_eq",
    "arxiv_in_raw",
    "doi_in_raw",
)

TOP1_CANDIDATE_FEATURES = (
    # Identifier signals
    "arxiv_eq",
    "doi_eq",
    "arxiv_in_raw",
    "doi_in_raw",
    "id_conflict",
    # Title similarity signals
    "title_exact",
    "title_word_tfidf_cos",
    "title_char_tfidf_cos",
    "title_jw",
    "title_token_jaccard",
    "title_bm25_rr",
    "raw_cand_title_cov",
    "title_len_ratio",
    # Author signals
    "first_surname_jw",
    "surname_containment",
    "author_jaccard",
    # Year / venue signals
    "year_diff_abs",
    "year_diff_arxiv",
    "year_exact",
    "venue_token_jaccard",
)

MARGIN_PAIR_FEATURES = (
    "title_word_tfidf_cos",
    "raw_cand_title_cov",
    "first_surname_jw",
    "surname_containment",
)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute 10-bin Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if len(y_true) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def tune_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float]:
    """Select the decision threshold maximizing binary F1 on a validation set."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5, 0.0
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1s = (2.0 * prec[:-1] * rec[:-1]) / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def evaluate_binary_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Evaluate probabilistic discrimination, calibration, and thresholded F1."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)
    has_both = len(np.unique(y_true)) == 2
    return {
        "n_entries": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "positive_rate": round(float(y_true.mean()), 4) if len(y_true) else 0.0,
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4) if has_both else 0.0,
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4) if has_both else 0.0,
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4) if len(y_true) else 0.0,
        "ece_10bin": round(expected_calibration_error(y_true, y_prob, n_bins=10), 4),
        "threshold": round(float(threshold), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }


def build_entry_meta_features(
    pair_df: pd.DataFrame,
    score_col: str = "m4_score",
    exclude_id_features: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate pair-level Stage-1 scores into leak-free entry-level meta-features."""
    if pair_df.empty:
        return pd.DataFrame(), []

    req = {"arxiv_id", "citation_key", "ref_arxiv_id", score_col}
    missing = req - set(pair_df.columns)
    if missing:
        raise ValueError(f"pair_df missing required columns: {sorted(missing)}")

    df = pair_df.sort_values(
        by=["arxiv_id", "citation_key", score_col, "ref_arxiv_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()

    entry_keys = ["arxiv_id", "citation_key"]
    df["_rank"] = df.groupby(entry_keys, sort=False).cumcount()

    scores = df[score_col].to_numpy(dtype=float)
    entry_grp = df.groupby(entry_keys, sort=False)[score_col]
    s_max = entry_grp.transform("max").to_numpy(dtype=float)
    s_mean = entry_grp.transform("mean").to_numpy(dtype=float)
    s_std = entry_grp.transform("std").fillna(0.0).to_numpy(dtype=float)
    n_cands = entry_grp.transform("count").to_numpy(dtype=float)

    shifted = np.clip(scores - s_max, -40.0, 0.0)
    exp_s = np.exp(shifted)
    df["_exp_s"] = exp_s
    sum_exp = df.groupby(entry_keys, sort=False)["_exp_s"].transform("sum").to_numpy(dtype=float)
    p_soft = exp_s / np.maximum(sum_exp, 1e-12)
    df["_p_soft"] = p_soft
    df["_neg_p_log_p"] = -p_soft * np.log(np.maximum(p_soft, 1e-12))
    entropy_series = df.groupby(entry_keys, sort=False)["_neg_p_log_p"].sum()

    has_label = "label" in df.columns
    if has_label:
        y_has_series = (df.groupby(entry_keys, sort=False)["label"].max() > 0).astype(int)
        pos_rows = df[df["label"] > 0].drop_duplicates(subset=entry_keys)
        true_ref_map = pos_rows.set_index(entry_keys)["ref_arxiv_id"]
    else:
        y_has_series = None
        true_ref_map = None

    top1 = df[df["_rank"] == 0].copy()
    top1["top1_ref_arxiv_id"] = top1["ref_arxiv_id"].astype(str)
    top1["top1_score"] = top1[score_col].astype(float)
    top1["score_mean"] = s_mean[df["_rank"].to_numpy() == 0]
    top1["score_std"] = s_std[df["_rank"].to_numpy() == 0]
    top1["n_cands_entry"] = n_cands[df["_rank"].to_numpy() == 0]
    top1["score_z_top1"] = (top1["top1_score"] - top1["score_mean"]) / np.maximum(
        top1["score_std"], 1e-6
    )
    top1["top1_softmax_prob"] = top1["_p_soft"].astype(float)
    top1["score_entropy"] = (
        top1.set_index(entry_keys).index.map(entropy_series).to_numpy(dtype=float)
    )
    top1["norm_entropy"] = top1["score_entropy"] / np.maximum(
        np.log(np.maximum(top1["n_cands_entry"], 2.0)), 1e-6
    )

    top2 = df[df["_rank"] == 1][
        entry_keys
        + [score_col]
        + [c for c in MARGIN_PAIR_FEATURES if c in df.columns]
    ].copy()
    rename_t2 = {score_col: "_top2_score"}
    for c in MARGIN_PAIR_FEATURES:
        if c in top2.columns:
            rename_t2[c] = f"_top2_{c}"
    top2 = top2.rename(columns=rename_t2)

    top1 = top1.merge(top2, on=entry_keys, how="left")
    top1["has_second_cand"] = top1["_top2_score"].notna().astype(float)
    top1["top2_score"] = top1["_top2_score"].fillna(top1["top1_score"] - 5.0)
    top1["score_margin_1_2"] = top1["top1_score"] - top1["top2_score"]

    meta_cols: List[str] = [
        "top1_score",
        "top2_score",
        "score_margin_1_2",
        "score_mean",
        "score_std",
        "score_z_top1",
        "top1_softmax_prob",
        "score_entropy",
        "norm_entropy",
        "n_cands_entry",
        "has_second_cand",
    ]

    id_set = set(ID_PAIR_FEATURES) if exclude_id_features else set()
    for col in TOP1_CANDIDATE_FEATURES:
        if col in top1.columns and col not in id_set:
            out_name = f"top1_{col}"
            top1[out_name] = top1[col].astype(float)
            meta_cols.append(out_name)

    for col in MARGIN_PAIR_FEATURES:
        t2_col = f"_top2_{col}"
        if col in top1.columns and t2_col in top1.columns and col not in id_set:
            out_name = f"margin_1_2_{col}"
            top1[out_name] = top1[col].astype(float) - top1[t2_col].fillna(0.0).astype(float)
            meta_cols.append(out_name)

    # Publication context & candidate contention features (bridges Stage-1 -> M5)
    pub_entry_cnt = top1.groupby("arxiv_id")["citation_key"].transform("count").astype(float)
    pub_cand_cnt = df.groupby("arxiv_id")["ref_arxiv_id"].nunique()
    top1["n_entries_pub"] = pub_entry_cnt
    top1["n_cands_pub"] = top1["arxiv_id"].map(pub_cand_cnt).astype(float)
    top1["cand_to_entry_ratio"] = top1["n_cands_pub"] / np.maximum(top1["n_entries_pub"], 1.0)

    claim_keys = ["arxiv_id", "top1_ref_arxiv_id"]
    top1 = top1.sort_values(
        by=["arxiv_id", "top1_ref_arxiv_id", "top1_score", "citation_key"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    claim_grp = top1.groupby(claim_keys, sort=False)["top1_score"]
    top1["top1_cand_claim_count"] = claim_grp.transform("count").astype(float)
    claim_rank = top1.groupby(claim_keys, sort=False).cumcount()
    top1["is_best_claimant"] = (claim_rank == 0).astype(float)

    best_claim_score = claim_grp.transform("max").astype(float)
    second_claim_map = top1[claim_rank == 1].set_index(claim_keys)["top1_score"]
    second_claim_score = (
        top1.set_index(claim_keys).index.map(second_claim_map).to_numpy(dtype=float)
    )
    other_best = np.where(
        claim_rank.to_numpy() == 0,
        np.where(
            np.isnan(second_claim_score),
            top1["top1_score"].to_numpy() - 5.0,
            second_claim_score,
        ),
        best_claim_score.to_numpy(),
    )
    top1["claimant_score_gap"] = top1["top1_score"].to_numpy() - other_best

    top1 = top1.sort_values(by=entry_keys, kind="mergesort").reset_index(drop=True)

    meta_cols.extend([
        "n_entries_pub",
        "n_cands_pub",
        "cand_to_entry_ratio",
        "top1_cand_claim_count",
        "is_best_claimant",
        "claimant_score_gap",
    ])

    keep_meta = ["arxiv_id", "citation_key", "top1_ref_arxiv_id"]
    if "partition" in top1.columns:
        keep_meta.append("partition")
    if "rule" in top1.columns:
        keep_meta.append("rule")

    out = top1[keep_meta + meta_cols].copy()
    if y_has_series is not None:
        idx = out.set_index(entry_keys).index
        out["y_has_match"] = idx.map(y_has_series).to_numpy(dtype=int)
        out["true_ref_arxiv_id"] = idx.map(true_ref_map).to_numpy()
    return out, meta_cols


class PlattCalibrator:
    """1D logistic (Platt) probability calibrator fit strictly on OOF scores."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=500)
        self._fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        p = np.clip(np.asarray(raw_probs, dtype=float), 1e-6, 1.0 - 1e-6)
        z = logit(p).reshape(-1, 1)
        y = np.asarray(y_true, dtype=int)
        if len(np.unique(y)) == 2:
            self._lr.fit(z, y)
            self._fitted = True
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw_probs, dtype=float), 1e-6, 1.0 - 1e-6)
        if not self._fitted:
            return p
        z = logit(p).reshape(-1, 1)
        return np.clip(self._lr.predict_proba(z)[:, 1], 1e-6, 1.0 - 1e-6)


@dataclass
class HasMatchResult:
    train_entry_df: pd.DataFrame
    valid_entry_df: pd.DataFrame
    test_entry_df: Optional[pd.DataFrame]
    meta_feature_cols: List[str]
    best_model_name: str
    threshold_t: float
    metrics: Dict[str, Any]
    top_features_gain: Dict[str, float]


def _make_lgbm_has_match(seed: int = 42) -> Any:
    if lgb is None:
        raise RuntimeError("lightgbm is required for H2_lgbm Has-Match classifier")
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _make_logreg_has_match() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])


def train_and_evaluate_has_match(
    train_pairs: pd.DataFrame,
    valid_pairs: pd.DataFrame,
    test_pairs: Optional[pd.DataFrame] = None,
    score_col: str = "m4_score",
    n_splits: int = 5,
    seed: int = 42,
    exclude_id_features: bool = False,
) -> HasMatchResult:
    """Train H0 (threshold), H1 (LogReg), H2 (calibrated LightGBM) Has-Match models."""
    tr_ent, meta_cols = build_entry_meta_features(
        train_pairs, score_col=score_col, exclude_id_features=exclude_id_features
    )
    va_ent, _ = build_entry_meta_features(
        valid_pairs, score_col=score_col, exclude_id_features=exclude_id_features
    )
    te_ent = (
        build_entry_meta_features(
            test_pairs, score_col=score_col, exclude_id_features=exclude_id_features
        )[0]
        if test_pairs is not None and not test_pairs.empty
        else None
    )

    X_tr = tr_ent[meta_cols].to_numpy(dtype=float)
    y_tr = tr_ent["y_has_match"].to_numpy(dtype=int)
    g_tr = tr_ent["arxiv_id"].to_numpy()

    X_va = va_ent[meta_cols].to_numpy(dtype=float)
    y_va = va_ent["y_has_match"].to_numpy(dtype=int)

    X_te = te_ent[meta_cols].to_numpy(dtype=float) if te_ent is not None else None
    y_te = te_ent["y_has_match"].to_numpy(dtype=int) if te_ent is not None else None

    unique_pubs = len(np.unique(g_tr))
    k_folds = min(n_splits, max(2, unique_pubs))
    gkf = GroupKFold(n_splits=k_folds)

    # H0: calibrated sigmoid of top1_score alone
    h0_cal = PlattCalibrator().fit(expit(tr_ent["top1_score"].to_numpy(dtype=float)), y_tr)
    h0_tr_p = h0_cal.predict(expit(tr_ent["top1_score"].to_numpy(dtype=float)))
    h0_va_p = h0_cal.predict(expit(va_ent["top1_score"].to_numpy(dtype=float)))
    h0_t, _ = tune_f1_threshold(y_va, h0_va_p)

    # H1: L2 Logistic Regression (GroupKFold OOF)
    h1_oof = np.zeros(len(tr_ent), dtype=float)
    for tr_idx, oof_idx in gkf.split(X_tr, y_tr, g_tr):
        m = _make_logreg_has_match()
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        h1_oof[oof_idx] = m.predict_proba(X_tr[oof_idx])[:, 1]
    h1_full = _make_logreg_has_match()
    h1_full.fit(X_tr, y_tr)
    h1_va_p = h1_full.predict_proba(X_va)[:, 1]
    h1_te_p = h1_full.predict_proba(X_te)[:, 1] if X_te is not None else None
    h1_t, _ = tune_f1_threshold(y_va, h1_va_p)

    # H2: Regularized LightGBM + OOF Platt calibration
    h2_oof_raw = np.zeros(len(tr_ent), dtype=float)
    for tr_idx, oof_idx in gkf.split(X_tr, y_tr, g_tr):
        m = _make_lgbm_has_match(seed=seed)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        h2_oof_raw[oof_idx] = m.predict_proba(X_tr[oof_idx])[:, 1]

    h2_cal = PlattCalibrator().fit(h2_oof_raw, y_tr)
    h2_oof_cal = h2_cal.predict(h2_oof_raw)

    h2_full = _make_lgbm_has_match(seed=seed)
    h2_full.fit(X_tr, y_tr)
    h2_va_cal = h2_cal.predict(h2_full.predict_proba(X_va)[:, 1])
    h2_te_cal = (
        h2_cal.predict(h2_full.predict_proba(X_te)[:, 1]) if X_te is not None else None
    )
    h2_t, _ = tune_f1_threshold(y_va, h2_va_cal)

    gains = h2_full.booster_.feature_importance(importance_type="gain")
    top_gain = {
        meta_cols[i]: round(float(gains[i]), 2)
        for i in np.argsort(gains)[::-1][:10]
    }

    metrics: Dict[str, Any] = {
        "valid": {
            "H0_score_threshold": evaluate_binary_probabilities(y_va, h0_va_p, h0_t),
            "H1_logreg": evaluate_binary_probabilities(y_va, h1_va_p, h1_t),
            "H2_lgbm_calibrated": evaluate_binary_probabilities(y_va, h2_va_cal, h2_t),
        },
        "train_oof": {
            "H0_score_threshold": evaluate_binary_probabilities(y_tr, h0_tr_p, h0_t),
            "H1_logreg": evaluate_binary_probabilities(y_tr, h1_oof, h1_t),
            "H2_lgbm_calibrated": evaluate_binary_probabilities(y_tr, h2_oof_cal, h2_t),
        },
    }

    candidates = [
        ("H1_logreg", h1_oof, h1_va_p, h1_te_p, h1_t),
        ("H2_lgbm_calibrated", h2_oof_cal, h2_va_cal, h2_te_cal, h2_t),
    ]
    best_name, best_tr, best_va, best_te, best_t = max(
        candidates,
        key=lambda item: (
            metrics["valid"][item[0]]["pr_auc"],
            -metrics["valid"][item[0]]["brier"],
        ),
    )

    tr_ent["p_has_match"] = best_tr
    va_ent["p_has_match"] = best_va
    if te_ent is not None and best_te is not None and y_te is not None:
        te_ent["p_has_match"] = best_te
        metrics["test"] = {
            best_name: evaluate_binary_probabilities(y_te, best_te, best_t),
        }

    return HasMatchResult(
        train_entry_df=tr_ent,
        valid_entry_df=va_ent,
        test_entry_df=te_ent,
        meta_feature_cols=meta_cols,
        best_model_name=best_name,
        threshold_t=best_t,
        metrics=metrics,
        top_features_gain=top_gain,
    )


def export_suspected_missed_matches(
    entry_df: pd.DataFrame,
    out_path: Path,
    min_prob: float = 0.85,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Export top rule_none (y_has_match == 0) entries with high predicted P(has_match)."""
    if entry_df.empty or "y_has_match" not in entry_df.columns:
        return {"n_flagged": 0, "out_path": str(out_path)}

    suspected = entry_df[
        (entry_df["y_has_match"] == 0) & (entry_df["p_has_match"] >= min_prob)
    ].sort_values(by="p_has_match", ascending=False)

    n_total_flagged = int(len(suspected))
    top_rows = suspected.head(top_k)

    cols_to_dump = [
        c
        for c in (
            "arxiv_id",
            "citation_key",
            "top1_ref_arxiv_id",
            "p_has_match",
            "top1_score",
            "score_margin_1_2",
            "top1_title_word_tfidf_cos",
            "top1_raw_cand_title_cov",
            "top1_first_surname_jw",
            "top1_surname_containment",
            "top1_year_diff_arxiv",
            "top1_cand_claim_count",
            "is_best_claimant",
        )
        if c in top_rows.columns
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in top_rows[cols_to_dump].iterrows():
            rec = {
                k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                for k, v in row.items()
            }
            f.write(json.dumps(rec) + "\n")

    return {
        "n_flagged_ge_min_prob": n_total_flagged,
        "min_prob": min_prob,
        "exported_top_k": int(len(top_rows)),
        "out_path": str(out_path),
    }