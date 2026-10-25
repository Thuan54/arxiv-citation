import json
import pandas as pd
import numpy as np

def has_valid_path(series: pd.Series) -> pd.Series:
    """Returns True where latex_source_path is neither NaN/None nor an empty string."""
    return series.notna() & (series.astype(str).str.strip() != "")

def build_book_summaries(crawled_jsonl_path: str, out_dir: str = "data"):
    records = []
    with open(crawled_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 1. Flatten to Version-Level Table (1 row per (arxiv_id, version))
    df_versions = pd.json_normalize(
        records,
        record_path=["versions"],
        meta=["arxiv_id"]
    )

    # Clean & compute version-level features
    df_versions["version_num"] = (
        df_versions["version"].astype(str).str.replace("v", "", regex=False).astype(int)
    )
    df_versions["num_authors"] = df_versions["metadata.authors"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df_versions["num_categories"] = df_versions["metadata.categories"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df_versions["primary_category"] = df_versions["metadata.categories"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )
    df_versions["num_raw_refs"] = df_versions["raw_references"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df_versions["abstract_words"] = (
        df_versions["metadata.abstract"].fillna("").astype(str).str.split().str.len()
    )
    df_versions["submission_dt"] = pd.to_datetime(
        df_versions["metadata.submission_date"], errors="coerce"
    )

    # 2. Aggregate to Paper-Level Table (1 row per unique arxiv_id)
    df_versions = df_versions.sort_values(["arxiv_id", "version_num"])
    df_papers = df_versions.groupby("arxiv_id", as_index=False).agg(
        num_versions=("version_num", "max"),
        crawled_versions=("version", "count"),
        versions_with_source=("latex_source_path", lambda s: int(has_valid_path(s).sum())),
        primary_category=("primary_category", "last"),
        num_categories=("num_categories", "last"),
        num_authors=("num_authors", "last"),
        abstract_words=("abstract_words", "last"),
        v1_refs=("num_raw_refs", "first"),
        latest_refs=("num_raw_refs", "last"),
        first_date=("submission_dt", "min"),
        latest_date=("submission_dt", "max"),
        missing_title=("metadata.title", lambda s: s.isna().any() or (s.astype(str).str.strip() == "").any()),
        missing_abstract=("metadata.abstract", lambda s: s.isna().any() or (s.astype(str).str.strip() == "").any()),
    )

    df_papers["revision_span_days"] = (
        (df_papers["latest_date"] - df_papers["first_date"]).dt.days
    )
    df_papers["ref_growth"] = df_papers["latest_refs"] - df_papers["v1_refs"]

    # Drop heavy text/list columns so the Parquet files stay tiny (< 5 MB) for GitHub
    cols_to_drop = ["raw_references", "metadata.abstract", "metadata.authors"]
    df_versions_light = df_versions.drop(columns=[c for c in cols_to_drop if c in df_versions.columns])
    
    df_papers.to_parquet(f"{out_dir}/papers_summary.parquet", index=False)
    df_versions_light.to_parquet(f"{out_dir}/versions_summary.parquet", index=False)
    print(f"Exported {len(df_papers)} papers and {len(df_versions_light)} versions to {out_dir}/")