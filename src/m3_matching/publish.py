"""Phase 7: Hugging Face 6-table sharded Parquet export, validator, and Dataset Card.

Builds the final submission dataset directory matching Milestone 1, 2, and 3 specs:

    <out_dir>/
    ├── README.md                 # Hugging Face YAML configs + dataset card
    ├── dataset_manifest.json     # Row counts, shard counts, byte sizes, validation report
    └── data/
        ├── papers/               # part-00000.parquet, ...
        ├── references/           # part-00000.parquet, ...
        ├── elements/             # part-00000.parquet, ... (streamed in bounded shards)
        ├── hierarchy/            # part-00000.parquet, ... (streamed in bounded shards)
        ├── bibtex/               # part-00000.parquet, ...
        └── predictions/          # part-00000.parquet, ... (train/valid/test pubs ONLY)

Usage:
    # 1. After freezing models and running --eval-test ONCE:
    python -m src.m3_matching.publish --out-dir data/hf_dataset

    # 2. Dry-run before running --eval-test (validates schemas + M1/M2 + train/valid):
    python -m src.m3_matching.publish --out-dir data/hf_dataset_dryrun --allow-missing-test

    # 3. Re-validate an existing exported dataset folder without rewriting shards:
    python -m src.m3_matching.publish --out-dir data/hf_dataset --validate-only

    # 4. Upload validated dataset folder to Hugging Face Hub (uses HF_TOKEN or `hf auth login`):
    python -m src.m3_matching.publish --out-dir data/hf_dataset --validate-only --push-to-hub <org-or-user>/<dataset-name>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - surfaced clearly at runtime
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


TABLE_NAMES: Tuple[str, ...] = (
    "papers",
    "references",
    "elements",
    "hierarchy",
    "bibtex",
    "predictions",
)

VALID_PARTITIONS: Tuple[str, ...] = ("train", "valid", "test")
_YEAR_RE = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")
_VERSION_RE = re.compile(r"v?(\d+)", re.IGNORECASE)


def _require_pyarrow() -> None:
    if pa is None or pq is None:
        raise RuntimeError(
            "pyarrow is required for Parquet export and validation. "
            "Install with: pip install pyarrow"
        )


def get_table_schemas() -> Dict[str, "pa.Schema"]:
    """Return exact PyArrow schemas for all six published tables."""
    _require_pyarrow()
    dict_str = pa.dictionary(pa.int32(), pa.string())
    str_list = pa.list_(pa.string())

    return {
        "papers": pa.schema([
            pa.field("arxiv_id", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=True),
            pa.field("authors", str_list, nullable=True),
            pa.field("submission_date", pa.string(), nullable=True),
            pa.field("revised_dates", str_list, nullable=True),
            pa.field("venue", pa.string(), nullable=True),
        ]),
        "references": pa.schema([
            pa.field("arxiv_id", pa.string(), nullable=False),
            pa.field("ref_arxiv_id", pa.string(), nullable=False),
            pa.field("ref_title", pa.string(), nullable=True),
            pa.field("ref_authors", str_list, nullable=True),
            pa.field("ref_submission_date", pa.string(), nullable=True),
            pa.field("ref_s2_id", pa.string(), nullable=True),
        ]),
        "elements": pa.schema([
            pa.field("element_id", pa.string(), nullable=False),
            pa.field("arxiv_id", pa.string(), nullable=False),
            pa.field("kind", dict_str, nullable=True),
            pa.field("content", pa.string(), nullable=True),
        ]),
        "hierarchy": pa.schema([
            pa.field("arxiv_id", pa.string(), nullable=False),
            pa.field("version", pa.int32(), nullable=False),
            pa.field("element_id", pa.string(), nullable=False),
            pa.field("parent_id", pa.string(), nullable=False),
        ]),
        "bibtex": pa.schema([
            pa.field("arxiv_id", pa.string(), nullable=False),
            pa.field("citation_key", pa.string(), nullable=False),
            pa.field("entry_type", dict_str, nullable=True),
            pa.field("title", pa.string(), nullable=True),
            pa.field("authors", str_list, nullable=True),
            pa.field("year", pa.int32(), nullable=True),
            pa.field("doi", pa.string(), nullable=True),
            pa.field("journal", pa.string(), nullable=True),
            pa.field("url", pa.string(), nullable=True),
        ]),
        "predictions": pa.schema([
            pa.field("arxiv_id", pa.string(), nullable=False),
            pa.field("partition", dict_str, nullable=False),
            pa.field("citation_key", pa.string(), nullable=False),
            pa.field("groundtruth_arxiv_id", pa.string(), nullable=True),
            pa.field("cand_1", pa.string(), nullable=True),
            pa.field("cand_2", pa.string(), nullable=True),
            pa.field("cand_3", pa.string(), nullable=True),
            pa.field("cand_4", pa.string(), nullable=True),
            pa.field("cand_5", pa.string(), nullable=True),
        ]),
    }


def clean_optional_str(value: Any) -> Optional[str]:
    """Convert nullable scalar value to stripped string or None (handles NaN / empty)."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text


def clean_required_str(value: Any) -> Optional[str]:
    """Convert non-null identifier to stripped string; returns None only if null/blank."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def clean_str_list(value: Any) -> List[str]:
    """Coerce author/date collections into a flat list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            s = clean_optional_str(item)
            if s is not None:
                out.append(s)
        return out
    try:
        out = []
        for item in list(value):
            s = clean_optional_str(item)
            if s is not None:
                out.append(s)
        return out
    except TypeError:
        s = clean_optional_str(value)
        return [s] if s is not None else []


def parse_version_int(value: Any) -> Optional[int]:
    """Parse version identifiers like 1, '1', 'v1', 'V2' -> int."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        iv = int(value)
        return iv if iv >= 0 and float(iv) == value else None
    m = _VERSION_RE.fullmatch(str(value).strip())
    return int(m.group(1)) if m else None


def parse_year_int(value: Any) -> Optional[int]:
    """Parse nullable publication year into int32 range or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1000 <= value <= 2100 else None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        iv = int(value)
        return iv if 1000 <= iv <= 2100 else None
    s = str(value).strip()
    if not s:
        return None
    m = _YEAR_RE.search(s)
    if not m:
        return None
    iv = int(m.group(1))
    return iv if 1000 <= iv <= 2100 else None


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream JSONL records line-by-line with clear error reporting."""
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_no}")
            yield obj


def normalize_paper_row(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    aid = clean_required_str(raw.get("arxiv_id"))
    if not aid:
        return None
    return {
        "arxiv_id": aid,
        "title": clean_optional_str(raw.get("title")),
        "authors": clean_str_list(raw.get("authors")),
        "submission_date": clean_optional_str(raw.get("submission_date")),
        "revised_dates": clean_str_list(raw.get("revised_dates")),
        "venue": clean_optional_str(raw.get("venue")),
    }


def normalize_reference_row(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    aid = clean_required_str(raw.get("arxiv_id"))
    ref_aid = clean_required_str(raw.get("ref_arxiv_id"))
    if not aid or not ref_aid:
        return None
    return {
        "arxiv_id": aid,
        "ref_arxiv_id": ref_aid,
        "ref_title": clean_optional_str(raw.get("ref_title")),
        "ref_authors": clean_str_list(raw.get("ref_authors")),
        "ref_submission_date": clean_optional_str(raw.get("ref_submission_date")),
        "ref_s2_id": clean_optional_str(raw.get("ref_s2_id")),
    }


def normalize_element_row(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    eid = clean_required_str(raw.get("element_id"))
    aid = clean_required_str(raw.get("arxiv_id"))
    if not eid or not aid:
        return None
    content_val = raw.get("content")
    if content_val is None or (isinstance(content_val, float) and math.isnan(content_val)):
        content = None
    else:
        content = str(content_val)
    return {
        "element_id": eid,
        "arxiv_id": aid,
        "kind": clean_optional_str(raw.get("kind")),
        "content": content,
    }


def normalize_hierarchy_row(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    aid = clean_required_str(raw.get("arxiv_id"))
    ver = parse_version_int(raw.get("version"))
    eid = clean_required_str(raw.get("element_id"))
    pid = clean_required_str(raw.get("parent_id"))
    if not aid or ver is None or not eid or not pid:
        return None
    return {
        "arxiv_id": aid,
        "version": ver,
        "element_id": eid,
        "parent_id": pid,
    }


def normalize_bibtex_row(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    aid = clean_required_str(raw.get("arxiv_id"))
    ck = clean_required_str(raw.get("citation_key"))
    if not aid or not ck:
        return None
    entry_type = clean_optional_str(raw.get("entry_type"))
    if entry_type is not None:
        entry_type = entry_type.lower()
    return {
        "arxiv_id": aid,
        "citation_key": ck,
        "entry_type": entry_type,
        "title": clean_optional_str(raw.get("title")),
        "authors": clean_str_list(raw.get("authors")),
        "year": parse_year_int(raw.get("year")),
        "doi": clean_optional_str(raw.get("doi")),
        "journal": clean_optional_str(raw.get("journal")),
        "url": clean_optional_str(raw.get("url")),
    }


def normalize_prediction_row(
    raw: Dict[str, Any],
    default_partition: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    aid = clean_required_str(raw.get("arxiv_id"))
    ck = clean_required_str(raw.get("citation_key"))
    part = clean_required_str(raw.get("partition")) or default_partition
    if not aid or not ck or part not in VALID_PARTITIONS:
        return None
    gt = clean_optional_str(raw.get("groundtruth_arxiv_id"))

    seen_cands: Set[str] = set()
    cands: List[Optional[str]] = []
    for k in range(1, 6):
        c = clean_optional_str(raw.get(f"cand_{k}"))
        if c is not None and c not in seen_cands:
            seen_cands.add(c)
            cands.append(c)
    while len(cands) < 5:
        cands.append(None)

    return {
        "arxiv_id": aid,
        "partition": part,
        "citation_key": ck,
        "groundtruth_arxiv_id": gt,
        "cand_1": cands[0],
        "cand_2": cands[1],
        "cand_3": cands[2],
        "cand_4": cands[3],
        "cand_5": cands[4],
    }


def load_split_partitions(splits_path: Optional[Path]) -> Dict[str, str]:
    """Load arxiv_id -> partition ('train'|'valid'|'test') mapping from splits JSON."""
    if splits_path is None or not splits_path.exists():
        return {}
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    for part in VALID_PARTITIONS:
        pubs = payload.get(part)
        if not isinstance(pubs, list):
            continue
        for pub in pubs:
            aid = clean_required_str(pub)
            if not aid:
                continue
            if aid in mapping and mapping[aid] != part:
                raise ValueError(
                    f"Split leakage in {splits_path}: {aid} in both {mapping[aid]} and {part}"
                )
            mapping[aid] = part
    return mapping


def compute_mrr5_from_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute dual MRR@5 (all entries vs matchable-only) from prediction rows."""
    n_entries = 0
    n_matchable = 0
    rr_sum = 0.0
    hit1 = 0
    hit5 = 0

    for row in rows:
        n_entries += 1
        gt = clean_optional_str(row.get("groundtruth_arxiv_id"))
        if gt is None:
            continue
        n_matchable += 1
        for rank in range(1, 6):
            if clean_optional_str(row.get(f"cand_{rank}")) == gt:
                rr_sum += 1.0 / rank
                if rank == 1:
                    hit1 += 1
                hit5 += 1
                break

    return {
        "n_entries": n_entries,
        "n_matchable": n_matchable,
        "n_no_match": n_entries - n_matchable,
        "oracle_ceiling": round(n_matchable / n_entries, 4) if n_entries else 0.0,
        "mrr5_all": round(rr_sum / n_entries, 4) if n_entries else 0.0,
        "mrr5_matchable": round(rr_sum / n_matchable, 4) if n_matchable else 0.0,
        "hit1_matchable": round(hit1 / n_matchable, 4) if n_matchable else 0.0,
        "hit5_matchable": round(hit5 / n_matchable, 4) if n_matchable else 0.0,
    }


def _rows_to_arrow_table(rows: Sequence[Dict[str, Any]], schema: "pa.Schema") -> "pa.Table":
    """Convert a batch of normalized row dicts into a strictly-typed PyArrow Table."""
    _require_pyarrow()
    arrays = []
    for field in schema:
        col_values = [row[field.name] for row in rows]
        if pa.types.is_dictionary(field.type):
            str_arr = pa.array(col_values, type=pa.string())
            arr = str_arr.dictionary_encode().cast(field.type)
        else:
            arr = pa.array(col_values, type=field.type)
        arrays.append(arr)
    return pa.Table.from_arrays(arrays, schema=schema)


def write_sharded_parquet(
    rows_iter: Iterable[Dict[str, Any]],
    schema: "pa.Schema",
    table_dir: Path,
    shard_rows: int = 250_000,
    compression: str = "zstd",
) -> Dict[str, Any]:
    """Stream rows into cleanly replaced part-XXXXX.parquet shards."""
    _require_pyarrow()
    if shard_rows <= 0:
        raise ValueError("shard_rows must be positive")

    if table_dir.exists():
        shutil.rmtree(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    shard_idx = 0
    total_bytes = 0
    shard_files: List[str] = []
    batch: List[Dict[str, Any]] = []

    def _flush(items: Sequence[Dict[str, Any]]) -> None:
        nonlocal shard_idx, total_bytes
        out_path = table_dir / f"part-{shard_idx:05d}.parquet"
        table = _rows_to_arrow_table(items, schema)
        pq.write_table(
            table,
            out_path,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        total_bytes += out_path.stat().st_size
        shard_files.append(out_path.name)
        shard_idx += 1

    for row in rows_iter:
        batch.append(row)
        total_rows += 1
        if len(batch) >= shard_rows:
            _flush(batch)
            batch.clear()

    if batch or shard_idx == 0:
        _flush(batch)
        batch.clear()

    return {
        "rows": total_rows,
        "shards": shard_idx,
        "bytes": total_bytes,
        "files": shard_files,
    }


def _stream_deduped_jsonl(
    path: Path,
    normalizer: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    key_fn: Optional[Callable[[Dict[str, Any]], Any]],
    stats: Dict[str, int],
) -> Iterator[Dict[str, Any]]:
    """Stream, normalize, and optionally deduplicate JSONL records in bounded batch RAM."""
    seen: Optional[Set[Any]] = set() if key_fn is not None else None
    for raw in iter_jsonl(path):
        stats["input_rows"] += 1
        row = normalizer(raw)
        if row is None:
            stats["skipped_invalid"] += 1
            continue
        if seen is not None and key_fn is not None:
            key = key_fn(row)
            if key in seen:
                stats["dropped_duplicates"] += 1
                continue
            seen.add(key)
        stats["exported_rows"] += 1
        yield row


def _iter_prediction_source_rows(
    models_dir: Path,
    splits_map: Dict[str, str],
    allow_missing_test: bool,
    stats: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Read train_oof, valid, and test prediction Parquets and strictly normalize."""
    _require_pyarrow()
    expected_files: List[Tuple[str, Path]] = [
        ("train", models_dir / "train_oof_predictions.parquet"),
        ("valid", models_dir / "valid_predictions.parquet"),
        ("test", models_dir / "test_predictions.parquet"),
    ]

    missing = [
        (part, p) for part, p in expected_files
        if not p.exists() and not (part == "test" and allow_missing_test)
    ]
    if missing:
        missing_desc = ", ".join(f"{part}={p}" for part, p in missing)
        hint = ""
        if any(part == "test" for part, _ in missing):
            hint = (
                " Run `python -m src.m3_matching.models --eval-test` once after "
                "freezing hyperparameters, or pass `--allow-missing-test` for a dry run."
            )
        raise FileNotFoundError(f"Missing prediction files: {missing_desc}.{hint}")

    rows: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[str, str]] = set()

    for expected_part, path in expected_files:
        if not path.exists():
            continue
        tbl = pq.read_table(path)
        for raw in tbl.to_pylist():
            stats["input_rows"] += 1
            row = normalize_prediction_row(raw, default_partition=expected_part)
            if row is None:
                stats["skipped_invalid"] += 1
                continue
            if splits_map:
                assigned_part = splits_map.get(row["arxiv_id"])
                if assigned_part is None:
                    stats["filtered_outside_splits"] += 1
                    continue
                if assigned_part != row["partition"]:
                    raise ValueError(
                        f"Partition mismatch for {row['arxiv_id']}: "
                        f"file has {row['partition']!r}, splits JSON has {assigned_part!r}"
                    )
            pk = (row["arxiv_id"], row["citation_key"])
            if pk in seen_keys:
                stats["dropped_duplicates"] += 1
                continue
            seen_keys.add(pk)
            rows.append(row)

    part_order = {"train": 0, "valid": 1, "test": 2}
    rows.sort(key=lambda r: (part_order[r["partition"]], r["arxiv_id"], r["citation_key"]))
    stats["exported_rows"] = len(rows)
    return rows


def export_all_tables(
    out_dir: Path,
    papers_path: Path,
    references_path: Path,
    elements_path: Path,
    hierarchy_path: Path,
    bibtex_path: Path,
    models_dir: Path,
    splits_path: Optional[Path] = None,
    shard_rows: int = 250_000,
    compression: str = "zstd",
    allow_missing_test: bool = False,
) -> Dict[str, Any]:
    """Export all six sharded Parquet tables into `<out_dir>/data/<table>/`."""
    _require_pyarrow()
    schemas = get_table_schemas()
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for label, path in [
        ("papers", papers_path),
        ("references", references_path),
        ("elements", elements_path),
        ("hierarchy", hierarchy_path),
        ("bibtex", bibtex_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required {label} source file not found: {path}")

    splits_map = load_split_partitions(splits_path)
    export_summary: Dict[str, Any] = {}

    p_stats = {"input_rows": 0, "exported_rows": 0, "skipped_invalid": 0, "dropped_duplicates": 0}
    p_meta = write_sharded_parquet(
        _stream_deduped_jsonl(papers_path, normalize_paper_row, lambda r: r["arxiv_id"], p_stats),
        schemas["papers"],
        data_dir / "papers",
        shard_rows=shard_rows,
        compression=compression,
    )
    export_summary["papers"] = {**p_meta, **p_stats}

    r_stats = {"input_rows": 0, "exported_rows": 0, "skipped_invalid": 0, "dropped_duplicates": 0}
    r_meta = write_sharded_parquet(
        _stream_deduped_jsonl(
            references_path,
            normalize_reference_row,
            lambda r: (r["arxiv_id"], r["ref_arxiv_id"]),
            r_stats,
        ),
        schemas["references"],
        data_dir / "references",
        shard_rows=shard_rows,
        compression=compression,
    )
    export_summary["references"] = {**r_meta, **r_stats}

    e_stats = {"input_rows": 0, "exported_rows": 0, "skipped_invalid": 0, "dropped_duplicates": 0}
    e_meta = write_sharded_parquet(
        _stream_deduped_jsonl(elements_path, normalize_element_row, lambda r: r["element_id"], e_stats),
        schemas["elements"],
        data_dir / "elements",
        shard_rows=shard_rows,
        compression=compression,
    )
    export_summary["elements"] = {**e_meta, **e_stats}

    h_stats = {"input_rows": 0, "exported_rows": 0, "skipped_invalid": 0, "dropped_duplicates": 0}
    h_meta = write_sharded_parquet(
        _stream_deduped_jsonl(
            hierarchy_path,
            normalize_hierarchy_row,
            lambda r: (r["arxiv_id"], r["version"], r["element_id"], r["parent_id"]),
            h_stats,
        ),
        schemas["hierarchy"],
        data_dir / "hierarchy",
        shard_rows=shard_rows,
        compression=compression,
    )
    export_summary["hierarchy"] = {**h_meta, **h_stats}

    b_stats = {"input_rows": 0, "exported_rows": 0, "skipped_invalid": 0, "dropped_duplicates": 0}
    b_meta = write_sharded_parquet(
        _stream_deduped_jsonl(
            bibtex_path,
            normalize_bibtex_row,
            lambda r: (r["arxiv_id"], r["citation_key"]),
            b_stats,
        ),
        schemas["bibtex"],
        data_dir / "bibtex",
        shard_rows=shard_rows,
        compression=compression,
    )
    export_summary["bibtex"] = {**b_meta, **b_stats}

    pred_stats = {
        "input_rows": 0,
        "exported_rows": 0,
        "skipped_invalid": 0,
        "dropped_duplicates": 0,
        "filtered_outside_splits": 0,
    }
    pred_rows = _iter_prediction_source_rows(
        models_dir=models_dir,
        splits_map=splits_map,
        allow_missing_test=allow_missing_test,
        stats=pred_stats,
    )
    pred_meta = write_sharded_parquet(
        pred_rows,
        schemas["predictions"],
        data_dir / "predictions",
        shard_rows=shard_rows,
        compression=compression,
    )
    export_summary["predictions"] = {**pred_meta, **pred_stats}

    return export_summary


def _iter_parquet_batches(
    table_dir: Path,
    columns: Optional[Sequence[str]] = None,
    batch_size: int = 100_000,
) -> Iterator[Dict[str, List[Any]]]:
    """Stream batches of Python lists from all shards in a table directory."""
    _require_pyarrow()
    shards = sorted(table_dir.glob("*.parquet"))
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rb in pf.iter_batches(batch_size=batch_size, columns=list(columns) if columns else None):
            yield rb.to_pydict()


def _schema_matches(actual: "pa.Schema", expected: "pa.Schema") -> Tuple[bool, List[str]]:
    """Check exact column names, order, primitive/list/dictionary types, and nullability."""
    diffs: List[str] = []
    if len(actual) != len(expected):
        diffs.append(f"column count mismatch: got {len(actual)}, expected {len(expected)}")
        return False, diffs
    for a_field, e_field in zip(actual, expected):
        if a_field.name != e_field.name:
            diffs.append(f"column name mismatch: got {a_field.name!r}, expected {e_field.name!r}")
        if a_field.type != e_field.type:
            diffs.append(
                f"column {e_field.name!r} type mismatch: got {a_field.type}, expected {e_field.type}"
            )
        if not e_field.nullable and a_field.nullable:
            diffs.append(f"column {e_field.name!r} must be non-nullable")
    return len(diffs) == 0, diffs


def validate_dataset(
    out_dir: Path,
    splits_path: Optional[Path] = None,
    allow_missing_test: bool = False,
    check_elements_fk: bool = True,
    strict_papers_fk: bool = True,
) -> Dict[str, Any]:
    """Validate schemas, PK uniqueness, FK integrity, and prediction metrics on disk."""
    _require_pyarrow()
    schemas = get_table_schemas()
    data_dir = out_dir / "data"

    errors: List[str] = []
    warnings: List[str] = []
    table_stats: Dict[str, Dict[str, Any]] = {}

    for name in TABLE_NAMES:
        tdir = data_dir / name
        shards = sorted(tdir.glob("*.parquet")) if tdir.exists() else []
        if not shards:
            errors.append(f"[{name}] no Parquet shards found in {tdir}")
            continue
        total_rows = 0
        total_bytes = 0
        for shard in shards:
            pf = pq.ParquetFile(shard)
            actual_schema = pf.schema_arrow
            ok, diffs = _schema_matches(actual_schema, schemas[name])
            if not ok:
                errors.append(f"[{name}] schema mismatch in {shard.name}: {'; '.join(diffs)}")
            total_rows += pf.metadata.num_rows
            total_bytes += shard.stat().st_size
        table_stats[name] = {
            "rows": total_rows,
            "shards": len(shards),
            "bytes": total_bytes,
            "mb": round(total_bytes / (1024 * 1024), 3),
        }

    if errors:
        return {
            "passed": False,
            "errors": errors,
            "warnings": warnings,
            "tables": table_stats,
        }

    paper_ids: Set[str] = set()
    dup_papers = 0
    for batch in _iter_parquet_batches(data_dir / "papers", columns=["arxiv_id"]):
        for aid in batch["arxiv_id"]:
            if not aid:
                errors.append("[papers] null/empty arxiv_id encountered")
                continue
            if aid in paper_ids:
                dup_papers += 1
            else:
                paper_ids.add(aid)
    if dup_papers:
        errors.append(f"[papers] {dup_papers} duplicate arxiv_id values found")

    ref_pairs_by_pub: Dict[str, Set[str]] = defaultdict(set)
    dup_refs = 0
    orphan_ref_pubs: Set[str] = set()
    for batch in _iter_parquet_batches(
        data_dir / "references", columns=["arxiv_id", "ref_arxiv_id"]
    ):
        for aid, raid in zip(batch["arxiv_id"], batch["ref_arxiv_id"]):
            if not aid or not raid:
                errors.append("[references] null/empty arxiv_id or ref_arxiv_id encountered")
                continue
            if aid not in paper_ids:
                orphan_ref_pubs.add(aid)
            pub_refs = ref_pairs_by_pub[aid]
            if raid in pub_refs:
                dup_refs += 1
            else:
                pub_refs.add(raid)
    if dup_refs:
        errors.append(f"[references] {dup_refs} duplicate (arxiv_id, ref_arxiv_id) rows found")
    if orphan_ref_pubs:
        msg = (
            f"[references] {len(orphan_ref_pubs)} citing arxiv_id values not in papers "
            f"(sample: {sorted(orphan_ref_pubs)[:5]})"
        )
        (errors if strict_papers_fk else warnings).append(msg)

    bib_keys_by_pub: Dict[str, Set[str]] = defaultdict(set)
    dup_bib = 0
    orphan_bib_pubs: Set[str] = set()
    for batch in _iter_parquet_batches(
        data_dir / "bibtex", columns=["arxiv_id", "citation_key"]
    ):
        for aid, ck in zip(batch["arxiv_id"], batch["citation_key"]):
            if not aid or not ck:
                errors.append("[bibtex] null/empty arxiv_id or citation_key encountered")
                continue
            if aid not in paper_ids:
                orphan_bib_pubs.add(aid)
            pub_cks = bib_keys_by_pub[aid]
            if ck in pub_cks:
                dup_bib += 1
            else:
                pub_cks.add(ck)
    if dup_bib:
        errors.append(f"[bibtex] {dup_bib} duplicate (arxiv_id, citation_key) rows found")
    if orphan_bib_pubs:
        msg = (
            f"[bibtex] {len(orphan_bib_pubs)} arxiv_id values not in papers "
            f"(sample: {sorted(orphan_bib_pubs)[:5]})"
        )
        (errors if strict_papers_fk else warnings).append(msg)

    element_ids: Set[str] = set()
    dup_elements = 0
    orphan_elem_pubs: Set[str] = set()
    if check_elements_fk:
        for batch in _iter_parquet_batches(
            data_dir / "elements", columns=["element_id", "arxiv_id"]
        ):
            for eid, aid in zip(batch["element_id"], batch["arxiv_id"]):
                if not eid or not aid:
                    errors.append("[elements] null/empty element_id or arxiv_id encountered")
                    continue
                if aid not in paper_ids:
                    orphan_elem_pubs.add(aid)
                if eid in element_ids:
                    dup_elements += 1
                else:
                    element_ids.add(eid)
        if dup_elements:
            errors.append(f"[elements] {dup_elements} duplicate element_id rows found")
        if orphan_elem_pubs:
            msg = (
                f"[elements] {len(orphan_elem_pubs)} arxiv_id values not in papers "
                f"(sample: {sorted(orphan_elem_pubs)[:5]})"
            )
            (errors if strict_papers_fk else warnings).append(msg)

        missing_hier_elem = 0
        missing_hier_parent = 0
        orphan_hier_pubs: Set[str] = set()
        for batch in _iter_parquet_batches(
            data_dir / "hierarchy", columns=["arxiv_id", "version", "element_id", "parent_id"]
        ):
            for aid, ver, eid, pid in zip(
                batch["arxiv_id"], batch["version"], batch["element_id"], batch["parent_id"]
            ):
                if not aid or ver is None or not eid or not pid:
                    errors.append("[hierarchy] null/empty required column encountered")
                    continue
                if aid not in paper_ids:
                    orphan_hier_pubs.add(aid)
                if eid not in element_ids:
                    missing_hier_elem += 1
                if pid not in element_ids:
                    missing_hier_parent += 1
        if missing_hier_elem:
            errors.append(
                f"[hierarchy] {missing_hier_elem} rows have element_id missing from elements"
            )
        if missing_hier_parent:
            errors.append(
                f"[hierarchy] {missing_hier_parent} rows have parent_id missing from elements"
            )
        if orphan_hier_pubs:
            msg = (
                f"[hierarchy] {len(orphan_hier_pubs)} arxiv_id values not in papers "
                f"(sample: {sorted(orphan_hier_pubs)[:5]})"
            )
            (errors if strict_papers_fk else warnings).append(msg)

    splits_map = load_split_partitions(splits_path)
    pred_seen_keys: Set[Tuple[str, str]] = set()
    pub_to_partitions: Dict[str, Set[str]] = defaultdict(set)
    rows_by_partition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    dup_pred_pk = 0
    invalid_partitions: Set[str] = set()
    missing_pred_bib = 0
    missing_pred_gt_ref = 0
    missing_pred_cand_ref = 0
    dup_within_row_cands = 0
    outside_splits_pubs: Set[str] = set()
    pred_not_in_papers: Set[str] = set()

    for batch in _iter_parquet_batches(data_dir / "predictions"):
        n_batch = len(batch["arxiv_id"])
        for i in range(n_batch):
            row = {col: batch[col][i] for col in batch}
            aid = row["arxiv_id"]
            part = row["partition"]
            ck = row["citation_key"]
            gt = row["groundtruth_arxiv_id"]
            cands = [row[f"cand_{k}"] for k in range(1, 6)]

            if part not in VALID_PARTITIONS:
                invalid_partitions.add(str(part))
                continue

            pub_to_partitions[aid].add(part)
            rows_by_partition[part].append(row)

            if aid not in paper_ids:
                pred_not_in_papers.add(aid)

            if splits_map and splits_map.get(aid) != part:
                outside_splits_pubs.add(aid)

            pk = (aid, ck)
            if pk in pred_seen_keys:
                dup_pred_pk += 1
            else:
                pred_seen_keys.add(pk)

            if ck not in bib_keys_by_pub.get(aid, set()):
                missing_pred_bib += 1

            pub_refs = ref_pairs_by_pub.get(aid, set())
            if gt is not None and gt not in pub_refs:
                missing_pred_gt_ref += 1

            non_null_cands = [c for c in cands if c is not None]
            if len(non_null_cands) != len(set(non_null_cands)):
                dup_within_row_cands += 1
            for c in non_null_cands:
                if c not in pub_refs:
                    missing_pred_cand_ref += 1

    leaked_pubs = sorted(aid for aid, parts in pub_to_partitions.items() if len(parts) > 1)
    if invalid_partitions:
        errors.append(f"[predictions] invalid partition values: {sorted(invalid_partitions)}")
    if leaked_pubs:
        errors.append(
            f"[predictions] {len(leaked_pubs)} publications appear in multiple partitions "
            f"(sample: {leaked_pubs[:5]})"
        )
    if dup_pred_pk:
        errors.append(f"[predictions] {dup_pred_pk} duplicate (arxiv_id, citation_key) rows found")
    if pred_not_in_papers:
        errors.append(
            f"[predictions] {len(pred_not_in_papers)} arxiv_id values missing from papers "
            f"(sample: {sorted(pred_not_in_papers)[:5]})"
        )
    if outside_splits_pubs:
        errors.append(
            f"[predictions] {len(outside_splits_pubs)} publications mismatch/outside splits JSON "
            f"(sample: {sorted(outside_splits_pubs)[:5]})"
        )
    if missing_pred_bib:
        errors.append(
            f"[predictions] {missing_pred_bib} (arxiv_id, citation_key) pairs missing from bibtex"
        )
    if missing_pred_gt_ref:
        errors.append(
            f"[predictions] {missing_pred_gt_ref} non-null groundtruth_arxiv_id values "
            "missing from references for the same arxiv_id"
        )
    if missing_pred_cand_ref:
        errors.append(
            f"[predictions] {missing_pred_cand_ref} non-null cand_1..cand_5 values "
            "missing from references for the same arxiv_id"
        )
    if dup_within_row_cands:
        errors.append(
            f"[predictions] {dup_within_row_cands} rows contain duplicate candidates across cand_1..cand_5"
        )

    for required_part in VALID_PARTITIONS:
        if required_part == "test" and allow_missing_test:
            continue
        if not rows_by_partition.get(required_part):
            errors.append(f"[predictions] partition {required_part!r} has 0 rows")

    partition_metrics: Dict[str, Dict[str, Any]] = {}
    for part in VALID_PARTITIONS:
        part_rows = rows_by_partition.get(part, [])
        if not part_rows:
            continue
        part_pubs = {r["arxiv_id"] for r in part_rows}
        partition_metrics[part] = {
            "n_publications": len(part_pubs),
            **compute_mrr5_from_rows(part_rows),
        }

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "tables": table_stats,
        "predictions_by_partition": partition_metrics,
        "unique_papers": len(paper_ids),
        "unique_elements": len(element_ids) if check_elements_fk else None,
    }


def render_dataset_card(
    validation_report: Dict[str, Any],
    dataset_name: str = "arXiv LaTeX Citation & Reference Matching Corpus",
    repo_id: str = "<org-or-user>/<dataset-repo>",
    code_repo_url: str = "https://github.com/Thuan54/arxiv-citation",
    license_id: str = "cc-by-4.0",
) -> str:
    """Render the Hugging Face Dataset Card (`README.md`) with exact YAML configs."""
    yaml_lines = [
        "---",
        f"license: {license_id}",
        "task_categories:",
        "- text-classification",
        "- tabular-classification",
        "language:",
        "- en",
        "tags:",
        "- arxiv",
        "- citations",
        "- bibtex",
        "- latex",
        "- entity-resolution",
        "- reference-matching",
        "configs:",
    ]
    for name in TABLE_NAMES:
        yaml_lines.extend([
            f"- config_name: {name}",
            "  data_files:",
            "  - split: train",
            f"    path: data/{name}/*.parquet",
        ])
    yaml_lines.append("---")

    tables_info = validation_report.get("tables", {})
    table_descriptions = {
        "papers": "Crawled arXiv publication metadata (title, authors, dates, venue)",
        "references": "Crawled arXiv-keyed reference candidates per citing publication",
        "elements": "Deduplicated hierarchical LaTeX elements across all paper versions",
        "hierarchy": "Per-version node-parent edges linking elements into document trees",
        "bibtex": "Unified deduplicated BibTeX entries extracted from LaTeX/BibTeX sources",
        "predictions": "Milestone 3 reference matching top-5 ranked predictions (train/valid/test)",
    }

    summary_rows = [
        "| Table | Rows | Shards | Size (MB) | Description |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]
    total_rows = 0
    total_mb = 0.0
    for name in TABLE_NAMES:
        info = tables_info.get(name, {})
        rows = int(info.get("rows", 0))
        shards = int(info.get("shards", 0))
        mb = float(info.get("mb", 0.0))
        total_rows += rows
        total_mb += mb
        summary_rows.append(
            f"| `{name}` | {rows:,} | {shards} | {mb:.2f} | {table_descriptions[name]} |"
        )
    summary_rows.append(f"| **Total** | **{total_rows:,}** | — | **{total_mb:.2f}** | — |")

    part_metrics = validation_report.get("predictions_by_partition", {})
    metrics_section_lines: List[str] = []
    if part_metrics:
        metrics_section_lines.extend([
            "## Reference Matching Results (`predictions`)",
            "",
            "Evaluated directly on the published `predictions` Parquet shards using both",
            "all-labeled-entry MRR@5 (where entries without a matching arXiv reference score `0`,",
            "yielding an oracle ceiling equal to the matchable fraction) and matchable-only MRR@5:",
            "",
            "| Partition | Pubs | Entries | Matchable | Oracle Ceiling | MRR@5 (All) | MRR@5 (Matchable) | Hit@1 (Matchable) | Hit@5 (Matchable) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for part in VALID_PARTITIONS:
            m = part_metrics.get(part)
            if not m:
                continue
            metrics_section_lines.append(
                f"| `{part}` | {m['n_publications']:,} | {m['n_entries']:,} | "
                f"{m['n_matchable']:,} | {m['oracle_ceiling']:.4f} | "
                f"{m['mrr5_all']:.4f} | {m['mrr5_matchable']:.4f} | "
                f"{m['hit1_matchable']:.4f} | {m['hit5_matchable']:.4f} |"
            )
        metrics_section_lines.append("")

    generated_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    body = [
        *yaml_lines,
        "",
        f"# {dataset_name}",
        "",
        "Six-table sharded Parquet corpus spanning arXiv metadata crawling (Milestone 1),",
        "hierarchical multi-version LaTeX parsing and BibTeX deduplication (Milestone 2),",
        "and classical machine-learning reference matching (Milestone 3).",
        "",
        f"- **Source Code Repository:** {code_repo_url}",
        "- **Format:** Sharded Apache Parquet (flat primitive columns + one-level `list<string>`, streaming-ready)",
        f"- **Generated:** {generated_date}",
        "",
        "## Dataset Summary",
        "",
        *summary_rows,
        "",
        *metrics_section_lines,
        "## Loading with Hugging Face `datasets` (Streaming)",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        f'preds = load_dataset("{repo_id}", "predictions", split="train", streaming=True)',
        "for row in preds.take(3):",
        "    print(row)",
        "```",
        "",
        "Each table is exposed as its own configuration (`papers`, `references`, `elements`,",
        "`hierarchy`, `bibtex`, `predictions`) with a single `train` file split matching the",
        "Hub standard layout (`data/<table>/*.parquet`). For `predictions`, the ML split",
        "assignment (`train`, `valid`, `test`) is stored in the `partition` column.",
        "",
        "## Table Schemas",
        "",
        "- **`papers`**: `arxiv_id` (string, non-null), `title` (string), `authors` (list<string>), `submission_date` (string), `revised_dates` (list<string>), `venue` (string)",
        "- **`references`**: `arxiv_id` (string, non-null), `ref_arxiv_id` (string, non-null), `ref_title` (string), `ref_authors` (list<string>), `ref_submission_date` (string), `ref_s2_id` (string)",
        "- **`elements`**: `element_id` (string, non-null), `arxiv_id` (string, non-null), `kind` (dictionary<int32, string>), `content` (string)",
        "- **`hierarchy`**: `arxiv_id` (string, non-null), `version` (int32, non-null), `element_id` (string, non-null), `parent_id` (string, non-null)",
        "- **`bibtex`**: `arxiv_id` (string, non-null), `citation_key` (string, non-null), `entry_type` (dictionary<int32, string>), `title` (string), `authors` (list<string>), `year` (int32, nullable), `doi` (string), `journal` (string), `url` (string)",
        "- **`predictions`**: `arxiv_id` (string, non-null), `partition` (dictionary<int32, string>, non-null), `citation_key` (string, non-null), `groundtruth_arxiv_id` (string, nullable), `cand_1`..`cand_5` (string, nullable)",
        "",
        "## Modeling Overview",
        "",
        "Reference matching uses a 100% classical ML pipeline (no LLMs or neural embeddings):",
        "37 leak-free entry-reference similarity features (TF-IDF title cosine, character n-gram cosine,",
        "within-publication BM25 rank, author surname Jaro-Winkler & containment, year proximity,",
        "identifier signals, and raw-BibTeX fallback coverage), a LightGBM LambdaRank stage-1 ranker,",
        "a calibrated LightGBM `has_match` classifier with candidate-contention features, and a",
        "two-step calibrated Hungarian collective assignment (M5) to resolve within-paper collisions.",
        "",
    ]
    return "\n".join(body)


def push_dataset_to_hub(
    out_dir: Path,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload validated 6-table sharded Parquet dataset",
) -> str:
    """Upload the validated dataset directory (`README.md` + `data/`) to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --push-to-hub. "
            "Install with: pip install huggingface_hub"
        ) from exc

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["README.md", "data/**/*.parquet"],
        commit_message=commit_message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def _resolve_first_existing(candidates: Sequence[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7: Export & validate the 6-table Hugging Face sharded Parquet dataset."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/hf_dataset"))
    parser.add_argument(
        "--papers", type=Path, default=Path("data/metadata/papers_table.jsonl")
    )
    parser.add_argument(
        "--references", type=Path, default=Path("data/metadata/references_table.jsonl")
    )
    parser.add_argument(
        "--elements",
        type=Path,
        default=None,
        help="Path to elements.jsonl (auto-detects data/processed/ or data/parsed/)",
    )
    parser.add_argument(
        "--hierarchy",
        type=Path,
        default=None,
        help="Path to hierarchy.jsonl (auto-detects data/processed/ or data/parsed/)",
    )
    parser.add_argument(
        "--bibtex",
        type=Path,
        default=None,
        help="Path to bibtex.jsonl (auto-detects data/parsed/ or data/processed/)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("data/matching/models"),
        help="Directory holding train_oof/valid/test_predictions.parquet",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=None,
        help="Path to splits_auto.json or splits.json (auto-detected if omitted)",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=250_000,
        help="Maximum rows per Parquet shard (default: 250000)",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=("zstd", "snappy", "gzip", "none"),
        help="Parquet compression codec (default: zstd)",
    )
    parser.add_argument(
        "--allow-missing-test",
        action="store_true",
        help="Allow dry-run export/validation when test_predictions.parquet is not yet generated",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip Parquet export and validate an already-exported --out-dir",
    )
    parser.add_argument(
        "--skip-elements-fk",
        action="store_true",
        help="Skip in-memory element_id set check (only for severely RAM-constrained environments)",
    )
    parser.add_argument(
        "--warn-orphan-papers",
        action="store_true",
        help="Downgrade missing M1/M2 arxiv_id-in-papers checks from errors to warnings",
    )
    parser.add_argument(
        "--repo-id",
        default="<org-or-user>/<dataset-repo>",
        help="Hugging Face dataset repo ID displayed in README.md usage snippet",
    )
    parser.add_argument(
        "--code-repo-url",
        default="https://github.com/Thuan54/arxiv-citation",
        help="Public GitHub repository URL displayed in README.md",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="If set (e.g. 'username/arxiv-citation-dataset'), upload validated folder to HF Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face dataset repository as private (use only while staging)",
    )
    args = parser.parse_args(argv)

    elements_path = args.elements or _resolve_first_existing(
        [Path("data/processed/elements.jsonl"), Path("data/parsed/elements.jsonl")]
    )
    hierarchy_path = args.hierarchy or _resolve_first_existing(
        [Path("data/processed/hierarchy.jsonl"), Path("data/parsed/hierarchy.jsonl")]
    )
    bibtex_path = args.bibtex or _resolve_first_existing(
        [Path("data/parsed/bibtex.jsonl"), Path("data/processed/bibtex.jsonl")]
    )
    splits_path = args.splits
    if splits_path is None:
        candidate_splits = [
            Path("data/matching/splits_auto.json"),
            Path("data/matching/splits.json"),
        ]
        existing = [p for p in candidate_splits if p.exists()]
        splits_path = existing[0] if existing else None

    export_summary: Optional[Dict[str, Any]] = None
    if not args.validate_only:
        export_summary = export_all_tables(
            out_dir=args.out_dir,
            papers_path=args.papers,
            references_path=args.references,
            elements_path=elements_path,
            hierarchy_path=hierarchy_path,
            bibtex_path=bibtex_path,
            models_dir=args.models_dir,
            splits_path=splits_path,
            shard_rows=args.shard_rows,
            compression=args.compression,
            allow_missing_test=args.allow_missing_test,
        )

    validation = validate_dataset(
        out_dir=args.out_dir,
        splits_path=splits_path,
        allow_missing_test=args.allow_missing_test,
        check_elements_fk=not args.skip_elements_fk,
        strict_papers_fk=not args.warn_orphan_papers,
    )

    display_repo_id = args.push_to_hub or args.repo_id
    readme_text = render_dataset_card(
        validation_report=validation,
        repo_id=display_repo_id,
        code_repo_url=args.code_repo_url,
    )
    (args.out_dir / "README.md").write_text(readme_text, encoding="utf-8")

    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "export": export_summary,
        "validation": validation,
    }
    (args.out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))

    if not validation["passed"]:
        print(
            f"\nVALIDATION FAILED with {len(validation['errors'])} error(s). "
            "Fix errors above before uploading to Hugging Face.",
            file=sys.stderr,
        )
        return 1

    if args.push_to_hub:
        url = push_dataset_to_hub(
            out_dir=args.out_dir,
            repo_id=args.push_to_hub,
            private=args.private,
        )
        print(f"\nSuccessfully uploaded dataset to: {url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())