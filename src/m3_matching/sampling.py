"""Phase 2: Publication pools and frozen, seeded split manifest.

Supports --auto-only mode to stratify auto-labelled publications strictly by
pub_id into train / valid / test (default 70% / 15% / 15%), ensuring valid and
test have enough publications (~112 each on 748 pubs) for reliable Has-Match
calibration, M5 tuning, and evaluation without manual labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_SPLITS_PATH = Path("data/matching/splits_auto.json")
DEFAULT_AUTO_LABELS_PATH = Path("data/matching/auto_labels.jsonl")


def _compute_manifest_sha256(splits: Dict[str, List[str]]) -> str:
    canonical = json.dumps(
        {k: sorted(v) for k, v in sorted(splits.items())},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_splits(splits: Dict[str, List[str]]) -> None:
    """Assert strictly disjoint train/valid/test publication sets."""
    required = {"train", "valid", "test"}
    if set(splits.keys()) != required:
        raise ValueError(f"Expected splits keys {required}, got {set(splits.keys())}")
    seen: Dict[str, str] = {}
    for part in ("train", "valid", "test"):
        pubs = splits[part]
        if len(pubs) != len(set(pubs)):
            raise ValueError(f"Duplicate pub_id inside split '{part}'")
        for p in pubs:
            if p in seen:
                raise ValueError(
                    f"Publication '{p}' appears in both '{seen[p]}' and '{part}'"
                )
            seen[p] = part


def _quantile_bucket(value: float, sorted_vals: Sequence[float], n_bins: int = 3) -> int:
    if not sorted_vals or n_bins <= 1:
        return 0
    n = len(sorted_vals)
    for b in range(1, n_bins):
        idx = min(n - 1, int(round(b * n / n_bins)))
        if value <= sorted_vals[idx]:
            return b - 1
    return n_bins - 1


def stratified_pub_split(
    pub_stats: Dict[str, Dict[str, float]],
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Stratified publication-level split by (entry_count_bucket, match_rate_bucket)."""
    if abs((train_frac + valid_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + valid_frac + test_frac must equal 1.0")
    if len(pub_stats) < 3:
        raise ValueError(f"Need at least 3 publications to split, got {len(pub_stats)}")

    rng = random.Random(seed)
    all_entries = sorted(v.get("n_entries", 0.0) for v in pub_stats.values())
    all_rates = sorted(v.get("match_rate", 0.0) for v in pub_stats.values())

    strata: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for pub_id in sorted(pub_stats.keys()):
        st = pub_stats[pub_id]
        b_ent = _quantile_bucket(st.get("n_entries", 0.0), all_entries, n_bins=3)
        b_rate = _quantile_bucket(st.get("match_rate", 0.0), all_rates, n_bins=3)
        strata[(b_ent, b_rate)].append(pub_id)

    splits: Dict[str, List[str]] = {"train": [], "valid": [], "test": []}
    leftovers: List[str] = []

    for key in sorted(strata.keys()):
        bucket = list(strata[key])
        rng.shuffle(bucket)
        n = len(bucket)
        n_valid = int(round(n * valid_frac))
        n_test = int(round(n * test_frac))
        if n_valid + n_test >= n and n >= 3:
            n_valid = max(1, min(n_valid, n // 3))
            n_test = max(1, min(n_test, n // 3))
        if n < 3:
            leftovers.extend(bucket)
            continue
        splits["valid"].extend(bucket[:n_valid])
        splits["test"].extend(bucket[n_valid : n_valid + n_test])
        splits["train"].extend(bucket[n_valid + n_test :])

    if leftovers:
        rng.shuffle(leftovers)
        total_now = sum(len(v) for v in splits.values()) + len(leftovers)
        target_valid = max(1, int(round(total_now * valid_frac)))
        target_test = max(1, int(round(total_now * test_frac)))
        for p in leftovers:
            if len(splits["valid"]) < target_valid:
                splits["valid"].append(p)
            elif len(splits["test"]) < target_test:
                splits["test"].append(p)
            else:
                splits["train"].append(p)

    for part in ("valid", "test"):
        if not splits[part] and len(splits["train"]) > 1:
            splits[part].append(splits["train"].pop())

    for part in splits:
        splits[part].sort()

    validate_splits(splits)
    return splits


def load_pub_stats_from_labels(labels_path: Path) -> Dict[str, Dict[str, float]]:
    """Aggregate per-publication entry counts and match rates from labels JSONL."""
    counts: Dict[str, int] = Counter()
    matched: Dict[str, int] = Counter()
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pub_id = str(row["arxiv_id"])
            counts[pub_id] += 1
            if row.get("true_ref_arxiv_id") is not None:
                matched[pub_id] += 1
    return {
        pub_id: {
            "n_entries": float(n),
            "n_matched": float(matched[pub_id]),
            "match_rate": float(matched[pub_id]) / max(1, n),
        }
        for pub_id, n in counts.items()
    }


def build_auto_only_manifest(
    pub_stats: Dict[str, Dict[str, float]],
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create a complete splits manifest for auto-only mode."""
    splits = stratified_pub_split(
        pub_stats,
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
        seed=seed,
    )
    pub_to_split: Dict[str, str] = {}
    summary: Dict[str, Dict[str, float]] = {}
    for part in ("train", "valid", "test"):
        pubs = splits[part]
        for p in pubs:
            pub_to_split[p] = part
        n_ent = sum(pub_stats[p]["n_entries"] for p in pubs)
        n_mat = sum(pub_stats[p].get("n_matched", 0.0) for p in pubs)
        summary[part] = {
            "n_publications": len(pubs),
            "n_entries": int(n_ent),
            "n_matched": int(n_mat),
            "match_rate_pct": round(100.0 * n_mat / max(1.0, n_ent), 2),
        }

    all_pubs = sorted(pub_stats.keys())
    return {
        "mode": "auto_only",
        "seed": seed,
        "ratios": {
            "train": train_frac,
            "valid": valid_frac,
            "test": test_frac,
        },
        "manifest_sha256": _compute_manifest_sha256(splits),
        "manual_pool": [],
        "auto_pool": all_pubs,
        "splits": splits,
        "pub_to_split": pub_to_split,
        "summary": summary,
    }


def load_pub_to_split(splits_path: Path) -> Dict[str, str]:
    """Read pub_id -> partition mapping from any splits.json format."""
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    if isinstance(data.get("pub_to_split"), dict) and data["pub_to_split"]:
        return {str(k): str(v) for k, v in data["pub_to_split"].items()}
    splits = data.get("splits", data)
    mapping: Dict[str, str] = {}
    for part in ("train", "valid", "test"):
        for pub_id in splits.get(part, []):
            mapping[str(pub_id)] = part
    if not mapping:
        raise ValueError(f"Could not parse train/valid/test splits from {splits_path}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen publication-level splits manifest (supports --auto-only)."
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Rebalance auto-labelled publications into 70/15/15 train/valid/test.",
    )
    parser.add_argument(
        "--from-labels",
        type=Path,
        default=DEFAULT_AUTO_LABELS_PATH,
        help="Path to auto_labels.jsonl when using --auto-only.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SPLITS_PATH,
        help="Output path for splits manifest JSON (default: data/matching/splits_auto.json).",
    )
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--valid-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.auto_only:
        raise SystemExit(
            "Pass --auto-only --from-labels data/matching/auto_labels.jsonl "
            "to generate the rebalanced auto-only split manifest."
        )

    if not args.from_labels.exists():
        raise SystemExit(f"Labels file not found: {args.from_labels}")

    pub_stats = load_pub_stats_from_labels(args.from_labels)
    manifest = build_auto_only_manifest(
        pub_stats,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Saved manifest ({manifest['manifest_sha256'][:12]}) -> {args.out}")


if __name__ == "__main__":
    main()