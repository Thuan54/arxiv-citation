"""Phase 2: Publication pools and frozen, seeded split manifests.

Supports TWO workflows with one unified JSON schema:
1. Standard Milestone 3 split (`data/matching/splits.json`):
   Reads `data/matching/eligibility.json`, selects `>= 5 * k` manual-eligible
   publications (m >= 20, n >= 5) and `>= auto_pct` (floor >= 10%, min 2 * k)
   of remaining auto-eligible publications, places `k` manual + `k` auto in
   `valid` and `test`, and the remainder in `train`.
2. Rebalanced auto-only split (`--auto-only`, `data/matching/splits_auto.json`):
   Stratifies all auto-labelled publications strictly by `arxiv_id` into
   70/15/15 `train` / `valid` / `test`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_ELIGIBILITY_PATH = Path("data/matching/eligibility.json")
DEFAULT_STANDARD_SPLITS_PATH = Path("data/matching/splits.json")
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


def _stratified_sample_by_entries(
    rows: Sequence[Dict[str, Any]],
    target_n: int,
    rng: random.Random,
) -> List[str]:
    """Sample `target_n` publications evenly across entry-count terciles."""
    if target_n <= 0:
        return []
    if target_n >= len(rows):
        return sorted(str(r["arxiv_id"]) for r in rows)

    all_m = sorted(float(r.get("m_entries", 0)) for r in rows)
    buckets: Dict[int, List[str]] = defaultdict(list)
    for r in sorted(rows, key=lambda x: str(x["arxiv_id"])):
        b = _quantile_bucket(float(r.get("m_entries", 0)), all_m, n_bins=3)
        buckets[b].append(str(r["arxiv_id"]))

    chosen: List[str] = []
    remaining: List[str] = []
    total = len(rows)
    for b in sorted(buckets):
        ids = list(buckets[b])
        rng.shuffle(ids)
        take = min(len(ids), int(round(target_n * len(ids) / total)))
        chosen.extend(ids[:take])
        remaining.extend(ids[take:])

    rng.shuffle(remaining)
    while len(chosen) < target_n and remaining:
        chosen.append(remaining.pop())
    if len(chosen) > target_n:
        rng.shuffle(chosen)
        chosen = chosen[:target_n]
    return sorted(chosen)


def build_standard_manifest(
    eligibility_payload: Dict[str, Any],
    members: int = 4,
    manual_per_member: int = 5,
    auto_pct: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build standard Milestone 3 `splits.json` (manual + auto pools)."""
    if members < 1:
        raise ValueError("members must be >= 1")
    if manual_per_member < 5:
        raise ValueError("Milestone 3 requires manual_per_member >= 5")
    if auto_pct < 0.10 or auto_pct > 1.0:
        raise ValueError("Milestone 3 requires 0.10 <= auto_pct <= 1.0")

    per_pub = eligibility_payload.get("per_publication")
    if not isinstance(per_pub, list) or not per_pub:
        raise ValueError("eligibility.json missing non-empty 'per_publication' list")

    by_id = {str(r["arxiv_id"]): r for r in per_pub}
    manual_eligible = [r for r in per_pub if r.get("manual_eligible")]
    n_manual = members * manual_per_member
    if len(manual_eligible) < n_manual:
        raise ValueError(
            f"Need {n_manual} manual-eligible publications (m>=20, n>=5), "
            f"only found {len(manual_eligible)}"
        )

    rng = random.Random(seed)
    manual_pool = _stratified_sample_by_entries(manual_eligible, n_manual, rng)
    manual_set = set(manual_pool)

    remaining_auto_eligible = [
        r for r in per_pub
        if r.get("auto_eligible") and str(r["arxiv_id"]) not in manual_set
    ]
    n_auto = max(2 * members, int(math.ceil(auto_pct * len(remaining_auto_eligible))))
    if len(remaining_auto_eligible) < n_auto:
        raise ValueError(
            f"Need {n_auto} auto-eligible publications outside manual pool, "
            f"only found {len(remaining_auto_eligible)}"
        )
    auto_pool = _stratified_sample_by_entries(remaining_auto_eligible, n_auto, rng)

    m_shuf = list(manual_pool)
    a_shuf = list(auto_pool)
    rng.shuffle(m_shuf)
    rng.shuffle(a_shuf)

    partitions: Dict[str, Dict[str, List[str]]] = {
        "valid": {
            "manual": sorted(m_shuf[:members]),
            "auto": sorted(a_shuf[:members]),
        },
        "test": {
            "manual": sorted(m_shuf[members : 2 * members]),
            "auto": sorted(a_shuf[members : 2 * members]),
        },
        "train": {
            "manual": sorted(m_shuf[2 * members :]),
            "auto": sorted(a_shuf[2 * members :]),
        },
    }

    splits: Dict[str, List[str]] = {
        part: sorted(partitions[part]["manual"] + partitions[part]["auto"])
        for part in ("train", "valid", "test")
    }
    validate_splits(splits)

    pub_to_split: Dict[str, str] = {}
    summary: Dict[str, Dict[str, int]] = {}
    for part in ("train", "valid", "test"):
        for p in splits[part]:
            pub_to_split[p] = part
        m_ids = partitions[part]["manual"]
        a_ids = partitions[part]["auto"]
        summary[part] = {
            "manual_pubs": len(m_ids),
            "auto_pubs": len(a_ids),
            "total_pubs": len(splits[part]),
            "manual_entries": int(sum(by_id[p]["m_entries"] for p in m_ids)),
            "auto_entries": int(sum(by_id[p]["m_entries"] for p in a_ids)),
        }

    return {
        "mode": "standard",
        "seed": seed,
        "members": members,
        "manual_per_member": manual_per_member,
        "auto_pct": auto_pct,
        "manifest_sha256": _compute_manifest_sha256(splits),
        "manual_pool": manual_pool,
        "auto_pool": auto_pool,
        "partitions": partitions,
        "splits": splits,
        "pub_to_split": pub_to_split,
        "summary": summary,
    }


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
    partitions: Dict[str, Dict[str, List[str]]] = {}
    for part in ("train", "valid", "test"):
        pubs = splits[part]
        partitions[part] = {"manual": [], "auto": list(pubs)}
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
        "partitions": partitions,
        "splits": splits,
        "pub_to_split": pub_to_split,
        "summary": summary,
    }


def load_pub_to_split(splits_path: Path) -> Dict[str, str]:
    """Read pub_id -> partition mapping from any splits.json or splits_auto.json format."""
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    if isinstance(data.get("pub_to_split"), dict) and data["pub_to_split"]:
        return {str(k): str(v) for k, v in data["pub_to_split"].items()}
    mapping: Dict[str, str] = {}
    if isinstance(data.get("partitions"), dict):
        for part in ("train", "valid", "test"):
            pdict = data["partitions"].get(part, {})
            if isinstance(pdict, dict):
                for src in ("manual", "auto"):
                    for pub_id in pdict.get(src, []):
                        mapping[str(pub_id)] = part
        if mapping:
            return mapping
    splits = data.get("splits", data)
    for part in ("train", "valid", "test"):
        for pub_id in splits.get(part, []):
            mapping[str(pub_id)] = part
    if not mapping:
        raise ValueError(f"Could not parse train/valid/test splits from {splits_path}")
    return mapping


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen publication-level splits.json (standard) or splits_auto.json (--auto-only)."
    )
    parser.add_argument(
        "--eligibility",
        type=Path,
        default=DEFAULT_ELIGIBILITY_PATH,
        help="Path to eligibility.json from data.py (standard splits.json mode).",
    )
    parser.add_argument(
        "--members",
        type=int,
        default=4,
        help="Number of group members k (standard mode: >=5*k manual pubs, k manual + k auto in valid/test).",
    )
    parser.add_argument(
        "--manual-per-member",
        type=int,
        default=5,
        help="Manual publications per member (spec floor: 5).",
    )
    parser.add_argument(
        "--auto-pct",
        type=float,
        default=0.15,
        help="Fraction of remaining auto-eligible publications to sample (spec floor: 0.10).",
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
        default=None,
        help="Output JSON path (defaults to splits.json in standard mode, splits_auto.json in --auto-only).",
    )
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--valid-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.auto_only:
        out_path = args.out or DEFAULT_SPLITS_PATH
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
    else:
        out_path = args.out or DEFAULT_STANDARD_SPLITS_PATH
        if not args.eligibility.exists():
            raise SystemExit(
                f"Eligibility file not found: {args.eligibility}. "
                "Run `python -m src.m3_matching.data` first."
            )
        elig = json.loads(args.eligibility.read_text(encoding="utf-8"))
        manifest = build_standard_manifest(
            elig,
            members=args.members,
            manual_per_member=args.manual_per_member,
            auto_pct=args.auto_pct,
            seed=args.seed,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Saved {manifest['mode']} manifest ({manifest['manifest_sha256'][:12]}) -> {out_path}")


if __name__ == "__main__":
    main()