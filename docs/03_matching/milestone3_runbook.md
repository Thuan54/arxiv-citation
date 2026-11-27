# Milestone 3 — End-to-End Reference Matching Runbook

Complete execution guide for **Milestone 3 (Reference Matching Pipeline & Hugging Face Dataset Submission)** using **100% classical machine learning** (no LLMs, no pretrained neural embeddings, no external API lookups during labelling or matching).

All commands assume you are running from the **repository root** (`arxiv-citation/`).

---

## 1. Prerequisites & Environment Setup

### 1.1 Required Milestone 1 & Milestone 2 Outputs

Verify that all six upstream files exist before starting:

```bash
ls -lh \
  data/metadata/papers_table.jsonl \
  data/metadata/references_table.jsonl \
  data/metadata/s2_references.jsonl \
  data/processed/elements.jsonl \
  data/processed/hierarchy.jsonl \
  data/parsed/bibtex.jsonl
```

| File | Milestone | Used In |
| :--- | :---: | :--- |
| `data/metadata/papers_table.jsonl` | M1 | Phase 7 (`papers` Parquet table & FK validation) |
| `data/metadata/references_table.jsonl` | M1 | Phases 1, 3, 4, 6, 7 (`references` table & candidate pool) |
| `data/metadata/s2_references.jsonl` | M1 | Phase 1 (enriches candidates with DOI / S2 external IDs) |
| `data/processed/elements.jsonl` | M2 | Phase 3 (`labeltool.py`) & Phase 7 (`elements` table) |
| `data/processed/hierarchy.jsonl` | M2 | Phase 7 (`hierarchy` table & FK validation) |
| `data/parsed/bibtex.jsonl` | M2 | Phases 1, 3, 6, 7 (`bibtex` table & BibTeX entries) |
| `data/parsed/bibtex_entries.jsonl` *(optional)* | M2 | Phase 1 (attaches raw BibTeX text for fallback features) |

> **Path Note:** Milestone 2 writes deduplicated BibTeX to `data/parsed/bibtex.jsonl` (not `data/processed/bibtex.jsonl`), while elements and hierarchy live under `data/processed/`.

### 1.2 Python Environment

Install all Milestone 3 + Phase 7 packages:

```bash
pip install numpy pandas pyarrow scikit-learn scipy lightgbm rapidfuzz matplotlib pytest huggingface_hub datasets
```

---

## 2. Pipeline Architecture & Choosing a Split Workflow

```text
M1 & M2 Tables (papers, references, s2_references, elements, hierarchy, parsed/bibtex)
        │
        ▼
Phase 0–1  (data.py)          ──► eligibility.json + preprocessed_{bibtex,references}.jsonl
        │
        ▼
Phase 2A   (sampling.py)      ──► splits.json  (Standard spec: manual pool + auto pool)
        │
        ├──► Workflow A (Manual + Auto)               ├──► Workflow B (Auto-Only 70/15/15)
        │    1. autolabel.py -> auto_labels.jsonl     │    1. autolabel.py -> auto_labels.jsonl
        │    2. labeltool.py -> labels.jsonl          │    2. sampling.py --auto-only
        │       (uses splits.json)                    │       -> splits_auto.json
        └──────────────────────┬──────────────────────┘
                               ▼
Phase 4    (features.py)      ──► pair_features.parquet (37 leak-free pair features)
                               │
                               ▼
Phase 5    (models.py)        ──► B1/M3/M4 + Stage-1.5 Has-Match + Calibrated M5
                               │
                               ▼
Phase 6    (inspect_errors.py + evaluate.py) ──► Error cards, 100-entry audit, CIs & 4 plots
                               │
                               ▼
Final Test (models.py --eval-test) ──► test_predictions.parquet (RUN ONCE AFTER FREEZING!)
                               │
                               ▼
Phase 7    (publish.py)       ──► 6 sharded Parquet tables + README.md + Validator + HF Hub
```

### Workflow A vs. Workflow B

| Aspect | Workflow A — Standard Spec (`splits.json`) | Workflow B — Auto-Only (`splits_auto.json`) |
| :--- | :--- | :--- |
| **Ground Truth** | Manual labels (`>= 5*k` pubs) + Auto labels (`>= 10%` remaining) | Auto labels only (`auto_labels.jsonl`) |
| **Split Rule** | `valid`: `k` manual + `k` auto; `test`: `k` manual + `k` auto; `train`: rest | Stratified 70% `train` / 15% `valid` / 15% `test` by `(entry_count, match_rate)` |
| **When to Use** | Full course submission with completed human annotation sheets | Training/evaluating with large valid & test sets (`~112` pubs each) using auto labels only |
| **Labels File** | `data/matching/labels.jsonl` (merged manual + auto) | `data/matching/auto_labels.jsonl` |
| **Splits File** | `data/matching/splits.json` | `data/matching/splits_auto.json` |

> **Important:** Even in **Workflow B**, you **always run Phase 2A (`splits.json`) first**, because `autolabel.py` reads the automatic publication pool from `data/matching/splits.json`. Then Phase 2B rebalances those auto-labelled publications into `data/matching/splits_auto.json`.

---

## 3. Step-by-Step Execution Guide

### Phase 0–1 — Cleaning Tests & Corpus Eligibility Audit

Verify text/LaTeX normalization and join Milestone 1 + Milestone 2 tables:

```bash
pytest tests/test_clean.py -v
python -m src.m3_matching.data \
  --bibtex data/parsed/bibtex.jsonl \
  --bibtex-entries data/parsed/bibtex_entries.jsonl \
  --references data/metadata/references_table.jsonl \
  --s2-references data/metadata/s2_references.jsonl \
  --out-dir data/matching
```

**Outputs (`data/matching/`):**
- `eligibility.json` — per-publication counts (`m_entries`, `n_references`, `manual_eligible`, `auto_eligible`)
- `preprocessed_bibtex.jsonl` — normalized BibTeX entries with extracted arXiv IDs, DOIs, surnames, and raw text
- `preprocessed_references.jsonl` — normalized arXiv-keyed candidate references enriched with S2 metadata

**Checkpoint:**
- `total_publications_seen`: `~4,844`
- `total_bibtex_entries`: `~223,524` (`duplicate_bibtex_keys_dropped == 0`)
- `total_reference_candidates`: `~86,369` (`duplicate_reference_ids_dropped == 0`)
- `auto_eligible_publications`: `~4,005` (`m >= 1, n >= 1`)

---

### Phase 2A — Create Standard `data/matching/splits.json` (Required for Both Workflows)

> **Prerequisite Check:** Ensure your `src/m3_matching/sampling.py` includes dual-mode support (standard `splits.json` + `--auto-only`). If running without `--auto-only` exits with `"Pass --auto-only..."`, replace `src/m3_matching/sampling.py` with the verified dual-mode version in [Appendix A](#appendix-a--dual-mode-srcm3_matchingsamplingpy).

Generate the frozen standard split manifest from `eligibility.json` (replace `--members 4` with your actual team size `k`):

```bash
python -m src.m3_matching.sampling \
  --eligibility data/matching/eligibility.json \
  --members 3 \
  --manual-per-member 5 \
  --auto-pct 0.15 \
  --seed 42 \
  --out data/matching/splits.json
```

*(To reproduce a larger ~748-publication auto pool on 4,005 eligible papers, use `--auto-pct 0.188`.)*

**Output:** `data/matching/splits.json` containing:
- `partitions`: `{"train": {"manual": [...], "auto": [...]}, "valid": {...}, "test": {...}}`
- `splits`: `{"train": [...], "valid": [...], "test": [...]}`
- `pub_to_split`: `{arxiv_id: "train" | "valid" | "test"}`
- `manifest_sha256` & `summary`

**Checkpoint (for `k = 4`, `--manual-per-member 5`, `--auto-pct 0.15`):**
- `manual_pool`: `20` publications (every paper has `m >= 20` and `n >= 5`, yielding `>= 400` manual entries)
- `valid`: `4` manual + `4` auto; `test`: `4` manual + `4` auto; `train`: `12` manual + all remaining auto
- Zero overlap across `train`, `valid`, and `test` (automatically asserted by `validate_splits`)

---

### Phase 3 — Labelling (Automatic Pool + Optional Manual Pool)

#### Step 3A — Automatic Labelling (Required for Both Workflows)

Labels every BibTeX entry in the `auto` pool defined in `data/matching/splits.json`:

```bash
python -m src.m3_matching.autolabel \
  --splits data/matching/splits.json \
  --bibtex data/matching/preprocessed_bibtex.jsonl \
  --references data/matching/preprocessed_references.jsonl \
  --out data/matching/auto_labels.jsonl
```

**Output:** `data/matching/auto_labels.jsonl`

**Checkpoint:**
- Match rate around `~38–41%` across entries (remaining entries are legitimately non-arXiv books/journals/proceedings)
- High unique candidate coverage (`~88–91%` of crawled arXiv references matched ~1-to-1)

#### Step 3B — Manual Labelling Track (Workflow A Only — Skip if Using Auto-Only)

1. **Generate per-publication annotation sheets + 20% double-labelling audit sheet:**
   ```bash
   python -m src.m3_matching.labeltool sheets \
     --splits data/matching/splits.json \
     --raw-bibtex data/parsed/bibtex.jsonl \
     --raw-references data/metadata/references_table.jsonl \
     --elements data/processed/elements.jsonl \
     --out-dir data/matching/labelling
   ```
2. **Annotate** the CSV files inside `data/matching/labelling/manual_sheets/` and the overlap file `data/matching/labelling/double_label/double_label_20pct.csv`.
3. **Compute inter-annotator agreement (Cohen's $\kappa$):**
   ```bash
   python -m src.m3_matching.labeltool kappa \
     --sheets-dir data/matching/labelling/manual_sheets \
     --double-csv data/matching/labelling/double_label/double_label_20pct.csv
   ```
   *(Target: $\kappa \ge 0.90$; adjudicate all disagreements before merging.)*
4. **Merge manual sheets + auto labels into unified ground truth (`labels.jsonl`):**
   ```bash
   python -m src.m3_matching.labeltool merge \
     --splits data/matching/splits.json \
     --sheets-dir data/matching/labelling/manual_sheets \
     --auto-labels data/matching/auto_labels.jsonl \
     --out data/matching/labels.jsonl
   ```

---

### Phase 2B — Create Rebalanced `data/matching/splits_auto.json` (Workflow B Only)

Skip this step if you are running **Workflow A** (`splits.json`).

For **Workflow B (Auto-Only)**, rebalance all auto-labelled publications into a stratified **70% / 15% / 15%** publication-level split:

```bash
python -m src.m3_matching.sampling \
  --auto-only \
  --from-labels data/matching/auto_labels.jsonl \
  --train-frac 0.70 \
  --valid-frac 0.15 \
  --test-frac 0.15 \
  --seed 42 \
  --out data/matching/splits_auto.json
```

**Checkpoint (on 748 auto publications):**
- `524 train / 112 valid / 112 test` publications
- Balanced match rate (~`38–40%`) across all three splits
- Zero `arxiv_id` overlap across splits and printed 12-char `manifest_sha256`

---

### Phase 4 — Pairwise Feature Engineering (37 Leak-Free Features)

Run the command matching your chosen workflow:

- **Workflow A (Standard `splits.json` + merged `labels.jsonl`):**
  ```bash
  python -m src.m3_matching.features \
    --splits data/matching/splits.json \
    --labels data/matching/labels.jsonl \
    --out data/matching/pair_features.parquet
  ```

- **Workflow B (Auto-Only `splits_auto.json` + `auto_labels.jsonl`):**
  ```bash
  python -m src.m3_matching.features \
    --splits data/matching/splits_auto.json \
    --labels data/matching/auto_labels.jsonl \
    --out data/matching/pair_features.parquet
  ```

> **Tip:** All 37 features are computed strictly within each publication, so you do **not** need to re-run `features.py` if you only re-partition the same set of publications across `splits.json` / `splits_auto.json`.

**Checkpoint:**
- `data/matching/pair_features.parquet` created with `37` numeric pair features across Identifiers, Title, Authors, Year/Venue, and Raw-Text Fallbacks.

---

### Phase 5 — Modeling (`B1/M3/M4`, Stage-1.5 `Has-Match`, Calibrated `M5`)

First run the Phase 5 & 6 unit test suite:

```bash
pytest tests/test_has_match_m5.py -v
```

Then train and evaluate on `train` (GroupKFold OOF) and `valid` (**without** touching `test`):

```bash
# Set SPLITS and LABELS for your chosen workflow:
# Workflow A: SPLITS=data/matching/splits.json       LABELS=data/matching/labels.jsonl
# Workflow B: SPLITS=data/matching/splits_auto.json  LABELS=data/matching/auto_labels.jsonl

python -m src.m3_matching.models \
  --features data/matching/pair_features.parquet \
  --splits data/matching/splits_auto.json \
  --labels data/matching/auto_labels.jsonl \
  --out-dir data/matching/models
```

**Outputs (`data/matching/models/`):**
- `eval_report.json` — dual MRR@5 (`mrr5_all` + `mrr5_matchable`), `has_match` ROC/PR-AUC/F1/ECE, `M5` collision metrics, no-ID ablation, and asymmetric hard-citation stress benchmark
- `valid_scored_pairs.parquet`, `valid_has_match_entries.parquet`, `stress_scored_pairs.parquet`
- `train_oof_predictions.parquet`, `valid_predictions.parquet`
- `suspected_missed_matches_valid.jsonl`, `suspected_missed_matches_train_oof.jsonl`

**Checkpoints:**
1. `train_oof_M5.gated_recall_at_1` matches `valid` closely (~`0.99`, verifying fold-consistent z-score standardization)
2. `m5_collisions_valid.m5_duplicate_assigned_rate == 0.0` and `m5_collision_hit1 >= m4_collision_hit1`
3. `has_match.metrics.valid.H2_lgbm_calibrated.ece_10bin < 0.01`
4. `stress_benchmark.asymmetric_hard_stress` shows monotonic separation `B1 < M3 < M4 <= M5`

---

### Phase 6 — Error Analysis, Human Audit & Poster Plots

#### Phase 6A — Error Cards, Collision Case Studies & 100-Entry Audit Sample

Note the explicit `--bibtex data/parsed/bibtex.jsonl` flag:

```bash
python -m src.m3_matching.inspect_errors \
  --models-dir data/matching/models \
  --bibtex data/parsed/bibtex.jsonl \
  --references data/metadata/references_table.jsonl \
  --out-dir data/matching/analysis
```

**Outputs (`data/matching/analysis/`):**
- `m4_m5_errors_and_collisions.md` & `m4_m5_errors.jsonl` — side-by-side cards for every `M4`/`M5` miss and resolved collision on `valid`
- `suspected_missed_matches_readable.md` — human-readable cards for high-confidence `rule_none` discoveries
- `audit_sample_100.csv` — seeded stratified sample (`50` auto-matched + `50` `rule_none`)

**Optional (~15 min) Human Audit Scoring (Highly Recommended for Auto-Only Poster Credibility):**
Fill `human_verdict` (`match` or `no_match`) and `human_true_ref` (when `match`) in `audit_sample_100.csv`, then run:

```bash
python -m src.m3_matching.inspect_errors \
  --score-audit data/matching/analysis/audit_sample_100.csv
```

Outputs Horvitz-Thompson weighted unbiased estimates of `auto_label_precision`, `rule_none_false_negative_rate`, `estimated_true_match_rate`, `audited_mrr5_all_weighted`, and `audited_mrr5_matchable_weighted`.

#### Phase 6B — Publication-Clustered 95% Bootstrap CIs, Ablations & Poster Plots

```bash
python -m src.m3_matching.evaluate \
  --models-dir data/matching/models \
  --features data/matching/pair_features.parquet \
  --splits data/matching/splits_auto.json \
  --plots-dir data/matching/plots \
  --out-json data/matching/analysis/phase6_evaluation.json
```

*(Pass `--skip-ablations` for a fast re-plot run without retraining drop-one-feature-group models.)*

**Outputs:**
- `data/matching/analysis/phase6_evaluation.json` — 95% publication-clustered bootstrap CIs (`1,000` resamples), paired significance tests, and drop-one-feature-group ablation deltas
- `data/matching/plots/` (300 DPI `.png` + vector `.svg`, Portrait A2 ready):
  1. `01_model_ladder_clean_vs_stress.{png,svg}`
  2. `02_has_match_pr_and_calibration.{png,svg}`
  3. `03_feature_importance_and_ablation.{png,svg}`
  4. `04_collision_and_score_separation.{png,svg}`

---

### Final Test Freeze — One-Time Held-Out Test Evaluation

> **Hard Rule:** Only run this **once** after all features, thresholds (`t*`), and `M5` hyperparameters (`T`, `tau`, `alpha`) are permanently frozen. Never retune after inspecting `test_ladder`.

```bash
python -m src.m3_matching.models \
  --features data/matching/pair_features.parquet \
  --splits data/matching/splits_auto.json \
  --labels data/matching/auto_labels.jsonl \
  --out-dir data/matching/models \
  --eval-test
```

*(Replace `--splits` and `--labels` with `splits.json` / `labels.jsonl` if running Workflow A.)*

**Outputs:**
- Adds `test_ladder` and `m5_collisions_test` to `data/matching/models/eval_report.json`
- Writes `data/matching/models/test_predictions.parquet`

---

### Phase 7 — Hugging Face 6-Table Sharded Parquet Export, Validation & Upload

#### Step 7.1 — Run Phase 7 Unit Tests

```bash
pytest tests/test_publish.py -v
```

#### Step 7.2 — Optional Dry-Run Before Freezing Test Set

If you want to verify all Milestone 1 & 2 tables (`papers`, `references`, `elements`, `hierarchy`, `bibtex`) + `train`/`valid` predictions before running `--eval-test`:

```bash
python -m src.m3_matching.publish \
  --out-dir data/hf_dataset_dryrun \
  --splits data/matching/splits_auto.json \
  --allow-missing-test
```

#### Step 7.3 — Final 6-Table Export, Schema/PK/FK Validation & Dataset Card

```bash
python -m src.m3_matching.publish \
  --out-dir data/hf_dataset \
  --splits data/matching/splits_auto.json \
  --repo-id <your-hf-username>/<your-dataset-repo>
```

**Outputs (`data/hf_dataset/`):**
```text
data/hf_dataset/
├── README.md                 # YAML configs mapping all 6 tables + verified stats & MRR@5
├── dataset_manifest.json     # Export summary + full validation report (not uploaded to HF Hub)
└── data/
    ├── papers/part-00000.parquet
    ├── references/part-00000.parquet
    ├── elements/part-00000.parquet ...
    ├── hierarchy/part-00000.parquet ...
    ├── bibtex/part-00000.parquet
    └── predictions/part-00000.parquet
```

**Checkpoint:** `dataset_manifest.json` prints `"passed": true` with `"errors": []`, and `predictions_by_partition` matches `eval_report.json` on `train`, `valid`, and `test`.

#### Step 7.4 — Verify Local `datasets` Streaming Read Across All 6 Tables

```bash
python -c '
from datasets import load_dataset
for cfg in ["papers", "references", "elements", "hierarchy", "bibtex", "predictions"]:
    ds = load_dataset("data/hf_dataset", cfg, split="train", streaming=True)
    first = next(iter(ds))
    print(f"{cfg:12s} OK -> {list(first.keys())}")
'
```

#### Step 7.5 — Authenticate & Upload to Hugging Face Hub

Authenticate once via CLI (`hf auth login`) or environment variable (`export HF_TOKEN=hf_...`), then upload only the validated `README.md` and `data/**/*.parquet`:

```bash
python -m src.m3_matching.publish \
  --out-dir data/hf_dataset \
  --splits data/matching/splits_auto.json \
  --validate-only \
  --push-to-hub <your-hf-username>/<your-dataset-repo>
```

Verify the live Hugging Face Dataset Viewer opens all 6 configs cleanly and that the repository is **Public** before submitting the URL on Moodle.

---

### Phase 8 — Repository Hygiene & Portrait A2 Poster Checklist

1. **Git Repository Hygiene:** Commit only source files (`src/**/*.py`, `tests/**/*.py`, `requirements.txt`, `README.md`). Confirm `data/` and `.parquet`/`.jsonl` files are excluded via `.gitignore`.
2. **Portrait A2 Poster (`.pptx`, single page):**
   - M1 → M2 → M3 corpus funnel (`4,844` papers → `223,524` BibTeX entries → `86,369` arXiv references)
   - Feature engineering rationale + `Plot 03` (`03_feature_importance_and_ablation.svg`)
   - Clean vs. Hard-Stress model ladder + `Plot 01` (`01_model_ladder_clean_vs_stress.svg`)
   - Stage-1.5 Has-Match calibration & matchability separation + `Plot 02` & `Plot 04`
   - M5 Hungarian collision resolution case study from `m4_m5_errors_and_collisions.md`
   - Final held-out Test dual MRR@5 + 100-entry human audit summary

---

## 4. Troubleshooting Playbook

| Symptom / Error | Root Cause | Fix |
| :--- | :--- | :--- |
| `Pass --auto-only --from-labels ...` when trying to create `splits.json` | Older `sampling.py` only implemented `--auto-only` mode | Replace `src/m3_matching/sampling.py` with [Appendix A](#appendix-a--dual-mode-srcm3_matchingsamplingpy) so both standard `splits.json` and `--auto-only` `splits_auto.json` work |
| `FileNotFoundError: data/processed/bibtex.jsonl` in `inspect_errors.py` | Default CLI path points to `data/processed/`, but Milestone 2 writes `data/parsed/bibtex.jsonl` | Always pass `--bibtex data/parsed/bibtex.jsonl` |
| `KeyError: 'partitions'` in `features.py` when passing `splits_auto.json` | Older `splits_auto.json` omitted the `partitions` key | Re-generate `splits_auto.json` using the updated dual-mode `sampling.py` in Appendix A (which writes `partitions`, `splits`, and `pub_to_split`) |
| `FileNotFoundError: Missing prediction files: test=...` in `publish.py` | `test_predictions.parquet` has not been created yet | Run `models.py --eval-test` once after freezing, or pass `--allow-missing-test` for a dry run |
| `[predictions] ... publications mismatch/outside splits JSON` in `publish.py` | `publish.py` read top-level split keys instead of `pub_to_split` / `partitions` / `splits` | Apply the 10-line `load_split_partitions` compatibility patch in [Appendix B](#appendix-b--publishpy-unified-split-loader-patch) |
| `[references/elements/bibtex] N arxiv_id values not in papers` in `publish.py` | `papers_table.jsonl` is stale relative to M2 outputs | Re-run `python -m src.m1_crawling.export_tables` first; if an intentional M1 edge case remains, inspect sample IDs or pass `--warn-orphan-papers` |
| `401 Unauthorized` on `--push-to-hub` | Missing Hugging Face write token | Run `hf auth login` (with a **Write** token) or `export HF_TOKEN=hf_...` in your shell — never hardcode tokens in code |

---

## 5. Non-Negotiable Guardrails

1. **100% Classical ML:** Zero LLMs, zero pretrained neural embeddings, zero external API calls during labelling or matching.
2. **Zero Publication Leakage:** Every `arxiv_id` belongs to strictly one partition (`train`, `valid`, or `test`); all training meta-features use `GroupKFold(by=arxiv_id)` out-of-fold predictions.
3. **One-Time Test Evaluation:** Never tune features, thresholds, or M5 parameters after running `--eval-test`.
4. **Dual MRR@5 Transparency:** Always report both **All-Labelled MRR@5** (with its **Oracle Ceiling**) and **Matchable-Only MRR@5**, alongside **Has-Match F1 / ECE**.

---

## Appendix A — Dual-Mode `src/m3_matching/sampling.py`

Drop this into `src/m3_matching/sampling.py` so both **standard `splits.json`** (from `eligibility.json`) and **auto-only `splits_auto.json`** (`--auto-only`) work seamlessly with a unified JSON schema:

```python
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
```

---

## Appendix B — `publish.py` Unified Split Loader Patch

Replace `load_split_partitions` in `src/m3_matching/publish.py` with this version so it reads `pub_to_split`, `partitions.{part}.{manual,auto}`, `splits.{part}`, and top-level `{part}` identically:

```python
def load_split_partitions(splits_path: Optional[Path]) -> Dict[str, str]:
    """Load arxiv_id -> partition ('train'|'valid'|'test') from any splits manifest."""
    if splits_path is None or not splits_path.exists():
        return {}
    payload = json.loads(splits_path.read_text(encoding="utf-8"))

    if isinstance(payload.get("pub_to_split"), dict) and payload["pub_to_split"]:
        out: Dict[str, str] = {}
        for pub, part in payload["pub_to_split"].items():
            aid = clean_required_str(pub)
            if aid and part in VALID_PARTITIONS:
                out[aid] = str(part)
        return out

    mapping: Dict[str, str] = {}
    if isinstance(payload.get("partitions"), dict):
        for part in VALID_PARTITIONS:
            pdict = payload["partitions"].get(part, {})
            if isinstance(pdict, dict):
                for src in ("manual", "auto"):
                    for pub in pdict.get(src, []):
                        aid = clean_required_str(pub)
                        if not aid:
                            continue
                        if aid in mapping and mapping[aid] != part:
                            raise ValueError(f"Split leakage in {splits_path}: {aid}")
                        mapping[aid] = part
        if mapping:
            return mapping

    container = payload.get("splits", payload)
    for part in VALID_PARTITIONS:
        pubs = container.get(part)
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
```
