# Project Summary: Bibliographic Entity Resolution for arXiv Publications

## 1. Project Overview

### Objective
The goal of this project is to resolve bibliographic citations extracted from LaTeX `.bbl` files (Milestone 2) against candidate reference records crawled from Semantic Scholar (Milestone 3). For each citing publication (`arxiv_id`), we construct an $m \times n$ candidate matrix pairing its $m$ BibTeX `citation_key` entries with $n$ Semantic Scholar arXiv candidates (`ref_arxiv_id`), enabling supervised ranking models evaluated by $\text{MRR@5}$.

### Target Data Schema (`labels.jsonl`)
Every labeled record adheres to the schema defined in `labels.md`:

| Field | Type | Description |
| :--- | :--- | :--- |
| **`arxiv_id`** | `string` (non-null) | Identifier of the citing arXiv publication used to group candidate matrices internally. |
| **`citation_key`** | `string` (non-null) | Normalized BibTeX entry key extracted in Milestone 2. |
| **`true_ref_arxiv_id`** | `string` (nullable) | Ground-truth matched arXiv ID from Semantic Scholar, or `null` when no arXiv match exists. |
| **`source`** | `string` | Distinguishes verified manual labels (`"manual"`) from heuristic matches (`"auto"`). |
| **`annotator`** | `string` (nullable) | Identifier of the team member who annotated the record (e.g., `"member_1"`), or `null` for automated labels. |
| **`confidence`** | `float` | Certainty score (`1.0` for verified manual annotations; heuristic similarity score when generated automatically). |
| **`unsure`** | `boolean` | Quality-audit flag indicating whether an annotation is uncertain. |

---

## 2. What We Have Done

### A. Diagnosed & Resolved File-Ingestion Issues
* **Root-Cause Isolation:** Identified that generic `.csv` attachments (`bib_tmp_*.csv` / `ref_tmp_*.csv`) were arriving with $0$-byte payloads due to a chat interface attachment-handling bug rather than any content policy restriction.
* **Workflow Optimization:** Established two reliable transfer methods—pasting raw CSV text directly into prompts and naming files with explicit arXiv ID prefixes (`<arxiv_id>_bib.csv` and `<arxiv_id>_ref.csv`)—which allowed us to process batches of $3$ to $5$ publications cleanly in a single turn.

### B. Audited Milestone 2 BibTeX Deduplication & Corpus Statistics
We audited two iterations of `bibtex_stats.json` across the $5,000$-publication corpus ($8,086$ total versions; $4,573$ publications with bibliographies and $427$ without):
* **Pipeline Improvements Verified:** Between the first and second runs, parse errors dropped from $142$ to $0$, wall-clock runtime improved by $2.13\times$ ($447.7\text{ s} \rightarrow 210.5\text{ s}$), and peak memory decreased from $49.8\text{ MB}$ to $36.3\text{ MB}$.
* **Deduplication Accounting Confirmed:** In the final run, $379,912$ raw entries were reduced by $38.35\%$ (removing $3,623$ within-version duplicates and $142,061$ cross-version duplicates) to **$234,228$ unique `(arxiv_id, citation_key)` entries**, averaging $51.22$ references per eligible publication.
* **Domain Data Characteristics Identified:**
  * **Missing Titles ($22.84\%$):** Only $77.16\%$ of deduplicated entries include a `title` (whereas `authors` cover $96.87\%$ and `year` covers $98.83\%$), because APS and astronomy journal styles frequently omit article titles.
  * **Generic Styling ($72.51\%$):** Most raw entries ($275,474 / 379,912$) come from unstructured `\bibitem` blocks rather than `\bibinfo` or BibTeX macros, requiring text normalization before feature extraction.

### C. Built Local Python Matching Tooling & Disambiguation Protocols
* **Reusable Local Scripts:** Created Python scripts supporting Unicode accent stripping, title/author normalization, interactive borderline verification, and automated heuristic labeling.
* **Collision Disambiguation:** Established strict verification rules using author last-name overlap, publication year alignment, and DOI/URL matching to avoid false positives on shared generic titles (e.g., distinguishing textbooks like *Gravitation* or *Convex Optimization* from research papers with identical titles) and to handle within-version duplicate keys (such as main-text and supplement keys).

### D. Completed the Manual Ground-Truth Pool
* Across our labeling sessions, we completed **$27$ publications** ($26$ with $Q \ge 20$ entries) totaling **$1,438$ labeled entries** ($767$ positive arXiv matches and $671$ negative entries), surpassing the manual quota requirements ($5 \times k$ publications with $Q \ge 20$ entries and $100 \times k$ total entries).

---

## 3. What We Are Planning to Do (Next Steps)

1. **Consolidate & Validate `labels.jsonl`:**
   * Merge all per-publication `labels_<arxiv_id>.json` files into a unified `labels.jsonl` dataset.
   * Verify schema compliance, ensure no duplicate `(arxiv_id, citation_key)` pairs exist, and confirm that none of the $286$ ambiguous BibTeX keys leak into the `valid` or `test` splits.
2. **Generate the Automatic Labeling Pool (`source: "auto"`):**
   * Run the heuristic matching script across the designated automatic subset of remaining publications, recording heuristic similarity scores in `confidence` and setting `annotator: null`.
3. **Create Group-Aware Data Splits (`train` / `valid` / `test`):**
   * Partition the dataset strictly by `arxiv_id` so all candidate pairs belonging to a citing paper remain together in the same split, keeping verified manual annotations in `valid` and `test`.
4. **Engineer Pairwise Ranking Features:**
   * Extract multi-signal features for each `(citation_key, candidate_ref_arxiv_id)` pair:
     * **Identifier & URL Signals:** Exact arXiv ID or DOI substring match in `url`/`doi`.
     * **Title Similarity:** Normalized Levenshtein/sequence ratio, token Jaccard, and TF-IDF cosine similarity (plus a `title_is_missing` indicator flag to handle the $22.84\%$ of entries lacking titles).
     * **Author & Temporal Signals:** Author last-name Jaccard/overlap count, first-author match, and absolute publication year difference ($|\Delta \text{year}|$).
5. **Train & Evaluate the Ranking Model:**
   * Train a learning-to-rank or pairwise classifier (such as LightGBM) grouped by `(arxiv_id, citation_key)` and evaluate performance on `valid` and `test` using **$\text{MRR@5}$**.