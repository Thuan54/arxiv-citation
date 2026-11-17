# Milestone 3 - Reference Matching: Execution Plan

Classical ML only: no LLMs, no pretrained or deep-learning models. Built on your Milestone 1 and 2 outputs. Written for a group of k members.

## 1. Summary

**The task.** For every BibTeX entry of a publication, rank the arXiv-keyed Semantic Scholar references of the same publication and output the top 5. Score: MRR@5 on the test publications. Everything happens inside one publication, so each entry has a short candidate list (about 18 references on average, if your roughly 90,000 reference rows are spread over about 5,000 papers).

**Recommended approach: learning-to-rank over a small candidate list.**

1. One shared cleaning module for entries and references.
2. Two label sources: a rule-based automatic matcher and careful manual labelling.
3. About 40 hand-built features per (entry, reference) pair: identifiers, title, authors, year, venue, plus raw-text fallbacks for entries the parser could not read well.
4. A gradient-boosted ranker (LightGBM LambdaRank), compared against simpler baselines.
5. A second, collective stage: a reference is normally cited by one entry, so the final ranking solves an assignment problem over the whole publication.
6. Evaluation on the official test set and by leave-publication-out cross-validation, because 2 test publications per member is a very noisy estimate.

**What a 10-paper sample of your data says** (indicative only; Phase 0 repeats it on the full corpus):

| Observation | Sample value | Consequence |
| --- | --- | --- |
| Papers usable for matching | 8 of 10 have at least 3 arXiv-keyed references; 1 has no BibTeX entries; 1 has 107 entries but no arXiv-keyed reference | Eligibility filter before any sampling |
| BibTeX entries per paper | 12 to 107, median 41 | Manual pool: prefer papers with 20-45 entries so every entry can be labelled |
| arXiv-keyed candidates per paper | 3 to 39, median 12 | Tiny candidate lists: no blocking step needed |
| Entries with an exact title partner | 122 of 399 (31%); 7 more at fuzzy similarity 0.9 or higher | About 7 entries in 10 are 'no reference' rows |
| Entries with a partner versus candidates | 129 partners for 130 candidates | Nearly every candidate seems to have an entry, which favours an assignment stage (verify with unique counts) |
| Order of the Semantic Scholar list | Spearman with arXiv id is -1.00 and with year -0.90 to -0.98 in all 7 papers tested | The list is sorted newest-first, NOT in bibliography order |

**Consequences.**

- Most rows will be 'no reference'. If those rows count in the MRR denominator, MRR@5 is capped near the share of entries that do have a reference. Report MRR on all labelled entries and on matchable entries only (Question 1 in section 3).
- Exact identifiers and exact titles solve the easy majority. The ML earns its keep on the residue: parse failures, retitled preprints, subtitles, missing years. Robust cleaning and raw-text fallback features matter more than model tuning.
- Do not build a 'position in the bibliography' feature from the Semantic Scholar order. The hypothesis was worth testing; the negative result is good poster material.
- Choose manual publications with few entries (20-45) so that every entry can be labelled. This removes the ambiguity of unlabelled rows in the predictions table.

## 2. Requirements checklist (from matching.md)

| Requirement | Quantity for k members | Planned in |
| --- | --- | --- |
| Manually labelled publications | at least 5k; each with at least 20 labelled entries; at least 100k labelled entries in total | Phase 3C |
| Automatically labelled publications | at least 10% of the publications outside the manual pool, and at least 2k; the matcher labels every entry | Phase 3B (plan 12-15% for safety) |
| Test set | k manual + k automatic publications | Phase 2 |
| Validation set | k manual + k automatic publications | Phase 2 |
| Training set | all other labelled publications (3k manual + the remaining automatic ones) | Phase 2 |
| Candidate matrix | m x n per publication, completed row-wise: exactly one yes, or an all-no row | Phase 3 |
| Model output | top-5 ranked references per labelled entry | Phase 5 |
| Metric | MRR@5; a correct reference outside the top 5 scores 0 | Phase 6 |
| predictions table | arxiv\_id, partition, citation\_key, groundtruth\_arxiv\_id, cand\_1 to cand\_5; rows only for train/valid/test publications | Phase 7 |
| Hugging Face dataset | 6 sharded Parquet tables, flat primitive columns, dataset card with one config per table, no loading script | Phase 7 |
| Repository | src with .py or .ipynb, requirements.txt, README covering environment, execution, feature and label documentation | Phase 8 |
| Poster | one pptx slide, portrait A2, presented at the session | Phase 8 |
| Grading note | methods using external knowledge (LLMs, pretrained models) are judged separately | This plan stays fully classical |

Example, k = 2: 10 manual publications (200+ labelled entries), test = 2 manual + 2 automatic, validation = 2 manual + 2 automatic, training = 6 manual + all remaining automatic ones.

## 3. Questions to settle with the instructors in week 1

1. **Entries with no true reference:** do they count in |Q| (they score 0 whatever the model does) or are they excluded? This sets the MRR ceiling and how test publications should be chosen. The plan reports both versions.
2. **Unlabelled entries inside a manual publication:** does predictions need a row for them? A null groundtruth is ambiguous between 'unlabelled' and 'no reference'. The plan sidesteps this by labelling every entry of manual publications.
3. **Auto-labelled validation/test publications:** may they be audited and corrected by hand? The plan assumes yes and reports it.
4. **Allowed text statistics:** is TF-IDF/LSA fitted only on your own corpus acceptable under 'no external knowledge'? The plan assumes yes and uses no pretrained resource of any kind.
5. **element\_id format:** the spec example uses 2310-12345#root1 (dot replaced by a hyphen) while the current parser keeps the dot. Check which one the published dataset must use.
6. **Group size k and the deadline,** to fix the schedule in section 6.

## 4. Pipeline and repository layout

```
M1: papers + s2_references --> references table (arXiv-keyed)
M2: bibtex table (entries unified over versions)
        |
Phase 0 join + eligibility --> Phase 1 cleaning --> Phase 2 pools + frozen splits
        |
Phase 3 labels (manual + automatic) --> Phase 4 pair features --> Phase 5 ranker + collective stage
        |
Phase 6 evaluation --> Phase 7 predictions + 6 Parquet tables + dataset card --> Hugging Face
Phase 8 README + poster
```

```
src/03_matching/
  data.py         load and join bibtex + references, eligibility statistics
  clean.py        normalisation of titles, authors, years, ids, venues
  sampling.py     publication pools and the frozen, seeded split manifest
  autolabel.py    rule-based matcher -> labels.jsonl (source = auto)
  labeltool.py    manual labelling sheets, label-file validation, agreement
  features.py     pair features and stage-2 collective features
  models.py       baselines, LightGBM ranker, assignment post-processing
  evaluate.py     MRR@5, Hit@k, bootstrap, ablations, error taxonomy
  predictions.py  builds the predictions table (out-of-fold for train rows)
  publish.py      Parquet schemas, validators, shards, dataset card, upload
  poster/         figure scripts and the python-pptx builder
tests/            unit tests per module
```

Allowed toolbox: regex, rapidfuzz/difflib, scikit-learn (TF-IDF, SVD, logistic regression, random forest), LightGBM or XGBoost, scipy (Hungarian assignment), pandas/pyarrow, matplotlib, SHAP. Not used: pretrained embeddings, neural networks, LLM assistance for labelling or code that labels, extra bibliographic APIs (Crossref, OpenAlex).

## 5. Plan by phase

### Phase 0 - Data readiness (days 1-2)

**Goal:** a trustworthy join of the bibtex table and the references table, with statistics.

- Re-run Milestone 2 reference extraction on the full corpus; list publications with 0 entries (the sample had one) and fix the cause before sampling, since each such paper wastes a pool slot.
- Re-run the Semantic Scholar retry for papers whose reference list is null or empty.
- Per publication compute m (entries), n (references with an arXiv id, de-duplicated by arXiv id, version suffix removed), share of entries with title, authors, year, DOI, arXiv id in url/eprint, bibliography style.
- Eligibility rule: m of at least 20 and n of at least 5 for manual candidates; m of at least 1 and n of at least 1 for the automatic pool.
- **Output:** eligibility table, histograms of m, n and entries-per-candidate (poster material).
- **Done when:** (arxiv\_id, citation\_key) is unique, candidate lists are unique by arXiv id, and the eligible count is known.

### Phase 1 - Cleaning module (days 2-4)

**Goal:** one normalisation used by labelling, features and tests.

- Titles: lowercase, accent folding, LaTeX and math stripped, punctuation removed, stop-word-free token list plus a stop-word-kept string for character n-grams.
- Authors: surname key (particles handled), initials, et-al flag, order kept; natural order 'Given Family' as produced by the Milestone 2 parser.
- Identifiers: DOI lower-cased without URL prefix; arXiv id without version, old-style ids supported; search in url, eprint, journal text and the raw bibitem.
- Years: integer or missing; venues: lowercase, years and ordinals removed.
- Unit tests built from real hard cases (accents, math in titles, subtitles, corporate authors, missing titles).
- **Done when:** the tests pass and a notebook shows before/after examples.

### Phase 2 - Choose and freeze the publications (day 4-5)

- **Manual pool (5k publications plus spares):** m from 20 to 45, n at least 10, mixed bibliography styles and arXiv categories, no obvious parse failures. Stratify so the pool is not only the easy papers.
- **Automatic pool:** a seeded random 12-15% of the remaining eligible publications, stratified by category and size, at least 2k.
- **Pick validation and test publications now** (k manual + k automatic each, with n at least 10 so they contain positives) and write them to `splits.json`. Record the seed and a hash of the file in the README.
- **Rule:** test publications are used exactly once, at the very end. No feature or threshold is tuned on them.

### Phase 3 - Labelling

**3A. Label format.** One line per entry in `labels.jsonl` (see Appendix A): arxiv\_id, citation\_key, true\_ref\_arxiv\_id or null, source (manual or auto), annotator, confidence, rule. The candidate matrix is derived from it: a one-hot row for a matched entry, an all-no row otherwise. Many entries may map to one reference.

**3B. Automatic matcher (regex and string similarity).** A cascade, first rule that fires wins:

1. arXiv id found in the entry (eprint, url, raw text) equals a candidate: confidence 1.0.
2. DOI equals the DOI stored in the candidate's external ids: 1.0.
3. Normalised title equal (at least 3 words) and first-author surname compatible or year within 2: 0.95.
4. Title token-set similarity at least 0.92, same first-author surname, year within 1: 0.85.
5. Otherwise: no reference (all-no row). Tune the thresholds on the manual publications for precision of at least 0.98 and report the recall. Guard against circular learning: (i) the matcher uses identifiers and strict rules while the model uses many extra soft features; (ii) train-on-auto versus train-on-manual, both evaluated on the manual test publication, is the unbiased estimate; (iii) hand-audit 100 random auto labels to report their precision; (iv) audit every label of the automatic validation and test publications.

**3C. Manual labelling.**

- Generate one sheet per publication: parsed fields, raw bibitem, the citing sentence from the Milestone 2 hierarchy, the 8 most similar candidates and a free search over all candidates, plus a 'none' option.
- Each member labels their own 5 publications, entirely (every entry). Estimate 45-75 minutes per publication, about 5-6 hours per member (assumption, time one publication first).
- Double-label 20% of entries by a second member; report Cohen's kappa; adjudicate disagreements and refine the guidelines (Appendix A).
- Suggestions can anchor annotators: always allow the free search and keep the double labelling.
- No LLM assistance in any labelling step.

### Phase 4 - Feature engineering (days 5-10)

One row per (entry, candidate) pair. Every feature needs a stated idea and evidence on the poster: positive-versus-negative distributions, single-feature AUC or MRR, missing-value rate, and the ablation drop.

| Group | Features | Idea |
| --- | --- | --- |
| Identifiers | doi\_eq, arxiv\_eq, arxiv\_id\_in\_raw\_text, url\_contains\_id | Decisive when present; show how often they are present |
| Title | exact after cleaning, token Jaccard, TF-IDF cosine (words, and character 3-5 grams), token-set ratio, partial ratio, longest-common-subsequence ratio, length ratio, subtitle/prefix containment, shared numbers, BM25 rank | Different views of the same title; character n-grams survive typos and math |
| Raw-text fallback | candidate title found inside the raw bibitem (partial ratio), candidate first surname in raw text, candidate year in raw text | Works when the parser lost the title or authors (title-less astronomy style) |
| Authors | first-surname equality and Jaro-Winkler, surname-set Jaccard, author-count ratio, initials compatibility, et-al flag | First author is the strongest, truncated lists need tolerant overlap |
| Year | absolute difference, equal, missing flags, arXiv id year versus entry year | Preprint and published years differ by 0-2 |
| Venue and type | venue similarity, entry type, entry-is-arXiv, candidate venue is arXiv | Separates preprint entries from journal entries |
| Meta | parse completeness (number of non-empty fields), bibliography style, n candidates, candidate year | Lets the model trust a feature less when its source field is weak |
| Collective (stage 2) | row rank, row margin, row softmax, column rank, column margin, column softmax, mutual best, assigned by Hungarian | Uses the near one-to-one structure between entries and references |
| Optional context | number of citations of the entry in the text, section of first citation | From the Milestone 2 hierarchy; expect a small gain, keep only if the ablation shows it |

No identifier-like features (raw tokens, publication id) so the model cannot memorise papers.

### Phase 5 - Modelling (days 10-17)

- **Framing:** ranking. Group = one entry; rows = its candidates; label 1 for the true reference. Entries with an all-no row have no positive: they are excluded from the ranker and used for a separate has-match classifier.
- **Model ladder:** (B0) the automatic rule matcher; (B1) TF-IDF cosine on title+authors+year; (B2) a weighted sum of the best features tuned by cross-validation; (M3) logistic regression; (M4) LightGBM LambdaRank, NDCG at 5 as the early-stopping metric, monotone constraints on similarity features; (M5) M4 plus the collective stage.
- **Collective stage:** build the m x n score matrix of M4, then (a) add row and column softmax and rank features and train a second small model on out-of-fold scores; (b) solve a Hungarian assignment with dummy 'none' columns whose cost is the threshold tau; (c) final ranking puts the assigned reference first, then the rest by competition-adjusted score. Tune tau and the adjustment weight on out-of-fold predictions.
- **Protocol:** GroupKFold by publication over the training and validation publications for model selection; early stopping on validation; official test used once. Out-of-fold predictions fill the train rows of the predictions table so they are honest. Fixed seeds, small random search over LightGBM parameters (leaves, learning rate, minimum child samples, regularisation).
- **Always output 5 candidates** when at least 5 exist: extra candidates cannot lower MRR@5.
- **Interpretability:** SHAP summary for the poster.

### Phase 6 - Evaluation and analysis (days 15-21)

- **Metrics:** MRR@5 as defined (all labelled entries); MRR@5 on matchable entries; Hit@1 and Hit@5; the oracle ceiling (share of entries whose true reference exists). Report manual-test, auto-test and combined separately.
- **Rigour:** bootstrap confidence intervals over entries and over publications; leave-publication-out cross-validation next to the 2-publication test.
- **Analyses:** feature-group ablations; model ladder table; has-match classifier quality; error taxonomy (parse failure, retitled preprint, subtitle, wrong year, duplicate reference, math title) with counts and examples; performance by bibliography style; learning curve (number of labelled publications); train-on-auto versus train-on-manual; label-noise experiment (add noise to auto labels); runtime.
- **Sanity checks:** no publication appears in two partitions; no ids as features; test publications untouched until the final run.

### Phase 7 - Predictions table and dataset (days 20-26)

- **predictions:** one row per entry of each train/valid/test publication. arxiv\_id, citation\_key (strings, non-null); partition (dictionary string: train, valid, test); groundtruth\_arxiv\_id (nullable string); cand\_1 to cand\_5 (nullable strings). Every candidate must be the arXiv id of a reference of the same publication, with no duplicates inside a row.
- **Other tables:** papers, references, bibtex, elements, hierarchy from Milestones 1 and 2. Explicit pyarrow schema for each: flat columns only, list of strings for authors and revised dates, dictionary strings for kind, entry\_type and partition, nullable int for year, no nested objects or JSON blobs.
- **Validators (run in CI and before upload):** schema equals the declared schema; (arxiv\_id, citation\_key) unique in bibtex; every predictions key exists in bibtex; every hierarchy element\_id and parent\_id exists in elements; every candidate exists in references for that publication; row counts per table logged.
- **Publishing:** shards of a few hundred MB, data/\<table>/\*.parquet, a README dataset card with one config per table, upload with huggingface\_hub, then check `load_dataset(repo, config, streaming=True)` and the Hub viewer for every table. No loading script.

### Phase 8 - Documentation and poster (days 22-32)

- **README:** environment and Python version, exact commands from raw data to predictions, documentation of every feature (idea, formula, evidence) and of the label format and guidelines, seeds and the split manifest.
- **Poster:** one slide, portrait A2 (420 x 594 mm, 16.54 x 23.39 in) built with python-pptx from scripted figures so it can be regenerated. Layout: title band; pipeline diagram; data findings (7 in 10 entries have no reference, S2 order is recency); label statistics with kappa and auto-label precision; feature highlights with evidence; model ladder and MRR table with intervals; error taxonomy; dataset schema and statistics. Choose the story that makes the group distinct: robust parsing of many bibliography styles, cross-version deduplication, collective assignment, honest evaluation.
- **Print:** allow lead time for printing and rehearse the 3-minute pitch.

## 6. Schedule and roles

| Week | Focus |
| --- | --- |
| 1 | Questions to instructors; Phase 0 and 1; label guidelines; pick pools (Phase 2) |
| 2 | Auto matcher (3B); every member starts manual labelling; first features and baselines |
| 3 | Finish manual labels, double labelling and kappa; full features; LightGBM ranker |
| 4 | Collective stage, tuning, ablations, error analysis; freeze the model; run the test once |
| 5 | Predictions table, Parquet assembly, validators, upload; README; poster draft |
| 6 | Poster polish and printing; rehearsal; buffer for slips |

Roles (merge if k is small): data and dataset owner (Phases 0, 7); labelling and auto matcher (Phase 3); features and model (Phases 4, 5); analysis and poster (Phases 6, 8). Everybody labels their own 5 publications.

## 7. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Many papers have no usable entries or references | Phase 0 audit; fix Milestone 1 and 2 gaps before sampling |
| Small test set makes MRR noisy | Leave-publication-out CV, bootstrap intervals, report both |
| No-reference rows dominate and cap MRR | Settle Question 1; report ceiling and matchable-only MRR |
| Circular labels: model imitates the auto matcher | Train-on-auto versus train-on-manual; independent soft features; audits |
| Annotator disagreement or anchoring on suggestions | Guidelines, free search, 20% double labelling, kappa |
| Test leakage | Frozen manifest, test used once, no id features |
| Preprint versus published differences (title, year) | Year tolerance, subtitle features, raw-text fallback |
| Schedule slip around printing | Poster draft in week 5, printing lead time in week 6 |

## 8. Levers for the best result, in expected order of impact (hypotheses to confirm by ablation)

1. High-quality Milestone 2 parsing: titles and authors present for as many entries as possible.
2. Identifier shortcuts (arXiv id, DOI) and exact cleaned titles.
3. Raw-text fallback features for entries with failed parsing.
4. The collective assignment stage (each reference has about one entry).
5. Character n-gram title similarity and tolerant author features.
6. LambdaRank instead of pointwise classification.
7. More clean automatic labels for training.
8. Hyperparameter tuning (smallest effect).

## Appendix A - Label file and guidelines

`labels.jsonl`, one JSON object per entry: arxiv\_id, citation\_key, true\_ref\_arxiv\_id (string or null), source (manual or auto), annotator (name or rule id), confidence (0 to 1), unsure (boolean), note.

Guidelines:

1. A match means the same work, whatever the version or venue: a preprint and its published version are the same reference.
2. A different edition, translation, or a different paper by the same authors is not a match.
3. If the true reference's arXiv id is not in the candidate list, the label is null.
4. When unsure, mark unsure and decide in review; never guess silently.
5. Do not use an LLM or any external service to decide a label.

## Appendix B - Definition of done

- Manual labels: 5 publications per member, every entry labelled, at least 100 entries per member, kappa reported.
- Automatic labels: at least 10% of the publications outside the manual pool and at least 2k, with audited precision.
- Frozen split manifest with the seed; test used once.
- Feature documentation with evidence for every feature.
- predictions table passes all validators; six Parquet tables load with streaming.
- Poster pptx (portrait A2), README, repository and dataset link ready for Moodle.
