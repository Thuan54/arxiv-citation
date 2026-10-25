# Milestone 1: Crawler

## Purpose

Collect the arXiv publications assigned to our group — paper metadata, the LaTeX source of **every version**, and the reference lists — and profile the result. The corpus is the working input of Milestone 2 (hierarchical parsing) and Milestone 3 (reference matching), so the goal is a crawl that is complete, correctly organised and reproducible.

## Overview

The crawler is a sequential pipeline. For each assigned arXiv ID it discovers the versions, fetches their metadata, downloads and cleans each version's source, and extracts the raw bibliography. One JSON line per paper is appended to `papers.jsonl`; a paper that fails is written to `failures.jsonl` with the step and the error, and the run can be restarted at any time because finished IDs are skipped.

```text
generate IDs → probe versions → fetch metadata → download + extract source → extract references → papers.jsonl
                                                                                   └── on error → failures.jsonl
```

| Step | Module (`src/`) | Endpoint / method | Output |
|---|---|---|---|
| ID generation | `generate_ids.py` | sequential `YYMM.NNNNN` IDs between the assigned first and last ID | `data/inputs/arxiv_ids.txt` |
| Version discovery | `version_probe.py` | `arxiv.org/abs/{id}v{n}` is requested for n = 1, 2, … until HTTP 404 | list such as `["v1", "v2"]` |
| Metadata | `metadata.py` | arXiv API `export.arxiv.org/api/query?id_list=…` (Atom XML) | title, authors, abstract, categories, date, per version |
| Source | `source_fetcher.py` | `arxiv.org/e-print/{id}v{n}`, streamed to a temporary `.tar.gz`, extracted, image files deleted | `data/raw/{id}/{vN}/` |
| References | `reference_extractor.py` | `.bbl` file, or the `thebibliography` environment of a `.tex` file | list of raw `\bibitem` strings |
| Orchestration | `crawler.py` | loops over IDs, logs failures, resumes | `papers.jsonl`, `failures.jsonl` |

**Tools and libraries:** `requests` (HTTP), `xml.etree.ElementTree` (Atom parsing), `tarfile` (extraction), `re` (bibliography parsing), `pandas` / `seaborn` / `matplotlib` (analysis). OAI-PMH and the Kaggle snapshot were not needed: the IDs are assigned, so the arXiv API is enough to resolve them, and a repository-scale harvest is outside this milestone.

**Politeness.** Every request is followed by a 3-second pause (the arXiv API rule), a descriptive `User-Agent` (`HCMUS-IntroDS-Crawler/1.0 (academic project)`) identifies us, requests use timeouts, and one connection is used at a time.

**Record structure** (one line of `papers.jsonl`):

```text
arxiv_id
versions[]
  ├─ version              "v1", "v2", …
  ├─ metadata             title, authors[], abstract, categories[], submission_date
  ├─ latex_source_path    folder of the extracted source, or null when no source could be obtained
  └─ raw_references[]     raw \bibitem strings
```

## Achievements

What we have after this milestone:

- A crawler that resolves each assigned ID into **all of its versions** and stores, per version, the metadata, the extracted LaTeX source folder and the raw references.
- A corpus of **4,087 papers and 6,573 versions** (1.61 versions per paper on average) out of **5,000 assigned IDs**, i.e. a paper-level success rate of **81.7%**. **[TODO: confirm the assigned range — `crawler.py` is set to `2407.02839`–`2407.07838`, but the sample dates shown in the analysis notebook (28 Aug 2024) cannot belong to a July 2024 ID. Use the range of the data the notebook actually analyses.]**
- A resumable run with a failure log, so no ID is lost silently.
- A first quality profile of the corpus (`02_corpus_analysis.ipynb`): crawl completeness, missing fields, versions, references, categories and authors, meant to be executed at book-compilation time.

## Details about our work

- **Every version is crawled.** Versions are discovered by probing `/abs/{id}v{n}` until a 404, and each version is then downloaded individually, so revised papers keep their full history. Papers without obtainable source are **kept with an empty record** (`latex_source_path = null`) instead of being dropped.
- **Figures are removed to save space.** After extraction, files with image extensions (`.png`, `.jpg`, `.jpeg`, `.pdf`, `.eps`, `.gif`, `.bmp`, `.ps`) are deleted; the `figure` environments and `\includegraphics` lines inside the `.tex` files are untouched, so Milestone 2 can still parse the document structure.
- **Safe extraction.** Archive members are checked against path traversal before extraction, and the download goes to a temporary file that is always removed.
- **References are extracted from LaTeX** as raw `\bibitem` strings (`.bbl` first, then `thebibliography` in `.tex`). They are *not* yet resolved to arXiv IDs.
- **Report.** The statistics and figures are produced by a notebook that is meant to be executed when the Jupyter Book is compiled (`execute_notebooks` enabled in `_config.yml`), so they are not static screenshots.

### Corpus statistics (from `02_corpus_analysis.ipynb`)

| Statistic | Value |
|---|---|
| Papers / versions crawled | 4,087 / 6,573 |
| Versions per paper | mean 1.61, median 1, 99th percentile 5, maximum 10 |
| Versions with LaTeX source | 5,608 of 6,573 (85.32%) |
| Versions without source | 965 (14.68%) |
| Versions with source but 0 extracted references | 396 (6.02%) |
| Versions with 0 extracted references (total) | 1,361 (20.71%) |
| Extracted references per paper (latest version) | mean 40.8, median 34, 99th percentile 193, maximum 741 |
| Authors per paper | mean 6.59, median 3, maximum 1,140 |
| Metadata completeness (title, categories, date) | 100% |

The three largest primary categories are all computer science (`cs.CV`, `cs.LG`, `cs.CL`); physics and mathematics categories fill most of the rest of the top 12.

## Challenges

1. **Throughput is bounded by the 3-second rule.** For a paper with *n* versions the current pipeline waits up to `2n + 1` pauses (one per version probe, one for the metadata call, one per successful download), i.e. up to about `6n + 3` seconds. At the observed mean of 1.61 versions this is roughly 13 seconds per paper and **about 14 hours of enforced waiting** for 4,087 papers (estimate), before any transfer time.
2. **Source formats.** The extractor accepts only gzipped tar archives. Single-file sources and PDF-only submissions raise an error, the error is swallowed and the version is stored with `null`. As a result **14.68% of versions have no source** — about five times the ~3% PDF-only rate that arXiv reports — and the log does not say *why*, so real PDF-only versions cannot be told apart from recoverable failures.
3. **Bibliography formats.** The `\bibitem` extractor cannot read `biblatex` output or a stand-alone `.bib`, and reads only the first `.bbl` it finds. This leaves **20.71% of versions without references** (965 without source plus 396 with source).
4. **Version dates are wrong.** `metadata.py` stores the Atom `<published>` element, which the arXiv API defines as the date of version 1 whichever version is requested. Every version of a paper therefore has the same date: the revision span is 0 days for all multi-version papers and `revised_dates` cannot be filled correctly. The date of a specific version is in `<updated>`.
5. **Failed IDs.** 913 of the 5,000 assigned IDs are not in `papers.jsonl`. The notebook breaks them down by failing step and error message from `failures.jsonl`; IDs that do not exist or were removed are not crawler errors. **[TODO: paste the failure breakdown once the notebook has run on the full data.]**
6. **Figure removal by deletion.** Deleting a fixed list of image extensions leaves other non-source files (`.sty`, `.cls`, `.bst`, `.bbl`, …) in the corpus, whereas the project statement keeps only `.tex` and `.bib`. The archive size is not logged, so the size *before* removal cannot be reported yet.

## KPIs

| KPI (required by the project statement) | Result | Status |
|---|---|---|
| Papers scraped successfully / success rate | 4,087 of 5,000 — 81.7% | measured |
| All versions of each paper crawled | 6,573 versions, mean 1.61 per paper | measured (version dates still wrong, see Challenge 4) |
| Source coverage | 85.32% of versions | measured |
| Average references per paper | 40.8 raw `\bibitem` entries (latest version) | measured, not yet resolved |
| Average success rate of reference-metadata scraping (Semantic Scholar) | — | **not implemented in `src/` yet** |
| Average paper size before / after figure removal (bytes) | after: see the on-disk table of the notebook; before: **[TODO]** | partly measured |
| Running time: one paper, average per paper, entry discovery | **[TODO]** — estimate from the pauses alone: ≈ 13 s per paper, ≈ 14 h in total | to measure |
| Memory: peak RAM, average RAM, maximum disk, final output size | **[TODO]** | to measure |

**How the performance numbers will be measured** (auditable, on the Colab CPU-only testbed): wall-clock time per stage with `time.perf_counter()` written to a log; peak RAM with `/usr/bin/time -v` (or `psutil`); average RAM by sampling `psutil.Process().memory_info().rss` every few seconds in a background thread; maximum disk with `du -sb` taken before cleanup and at the end; final output size with `du -sb data/`. For the full system the runtime is the end-to-end wall time and the memory is the total of all processes.

## Next Steps

In order of expected impact on the grade:

1. **Add the Semantic Scholar crawl** (required): for each paper call `graph/v1/paper/arXiv:{id}` with flat `references.*` fields, honour the rate limit, keep *all* references locally and mark those with `externalIds.ArXiv`. This yields the `references` table and the reference-scraping success rate, and covers versions whose LaTeX source is missing.
2. **Fix the version dates:** read `<updated>` per version and `<arxiv:journal_ref>` for `venue`, and re-fetch metadata only (no source download needed).
3. **Switch discovery and download to the recommended endpoints:** resolve versions with one API `id_list` request on the base ID (the returned `vN` suffix is the latest version) and download from `export.arxiv.org/src/{id}vN` instead of probing `arxiv.org` pages, which risks automatic blocking. Batching IDs into the API request leaves the downloads as the only paced step: about 6,573 × 3 s ≈ 5.5 h, roughly 2.6× faster than now (estimate).
4. **Make the source step robust and observable:** handle non-tar sources, retry transient errors, and store a `source_status` / reason per version so that PDF-only, withdrawn and failed downloads are distinguishable; then re-run the failed IDs.
5. **Extract only `.tex` and `.bib`** from the archive (never writing other files) and record archive and extracted sizes; fix the 429/503 branch of the version probe so that a rate-limited version is retried instead of skipped.
6. **Reproducibility:** add `README.md` and `requirements.txt` to the code repository and make sure the notebook's helper (`utils/build_summary.py`) is in it.
7. Measure running time and memory on Colab and fill in the KPI table.

## Conclusion

The crawler already produces a usable corpus of 4,087 papers and 6,573 versions with metadata, source folders and raw references, and a notebook that profiles its quality. The analysis also shows what is still missing before the corpus is a sound base for Milestones 2 and 3: correct per-version dates, the Semantic Scholar reference metadata and the venue, a more complete source download (14.68% of versions are empty), and measured performance figures. Closing these gaps is the work of the next iteration; none of them needs the whole crawl to be repeated from scratch, except the source download for the versions that failed.
