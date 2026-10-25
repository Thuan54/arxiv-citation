# arXiv Full-Text Corpus Engineering & Reference Matching

:::{note}
**Document Version:** 1.0 (Milestone 1 Release)  
**Course:** Introduction to Data Science — Faculty of Information Technology, University of Science, VNU-HCM (HCMUS)
:::

## Overview & Project Objectives

This Jupyter Book documents our end-to-end data engineering and machine learning pipeline built for the **Introduction to Data Science** course project. 

While traditional bibliometric studies on **arXiv** rely solely on abstract-level metadata or hard-to-parse PDF files, this project constructs a structured dataset directly from **full-text LaTeX source archives** across all revision versions of each publication. 

Our primary objective is to take raw, unstructured academic preprints from discovery through hierarchical document parsing to an automated **Reference Matching Pipeline** evaluated using **Mean Reciprocal Rank at 5 ($\text{MRR}@5$)**.

---

## Project Roadmap & Milestones

The project is structured into three cumulative milestones:

- **Milestone 1 — Data Crawling & Corpus Analysis:**  
  Discover papers within our assigned arXiv ID range, harvest paper and reference metadata via the arXiv API / OAI-PMH and Semantic Scholar, download every revision's (`v1` to `vN`) full-text LaTeX source bundle, and perform an exploratory statistical audit of the raw corpus.
- **Milestone 2 — Hierarchical Parsing & Standardization:**  
  Parse raw `.tex` and `.bbl` source files into a unified hierarchical document tree (`elements` and `hierarchy`), normalize mathematical and textual notation, convert raw `\bibitem` strings into standardized BibTeX records, and deduplicate content across paper revisions.
- **Milestone 3 — Reference Matching Pipeline:**  
  Design and evaluate a machine learning ranking pipeline—covering candidate generation, feature engineering, labeling, and modeling—to link unlabelled in-text/BibTeX citations to their ground-truth arXiv identifiers, scored via $\text{MRR}@5$.

---

## About the Dataset

Our group's assigned partition covers **15,000 arXiv publications**( for a group of three with **5000 publications per student**). Unlike static single-snapshot datasets, our corpus captures the complete longitudinal revision history of every assigned paper:

* **Multi-Version Coverage:** Every available revision (`v1`, `v2`, ..., `vN`) is tracked independently to capture metadata updates, structural edits, and bibliography growth over time.
* **Full-Text LaTeX Sources:** Raw `.tar.gz` source trees are extracted into versioned directories (`latex_source_path`) alongside inline `\bibitem` reference strings (`raw_references`).
* **Reproducible Analytics:** All summary tables and visualizations in this book are dynamically compiled from our processed dataset during the Jupyter Book build process.

A comprehensive breakdown of the crawling architecture, rate-limit handling, and corpus quality audit is presented in **Milestone 1 — Data Crawling**.

---

## Team Members & Responsibilities

This project is developed by our team at the **Faculty of Information Technology, University of Science, VNU-HCM**:

[To be fill by USER later on]
| Student ID | Full Name | Primary Responsibilities (Milestone 1) |
| :---: | :--- | :--- |
| `xxyyyzzz` | **DAH** | Entry discovery, arXiv/Semantic Scholar crawler implementation, LaTeX source extraction |
| `xxyyyzzz` | **HMT** | Corpus quality audit, statistical distribution analysis, Jupyter Book architecture & narrative |
| `xxyyyzzz` | **LTP** | Data schema standardization, reference extraction validation, pipeline integration |

---

## Book Navigation
```{tableofcontents}
Introduction
Milestone 1: Crawler
Milestone 2: Parsing
Milestone 3: Matching
```

