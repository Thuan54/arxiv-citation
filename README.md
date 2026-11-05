# Arxiv cralwer improve from `main` branch
Handling more fields like `venue` and `raw_references`... more robust
## How to run it 

```bash
# create .venv
python -m venv .venv

# use .venv for Linux
soucre .venv/bin/activate

# install necessary pacakge
pip install -r requirements.txt

```

**You must go to `src/generate_ids.py` to change arxiv paper indexs rage to yours**
```python
# src/generate_ids.py
if __name__ == '__main__':
    generate_arxiv_ids(
        start_id=YOUR_START_ID,
        start_month=YOUR_START_MONTH,
        end_id=YOUR_END_MONTH,
        end_month='2024-08',
        output_path=IDS_FILE
    )
```

*If your **start_id > end_id** because **start_month $\ne$ end_month** then you must **run twice from [start_month, last_paper_of_month]** and **[first_paper_of_month, end_id]***

*You must also get **a Semnatic Scholar** for reference crawling and put it into `.env` file*

To crawl it, run this
```bash
# generate ids
python src/generate_ids.py
# crawl
python src/crawler.py
```

## Milestone 2: parse the LaTeX hierarchy

After the crawler has produced `data/metadata/papers.jsonl` and downloaded
sources under `data/raw/`, build the processed tables with:

```bash
python src/parse_corpus.py
```

The parser follows `\input` and `\include` from the `.tex` file containing
`\documentclass`, ignores commented-out includes and unused TeX files, and
writes the Milestone 2 tables to `data/processed/`:

- `elements.jsonl`: deduplicated document, heading, paragraph, sentence,
  formula, figure/table, and list nodes across versions.
- `hierarchy.jsonl`: per-version child-to-parent edges.
- `versions.jsonl`: parsing coverage, including versions with no TeX source,
  plus the corresponding version's existing `raw_references` from
  `papers.jsonl`.

The parser is intentionally dependency-free. Review the parsed output for
papers with custom LaTeX macros or unusual structures before using it as a
fully normalized corpus. It excludes bibliography sections and bibliography
commands from hierarchy content; references are taken from the crawler's
per-version `raw_references` rather than re-extracted from TeX or `.bib` files.