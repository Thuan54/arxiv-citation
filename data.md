
```
root/
├── data/
│   ├── metadata/
│   │   ├── papers.jsonl      # Crawler checkpoint with per-version source data
│   │   ├── papers_table.jsonl
│   │   └── references_table.jsonl
│   └── raw/
│       ├── 2310.06825/
│       │   ├── v1/
│       │   │   ├── main.tex
│       │   │   └── figures/
│       │   └── v2/
│       └── 2310.06826/
│           └── v1/
└── src/
    └── crawler.py
```


```json
{
  "arxiv_id": "2310.06825",
  "versions": [
    {
      "version": "v1",
      "metadata": {
        "title": "Paper Title",
        "authors": ["Author One", "Author Two"],
        "abstract": "Abstract text...",
        "categories": ["cs.LG", "cs.AI"],
        "submission_date": "2023-10-10"
      },
      "latex_source_path": "data/raw/2310.06825/v1/",
      "raw_references": [
        "{\"citedPaper\":{\"paperId\":\"...\",\"externalIds\":{\"ArXiv\":\"...\"},\"title\":\"Cited paper\"}}"
      ]
    },
    {
      "version": "v2",
      "metadata": { ... },
      "latex_source_path": null,
      "raw_references": []
    }
  ]
}
```

After crawling, the crawler writes two table-shaped JSONL files:

- `papers_table.jsonl`: one row per arXiv paper, combining title/authors,
  initial submission date, unique revision dates from later versions, and
  journal/conference venue from arXiv metadata.
- `references_table.jsonl`: one row per unique arXiv citation per citing paper.
  Cited-paper metadata is enriched from arXiv, and `ref_s2_id` is looked up
  through the Semantic Scholar batch API. Citations without a recognizable
  arXiv ID are omitted because `ref_arxiv_id` is required by the table schema.
  Missing enrichment values are `null` (and authors are an empty list).

Set `SEMANTIC_SCHOLAR_API_KEY` to use an API key for Semantic Scholar. Without
one, the public API is used; lookup failures are recorded in `failures.jsonl`.
During crawling, reference lookups are grouped into batch requests (up to 50
papers per request, matching the crawler batch size). Semantic Scholar provides
the `raw_references`; the source extractor does not parse TeX/BBL bibliographies.
If a batch response omits any references reported by
`referenceCount`, the worker fetches that paper's references through the
paginated references endpoint.

Each JSONL line is a table row. Paper rows use `arxiv_id`, `title`, `authors`,
`submission_date`, `revised_dates`, and `venue`. Reference rows use `arxiv_id`,
`ref_arxiv_id`, `ref_title`, `ref_authors`, `ref_submission_date`, and
`ref_s2_id`.
