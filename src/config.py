import os
from pathlib import Path

ROOT = Path.cwd()

MONITOR_ENABLED = False

IDS_FILE = ROOT / "data/inputs/arxiv_ids.txt"
JSONL_FILE = ROOT / "data/metadata/papers.jsonl"
PAPERS_TABLE_FILE = ROOT / "data/metadata/papers_table.jsonl"
REFERENCES_TABLE_FILE = ROOT / "data/metadata/references_table.jsonl"
FAILURES_FILE = ROOT / "data/metadata/failures.jsonl"
METRICS_FILE = ROOT / "data/metadata/metrics.json"
RAW_DIR = ROOT / "data/raw"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
SEMANTIC_SCHOLAR_REQUEST_INTERVAL = 1.0
SEMANTIC_SCHOLAR_BATCH_WAIT = 0.1
SEMANTIC_SCHOLAR_REFERENCE_PAGE_SIZE = 1000

BATCH_SIZE = 50

NUM_DOWNLOADERS = 4
NUM_EXTRACTORS = 4
EXTRACTOR_BACKEND = "thread" # "thread" or "process"

DOWNLOAD_QUEUE_SIZE =  BATCH_SIZE * 4
EXTRACTION_QUEUE_SIZE = BATCH_SIZE * 4


API_INTERVAL = 3.0

SOURCE_BURST_SIZE = 4
SOURCE_BURST_PAUSE = 1.0


API_TIMEOUT = 10
DOWNLOAD_TIMEOUT = 15
USER_AGENT = "HCMUS-IntroDS-Crawler/1.0 (academic project)"
SEMANTIC_SCHOLAR_API_TIMEOUT = 30

QUEUE_SIZE = BATCH_SIZE * 4

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".eps",
    ".gif",
    ".bmp",
    ".ps",
}