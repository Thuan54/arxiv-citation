from pathlib import Path

ROOT = Path.cwd()

IDS_FILE = ROOT / "data/inputs/arxiv_ids.txt"
JSONL_FILE = ROOT / "data/metadata/papers.jsonl"
FAILURES_FILE = ROOT / "data/metadata/failures.jsonl"
METRICS_FILE = ROOT / "data/metadata/metrics.json"
RAW_DIR = ROOT / "data/raw"

BATCH_SIZE = 50

NUM_DOWNLOADERS = 4
NUM_EXTRACTORS = 2

DOWNLOAD_QUEUE_SIZE =  BATCH_SIZE * 4
EXTRACTION_QUEUE_SIZE = BATCH_SIZE * 4


API_INTERVAL = 3.0

SOURCE_BURST_SIZE = 4
SOURCE_BURST_PAUSE = 1.0


API_TIMEOUT = 40
DOWNLOAD_TIMEOUT = 90
USER_AGENT = "HCMUS-IntroDS-Crawler/1.0 (academic project)"

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