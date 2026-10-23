import json
from pathlib import Path
from time import perf_counter
from typing import Callable, TypeVar

# Import the modules we built in the previous steps
from version_probe import clean_arxiv_id, probe_versions
from metadata import fetch_metadata
from source_fetcher import fetch_and_extract_source
from reference_extractor import extract_references
from generate_ids import generate_arxiv_ids

# Define paths based on our contract
IDS_FILE = Path("data/inputs/arxiv_ids.txt")
JSONL_FILE = Path("data/metadata/papers.jsonl")
FAILURES_FILE = Path("data/metadata/failures.jsonl")
RAW_DIR = "data/raw"
START_MONTH="2024-07"
START_ID=2839
END_MONTH="2024-07"
END_ID=7838

T = TypeVar("T")


def run_timed_step(step: str, operation: Callable[..., T], *args, **kwargs) -> T:
    started_at = perf_counter()
    try:
        return operation(*args, **kwargs)
    finally:
        elapsed = perf_counter() - started_at
        print(f"   ⏱ {step}: {elapsed:.2f}s")


def load_ids(file_path: Path) -> list[str]:
    """Reads the text file and returns a list of cleaned arXiv IDs."""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_ids = f.read().splitlines()

    # Clean each ID and remove empty lines
    return [clean_arxiv_id(rid) for rid in raw_ids if rid.strip()]

def get_processed_ids(jsonl_path: Path) -> set:
    """Reads the JSONL file and returns a set of already processed arXiv IDs for resumability."""
    processed_ids = set()

    if jsonl_path.exists():
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    paper_data = json.loads(line)
                    processed_ids.add(paper_data['arxiv_id'])

    return processed_ids

def write_jsonl(jsonl_path: Path, record: dict):
    """Appends a single paper record as a JSON line to the output file."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

def log_failure(
    failures_path: Path,
    arxiv_id: str,
    step: str,
    message: str
):
    """Appends a failure record to the failures JSONL file."""
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    failure_record = {
        "arxiv_id": arxiv_id,
        "failed_step": step,
        "error_message": message
    }

    with open(failures_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(failure_record) + '\n')



def main():
    crawl_started_at = perf_counter()
    print("Starting arXiv Crawler...")

    # Generate ids
    run_timed_step(
        "Generating IDs",
        generate_arxiv_ids,
        start_month=START_MONTH, start_id=START_ID,
        end_month=END_MONTH, end_id=END_ID,
        output_path=IDS_FILE
    )

    # 1. Load inputs and handle resumability
    all_ids = run_timed_step("Loading IDs", load_ids, IDS_FILE)
    processed_ids = run_timed_step(
        "Loading processed IDs", get_processed_ids, JSONL_FILE
    )

    print(
        f"Loaded {len(all_ids)} IDs. "
        f"{len(processed_ids)} already processed."
    )

    # Ensure output directories exist
    JSONL_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 2. Loop through IDs
    for i, arxiv_id in enumerate(all_ids):
        if arxiv_id in processed_ids:
            print(
                f"[{i + 1}/{len(all_ids)}] "
                f"Skipping {arxiv_id}, already processed."
            )
            continue

        print(
            f"[{i + 1}/{len(all_ids)}] Processing {arxiv_id}..."
        )
        paper_started_at = perf_counter()

        current_step = "Starting crawl"

        try:
            # Step 1: Probe versions
            current_step = "Probing versions"
            versions = run_timed_step(
                f"{current_step} for {arxiv_id}", probe_versions, arxiv_id
            )

            # Step 2: Fetch metadata
            current_step = "Fetching metadata"
            metadata_list = run_timed_step(
                f"{current_step} for {arxiv_id}",
                fetch_metadata, arxiv_id, versions
            )

            # Initialize the record for this paper
            paper_record = {
                "arxiv_id": arxiv_id,
                "versions": []
            }

            # Step 3 & 4: Download sources and extract references
            for j, version in enumerate(versions):
                try:
                    # Download and extract source
                    current_step = f"Downloading/extracting source for {version}"
                    latex_path = run_timed_step(
                        f"{current_step} ({arxiv_id})",
                        fetch_and_extract_source,
                        arxiv_id, version, base_raw_dir=RAW_DIR
                    )

                    # Extract raw references
                    current_step = f"Extracting references for {version}"
                    raw_refs = run_timed_step(
                        f"{current_step} ({arxiv_id})",
                        extract_references, latex_path
                    )
                
                except Exception as e:
                    # ⚠️ Graceful handling: Source is missing or failed.
                    # We DO NOT crash the paper. We just set path to null.
                    print(f"   ⚠️ Warning: Missing/failed source for {arxiv_id}{version}. Setting path to null.")
                    latex_path = None
                    raw_refs = []    


                # Append to the paper record according to our contract
                paper_record["versions"].append({
                    "version": version,
                    "metadata": metadata_list[j],
                    "latex_source_path": latex_path,
                    "raw_references": raw_refs
                })

            # If everything succeeds, write to JSONL
            run_timed_step(
                f"Writing record for {arxiv_id}",
                write_jsonl, JSONL_FILE, paper_record
            )

            print(
                f"Successfully processed {arxiv_id} "
                f"in {perf_counter() - paper_started_at:.2f}s"
            )

        except Exception as e:
            # Log the failure with the exact step that caused it
            print(f"Total failure at step '{current_step}' for {arxiv_id}: {str(e)}")
            log_failure(FAILURES_FILE, arxiv_id, current_step, str(e))
            print(
                f"   Total time for {arxiv_id}: "
                f"{perf_counter() - paper_started_at:.2f}s"
            )
            continue

    print(f"Crawler finished in {perf_counter() - crawl_started_at:.2f}s.")

if __name__ == "__main__":
    main()