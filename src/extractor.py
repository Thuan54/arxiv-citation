import gzip
import os
import shutil
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

from config import RAW_DIR, IMAGE_EXTENSIONS
from downloader import STOP
from helper import log_failure, format_error, relative_source_path
from logger import get_logger
from metric import metrics
from monitor import monitor
from state import set_source


logger = get_logger(__name__)


TEX_SUFFIXES = {".tex", ".ltx"}
BBL_SUFFIX = ".bbl"
BIB_SUFFIX = ".bib"
TEXT_SUFFIXES = TEX_SUFFIXES | {BBL_SUFFIX, BIB_SUFFIX}


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

def safe_extract(archive, destination):
    """Extract regular files and directories, skipping anything unsafe.

    Unsafe members (links, devices, paths escaping the destination) are
    skipped instead of failing the whole paper.
    """
    root = Path(destination).resolve()
    extract_kwargs = (
        {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    )

    for member in archive.getmembers():
        if not (member.isreg() or member.isdir()):
            metrics.inc("archive_members_skipped")
            continue

        target = (root / member.name).resolve()

        if not target.is_relative_to(root):
            metrics.inc("archive_members_skipped")
            logger.warning("Skipping unsafe archive path: %s", member.name)
            continue

        try:
            archive.extract(member, root, **extract_kwargs)
        except (tarfile.TarError, OSError, ValueError) as exc:
            metrics.inc("archive_members_failed")
            logger.warning(
                "Could not extract %s: %s", member.name, format_error(exc)
            )


def extract_source(tmp_path, staging_dir):
    staging = Path(staging_dir)

    if tarfile.is_tarfile(tmp_path):
        # Full tar archive: preserve the directory hierarchy.
        with tarfile.open(tmp_path, "r:*") as archive:
            safe_extract(archive, staging)

    elif is_gzip_file(tmp_path):
        # Single gzip-compressed file, such as main.tex.
        filename = get_gzip_original_filename(tmp_path) or "main.tex"

        with gzip.open(tmp_path, "rb") as source:
            content = source.read()

        if content.startswith(b"%PDF"):
            # PDF-only submission: no TeX source exists.
            filename = Path(filename).with_suffix(".pdf").name
        elif Path(filename).suffix.lower() not in TEXT_SUFFIXES:
            # The gzip header name may lack an extension; retain it as TeX.
            filename += ".tex"

        (staging / filename).write_bytes(content)

    else:
        raise ValueError(
            f"Unsupported source format: {tmp_path}"
        )

    return remove_images(staging)


def is_gzip_file(path):
    try:
        with gzip.open(path, "rb") as source:
            source.read(1)
        return True
    except (OSError, EOFError):
        return False


def get_gzip_original_filename(path):
    # Parse the optional original filename in the gzip header.
    with open(path, "rb") as source:
        header = source.read(10)

        if len(header) < 10 or header[:2] != b"\x1f\x8b":
            return None

        flags = header[3]

        if flags & 0x04:  # Extra field
            length_bytes = source.read(2)
            if len(length_bytes) != 2:
                return None
            source.seek(int.from_bytes(length_bytes, "little"), 1)

        if flags & 0x08:  # Original filename
            name = bytearray()

            while True:
                char = source.read(1)
                if not char or char == b"\x00":
                    break
                name.extend(char)

            return Path(name.decode("utf-8", errors="replace")).name

    return None


def remove_images(directory):
    removed = 0

    for root, _, files in os.walk(directory):
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                Path(root, filename).unlink(missing_ok=True)
                removed += 1
  
    metrics.inc("files_removed", removed)
    return removed
    


def _extract_item(item, worker_name, track=True):
    """Extract and publish one source, returning serializable results."""
    task = item["task"]
    tmp_path = item["tmp_path"]

    arxiv_id = task["arxiv_id"]
    version = task["version"]

    destination = RAW_DIR / arxiv_id / version
    destination.parent.mkdir(parents=True, exist_ok=True)

    staging_dir = None
    backup_dir = None
    source_path = None
    status = "failed"
    stage = "initialization"
    timings = {}
    files_removed = 0
    extracted = False
    error = None
    restore_error = None

    try:
        # A previously completed extraction can be reused.
        if item["reused"]:
            source_path = relative_source_path(destination)
            status = "success"
        else:
            stage = "extract_archive"
            if track:
                monitor.state(
                    worker_name,
                    "extracting_archive",
                    task=task["version_id"],
                )

            staging_dir = tempfile.mkdtemp(
                prefix=f".{version}-staging-",
                dir=destination.parent,
            )

            started = time.perf_counter()
            try:
                files_removed = extract_source(tmp_path, staging_dir)
            finally:
                timings["source_extraction"] = (
                    time.perf_counter() - started
                )

            # Publish only after the archive has been safely extracted.
            (Path(staging_dir) / ".complete").touch()
            stage = "publish"
            if track:
                monitor.state(
                    worker_name,
                    "publishing_directory",
                    task=task["version_id"],
                )

            if destination.exists():
                backup_dir = tempfile.mkdtemp(
                    prefix=f".{version}-backup-",
                    dir=destination.parent,
                )
                os.rmdir(backup_dir)
                os.replace(destination, backup_dir)

            try:
                os.replace(staging_dir, destination)
                staging_dir = None

            except Exception:
                if backup_dir and os.path.exists(backup_dir):
                    os.replace(backup_dir, destination)
                    backup_dir = None
                raise

            if backup_dir and os.path.exists(backup_dir):
                shutil.rmtree(backup_dir, ignore_errors=True)
                backup_dir = None

            source_path = relative_source_path(destination)
            status = "success"
            extracted = True

    except Exception as exc:
        # If publication fails after the old destination was
        # moved, attempt to restore it.
        if backup_dir and os.path.exists(backup_dir):
            if not destination.exists():
                try:
                    os.replace(backup_dir, destination)
                    backup_dir = None
                except OSError as restore_exc:
                    restore_error = (
                        f"{type(restore_exc).__name__}: {restore_exc}"
                    )

        source_path = None
        status = "failed"
        error = {
            "type": type(exc).__name__,
            "message": format_error(exc),
            "traceback": "".join(
                traceback.format_exception(
                    type(exc),
                    exc,
                    exc.__traceback__,
                )
            ),
        }

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

        if staging_dir and os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)

        if backup_dir and os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)

    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "source_path": source_path,
        "status": status,
        "timings": timings,
        "files_removed": files_removed,
        "extracted": extracted,
        "error": error,
        "restore_error": restore_error,
        "stage": stage,
    }


def _complete_extraction(result):
    arxiv_id = result["arxiv_id"]
    version = result["version"]

    set_source(
        arxiv_id,
        version,
        result["source_path"],
        result["status"],
    )
    metrics.inc("files_removed", result["files_removed"])
    for name, elapsed in result["timings"].items():
        metrics.record_timing(name, elapsed)

    if result["extracted"] and result["status"] == "success":
        metrics.inc("source_extractions_succeeded")

    error = result["error"]
    if error is not None:
        metrics.inc("source_versions_failed")
        log_failure(
            arxiv_id,
            f"Source extraction {version}",
            f"stage={result['stage']}; {error['message']}",
            exception_type=error["type"],
            traceback_text=error["traceback"],
            details={
                "version": version,
                "stage": result["stage"],
                "restore_error": result["restore_error"],
            },
        )
        logger.error(
            "Extraction failed | paper=%s | version=%s "
            "| stage=%s | %s",
            arxiv_id,
            version,
            result["stage"],
            error["message"],
        )


def extraction_task(item, worker_id=0):
    """Extract one downloaded source and update pipeline state."""
    worker_name = f"extractor-{worker_id}"
    result = _extract_item(item, worker_name)
    _complete_extraction(result)


def extraction_worker(worker_id, extraction_queue):
    """Extraction worker: consume downloaded archives."""
    logger.info("Extractor %s started", worker_id)

    try:
        while True:
            monitor.state(
                f"extractor-{worker_id}",
                "waiting_for_extraction_task",
            )
            item = extraction_queue.get()

            try:
                if item is STOP:
                    monitor.state(
                        f"extractor-{worker_id}",
                        "stopped",
                    )
                    return

                monitor.state(
                    f"extractor-{worker_id}",
                    "processing_extraction",
                    task=item["task"]["version_id"],
                )
                extraction_task(item, worker_id)
                monitor.completed("extraction_tasks")

            finally:
                extraction_queue.task_done()

    finally:
        logger.info("Extractor %s stopped", worker_id)