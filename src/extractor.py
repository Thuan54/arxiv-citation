from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
import multiprocessing
import tarfile
from pathlib import Path
import re
import tempfile
import os
import shutil
import time
import traceback


from downloader import STOP
from config import RAW_DIR, IMAGE_EXTENSIONS
from helper import log_failure, format_error, relative_source_path
from metric import metrics
from monitor import monitor
from state import set_source
from logger import get_logger


BIBITEM_PATTERN = re.compile(
    r"\\bibitem(?:\s*\[[^\]]*\])?\s*\{[^}]*\}"
    r".*?"
    r"(?=\\bibitem(?:\s*\[[^\]]*\])?\s*\{"
    r"|\\end\s*\{thebibliography\}|$)",
    re.DOTALL,
)

logger = get_logger(__name__)


def _strip_tex_comments(content):
    uncommented = []

    for line in content.splitlines(keepends=True):
        comment_start = None

        for index, character in enumerate(line):
            if character != "%":
                continue

            preceding_backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                preceding_backslashes += 1
                cursor -= 1

            if preceding_backslashes % 2 == 0:
                comment_start = index
                break

        if comment_start is not None:
            newline = "\n" if line.endswith("\n") else ""
            uncommented.append(line[:comment_start].rstrip("\r\n") + newline)
        else:
            uncommented.append(line)

    return "".join(uncommented)


def extract_raw_references(directory):
    """Extract unique raw bibitem entries from TeX and BBL files."""
    directory = Path(directory)
    references = []
    files_read_failed = 0

    source_files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tex", ".bbl"}
    )
    for source_file in source_files:
        try:
            content = source_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            # Continue scanning other TeX files.
            files_read_failed += 1
            continue

        content = _strip_tex_comments(content)
        for match in BIBITEM_PATTERN.finditer(content):
            reference = match.group(0).strip()
            normalized_reference = re.sub(r"\s+", " ", reference)

            if reference and normalized_reference not in seen:
                seen.add(normalized_reference)
                references.append(reference)

    return references, files_read_failed

def safe_extract(archive, destination):
  root = destination.resolve()

  for member in archive.getmembers():
      target = (root / member.name).resolve()

      if not target.is_relative_to(root):
          raise ValueError(f"Unsafe archive path: {member.name}")

      if member.issym() or member.islnk():
          raise ValueError(f"Archive links are not allowed: {member.name}")

  archive.extractall(root)

def extract_source(tmp_path, staging_dir):
    staging = Path(staging_dir)

    if tarfile.is_tarfile(tmp_path):
        # Full tar archive: preserve the directory hierarchy.
        with tarfile.open(tmp_path, "r:gz") as archive:
            safe_extract(archive, staging)

    elif is_gzip_file(tmp_path):
        # Single gzip-compressed file, such as main.tex.
        filename = get_gzip_original_filename(tmp_path)

        if not filename:
            filename = "main.tex"

        with gzip.open(tmp_path, "rb") as source:
            content = source.read()

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
    raw_references = []
    status = "failed"
    stage = "initialization"
    timings = {}
    files_removed = 0
    files_read_failed = 0
    extracted = False
    error = None
    restore_error = None

    try:
        # A previously completed extraction can be reused.
        if item["reused"]:
            stage = "reference_extraction"
            if track:
                monitor.state(
                    worker_name,
                    "extracting_references",
                    task=task["version_id"],
                )

            started = time.perf_counter()
            try:
                raw_references, files_read_failed = extract_raw_references(
                    destination
                )
            finally:
                timings["reference_extraction"] = (
                    time.perf_counter() - started
                )

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

            stage = "reference_extraction"
            if track:
                monitor.state(
                    worker_name,
                    "extracting_references",
                    task=task["version_id"],
                )

            started = time.perf_counter()
            try:
                raw_references, files_read_failed = extract_raw_references(
                    staging_dir
                )
            finally:
                timings["reference_extraction"] = (
                    time.perf_counter() - started
                )

            # Publish only after extraction and reference scanning
            # have completed successfully.
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
        raw_references = []
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
        "raw_references": raw_references,
        "timings": timings,
        "files_removed": files_removed,
        "files_read_failed": files_read_failed,
        "extracted": extracted,
        "error": error,
        "restore_error": restore_error,
        "stage": stage,
    }


def _complete_extraction(result):
    arxiv_id = result["arxiv_id"]
    version = result["version"]

    metrics.inc("files_removed", result["files_removed"])
    metrics.inc(
        "raw_references_extracted",
        len(result["raw_references"]),
    )
    metrics.inc(
        "reference_files_read_failed",
        result["files_read_failed"],
    )
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

    set_source(
        arxiv_id,
        version,
        result["source_path"],
        result["status"],
        result["raw_references"],
    )


def extraction_task(item, worker_id):
    """Extract, validate, and publish one downloaded source."""
    worker_name = f"extractor-{worker_id}"
    result = _extract_item(item, worker_name)
    _complete_extraction(result)
    return result


def _finish_process_task(future, worker_id, item, extraction_queue):
    worker_name = f"extractor-{worker_id}"
    task = item["task"]

    try:
        result = future.result()
        _complete_extraction(result)
    except Exception as exc:
        log_failure(
            task["arxiv_id"],
            f"Source extraction {task['version']}",
            format_error(exc),
            exception=exc,
            details={
                "version": task["version"],
                "worker_id": worker_id,
                "stage": "process_worker",
            },
        )
        metrics.inc("source_versions_failed")
        set_source(
            task["arxiv_id"],
            task["version"],
            None,
            "failed",
            [],
        )
        logger.exception(
            "Extraction process failed | paper=%s | version=%s",
            task["arxiv_id"],
            task["version"],
        )
    finally:
        monitor.state(worker_name, "waiting_for_extraction_task")
        monitor.completed("extraction_tasks")
        extraction_queue.task_done()


def _initialize_extraction_process(raw_dir):
    global RAW_DIR
    RAW_DIR = Path(raw_dir)


def process_extraction_worker(
    extraction_queue,
    worker_count,
    raw_dir=RAW_DIR,
):
    """Dispatch extraction tasks to child processes and collect results."""
    logger.info("Started %s extraction processes", worker_count)
    available_workers = list(range(worker_count))
    pending = {}
    stop_count = 0

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_extraction_process,
        initargs=(str(raw_dir),),
    ) as executor:
        while stop_count < worker_count:
            item = extraction_queue.get()

            if item is STOP:
                stop_count += 1
                extraction_queue.task_done()
                continue

            while not available_workers:
                completed, _ = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    worker_id, completed_item = pending.pop(future)
                    _finish_process_task(
                        future,
                        worker_id,
                        completed_item,
                        extraction_queue,
                    )
                    available_workers.append(worker_id)

            worker_id = available_workers.pop(0)
            worker_name = f"extractor-{worker_id}"
            task = item["task"]
            monitor.state(
                worker_name,
                "processing_extraction",
                task=task["version_id"],
                backend="process",
            )
            future = executor.submit(
                _extract_item,
                item,
                worker_name,
                False,
            )
            pending[future] = (worker_id, item)

        while pending:
            completed, _ = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                worker_id, item = pending.pop(future)
                _finish_process_task(
                    future,
                    worker_id,
                    item,
                    extraction_queue,
                )

    logger.info("Extraction processes stopped")


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
