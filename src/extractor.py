import gzip
import tarfile
from pathlib import Path
import re
import logging
import tempfile
import os
import shutil


from downloader import STOP
from config import RAW_DIR, IMAGE_EXTENSIONS
from helper import log_failure, format_error, relative_source_path
from metric import metrics
from state import set_source


BIBITEM_PATTERN = re.compile(
    r"\\bibitem(?:\s*\[[^\]]*\])?\s*\{[^}]*\}"
    r".*?"
    r"(?=\\bibitem(?:\s*\[[^\]]*\])?\s*\{"
    r"|\\end\s*\{thebibliography\}|$)",
    re.DOTALL,
)

logger = logging.getLogger(__name__)

def extract_raw_references(directory):
    """Extract complete raw bibitem entries from TeX files."""
    directory = Path(directory)
    references = []

    for tex_file in sorted(directory.rglob("*.tex")):
        try:
            content = tex_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            # Continue scanning other TeX files.
            metrics.inc("reference_files_read_failed")
            continue

        for match in BIBITEM_PATTERN.finditer(content):
            reference = match.group(0).strip()

            if reference:
                references.append(reference)

    metrics.inc("raw_references_extracted", len(references))
    return references

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

    remove_images(staging)


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


def extraction_task(item):
    """Extract, validate, and publish one downloaded source."""
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

    try:
        # A previously completed extraction can be reused.
        if item["reused"]:
            stage = "reference_extraction"

            with metrics.measure("reference_extraction"):
                raw_references = extract_raw_references(
                    destination
                )

            source_path = relative_source_path(destination)
            status = "success"
            return

        stage = "extract_archive"

        staging_dir = tempfile.mkdtemp(
            prefix=f".{version}-staging-",
            dir=destination.parent,
        )

        with metrics.measure("source_extraction"):
            extract_source(tmp_path, staging_dir)

        stage = "reference_extraction"

        with metrics.measure("reference_extraction"):
            raw_references = extract_raw_references(
                staging_dir
            )

        # Publish only after extraction and reference scanning
        # have completed successfully.
        (Path(staging_dir) / ".complete").touch()

        stage = "publish"

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
        metrics.inc("source_extractions_succeeded")

    except Exception as exc:
        # If publication fails after the old destination was
        # moved, attempt to restore it.
        if backup_dir and os.path.exists(backup_dir):
            if not destination.exists():
                try:
                    os.replace(backup_dir, destination)
                    backup_dir = None
                except OSError:
                    logger.exception(
                        "Could not restore backup for %s %s",
                        arxiv_id,
                        version,
                    )

        source_path = None
        raw_references = []
        status = "failed"

        log_failure(
            arxiv_id,
            f"Source extraction {version}",
            f"stage={stage}; {format_error(exc)}",
            exception=exc,
            details={
                "version": version,
                "stage": stage,
            },
        )

        metrics.inc("source_versions_failed")

        logger.error(
            "Extraction failed | paper=%s | version=%s "
            "| stage=%s | %s",
            arxiv_id,
            version,
            stage,
            format_error(exc),
        )

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

        if staging_dir and os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)

        if backup_dir and os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)

        set_source(
            arxiv_id,
            version,
            source_path,
            status,
            raw_references,
        )


def extraction_worker(worker_id, extraction_queue):
    """Extraction worker: consume downloaded archives."""
    logger.info("Extractor %s started", worker_id)

    try:
        while True:
            item = extraction_queue.get()

            try:
                if item is STOP:
                    return

                extraction_task(item)

            finally:
                extraction_queue.task_done()

    finally:
        logger.info("Extractor %s stopped", worker_id)
