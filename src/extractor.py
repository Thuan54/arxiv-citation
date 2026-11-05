import codecs
import gzip
import os
import re
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

from config import RAW_DIR, IMAGE_EXTENSIONS
from downloader import STOP
from helper import log_failure, format_error, relative_source_path
from logger import get_logger
from metric import metrics
from monitor import monitor
from state import set_source


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reference extraction
#
# arXiv sources keep their bibliography in several different places:
#   * inline \bibitem entries inside a .tex file
#   * a compiled .bbl file (BibTeX output, \bibitem format)
#   * a biblatex/biber .bbl file (\entry{...} ... \endentry format)
#   * \item entries inside thebibliography (no \bibitem at all)
#   * a raw .bib file with no .bbl (BibTeX database)
# All of them are handled below. Plain-text "References" sections with no
# LaTeX markup are NOT detected.
# ---------------------------------------------------------------------------

# Use the .bib database only when no other bibliography was found, because
# a .bib file also holds entries the paper never cited.
INCLUDE_BIBTEX_FALLBACK = True

TEX_SUFFIXES = {".tex", ".ltx"}
BBL_SUFFIX = ".bbl"
BIB_SUFFIX = ".bib"
TEXT_SUFFIXES = TEX_SUFFIXES | {BBL_SUFFIX, BIB_SUFFIX}

COMMENT_PATTERN = re.compile(r"(?<!\\)%.*")
BIBITEM_START = re.compile(r"\\bibitem(?![A-Za-z@])")
ITEM_START = re.compile(r"\\item(?![A-Za-z@])")
BIBLIOGRAPHY_END = re.compile(
    r"\\end\s*\{(?:thebibliography|references)\}"
)
THEBIBLIOGRAPHY_BLOCK = re.compile(
    r"\\begin\s*\{thebibliography\}.*?\\end\s*\{thebibliography\}",
    re.DOTALL,
)
BIBLATEX_ENTRY = re.compile(r"\\entry\s*\{.*?\\endentry", re.DOTALL)
BIB_ENTRY_START = re.compile(r"^[ \t]*@([A-Za-z]+)[ \t]*\{", re.MULTILINE)
SKIPPED_BIB_TYPES = {"comment", "string", "preamble"}

# Text right before "\bibitem" that means it is being (re)defined,
# not used as a bibliography entry.
MACRO_DEFINITION = re.compile(
    r"\\(?:re|provide)?newcommand\s*\{?\s*$|\\(?:[gex]?def|let)\s*$"
)


def _skip_group(text, pos, open_char, close_char, limit=None):
    """Return the index after the balanced group starting at text[pos]."""
    stop = len(text) if limit is None else min(len(text), pos + limit)
    depth = 0
    i = pos

    while i < stop:
        char = text[i]

        if char == "\\":
            i += 2
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return i + 1

        i += 1

    return None


def _skip_label(text, pos, limit=1000):
    """Skip an optional [label]; braces inside the label may contain ']'."""
    stop = min(len(text), pos + limit)
    braces = 0
    i = pos + 1

    while i < stop:
        char = text[i]

        if char == "\\":
            i += 2
            continue

        if char == "{":
            braces += 1
        elif char == "}":
            braces = max(0, braces - 1)
        elif char == "]" and braces == 0:
            return i + 1

        i += 1

    return None


def _bibitem_header_end(text, start):
    """Return where \\bibitem[label]{key} ends, or None if it is not one."""
    pos = start + len(r"\bibitem")
    size = len(text)

    while pos < size and text[pos].isspace():
        pos += 1

    if pos < size and text[pos] == "[":
        pos = _skip_label(text, pos)
        if pos is None:
            return None
        while pos < size and text[pos].isspace():
            pos += 1

    if pos < size and text[pos] == "{":
        return _skip_group(text, pos, "{", "}", limit=1000)

    return None


def _extract_bibitems(text):
    """Complete \\bibitem entries (BibTeX .bbl or inline in a .tex file)."""
    starts = []

    for match in BIBITEM_START.finditer(text):
        before = text[max(0, match.start() - 40):match.start()]

        if MACRO_DEFINITION.search(before):
            continue

        if _bibitem_header_end(text, match.start()) is None:
            continue

        starts.append(match.start())

    entries = []

    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)

        stop = BIBLIOGRAPHY_END.search(text, start, end)
        if stop:
            end = stop.start()

        entry = text[start:end].strip()
        if entry:
            entries.append(entry)

    return entries


def _extract_plain_items(text):
    """\\item entries in a thebibliography block that has no \\bibitem."""
    entries = []

    for block in THEBIBLIOGRAPHY_BLOCK.finditer(text):
        body = block.group(0)

        if BIBITEM_START.search(body):
            continue

        starts = [m.start() for m in ITEM_START.finditer(body)]

        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(body)
            entry = BIBLIOGRAPHY_END.sub("", body[start:end]).strip()

            if entry:
                entries.append(entry)

    return entries


def _extract_biblatex(text):
    """biblatex/biber .bbl entries: \\entry{key}{type}{...} ... \\endentry."""
    return [m.group(0).strip() for m in BIBLATEX_ENTRY.finditer(text)]


def _extract_bibtex(text):
    """Raw BibTeX database entries: @type{key, ...}."""
    entries = []
    pos = 0

    while True:
        match = BIB_ENTRY_START.search(text, pos)
        if match is None:
            break

        if match.group(1).lower() in SKIPPED_BIB_TYPES:
            pos = match.end()
            continue

        end = _skip_group(text, match.end() - 1, "{", "}", limit=200_000)
        if end is None:
            pos = match.end()
            continue

        entries.append(text[match.start():end].strip())
        pos = end

    return entries


def _read_text(path):
    """Read a source file, tolerating UTF-16 and legacy 8-bit encodings."""
    data = path.read_bytes()

    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")

    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # latin-1 never fails and keeps ASCII (and so every \bibitem) intact.
        return data.decode("latin-1")


def _candidate_files(directory):
    """Return (all_files_count, tex_files, bbl_files, bib_files)."""
    tex_files, bbl_files, bib_files = [], [], []
    total = 0

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        total += 1

        # Skip macOS resource-fork junk such as __MACOSX/._main.tex
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue

        suffix = path.suffix.lower()

        if suffix in TEX_SUFFIXES:
            tex_files.append(path)
        elif suffix == BBL_SUFFIX:
            bbl_files.append(path)
        elif suffix == BIB_SUFFIX:
            bib_files.append(path)

    return total, tex_files, bbl_files, bib_files


def extract_raw_references(directory, report=None):
    """Extract raw bibliography entries from an unpacked arXiv source.

    `report`, if given, is filled with what was found so callers can tell
    "no bibliography in the source" apart from "extractor missed it".
    """
    directory = Path(directory)
    total, tex_files, bbl_files, bib_files = _candidate_files(directory)

    references = []
    seen = set()
    kinds = Counter()

    def add(entries, kind):
        for entry in entries:
            key = re.sub(r"\s+", " ", entry)

            if key in seen:
                continue

            seen.add(key)
            references.append(entry)
            kinds[kind] += 1

    # .bbl first: it is the compiled, canonical bibliography.
    for path in bbl_files + tex_files:
        try:
            text = _read_text(path)
        except OSError:
            metrics.inc("reference_files_read_failed")
            continue

        text = COMMENT_PATTERN.sub("", text)

        add(_extract_bibitems(text), "bibitem")
        add(_extract_plain_items(text), "item")

        if path.suffix.lower() == BBL_SUFFIX:
            add(_extract_biblatex(text), "biblatex")

    if not references and INCLUDE_BIBTEX_FALLBACK:
        for path in bib_files:
            try:
                text = _read_text(path)
            except OSError:
                metrics.inc("reference_files_read_failed")
                continue

            # No comment stripping: '%' is not a comment inside .bib entries.
            add(_extract_bibtex(text), "bibtex")

    for kind, count in kinds.items():
        metrics.inc(f"raw_references_{kind}", count)

    metrics.inc("raw_references_extracted", len(references))

    if not references:
        metrics.inc("versions_without_references")

    if report is not None:
        report.update({
            "total_files": total,
            "tex_files": len(tex_files),
            "bbl_files": len(bbl_files),
            "bib_files": len(bib_files),
            "kinds": dict(kinds),
        })

    return references


def scan_references(directory, task):
    """Run reference extraction and log versions that yield nothing."""
    report = {}

    with metrics.measure("reference_extraction"):
        references = extract_raw_references(directory, report)

    if not references:
        # Not necessarily an error (PDF-only papers have no TeX source),
        # but the report shows which files existed, so misses are auditable.
        log_failure(
            task["arxiv_id"],
            f"Reference extraction {task['version']}",
            "No raw references found in source",
            details=report,
        )

    return references


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

def safe_extract(archive, destination):
    """Extract regular files and directories, skipping anything unsafe.

    Unsafe members (links, devices, paths escaping the destination) are
    skipped instead of failing the whole paper: one stray symlink used to
    discard the entire source and its references.
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
            # The gzip header name may lack an extension; without .tex
            # the reference scan would never look at this file.
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
    files_read_failed = 0
    extracted = False
    error = None
    restore_error = None

    try:
        # A previously completed extraction can be reused.
        if item["reused"]:
            stage = "reference_extraction"
            monitor.state(
                worker_name,
                "extracting_references",
                task=task["version_id"],
            )

            raw_references = scan_references(destination, task)

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
        raw_references = scan_references(staging_dir, task)

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
        "raw_references": [],
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

    set_source(
        arxiv_id,
        version,
        result["source_path"],
        result["status"],
    )
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