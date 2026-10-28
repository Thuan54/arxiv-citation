import xml.etree.ElementTree as ET
import requests

from config import API_INTERVAL, API_TIMEOUT, BATCH_SIZE, USER_AGENT
from helper import log_failure, parse_versioned_id
from metric import metrics
from rate_limiter import arxiv_limiter
from state import set_metadata

API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

def fetch_metadata_batch(tasks):
  """Fetch metadata for a bounded batch of version tasks."""
  expected = {task["version_id"]: task for task in tasks}

  try:
      with arxiv_limiter.request("api"):
        with metrics.measure("metadata_batch"):
            response = requests.get(
                API_URL,
                params={
                    "id_list": ",".join(expected),
                    "max_results": len(expected),
                },
                headers={"User-Agent": USER_AGENT},
                timeout=API_TIMEOUT,
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
      returned = set()

      for entry in root.findall("atom:entry", ATOM_NS):
          node = entry.find("atom:id", ATOM_NS)

          if node is None or not node.text:
              continue

          parsed = parse_versioned_id(node.text)

          if parsed is None:
              continue

          arxiv_id, version = parsed
          version_id = f"{arxiv_id}{version}"
          task = expected.get(version_id)

          if task is None:
              continue

          def text_value(tag):
              node = entry.find(f"atom:{tag}", ATOM_NS)
              return (
                  node.text.strip().replace("\n", " ")
                  if node is not None and node.text
                  else ""
              )


          metadata = {
              "title": text_value("title"),
              "authors": [
                  node.text.strip()
                  for author in entry.findall("atom:author", ATOM_NS)
                  if (node := author.find("atom:name", ATOM_NS))
                  is not None and node.text
              ],
              "abstract": text_value("summary"),
              "categories": [
                  node.get("term")
                  for node in entry.findall("atom:category", ATOM_NS)
                  if node.get("term")
              ],
              "submission_date": text_value("published")[:10] or None,
          }

          set_metadata(arxiv_id, version, metadata, "success")
          returned.add(version_id)
          metrics.inc("metadata_versions_succeeded")

      for version_id, task in expected.items():
          if version_id in returned:
              continue

          set_metadata(
              task["arxiv_id"], task["version"], None, "failed"
          )
          log_failure(
              task["arxiv_id"],
              f"Metadata {task['version']}",
              f"No matching entry returned for {version_id}",
          )
          metrics.inc("metadata_versions_failed")

  except Exception as exc:
      metrics.inc("metadata_batches_failed")

      for task in tasks:
          set_metadata(
              task["arxiv_id"], task["version"], None, "failed"
          )
          log_failure(
              task["arxiv_id"],
              f"Metadata {task['version']}",
              exc,
          )
          metrics.inc("metadata_versions_failed")


def metadata_worker(task_queue):
  """Consume individual tasks and fetch them in API-sized batches."""
  print("[Metadata Worker] Starting")
  batch = []

  while True:
      task = task_queue.get()

      try:
          if task is None:
              if batch:
                  fetch_metadata_batch(batch)
                  batch.clear()
              return

          batch.append(task)

          if len(batch) >= BATCH_SIZE:
              fetch_metadata_batch(batch)
              batch.clear()

      finally:
          task_queue.task_done()

  print("[Metadata Worker] Finished")