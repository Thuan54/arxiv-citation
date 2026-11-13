import os
import xml.etree.ElementTree as ET
from calendar import monthrange
from datetime import datetime
from pathlib import Path

import requests

from config import IDS_FILE, API_TIMEOUT, USER_AGENT


API_URL = "https://export.arxiv.org/api/query"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"


def _month_paper_count(month):
    month_start = month.strftime("%Y%m")
    month_end = month.replace(
        day=monthrange(month.year, month.month)[1]
    ).strftime("%Y%m%d")
    params = {
        "search_query": (
            f"submittedDate:[{month_start}010000 TO {month_end}2359]"
        ),
        "start": 0,
        "max_results": 1,
    }

    response = requests.get(
        API_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=API_TIMEOUT,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    total_results = root.find(f".//{OPENSEARCH_NS}totalResults")
    if total_results is None or total_results.text is None:
        raise ValueError(
            f"arXiv API response did not include totalResults for {month:%Y-%m}"
        )

    try:
        return int(total_results.text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid totalResults for {month:%Y-%m}: "
            f"{total_results.text!r}"
        ) from exc


def generate_arxiv_ids(
    start_month: str,
    start_id: int,
    end_month: str,
    end_id: int,
    output_path: str,
):
    """Generate IDs up to each month's arXiv result count."""
    start_dt = datetime.strptime(start_month, "%Y-%m")
    end_dt = datetime.strptime(end_month, "%Y-%m")

    if start_dt > end_dt:
        raise ValueError("start_month must be before or equal to end_month")
    if start_id < 1 or end_id < 1:
        raise ValueError("start_id and end_id must be positive integers")

    months = []
    current_dt = start_dt
    while current_dt <= end_dt:
        months.append(current_dt)
        if current_dt.month == 12:
            current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
        else:
            current_dt = current_dt.replace(month=current_dt.month + 1)

    month_counts = {
        month: _month_paper_count(month)
        for month in months
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_ids = 0

    with output.open("w", encoding="utf-8") as file:
        for month in months:
            yymm = month.strftime("%y%m")
            min_id = start_id if month == start_dt else 1
            max_id = end_id if month == end_dt else month_counts[month]
            max_id = min(max_id, month_counts[month])

            for paper_id in range(min_id, max_id + 1):
                file.write(f"{yymm}.{paper_id:05d}\n")
                total_ids += 1

    print(f"Generated {total_ids} IDs and saved to {output_path}")



if __name__ == '__main__':
    generate_arxiv_ids(
        start_id=16045,
        start_month='2024-08',
        end_id=17500,
        end_month='2024-08',
        output_path=IDS_FILE
    )