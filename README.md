# Arxiv cralwer improve from `main` branch
Handling more fields like `venue` and `raw_references`... more robust
## How to run it 

```bash
# create .venv
python -m venv .venv

# use .venv for Linux
soucre .venv/bin/activate

# install necessary pacakge
pip install -r requirements.txt

```

**You must go to `src/generate_ids.py` to change arxiv paper indexs rage to yours**
```python
# src/generate_ids.py
if __name__ == '__main__':
    generate_arxiv_ids(
        start_id=YOUR_START_ID,
        start_month=YOUR_START_MONTH,
        end_id=YOUR_END_MONTH,
        end_month='2024-08',
        output_path=IDS_FILE
    )
```

*If your **start_id > end_id** because **start_month $\ne$ end_month** then you must **run twice from [start_month, last_paper_of_month]** and **[first_paper_of_month, end_id]***

*You must also get **a Semnatic Scholar** for reference crawling and put it into `.env` file*

To crawl it, run this
```bash
# generate ids
python src/generate_ids.py
# crawl
python src/crawler.py
```