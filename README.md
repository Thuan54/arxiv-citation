#

## Preperation
The project was run on **python virtual environment**. To begin, create a virtual environment first  

```bash
# Create venv
python -m venv .venv

# Activate venv
# Linux
source .venv/bin/activate

# install packages
pip install -r requirements.txt
```

## Crawler
The guide below shows you how to run the crawler  
For **crawler version 1.0**, you will need to run a script with argument hard coded in the script `src/crawler.py` 
```python
# Change these variables to the ones you were given
# src/crawler.py
START_MONTH="2024-07"
START_ID=2839
END_MONTH="2024-07"
END_ID=3595
```

There are another way to do it, for **multiple crawlers**, **use second crawler version** in `my_arxiv_crawler`. First, you need to create a `.ipynb` at the **project root directory**, then insert a cell with the code below and execute it.

```python
from pathlib import Path
from my_arxiv_crawler.crawler import main


main(IDS_FILE = Path("data/inputs/arxiv_ids.txt"),
    JSONL_FILE = Path("data/metadata/papers.jsonl"),
    FAILURES_FILE = Path("data/metadata/failures.jsonl"),
    RAW_DIR = "data/raw",
    START_MONTH=YOU_START_MONTH,
    START_ID=YOUR_START_ID,
    END_MONTH=YOUR_END_MONTH,
    END_ID=YOUR_END_ID
)
```

We have created 3 of this example in the branch `custom_crawler`, you can check it out

*Note: respecting the polite policy in accessing arxiv for crawling purpose, it is not recommend to run multiple crawlers at once or crawling high rate which may overload arxiv server*

## Jupyter Book
For data visulization, we will run and render the result on **Jupyter Book** [This is goofy opening and need to fix]

```bash
# From project root, move directory to jupyter book
cd arxiv-jupyter-book

# Init jupyter book if it not exists
jupyter book init

# Build
jupyter book build

# Run it
jupyter book start
```


## Others
