import requests
import tarfile
import os
import tempfile
from pathlib import Path
import time


# File extensions to delete to remove figures and save space
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.pdf', '.eps', '.gif', '.bmp', '.ps'}


def fetch_and_extract_source(arxiv_id: str, version: str, base_raw_dir: str = "data/raw", sleep_time: float = 3.0) -> str:
    """
    Downloads, extracts, and cleans the LaTeX source for a specific arXiv paper version.
    Removes all image files after extraction.
    Returns the path to the extracted directory.
    """
    url = f"https://arxiv.org/e-print/{arxiv_id}{version}"
    headers = {"User-Agent": "HCMUS-IntroDS-Crawler/1.0 (academic project)"}
    
    # 1. Define the target directory based on our contract
    target_dir = Path(base_raw_dir) / arxiv_id / version
    target_dir.mkdir(parents=True, exist_ok=True)
    
    tmp_file_path = None
    try:
        # 2. Stream the download to avoid loading the whole tarball into memory
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        # 3. Download to a temporary file to ensure robust extraction
        # (If the network drops halfway, we don't want to extract a corrupted file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
            
        # 4. Extract the tarball safely
        with tarfile.open(tmp_file_path, "r:gz") as tar:
            # Security check: prevent path traversal vulnerabilities (CVE-2007-4559)
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            for member in tar.getmembers():
                member_path = os.path.join(str(target_dir), member.name)
                if not is_within_directory(str(target_dir), member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
                    
            # Extract all files into the target directory
            tar.extractall(str(target_dir))
        
        
        # CLEANUP: Remove all figure/image files to save space
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                    os.remove(os.path.join(root, file))
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error downloading source for {arxiv_id}{version}: {str(e)}")
    except tarfile.TarError as e:
        raise Exception(f"Tar extraction error for {arxiv_id}{version}: {str(e)}")
    finally:
        # 5. Clean up the temporary file to save disk space
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
    # Respect arXiv rate limits
    time.sleep(sleep_time)
    
    # Return the path so the main crawler can record it in the JSONL
    return str(target_dir)