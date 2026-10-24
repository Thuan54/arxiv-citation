import re
import time
import requests

def clean_arxiv_id(raw_id: str) -> str:
    """
    Cleans the arXiv ID by removing whitespace and any trailing version suffix.
    Examples: 
      '2310.06825v2' -> '2310.06825'
      'hep-th/9901001v1 ' -> 'hep-th/9901001'
    """
    raw_id = raw_id.strip()
    # Regex to remove 'v' followed by digits at the end of the string
    return re.sub(r'v\d+$', '', raw_id)

def probe_versions(arxiv_id: str, sleep_time: float = 3.0) -> list[str]:
    """
    Probes arXiv to find all available versions for a given base ID.
    Returns a list of version strings, e.g., ['v1', 'v2'].
    Raises an Exception if no versions are found or if a network error occurs.
    """
    versions = []
    # arXiv rarely has more than 20 versions, but we set a safe upper bound
    for i in range(1, 50): 
        version_str = f"v{i}"
        # We probe the abstract page as it is lightweight and reliable
        url = f"https://arxiv.org/abs/{arxiv_id}{version_str}"
        
        # arXiv requires a descriptive User-Agent
        headers = {"User-Agent": "HCMUS-IntroDS-Crawler/1.0 (academic project)"}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                versions.append(version_str)
            elif response.status_code == 404:
                # 404 means we've reached the end of available versions
                break
            elif response.status_code in [429, 503]:
                # Rate limited or service unavailable, wait longer and retry
                time.sleep(sleep_time * 2)
                continue
            else:
                raise Exception(f"Unexpected status code {response.status_code} for {url}")
                
        except requests.exceptions.RequestException as e:
            # This will be caught by the main crawler's try...except block 
            # and logged to failures.jsonl with the step "Probing versions"
            raise Exception(f"Network error while probing {url}: {str(e)}")
            
        # Respect arXiv rate limits (approx 3 seconds between requests)
        time.sleep(sleep_time)
        
    if not versions:
        raise Exception(f"No versions found for {arxiv_id}. The ID might be invalid or the paper was removed.")
        
    return versions