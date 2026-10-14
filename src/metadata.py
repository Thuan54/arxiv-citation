import requests
import xml.etree.ElementTree as ET
import time

def fetch_metadata(arxiv_id: str, versions: list[str], sleep_time: float = 3.0) -> list[dict]:
    """
    Fetches metadata for a list of versions from the arXiv API.
    Returns a list of dictionaries containing title, authors, abstract, categories, and submission_date.
    The order of the returned list matches the order of the input `versions`.
    """
    # 1. Construct the list of full IDs (e.g., '2310.06825v1,2310.06825v2')
    full_ids = [f"{arxiv_id}{v}" for v in versions]
    id_list_str = ",".join(full_ids)

    # 2. Query the API
    url = f"http://export.arxiv.org/api/query?id_list={id_list_str}&max_results={len(full_ids)}"
    headers = {"User-Agent": "HCMUS-IntroDS-Crawler/1.0 (academic project)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed for {arxiv_id}: {str(e)}")
        
    # 3. Parse the XML response
    # The arXiv API returns an Atom feed, which requires namespace handling
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(response.text)
    
    # 4. Extract data and map it back to the requested IDs
    fetched_data = {}
    for entry in root.findall('atom:entry', ns):
        # The ID in the feed is a URL like 'http://arxiv.org/abs/2310.06825v1'
        # We extract the ID part to use as a key
        id_url = entry.find('atom:id', ns).text
        entry_id = id_url.split('/abs/')[-1]
        
        # Extract and clean fields
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
        categories = [cat.get('term') for cat in entry.findall('atom:category', ns)]
        published = entry.find('atom:published', ns).text
        
        fetched_data[entry_id] = {
            "title": title,
            "authors": authors,
            "abstract": summary,
            "categories": categories,
            "submission_date": published
        }
        
    # 5. Reconstruct the list in the exact order of the input versions
    # This is critical so that metadata aligns with the correct version later
    metadata_list = []
    for full_id in full_ids:
        if full_id in fetched_data:
            metadata_list.append(fetched_data[full_id])
        else:
            # If probe_versions said it exists but API didn't return it, something is wrong
            raise Exception(f"Metadata missing for {full_id} in API response.")
            
    # Respect rate limits
    time.sleep(sleep_time)
    
    return metadata_list