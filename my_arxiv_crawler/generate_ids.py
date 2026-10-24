from datetime import datetime
from pathlib import Path

def generate_arxiv_ids(start_month: str, start_id: int, end_month: str, end_id: int, output_path: str):
    """
    Generates a sequential list of arXiv IDs in the format YYMM.NNNNN 
    and saves them to a text file, one per line.
    """
    # Parse the YYYY-MM strings into datetime objects
    start_dt = datetime.strptime(start_month, "%Y-%m")
    end_dt = datetime.strptime(end_month, "%Y-%m")
    
    current_dt = start_dt
    total_ids = 0
    
    # Ensure the output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        while current_dt <= end_dt:
            # Format the month as YYMM (e.g., 2024-07 -> 2407)
            yymm = current_dt.strftime("%y%m")
            
            # Determine the ID range for this specific month
            min_id = start_id if current_dt == start_dt else 1
            max_id = end_id if current_dt == end_dt else 99999
            
            for i in range(min_id, max_id + 1):
                # Format the ID with 5 leading zeros (e.g., 2839 -> 02839)
                arxiv_id = f"{yymm}.{i:05d}"
                f.write(arxiv_id + '\n')
                total_ids += 1
                
            # Advance to the next month
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
                
    print(f"Generated {total_ids} IDs and saved to {output_path}")