import re
from pathlib import Path

def extract_references(latex_source_path: str) -> list[str]:
    """
    Locates the bibliography in the extracted LaTeX files and extracts 
    the raw \bibitem strings.
    Returns a list of raw reference strings.
    """
    path = Path(latex_source_path)
    content = ""
    
    # 1. Look for .bbl files (BibTeX output)
    bbl_files = list(path.rglob("*.bbl"))
    if bbl_files:
        # Read the first .bbl file found
        content = bbl_files[0].read_text(encoding='utf-8', errors='ignore')
    else:
        # 2. Strategy B: Look for \begin{thebibliography} in .tex files
        tex_files = list(path.rglob("*.tex"))
        for tex_file in tex_files:
            text = tex_file.read_text(encoding='utf-8', errors='ignore')
            if "\\begin{thebibliography}" in text:
                content = text
                break
                
    # If no bibliography environment was found in either .bbl or .tex
    if not content:
        return [] 
        
    # 3. Extract the entire thebibliography environment
    # re.DOTALL allows the '.' to match newlines, which is crucial for multi-line entries
    match = re.search(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", 
        content, 
        re.DOTALL
    )
    
    if not match:
        return []
        
    bib_content = match.group(0)
    
    # 4. Split the environment into individual \bibitem entries
    # This regex finds a \bibitem and captures everything up to the next \bibitem 
    # or the \end{thebibliography} tag.
    entries = re.findall(
        r"\\bibitem.*?(?=\\bibitem|\\end\{thebibliography\})", 
        bib_content, 
        re.DOTALL
    )
    
    # 5. Clean up excessive whitespace
    cleaned_entries = [entry.strip() for entry in entries if entry.strip()]
    
    return cleaned_entries