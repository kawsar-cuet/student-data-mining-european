"""
Script to extract structure and formatting style from a reference paper PDF.
This will analyze the paper and provide a detailed template for reformatting.
"""

import fitz  # PyMuPDF
import re
from collections import defaultdict

def extract_pdf_text(pdf_path):
    """Extract text from PDF with page information."""
    doc = fitz.open(pdf_path)
    full_text = []
    page_texts = []
    
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        page_texts.append({
            'page': page_num,
            'text': text
        })
        full_text.append(text)
    
    return '\n'.join(full_text), page_texts

def identify_section_hierarchy(text):
    """Identify the main sections and their hierarchy."""
    # Common section patterns in academic papers
    section_patterns = [
        r'^(\d+\.?\s+)?([A-Z][A-Za-z\s]+)$',  # Numbered or unnumbered sections
        r'^([A-Z][A-Z\s]+)$',  # ALL CAPS sections
    ]
    
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check for common section headers
        common_sections = [
            'Abstract', 'Introduction', 'Literature Review', 'Related Work',
            'Methodology', 'Methods', 'Materials and Methods',
            'Results', 'Findings', 'Discussion', 'Results and Discussion',
            'Conclusion', 'Conclusions', 'References', 'Acknowledgment', 'Acknowledgments'
        ]
        
        for section in common_sections:
            if section.lower() in line.lower() and len(line) < 50:
                sections.append({
                    'title': line,
                    'line_number': i,
                    'type': section
                })
                break
        
        # Check for numbered sections (e.g., "2. Methodology" or "2.1 Data Collection")
        numbered_match = re.match(r'^(\d+(?:\.\d+)*\.?)\s+([A-Z][A-Za-z\s]+)', line)
        if numbered_match and len(line) < 100:
            sections.append({
                'title': line,
                'line_number': i,
                'number': numbered_match.group(1),
                'text': numbered_match.group(2)
            })
    
    return sections

def extract_methodology_structure(text):
    """Extract the detailed structure of the Methodology section."""
    # Find methodology section
    lines = text.split('\n')
    methodology_start = -1
    methodology_end = -1
    
    for i, line in enumerate(lines):
        if 'methodology' in line.lower() or 'methods' in line.lower():
            if len(line.strip()) < 50:  # Likely a header
                methodology_start = i
                break
    
    if methodology_start == -1:
        return None
    
    # Find the end (next major section)
    major_sections = ['results', 'findings', 'discussion', 'conclusion', 'evaluation']
    for i in range(methodology_start + 1, len(lines)):
        line = lines[i].strip().lower()
        if any(section in line for section in major_sections) and len(lines[i].strip()) < 50:
            methodology_end = i
            break
    
    if methodology_end == -1:
        methodology_end = len(lines)
    
    methodology_text = '\n'.join(lines[methodology_start:methodology_end])
    
    # Extract subsections
    subsections = []
    subsection_pattern = r'^(\d+\.\d+(?:\.\d+)?\.?)\s+([A-Z][A-Za-z\s]+)'
    
    for i, line in enumerate(lines[methodology_start:methodology_end]):
        match = re.match(subsection_pattern, line.strip())
        if match:
            subsections.append({
                'number': match.group(1),
                'title': match.group(2),
                'full_title': line.strip()
            })
    
    return {
        'text': methodology_text,
        'subsections': subsections,
        'start_line': methodology_start,
        'end_line': methodology_end
    }

def analyze_equations(text):
    """Analyze how equations are formatted."""
    # Look for equation patterns
    equation_patterns = []
    
    # Numbered equations (1), (2), etc.
    numbered_eqs = re.findall(r'\((\d+)\)', text)
    if numbered_eqs:
        equation_patterns.append(f"Numbered equations found: {len(set(numbered_eqs))} unique equation numbers")
    
    # Common math symbols
    math_symbols = {
        '∑': 'Summation',
        '∫': 'Integral',
        '∂': 'Partial derivative',
        '×': 'Multiplication',
        '≤': 'Less than or equal',
        '≥': 'Greater than or equal',
        '→': 'Arrow/Maps to',
        '∈': 'Element of',
    }
    
    found_symbols = []
    for symbol, name in math_symbols.items():
        if symbol in text:
            found_symbols.append(name)
    
    return {
        'numbered_equations': len(set(numbered_eqs)) if numbered_eqs else 0,
        'math_symbols_used': found_symbols
    }

def analyze_tables_and_figures(text):
    """Analyze table and figure formatting."""
    tables = re.findall(r'Table\s+(\d+)', text, re.IGNORECASE)
    figures = re.findall(r'Figure\s+(\d+)', text, re.IGNORECASE)
    
    # Find table/figure captions
    caption_pattern = r'(Table|Figure)\s+\d+[:.]\s*([^\n]+)'
    captions = re.findall(caption_pattern, text, re.IGNORECASE)
    
    return {
        'num_tables': len(set(tables)),
        'num_figures': len(set(figures)),
        'sample_captions': captions[:5]  # First 5 captions
    }

def analyze_references(text):
    """Determine citation style."""
    # Check for common citation patterns
    apa_pattern = r'\([A-Z][a-z]+,\s+\d{4}\)'  # (Author, Year)
    ieee_pattern = r'\[\d+\]'  # [1], [2], etc.
    harvard_pattern = r'\([A-Z][a-z]+\s+\d{4}\)'  # (Author 2020)
    
    apa_count = len(re.findall(apa_pattern, text))
    ieee_count = len(re.findall(ieee_pattern, text))
    harvard_count = len(re.findall(harvard_pattern, text))
    
    citation_style = "Unknown"
    if ieee_count > max(apa_count, harvard_count):
        citation_style = "IEEE (numbered)"
    elif apa_count > harvard_count:
        citation_style = "APA (author-year)"
    elif harvard_count > 0:
        citation_style = "Harvard (author year)"
    
    return {
        'style': citation_style,
        'ieee_citations': ieee_count,
        'apa_citations': apa_count
    }

def main():
    pdf_path = r"d:\MS program\Final Thesis\Final Thesis project\Existing paper\1.5 1-s2.0-S2666920X24000663-main - can follow- important one.pdf"
    
    print("=" * 80)
    print("ACADEMIC PAPER STRUCTURE ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing: {pdf_path}\n")
    
    # Extract text
    print("Extracting text from PDF...")
    full_text, page_texts = extract_pdf_text(pdf_path)
    
    print(f"✓ Extracted {len(page_texts)} pages\n")
    
    # Analyze structure
    print("=" * 80)
    print("1. DOCUMENT STRUCTURE")
    print("=" * 80)
    sections = identify_section_hierarchy(full_text)
    print(f"\nFound {len(sections)} potential sections:\n")
    for i, section in enumerate(sections[:20], 1):  # Show first 20
        print(f"  {i}. {section.get('title', 'N/A')}")
    
    # Methodology analysis
    print("\n" + "=" * 80)
    print("2. METHODOLOGY SECTION STRUCTURE")
    print("=" * 80)
    methodology = extract_methodology_structure(full_text)
    if methodology:
        print(f"\nMethodology section found (lines {methodology['start_line']}-{methodology['end_line']})")
        print(f"\nSubsections ({len(methodology['subsections'])}):\n")
        for subsection in methodology['subsections']:
            print(f"  {subsection['number']} {subsection['title']}")
        
        # Print first 1000 chars of methodology
        print("\n--- Methodology Preview (first 1000 characters) ---")
        print(methodology['text'][:1000])
        print("...")
    else:
        print("\nMethodology section not clearly identified")
    
    # Equations
    print("\n" + "=" * 80)
    print("3. MATHEMATICAL NOTATION")
    print("=" * 80)
    eq_analysis = analyze_equations(full_text)
    print(f"\nNumbered equations: {eq_analysis['numbered_equations']}")
    if eq_analysis['math_symbols_used']:
        print(f"Math symbols used: {', '.join(eq_analysis['math_symbols_used'])}")
    
    # Tables and Figures
    print("\n" + "=" * 80)
    print("4. TABLES AND FIGURES")
    print("=" * 80)
    tf_analysis = analyze_tables_and_figures(full_text)
    print(f"\nTables: {tf_analysis['num_tables']}")
    print(f"Figures: {tf_analysis['num_figures']}")
    if tf_analysis['sample_captions']:
        print("\nSample captions:")
        for caption_type, caption_text in tf_analysis['sample_captions']:
            print(f"  {caption_type}: {caption_text[:100]}")
    
    # References
    print("\n" + "=" * 80)
    print("5. CITATION STYLE")
    print("=" * 80)
    ref_analysis = analyze_references(full_text)
    print(f"\nDetected style: {ref_analysis['style']}")
    print(f"IEEE-style citations found: {ref_analysis['ieee_citations']}")
    print(f"APA-style citations found: {ref_analysis['apa_citations']}")
    
    # Extract full text to file for further analysis
    output_file = r"d:\MS program\Final Thesis\Final Thesis project\Existing paper\extracted_text.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print("\n" + "=" * 80)
    print(f"✓ Full text extracted to: extracted_text.txt")
    print("=" * 80)
    
    return full_text, sections, methodology

if __name__ == "__main__":
    main()
