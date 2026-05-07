# Author: Abeer Mansour

import pymupdf4llm
import os

def pdf_to_markdown(pdf_path, output_dir='markdown_output'):
    """Convert a single PDF to Markdown"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{filename}.md")
    
    # Convert PDF to Markdown
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    
    print(f"Converted: {pdf_path} -> {output_path}")
    return output_path

def batch_convert_pdfs(pdf_folder, output_dir='markdown_output'):
    """Convert all PDFs in a folder to Markdown"""
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            pdf_to_markdown(pdf_path, output_dir)
        except Exception as e:
            print(f"Error converting {pdf_file}: {e}")
    
    print(f"\nAll conversions complete! Markdown files saved in '{output_dir}'")

# Example usage:
if __name__ == "__main__":
    # Convert single PDF
    # pdf_to_markdown('paper.pdf')
    
    # Convert all PDFs in arxiv_pdfs folder
    batch_convert_pdfs('ssf_paper')