# Author: Abeer Mansour

import arxiv
from datetime import datetime, timedelta
import os


# Calculate date 2 years ago
two_years_ago = datetime.now() - timedelta(days=730)
date_filter = two_years_ago.strftime('%Y%m%d')

pdf_dir = 'arxiv_pdfs'
os.makedirs(pdf_dir, exist_ok=True)

categories = ['cs.AI', 'cs.LG', 'cs.CL']

papers = []

for category in categories:
    print(f"Fetching papers from {category}...")
    
    # Search query for each category
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=10,  
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    for result in search.results():

        if result.published.replace(tzinfo=None) >= two_years_ago:
            papers.append({
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'published': result.published,
                'categories': result.categories,
                'pdf_url': result.pdf_url,
                'entry_id': result.entry_id
            })


            pdf_filename = f"{result.entry_id.split('/')[-1]}.pdf"
            pdf_path = os.path.join(pdf_dir, pdf_filename)
            result.download_pdf(dirpath=pdf_dir, filename=pdf_filename)
            print(f"  Downloaded: {pdf_filename}")
        else:
            break  # Stop when papers are older than 2 years
    
    print(f"Retrieved {len([p for p in papers if category in p['categories']])} papers")

print(f"\nTotal papers retrieved: {len(papers)}")

# Example: Display first 3 papers
for i, paper in enumerate(papers[:3]):
    print(f"\n{i+1}. {paper['title']}")
    print(f"   Published: {paper['published'].date()}")
    print(f"   Categories: {', '.join(paper['categories'])}")