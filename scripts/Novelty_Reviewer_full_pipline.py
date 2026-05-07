# Author: Abeer Mansour

import requests
import json
import time
import os
import re

import datasets as hfds
import polars as pl
import transformers
from tqdm.auto import tqdm
from pathlib import Path
import torch 
import networkx as nx
from sklearn.cluster import DBSCAN

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import polars as pl

from typing import Dict, List, Optional
from datetime import datetime
import PyPDF2
import io
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup cache
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR


def generate_test_data():

    df_test = pl.read_parquet("../Dataset_construction/novelty_dataset_aggregated/test_dataset.parquet")
    print(f"loaded test df: {df_test.shape}")
    print(df_test.schema)

    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, legacy=False)

    # Convert Polars DataFrame to Hugging Face Dataset
    ds_test = hfds.Dataset.from_polars(df_test)

    # Define system and user prompt templates
    SYSTEM_PROMPT_TEMPLATE = """You are an expert reviewer for AI conferences. 
    Your task is to evaluate the NOVELTY of research papers according to reviewer guidelines. Be factual, concise, and balanced.
    You should carefully read the paper, judge whether the ideas are novel or not, 
    and provide a concise justification based only on the content of the paper.

    Return two things: Novelty Score and Short Novelty Review.
    """
    USER_PROMPT_TEMPLATE = """Review the following paper for novelty:

    {paper_text}

    Provide your evaluation in the following format:
    Novelty Score: 
    -1 = Not novel (work is incremental, derivative, or replicating existing approaches with minimal innovation.)
    0 = Limited novelty (work is somewhat standard, showing minor variations or applications of known methods without substantial conceptual or technical innovation.)
    1 = Moderately novel (some originality but overlap with prior work, or extension/combination of existing ideas.)
    2 = Highly novel (fundamentally new ideas, approaches, problem formulations, or insights that significantly advance the field.)

    Short Novelty Review: A 3-5 sentence containing your summary and reasoning about paper novelty/originality.
    """

    # Function to create messages in the chat format
    def create_messages(row):
    
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(paper_text=row['paper_text'])}
        ]
        return messages


    # Apply the create_messages function to the dataset
    ds_test = ds_test.map(lambda row: {
        'messages': create_messages(row)
    })


def calc_similarity_score(paper1, paper2, model_name='all-MiniLM-L6-v2'):

    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    paper1_embedding = model.encode(paper1)
    paper2_embedding = model.encode(paper2)
    
    # Calculate cosine similarity
    similarity_score = util.cos_sim(paper1_embedding, paper2_embedding).mean().item() * 100 
    print(f"similarity score: {similarity_score}")

    return similarity_score


class PDFDownloader:
    """Download and extract text from papers"""
    
    def __init__(self, cache_dir: str = "./paper_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_paper(self, paper_id: str, url: str) -> Optional[str]:
        """Download paper and extract text"""
        cache_path = self.cache_dir / f"{paper_id}.txt"
        
        if cache_path.exists():
            return cache_path.read_text(encoding='utf-8')
        
        # Try direct PDF URL
        if url and url.endswith('.pdf'):
            pdf_content = self._download_url(url)
            if pdf_content:
                text = self._extract_text(pdf_content)
                if text:
                    cache_path.write_text(text, encoding='utf-8')
                    return text
        
        # Try Semantic Scholar open access
        s2_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=openAccessPdf"
        try:
            resp = requests.get(s2_url, timeout=10).json()
            if resp.get('openAccessPdf', {}).get('url'):
                pdf_content = self._download_url(resp['openAccessPdf']['url'])
                if pdf_content:
                    text = self._extract_text(pdf_content)
                    if text:
                        cache_path.write_text(text, encoding='utf-8')
                        return text
        except:
            pass
        
        return None
    
    def _download_url(self, url: str) -> Optional[bytes]:
        try:
            resp = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
            return resp.content if resp.status_code == 200 else None
        except:
            return None
    
    def _extract_text(self, pdf_bytes: bytes) -> Optional[str]:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = " ".join(page.extract_text() for page in reader.pages[:50])
            text = re.sub(r'\s+', ' ', text).strip()
            return text if len(text) > 500 else None
        except:
            return None

class KnowledgeGraph:
    """Build and analyze knowledge graph of papers"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.embeddings = {}
    
    def add_paper(self, paper_id: str, knowledge: Dict, embedding: np.ndarray):
        """Add paper node to graph"""
        self.graph.add_node(paper_id, 
                           title=knowledge.get('title', ''),
                           knowledge=knowledge)
        self.embeddings[paper_id] = embedding
    
    def add_similarity_edge(self, paper1_id: str, paper2_id: str, 
                           similarity: float, threshold: float = 30.0):
        """Add edge if similarity above threshold"""
        if similarity >= threshold:
            self.graph.add_edge(paper1_id, paper2_id, weight=similarity)
    
    def find_most_similar_by_radius(self, manuscript_id: str, 
                                     radius: float = 50.0) -> List[tuple]:
        """Find papers within similarity radius of manuscript"""
        if manuscript_id not in self.graph:
            return []
        
        similar_papers = []
        for neighbor in self.graph.neighbors(manuscript_id):
            edge_data = self.graph[manuscript_id][neighbor]
            similarity = edge_data['weight']
            if similarity >= radius:
                similar_papers.append((
                    neighbor,
                    similarity,
                    self.graph.nodes[neighbor]['knowledge']
                ))
        
        return sorted(similar_papers, key=lambda x: x[1], reverse=True)
    
    def find_clusters(self, min_similarity: float = 40.0) -> List[List[str]]:
        """Find clusters of similar papers using community detection"""
        # Create subgraph with strong connections only
        strong_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                       if d['weight'] >= min_similarity]
        subgraph = self.graph.edge_subgraph(strong_edges)
        
        # Detect communities
        communities = list(nx.community.greedy_modularity_communities(subgraph))
        return [list(community) for community in communities]
    
    def get_paper_centrality(self) -> Dict[str, float]:
        """Calculate centrality scores for all papers"""
        if len(self.graph.edges()) == 0:
            return {node: 0.0 for node in self.graph.nodes()}
        
        # Weighted degree centrality
        centrality = {}
        for node in self.graph.nodes():
            total_weight = sum(d['weight'] for _, _, d 
                             in self.graph.edges(node, data=True))
            centrality[node] = total_weight
        
        return centrality


class LLMAnalyzer:
    """LLM for knowledge extraction and comparison"""
    
    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print(f"Loading {model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, device_map="auto", cache_dir=CACHE_DIR
        )
        print("Model loaded!")
    
    def extract_knowledge(self, text: str, title: str = "") -> Dict:
        """Extract structured knowledge from text"""
        prompt = f"""Extract information from this research paper and return ONLY valid JSON.

            Title: {title}
            Content: {text}

            Return this JSON structure:
            {{
                "core_ideas": ["list of main ideas"],
                "methods": ["list of methods"],
                "contributions": ["list of contributions"],
                "keywords": ["list of keywords"],
                "data_sources": ["list of data sources"],
                "experiments": ["list of experiments"]
            }}

            Return ONLY the JSON, no explanation."""

        response = self._generate(prompt, max_tokens=1000)
        print("Knowledge extraction response:", response)
        return self._parse_json(response, {
            "core_ideas": [], "methods": [], "contributions": [], "keywords": [], "data_sources": [], "experiments": []
        })
    
    def calculate_similarity(self, paper1: Dict, paper2: Dict) -> tuple[float, str]:
        """Calculate similarity between two papers"""
        prompt = f"""Compare these papers for ideas plagiarism and return ONLY valid JSON.

            Paper 1: {paper1['title']}
            Ideas: {'; '.join(paper1.get('core_ideas', []))}
            Methods: {'; '.join(paper1.get('methods', []))}
            Contributions: {'; '.join(paper1.get('contributions', []))}
            Keywords: {'; '.join(paper1.get('keywords', []))}
            Data Sources: {'; '.join(paper1.get('data_sources', []))}
            Experiments: {'; '.join(paper1.get('experiments', []))}

            Paper 2: {paper2['title']}
            Ideas: {'; '.join(paper2.get('core_ideas', []))}
            Methods: {'; '.join(paper2.get('methods', []))}
            Contributions: {'; '.join(paper2.get('contributions', []))}
            Keywords: {'; '.join(paper2.get('keywords', []))}
            Data Sources: {'; '.join(paper2.get('data_sources', []))}
            Experiments: {'; '.join(paper2.get('experiments', []))}

            Return this JSON:
            {{
                "overlap_percentage": 45,
                "similarity_aspects": "brief explanation"
            }}

            Return ONLY the JSON."""

        response = self._generate(prompt, max_tokens=200)
        result = self._parse_json(response, {"overlap_percentage": 0, "similarity_reason": "Unknown"})
        print("Similarity response:", result)
        return float(result.get('overlap_percentage', 0)), result.get('similarity_reason', 'Unknown')
    
    def _generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate LLM response"""
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=0.3,
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    def _parse_json(self, text: str, default: Dict) -> Dict:
        """Extract and parse JSON from response"""
        text = text.strip()
        
        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        # Extract JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        
        try:
            return json.loads(text.strip())
        except:
            return default


class ManuscriptAnalyzer:
    """Main analyzer"""
    
    def __init__(self, semantic_scholar_key: Optional[str] = None):
        self.llm = LLMAnalyzer()
        self.s2_key = semantic_scholar_key
        self.pdf_dl = PDFDownloader()
        self.kg = KnowledgeGraph()
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def analyze(self, manuscript_text, top_k: int = 5) -> str:
        """Analyze manuscript and generate report"""
        print("=" * 80)
        print("MANUSCRIPT SIMILARITY ANALYZER")
        print("=" * 80)
        
        # Load manuscript
        print("\n[1/3] Loading manuscript...")
        print(f"✓ Loaded {len(manuscript_text)} characters")
        
        # Extract manuscript knowledge
        print("\n[2/3] Extracting manuscript knowledge...")
        ms_knowledge = self.llm.extract_knowledge(manuscript_text, "New Manuscript")
        ms_knowledge['title'] = "New Manuscript"
        print(f"✓ Extracted {len(ms_knowledge['core_ideas'])} ideas, {len(ms_knowledge['methods'])} methods")
        print(ms_knowledge)
        
        # Search for similar papers
        papers = []
        queries = []
        # 1. Core idea (primary novelty signal)
        if ms_knowledge.get('core_ideas'):
            queries.append(
                ' '.join(ms_knowledge['core_ideas'][:3])
            )

        # 2. Method-centric query
        if ms_knowledge.get('methods'):
            queries.append(
                ' '.join(ms_knowledge['methods'][:3])
            )

        # 3. Idea + Method (captures methodological novelty)
        if ms_knowledge.get('core_ideas') and ms_knowledge.get('methods'):
            queries.append(
                f"{ms_knowledge['core_ideas'][0]} {ms_knowledge['methods'][0]}"
            )

        # 4. Contribution-oriented query
        if ms_knowledge.get('contributions'):
            queries.append(
                ' '.join(ms_knowledge['contributions'][:2])
            )

        # 5. Application / data-driven query
        if ms_knowledge.get('data_sources'):
            queries.append(
                f"{ms_knowledge['core_ideas'][0]} {ms_knowledge['data_sources'][0]}"
            )

        # 6. Experimental setup (helps detect incremental work)
        if ms_knowledge.get('experiments'):
            queries.append(
                ' '.join(ms_knowledge['experiments'][:2])
            )

        # 7. Keywords fallback (but expanded)
        if ms_knowledge.get('keywords'):
            queries.append(
                ' '.join(ms_knowledge['keywords'][:5])
            )

        # Deduplicate & truncate
        queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))

        for query in queries:
            print(f"Searching Semantic Scholar for: {query}...")
            query_papers = self._search_papers(query, limit=5)
            papers += query_papers
            time.sleep(1)  # Rate limiting

        # Remove duplicates and sort by citations
        unique_papers = {paper['paperId']: paper for paper in papers}
        papers = sorted(unique_papers.values(), 
                        key=lambda x: x.get('citationCount', 0), 
                        reverse=True)[:20]  # Keep top 20 by citations 

        print(f"✓ Found {len(papers)} papers")
        
        # Analyze papers and calculate similarities
        similarities = []
        for i, paper in enumerate(papers, 1):
            print(f"  [{i}/{len(papers)}] {paper['title']}...")
            
            # Try to get full text
            text = self.pdf_dl.download_paper(paper['paperId'], paper.get('url', ''))
            if not text:
                print(" No full text available, using abstract/title...")
                text = paper.get('abstract', paper['title'])
            
            if not text or len(text) < 200:
                print(" Text too short, skipping...")
                continue
            print(f"    Paper text length: {len(text)} characters")
            print(f"    Extracting knowledge...")
            # Extract knowledge
            knowledge = self.llm.extract_knowledge(text, paper['title'])
            knowledge['title'] = paper['title']
            knowledge['year'] = paper.get('year')
            knowledge['citations'] = paper.get('citationCount', 0)
            knowledge['url'] = paper.get('url')
            print("knowledge:", knowledge)
            # Calculate similarity
            if len(knowledge['methods']) != 0:
                similarity_score = calc_similarity_score(ms_knowledge['core_ideas']+ ms_knowledge['methods']+ ms_knowledge['contributions']+
                ms_knowledge['data_sources'] + ms_knowledge['experiments'],
                knowledge['core_ideas']+ knowledge['methods']+ knowledge['contributions']+ knowledge['data_sources'] + knowledge['experiments']) 
            
            overlap, reason = self.llm.calculate_similarity(ms_knowledge, knowledge)
            similarities.append((similarity_score, reason, knowledge))
            time.sleep(0.5)
        
        # Build knowledge graph
        print("\n[4/4] Building knowledge graph...")
        ms_id = "manuscript"
        ms_embedding = self.sentence_model.encode(
            ' '.join(ms_knowledge['core_ideas'] + ms_knowledge['methods'])
        )
        self.kg.add_paper(ms_id, ms_knowledge, ms_embedding)

        for similarity_score, reason, knowledge in similarities:
            paper_id = knowledge.get('title', '')[:50]  # Use truncated title as ID
            paper_embedding = self.sentence_model.encode(
                ' '.join(knowledge['core_ideas'] + knowledge['methods'])
            )
            self.kg.add_paper(paper_id, knowledge, paper_embedding)
            self.kg.add_similarity_edge(ms_id, paper_id, similarity_score, threshold=25.0)

        # Find similar papers using graph methods
        radius_similar = self.kg.find_most_similar_by_radius(ms_id, radius=35.0)
        print(f"✓ Found {len(radius_similar)} papers within similarity radius")

        # Get centrality scores
        centrality = self.kg.get_paper_centrality()
        print(f"✓ Calculated centrality for {len(centrality)} papers")

        # Combine rankings: similarity + centrality
        combined_scores = []
        for paper_id, sim_score, knowledge in radius_similar:
            cent_score = centrality.get(paper_id, 0)
            # Weighted combination: 70% similarity, 30% centrality
            combined = 0.7 * sim_score + 0.3 * (cent_score / 100)
            combined_scores.append((combined, sim_score, knowledge))

        combined_scores.sort(reverse=True, key=lambda x: x[0])

        # REPLACE the existing report generation line with:
        report = self._generate_report(ms_knowledge, combined_scores[:top_k])
        top_similarities = combined_scores[:top_k]

        self.visualize_graph()
        # Save report
        output_path = f"./outputs/similarity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        Path(output_path).write_text(report, encoding='utf-8')
        
        print(f"\n✓ Report saved to: {output_path}")
        
        return report, top_similarities

    def visualize_graph(self, output_path: str = "./outputs/knowledge_graph.png"):
        """Generate interactive graph visualization"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.kg.graph, k=2, iterations=50)
        
        # Draw nodes
        node_colors = ['red' if node == 'manuscript' else 'lightblue' 
                    for node in self.kg.graph.nodes()]
        nx.draw_networkx_nodes(self.kg.graph, pos, node_color=node_colors, 
                            node_size=500, alpha=0.8)
        
        # Draw edges with weights
        edges = self.kg.graph.edges()
        weights = [self.kg.graph[u][v]['weight']/10 for u, v in edges]
        nx.draw_networkx_edges(self.kg.graph, pos, width=weights, alpha=0.5)
        
        # Labels
        labels = {node: self.kg.graph.nodes[node]['title'][:30] 
                for node in self.kg.graph.nodes()}
        nx.draw_networkx_labels(self.kg.graph, pos, labels, font_size=8)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graph saved to: {output_path}")
    
    def _search_papers(self, query: str, limit: int = 15) -> List[Dict]:
        """Search Semantic Scholar"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query[:500],
            'limit': limit,
            'fields': 'paperId,title,year,authors,abstract,citationCount,url'
        }
        headers = {'x-api-key': self.s2_key} if self.s2_key else {}
        
        
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        papers = resp.json().get('data', [])
        print(f"Found papers for query: ", len(papers))
        print([p['title'] for p in papers])
        return papers
        
    
    def _generate_report(self, manuscript: Dict, similarities: List) -> str:
        """Generate similarity report"""
        lines = [
            "=" * 80,
            "MANUSCRIPT SIMILARITY REPORT",
            "=" * 80,
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n" + "-" * 80,
            "YOUR MANUSCRIPT",
            "-" * 80,
            f"\nCore Ideas: {', '.join(manuscript['core_ideas'][:5])}",
            f"Methods: {', '.join(manuscript['methods'][:4])}",
            f"Keywords: {', '.join(manuscript['keywords'][:7])}\n",
            "\n" + "=" * 80,
            "TOP 5 MOST SIMILAR PAPERS",
            "=" * 80
        ]
        
        for rank, (overlap, reason, paper) in enumerate(similarities, 1):
            lines.extend([
                f"\n{'-' * 80}",
                f"RANK #{rank} - {overlap:.1f}% SIMILARITY",
                f"{'-' * 80}",
                f"\nTitle: {paper['title']}",
                f"Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citations', 0)}",
                f"URL: {paper.get('url', 'N/A')}",
                f"\nSimilarity: {reason}",
                f"\nTheir Ideas: {', '.join(paper.get('core_ideas', [])[:4])}",
                f"Their Methods: {', '.join(paper.get('methods', [])[:3])}"
            ])
        
        lines.extend(["\n" + "=" * 80, "END OF REPORT", "=" * 80])
        return "\n".join(lines)


    def llm_novelty_check(self, manuscript_text, top_similar_papers: List = []):

        model_name = "AbeerMostafa/Novelty_Reviewer"

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, legacy=False, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True  # Required for some custom models
        )
        model.eval()


        USER_PROMPT_TEMPLATE = """
        You are an expert reviewer for AI conferences. 
        Your task is to evaluate the NOVELTY of research papers according to reviewer guidelines. Be factual, concise, and balanced.
        You should carefully read the paper, judge whether the ideas are novel or not, 
        and provide a concise justification based only on the content of the paper.

        Return two things: Novelty Score and Short Novelty Review.
        You MUST Provide your evaluation in the following format:
        Novelty Score: 
        -1 = Not novel (work is incremental, derivative, or replicating existing approaches with minimal innovation.)
        0 = Limited novelty (work is somewhat standard, showing minor variations or applications of known methods without substantial conceptual or technical innovation.)
        1 = Moderately novel (some originality but overlap with prior work, or extension/combination of existing ideas.)
        2 = Highly novel (fundamentally new ideas, approaches, problem formulations, or insights that significantly advance the field.)

        Short Novelty Review: A 3-5 sentence containing your summary and reasoning about paper novelty/originality.
        Review the following paper for novelty:

        {paper_text}

        take into account these similar papers:
        {similar_papers}

        
        """
   
        similar_papers_info = []
        for similarity_score, reason, knowledge in top_similar_papers:
            title = knowledge.get('title', 'Unknown')
            core_ideas = ', '.join(knowledge.get('core_ideas', [])[:3])
            methods = ', '.join(knowledge.get('methods', [])[:3])
            contributions = ', '.join(knowledge.get('contributions', [])[:2])
            keywords = ', '.join(knowledge.get('keywords', [])[:3])
            
            paper_summary = f"- {title}:\n"
            paper_summary += f"  Core Ideas: {core_ideas}\n"
            paper_summary += f"  Methods: {methods}\n"
            paper_summary += f"  Contributions: {contributions}\n"
            paper_summary += f"  Keywords: {keywords}"
            
            similar_papers_info.append(paper_summary)

        messages = [
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                paper_text=manuscript_text,
                similar_papers="\n\n".join(similar_papers_info)
            )}
        ]

        tokenized_output = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True, # Important for generation
            return_tensors="pt" # Return PyTorch tensors
        )

        in_id = tokenized_output[0].tolist()
        attn_mask = [1] * len(tokenized_output[0].tolist())

        input_ids = torch.tensor(in_id).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(attn_mask).unsqueeze(0).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.1,
                top_p=0.5,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        print(f"Generated Text: {generated_text}")

        return generated_text




if __name__ == "__main__":
    SEMANTIC_SCHOLAR_KEY = "1Vri9rxlc41Ke1OKCzL839Rs8Cljpmww1MPvXxRp"

    df_test = pl.read_parquet("test_dataset.parquet")
    ds_test = hfds.Dataset.from_polars(df_test)
    ds_test = ds_test.select(range(1))

    OUTPUT_BASE_DIR = Path("test_outputs_full_pipeline")
    OUTPUT_BASE_DIR.mkdir(exist_ok=True)
    
    analyzer = ManuscriptAnalyzer(semantic_scholar_key=SEMANTIC_SCHOLAR_KEY)

    generated_results = []

    for i, example in enumerate(tqdm(ds_test)):
        paper_id = example['paper_id']
        novelty_summary = example['novelty_summary']
        novelty_score = example['novelty_score']
        manuscript_text = example['paper_text']
        report, top_similarities = analyzer.analyze(manuscript_text, top_k=7)
        #print("\n" + report)

        generated_text = analyzer.llm_novelty_check(manuscript_text, top_similarities)

        generated_results.append({
            "paper_id": paper_id,
            "novelty_summary": novelty_summary,
            "novelty_score": novelty_score,
            "generated_text": generated_text
        })

    
    df_output = pl.DataFrame(generated_results)
    output_path = OUTPUT_BASE_DIR / "results.parquet"
    df_output.write_parquet(output_path)
