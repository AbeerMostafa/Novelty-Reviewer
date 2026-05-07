import os
import json
import requests
from pathlib import Path
import networkx as nx



MODEL = "meta-llama/llama-3.1-8b-instruct"
MARKDOWN_DIR = "markdown_output"

PROMPT = """Extract the following from this paper in JSON format:
{
  "title": "paper title",
  "core_ideas": ["idea1", "idea2"],
  "methods": ["method1", "method2"],
  "problems": ["problem1", "problem2"],
  "results": ["result1", "result2"],
  "builds_on": ["prior1", "prior2"],
  "innovations": ["innovation1", "innovation2"]
}
Return ONLY valid JSON.

Paper:
"""

def extract_features(paper_text):
    """Call Llama 3.1 via OpenRouter to extract features"""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT + paper_text[:6000]}],
            "temperature": 0.1
        }
    )
    result = response.json()
    print(result)
    
    content = result["choices"][0]["message"]["content"]
    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    return json.loads(content[json_start:json_end])

def build_graph(papers_data):
    """Build NetworkX knowledge graph"""
    G = nx.DiGraph()
    
    for paper in papers_data:
        title = paper.get("title", paper["filename"])
        G.add_node(title, type="PAPER")
        
        for idea in paper.get("core_ideas", []):
            G.add_node(idea, type="CORE_IDEA")
            G.add_edge(title, idea, relation="HAS_CORE_IDEA")
        
        for method in paper.get("methods", []):
            G.add_node(method, type="METHOD")
            G.add_edge(title, method, relation="USES_METHOD")
        
        for problem in paper.get("problems", []):
            G.add_node(problem, type="PROBLEM")
            G.add_edge(title, problem, relation="ADDRESSES_PROBLEM")
        
        for result in paper.get("results", []):
            G.add_node(result, type="RESULT")
            G.add_edge(title, result, relation="REPORTS_RESULT")
        
        for prior in paper.get("builds_on", []):
            G.add_node(prior, type="PRIOR_WORK")
            G.add_edge(title, prior, relation="BUILDS_ON")
        
        for innovation in paper.get("innovations", []):
            G.add_node(innovation, type="INNOVATION")
            G.add_edge(title, innovation, relation="INTRODUCES")
    
    return G

# Main execution
papers_data = []
for md_file in Path(MARKDOWN_DIR).glob("*.md"):
    print(f"Processing {md_file.name}...")
    text = md_file.read_text(encoding="utf-8")
    
    features = extract_features(text)
    features["filename"] = md_file.name
    papers_data.append(features)


# Build and save graph
G = build_graph(papers_data)
print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Save outputs
with open("extracted_features.json", "w") as f:
    json.dump(papers_data, f, indent=2)

nx.write_gexf(G, "knowledge_graph.gexf")  # For Gephi
nx.write_graphml(G, "knowledge_graph.graphml")  # For yEd, Cytoscape

# Export for Neo4j
with open("neo4j_nodes.csv", "w") as f:
    f.write("id,label,type\n")
    for node, data in G.nodes(data=True):
        f.write(f'"{node}","{node}",{data.get("type", "")}\n')

with open("neo4j_edges.csv", "w") as f:
    f.write("source,target,type\n")
    for u, v, data in G.edges(data=True):
        f.write(f'"{u}","{v}",{data.get("relation", "")}\n')

print("Saved: extracted_features.json, knowledge_graph.gexf, knowledge_graph.graphml, neo4j_*.csv")