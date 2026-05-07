import networkx as nx
import json
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load data
with open("extracted_features.json") as f:
    papers = json.load(f)

# Build paper similarity matrix
all_concepts = set()
for p in papers:
    all_concepts.update(p.get("core_ideas", []))
    all_concepts.update(p.get("methods", []))
    all_concepts.update(p.get("innovations", []))

concept_list = list(all_concepts)
vectors = []
for p in papers:
    vec = [0] * len(concept_list)
    p_concepts = set(p.get("core_ideas", []) + p.get("methods", []) + p.get("innovations", []))
    for i, c in enumerate(concept_list):
        if c in p_concepts:
            vec[i] = 1
    vectors.append(vec)

sim_matrix = cosine_similarity(vectors)

# Find similar papers
print("PAPER SIMILARITIES:\n")
for i in range(len(papers)):
    for j in range(i+1, len(papers)):
        if sim_matrix[i][j] > 0.1:
            print(f"{papers[i].get('title', papers[i]['filename'])[:60]}")
            print(f"  <-> {papers[j].get('title', papers[j]['filename'])[:60]}")
            print(f"  Similarity: {sim_matrix[i][j]:.2f}\n")

# Most common concepts
all_ideas = []
all_methods = []
for p in papers:
    all_ideas.extend(p.get("core_ideas", []))
    all_methods.extend(p.get("methods", []))

print("\nTOP CORE IDEAS:")
for idea, count in Counter(all_ideas).most_common(5):
    print(f"  {idea}: {count} papers")

print("\nTOP METHODS:")
for method, count in Counter(all_methods).most_common(5):
    print(f"  {method}: {count} papers")