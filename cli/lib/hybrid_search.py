import os
import textwrap

from cli.lib.chunked_semantic_search import ChunkedSemanticSearch
from cli.lib.search_keyword import InvertedIndex
from cli.lib.search_utils import INDEX_PATH, load_movies

class HybridSearch:
  def __init__(self, documents):
    self.documents = documents
    self.semantic_search = ChunkedSemanticSearch()
    self.semantic_search.load_or_create_chunk_embeddings(documents)

    self.idx = InvertedIndex()
    if not os.path.exists(INDEX_PATH):
      self.idx.build()
      self.idx.save()

  def _bm25_search(self, query, limit):
    self.idx.load()
    return self.idx.bm25_search(query, limit)

  def weighted_search(self, query, alpha, limit=5):
    search_limit = min(limit*500, len(self.documents))
    
    # print("search limit:" , search_limit)
  
    # keyword search
    keyword_results = self._bm25_search(query, search_limit)
    scores = [item["score"] for item in keyword_results]
    normalized_keyword_scores = normalize_scores(scores)
    for r, s in zip(keyword_results, normalized_keyword_scores):
      r["normalized_score"] = s

    # semantic search
    semantic_results = self.semantic_search.search_chunks(query, search_limit)
    scores = [item["score"] for item in semantic_results]
    normalized_semantic_scores = normalize_scores(scores)
    for r, s in zip(semantic_results, normalized_semantic_scores):
      r["normalized_score"] = s

    # normalize semantic scores
    semantic_scores = normalize_scores([r["score"] for r in semantic_results])
    for r, s in zip(semantic_results, semantic_scores):
        r["normalized_score"] = s

    # combine results from both keyword and semantic
    combined = {}  # dict of doc_id -> score info

    # add keyword results
    for item in keyword_results:
      doc_id = item["id"]
      score = item["normalized_score"]
      if doc_id not in combined:
          combined[doc_id] = {"norm_keyword_score": 0, "norm_semantic_score": 0}
      combined[doc_id]["norm_keyword_score"] = score

    # add semantic results
    for item in semantic_results:
      doc_id = item["doc_id"]
      score = item["normalized_score"]
      if doc_id not in combined:
          combined[doc_id] = {"norm_keyword_score": 0, "norm_semantic_score": 0}
      combined[doc_id]["norm_semantic_score"] = score

    # compute hybrid score for all docs
    for doc_id, scores in combined.items():
      kw = scores["norm_keyword_score"]
      sem = scores["norm_semantic_score"]
      scores["hybrid_score"] = hybrid_score(kw, sem, alpha)
    
    # sort by hybrid score (desc)
    sorted_docs = sorted(
      combined.items(),
      key=lambda x: x[1]["hybrid_score"],
      reverse=True
    )
    
    results = []
    for doc_id, scores in sorted_docs[:limit]:
      doc = self.documents[doc_id]
      results.append({
        "doc_id": doc_id,
        "title": doc["title"],
        "description": doc["description"],
        "keyword_score": scores["norm_keyword_score"],
        "semantic_score":  scores["norm_semantic_score"],
        "hybrid_score": scores["hybrid_score"]
      })
    
    return results

  def rrf_search(self, query, k, limit=10):
    raise NotImplementedError("RRF hybrid search is not implemented yet.")
  
  
def normalize_scores(inputs: list[float]) -> list[float]:
  if len(inputs) == 0:
    return []
  
  min_value = min(inputs)
  max_value = max(inputs)
  max_min_diff = max_value - min_value
  
  # print(f"min: {min_value}")
  # print(f"max: {max_value}")
  
  results = []
  if max_min_diff == 0:
    for _ in inputs:
      results.append(1.0) 
    return results
  
  for input in inputs:
    score = (input - min_value) / max_min_diff
    results.append(score)
    
  return results
  
def normalize_cmd(inputs: list[float]):
  scores = normalize_scores(inputs)
  for score in scores:
    print(f"* {score:.4f}")
  
  
# alpha (or "α") is just a constant that we can use to dynamically control 
# the weighting between the two scores

# Query Type	  Example	          Chosen Alpha	  Reason
# Exact match	  "The Revenant"	  0.8	            Title search needs keywords
# Conceptual	  "family movies"	  0.2	            Meaning matters more
# Mixed	        "2015 comedies"	  0.5	            Both year AND concept

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score
  
#This is why it's so important to tune your search system's constants based on 
# the types of data and queries you're working with in your application! It's not 
# a one-size-fits-all solution, but building configurability into your system 
# allows you to adjust it as needed.


def weighted_search_cmd(query: str, alpha: float, limit: int):
  documents = load_movies()  
  hs = HybridSearch(documents)
  
  results = hs.weighted_search(query, alpha, limit)
  
  for i, result in enumerate(results, 1):
    print(f"{i}. {result["title"]}")
    print(f"   Hybrid Score: {result["hybrid_score"]:.3f}")
    print(f"   BM25: {result["keyword_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}")
    # wrap description to a single line of max width
    wrapped = textwrap.wrap(result["description"], width=80)
    if wrapped:
        short_desc = wrapped[0] + "..."
    else:
        short_desc = ""

    print(f"   {short_desc}")
    