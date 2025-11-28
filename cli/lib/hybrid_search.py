import os

from cli.lib.chunked_semantic_search import ChunkedSemanticSearch
from cli.lib.search_keyword import InvertedIndex

class HybridSearch:
  def __init__(self, documents):
    self.documents = documents
    self.semantic_search = ChunkedSemanticSearch()
    self.semantic_search.load_or_create_chunk_embeddings(documents)

    self.idx = InvertedIndex()
    if not os.path.exists(self.idx.index_path):
      self.idx.build()
      self.idx.save()

  def _bm25_search(self, query, limit):
    self.idx.load()
    return self.idx.bm25_search(query, limit)

  def weighted_search(self, query, alpha, limit=5):
    raise NotImplementedError("Weighted hybrid search is not implemented yet.")

  def rrf_search(self, query, k, limit=10):
    raise NotImplementedError("RRF hybrid search is not implemented yet.")
  
  
  
  
def normalize_scores(inputs: list[float]):
  if len(inputs) == 0:
    return
  
  min_value = min(inputs)
  max_value = max(inputs)
  max_min_diff = max_value - min_value
  
  # print(f"min: {min_value}")
  # print(f"max: {max_value}")

  if max_min_diff == 0:
    for _ in inputs:
      score = 1.0
      print(f"* {score:.4f}")
    return
  
  for input in inputs:
    score = (input - min_value) / max_min_diff
    print(f"* {score:.4f}")
  
  
    