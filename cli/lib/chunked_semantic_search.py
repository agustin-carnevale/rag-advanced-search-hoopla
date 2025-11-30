import json
import os
import re
import textwrap

import numpy as np
from cli.lib.search_utils import CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH, load_movies
from cli.lib.semantic_search import SemanticSearch, cosine_similarity


class ChunkedSemanticSearch(SemanticSearch): 
  def __init__(self) -> None:
    super().__init__()
    self.chunk_embeddings = None
    self.chunk_metadata = None
    
  def build_chunk_embeddings(self, documents):
    self.documents = documents
    for doc in documents:
      self.document_map[doc["id"]] = doc
    
    chunks = []
    chunks_metadata = []
    for doc_index, doc in enumerate(self.documents):
      description = doc["description"].strip()
      if len(description) > 0:
        desc_chunks = semantic_chunk(description, 4, 1)
        chunks.extend(desc_chunks)
        for chunk_index, _ in enumerate(desc_chunks):
          chunks_metadata.append({
            "movie_idx": doc_index,
            "chunk_idx": chunk_index,
            "total_chunks": len(desc_chunks)
          })
          
          
    print(f"total_chunks: {len(chunks)}")      
    self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
    self.chunk_metadata = chunks_metadata
        
    with open(CHUNK_EMBEDDINGS_PATH, 'wb') as f:
      np.save(f, self.chunk_embeddings)
    
    with open(CHUNK_METADATA_PATH, 'w') as f:
      json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)
      
    return self.chunk_embeddings
        
        
  def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
    self.documents = documents
    for doc in documents:
      self.document_map[doc["id"]] = doc
    
    if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
      with open(CHUNK_EMBEDDINGS_PATH, 'rb') as f:
        self.chunk_embeddings = np.load(f)

      with open(CHUNK_METADATA_PATH, 'r') as f:
        data = json.load(f)
        self.chunk_metadata = data.get("chunks", [])
      
      return self.chunk_embeddings
      
    
    return self.build_chunk_embeddings(documents)
  
  def search_chunks(self, query: str, limit: int = 10):
    query_embedding = self.generate_embedding(query)
        
    scores = []
    for i, chunk_embedding in enumerate(self.chunk_embeddings):
      similarity_score = cosine_similarity(query_embedding,chunk_embedding)
      chunk_meta = self.chunk_metadata[i]
      scores.append({
          "chunk_idx": chunk_meta["chunk_idx"],
          "movie_idx": chunk_meta["movie_idx"],
          "score": similarity_score
      })
    
    movies_score_dic = {}
    
    for score_item in scores:
      movie_idx = score_item["movie_idx"]
      chunk_score = score_item["score"]
      if movie_idx not in movies_score_dic:
        movies_score_dic[movie_idx] = chunk_score
      elif chunk_score > movies_score_dic[movie_idx]:
        movies_score_dic[movie_idx] = chunk_score
    
    # Sort by value (score) in descending order
    movies_sorted_by_score = sorted(movies_score_dic.items(), key=lambda item: item[1], reverse=True)
    
    results = []
    for item in movies_sorted_by_score[:limit]:
      movie_idx = item[0]
      movie_score = item[1]
      doc = self.documents[movie_idx] 
      results.append({
        "score": movie_score,
        "title": doc["title"],
        "description":  doc["description"],
        "doc_id": movie_idx,
      })
      
    return results    
        
        
def semantic_chunk(text: str, max_chunk_size: int, overlap: int):
  # print(f"Semantically chunking {len(text)} characters")
  
  # remove leading and trailing whitespace
  text = text.strip()
  if len(text) == 0:
    return []
  
  
  raw_list_of_sentences = re.split(r"(?<=[.!?])\s+", text)
  list_of_sentences = []
  for s in raw_list_of_sentences:
    # remove leading and trailing whitespace
    sentence = s.strip()
    if len(sentence) > 0:
      list_of_sentences.append(sentence)
  
  chunks = []
  if len(list_of_sentences) > 0:
    chunk = " ".join(list_of_sentences[0:max_chunk_size])
    chunks.append(chunk)
    
  start = max_chunk_size - overlap
  while start+overlap < len(list_of_sentences):
    end = start + max_chunk_size
    chunk = " ".join(list_of_sentences[start:end])
    chunks.append(chunk)
    start = end - overlap
    
  return chunks
    

def embed_chunks_cmd():
  css = ChunkedSemanticSearch()
  documents = load_movies()  
  embeddings = css.load_or_create_chunk_embeddings(documents)

  print(f"Generated {len(embeddings)} chunked embeddings")
  
  
def search_chunked_cmd(query: str, limit: int):
  css = ChunkedSemanticSearch()
  documents = load_movies()  
  css.load_or_create_chunk_embeddings(documents)
  
  results = css.search_chunks(query, limit)
  
  for i, result in enumerate(results, 1):
    print(f"{i}. {result["title"]} (score: {result["score"]:.4f})")
    # wrap description to a single line of max width
    wrapped = textwrap.wrap(result["description"], width=80)
    if wrapped:
        short_desc = wrapped[0] + "..."
    else:
        short_desc = ""

    print(f"   {short_desc}")
    print()
    