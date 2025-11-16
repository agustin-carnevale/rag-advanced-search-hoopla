
import os
import textwrap
from sentence_transformers import SentenceTransformer
import numpy as np

from cli.lib.search_utils import EMBEDDINGS_PATH, load_movies

class SemanticSearch:
  def __init__(self):
    # Load the model (downloads automatically the first time)
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    self.embeddings = None
    self.documents = None
    self.document_map = {}
    pass
  
  def generate_embedding(self, text: str):
    text = text.strip()
    if len(text) == 0:
      raise ValueError("Invalid text input: empty text.")
    
    output = self.model.encode(text)
    if len(output) == 0:
      raise ValueError("Error creating embedding. No output was returned.")
    
    return output
  
  def build_embeddings(self, documents: list[dict]):
    self.documents = documents
    for doc in documents:
      self.document_map[doc["id"]] = doc

    doc_strings = []
    for doc in documents:
      doc_strings.append(f"{doc['title']}: {doc['description']}")
    
    self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
    
    with open(EMBEDDINGS_PATH, 'wb') as f:
      np.save(f, self.embeddings)
    
    return self.embeddings
  
  def load_or_create_embeddings(self, documents: list[dict]):
    self.documents = documents
    for doc in documents:
      self.document_map[doc["id"]] = doc
      
    if os.path.exists(EMBEDDINGS_PATH):
      with open(EMBEDDINGS_PATH, 'rb') as f:
        self.embeddings = np.load(f)

      if len(self.embeddings) == len(self.documents):
        return self.embeddings
    
    return self.build_embeddings(documents)

  def search(self, query, limit):
    if self.embeddings is None or self.embeddings.size == 0:
      raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
    
    query_embedding = self.generate_embedding(query)
    
    similarities = []
    for i, doc_embedding in enumerate(self.embeddings):
      similarity = cosine_similarity(query_embedding,doc_embedding)
      doc = self.documents[i]
      similarities.append((similarity, doc))
      
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    results = []
    for item in similarities[:limit]:
      results.append({
        "score": item[0],
        "title": item[1]["title"],
        "description":  item[1]["description"]
      })
      
    return results
    
    
      


def verify_model():
  ss = SemanticSearch()
  print(f"Model loaded: {ss.model._model_config}")
  print(f"Max sequence length: {ss.model.max_seq_length}")  

def embed_text(text: str):
  ss = SemanticSearch()
  embedding = ss.generate_embedding(text)
  
  print(f"Text: {text}")
  # print(f"Embedding: {embedding}")
  print(f"First 3 dimensions: {embedding[:3]}")
  print(f"Dimensions: {embedding.shape[0]}")
  
def verify_embeddings():
  ss = SemanticSearch()
  documents = load_movies()  
  embeddings = ss.load_or_create_embeddings(documents)

  print(f"Number of docs:   {len(documents)}")
  print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
  
  
def embed_query_text(query: str):
  ss = SemanticSearch()
  embedding = ss.generate_embedding(query)
  
  print(f"Query: {query}")
  print(f"First 5 dimensions: {embedding[:5]}")
  print(f"Shape: {embedding.shape}")
  
  
def cosine_similarity(vec1, vec2) -> float:
  dot_product = np.dot(vec1, vec2)
  norm1 = np.linalg.norm(vec1)
  norm2 = np.linalg.norm(vec2)

  if norm1 == 0 or norm2 == 0:
      return 0.0

  return dot_product / (norm1 * norm2)


def search_query(query, limit):
  ss = SemanticSearch()
  documents = load_movies()  
  ss.load_or_create_embeddings(documents)
  
  results = ss.search(query, limit)
  
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
    
  
def chunk_text(text: str, chunk_size: int):
  print(f"Chunking {len(text)} characters")
  
  list_of_words = text.rsplit()
  
  chunks = []
  for i in range(0, len(list_of_words), chunk_size):
      chunk = " ".join(list_of_words[i:i + chunk_size])
      chunks.append(chunk)

  for i, w in enumerate(chunks,1):
    print(f"{i}. {w}")