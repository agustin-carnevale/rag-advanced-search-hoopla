
import os
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