
from sentence_transformers import SentenceTransformer

class SemanticSearch:
  def __init__(self):
    # Load the model (downloads automatically the first time)
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    pass
  
  def generate_embedding(self, text: str):
    text = text.strip()
    if len(text) == 0:
      raise ValueError("Invalid text input: empty text.")
    
    output = self.model.encode(text)
    if len(output) == 0:
      raise ValueError("Error creating embedding. No output was returned.")
    
    return output


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