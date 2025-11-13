
from sentence_transformers import SentenceTransformer

class SemanticSearch:
  def __init__(self):
    # Load the model (downloads automatically the first time)
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    pass


def verify_model():
  ss = SemanticSearch()
  print(f"Model loaded: {ss.model._model_config}")
  print(f"Max sequence length: {ss.model.max_seq_length}")  
  # model.encode(text)