from .search_utils import (
  CACHE_DIR,
  DEFAULT_SEARCH_LIMIT,
  DOCMAP_PATH,
  INDEX_PATH,
  load_movies,
  load_stop_words,
)

import string
from nltk.stem import PorterStemmer

import os
import pickle


# inverted index based search
def search_cmd(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
  idx = InvertedIndex()
  idx.load()
  stops_words = load_stop_words()
  
  query_tokens = tokenize_text(query, stops_words)
  
  results = []
  for t in query_tokens:
    docs = idx.get_documents(t)
    results.extend(docs)
    if len(results) > limit:
      break
  
  return results[:limit]
  

# basic comparison to query iterating movie
def search_cmd_basic(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
  movies = load_movies()
  stops_words = load_stop_words()
  results = []
  
  query_tokens = tokenize_text(query, stops_words)
  for movie in movies:
    title_tokens = tokenize_text(movie["title"], stops_words)
    
    if has_matching_token(query_tokens, title_tokens):
      results.append(movie)
      if len(results) >= limit:
        break
      
  return results


class InvertedIndex:
  def __init__(self):
    self.index: dict[str, set[int]] = {}
    self.docmap: dict[int, object] = {}

  def __add_document(self, doc_id: int, text: str, stop_words: set[str]) -> None:
    tokens = tokenize_text(text, stop_words)
    for t in tokens:
      if t not in self.index:
        self.index[t] = set()
      self.index[t].add(doc_id)

  def get_documents(self, term: str) -> list[object]:
    results = []
    doc_ids = self.index.get(term)
    if doc_ids:
      for doc_id in doc_ids:
        doc = self.docmap.get(doc_id)
        if doc:
          results.append(doc)

    # sort results by ID (optional)
    results.sort(key=lambda d: d["id"])
    return results

  def build(self) -> None:
    movies = load_movies()
    stop_words = load_stop_words()

    for movie in movies:
      doc_id = movie["id"]
      text = f"{movie['title']} {movie['description']}"
      self.__add_document(doc_id, text, stop_words)
      self.docmap[doc_id] = movie

  def save(self) -> None:
    """Save index and docmap to disk using pickle."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open(INDEX_PATH, "wb") as f:
      pickle.dump(self.index, f)

    with open(DOCMAP_PATH, "wb") as f:
      pickle.dump(self.docmap, f)

  def load(self) -> None:
    """Load index and docmap from disk if they exist."""
    if os.path.exists(INDEX_PATH):
      with open(INDEX_PATH, "rb") as f:
        self.index = pickle.load(f)
    else:
      raise ValueError(f"Loading failed: '{INDEX_PATH}' is missing.")

    if os.path.exists(DOCMAP_PATH):
      with open(DOCMAP_PATH, "rb") as f:
        self.docmap = pickle.load(f)
    else:
      raise ValueError(f"Loading failed: '{DOCMAP_PATH}' is missing.")
    

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
  # Check if any token in list1 is a substring of any token in list2
  return any(q_token in t_token for q_token in query_tokens for t_token in title_tokens)


def preprocess_text(text: str) -> str:
  """
  Preprocesses text by converting to lowercase and removing special characters.
  
  Args:
    text: The text to preprocess
    
  Returns:
    The preprocessed text (lowercase with all punctuation removed)
  """
  # Convert to lowercase
  text = text.lower()
  
  # Create translation table to remove all punctuation
  translation_table = str.maketrans("", "", string.punctuation)
  
  # Remove punctuation using translation table
  text = text.translate(translation_table)
  
  return text

def tokenize_text(text: str, stops_words: list[str]) -> list[str]:
  """
  Splits text into valid tokens. Pre-processing text, and removing empty tokens.
  
  Args:
    text: The text to generate tokens from
    
  Returns:
    List of tokens for that text
  """
  
  # pre-process
  text = preprocess_text(text)
  
  # split into workds
  tokens = text.split(" ")
  
  # remove empty strings "", " ", and stop words like: "the", "in", etc
  tokens = [s for s in tokens if s.strip() != "" and s not in stops_words]
  
  # reduce each token to its root (stemmed form)
  stemmer = PorterStemmer()
  tokens = [stemmer.stem(t) for t in tokens]
  
  return tokens


