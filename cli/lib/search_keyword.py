from collections import defaultdict
import math
from typing import Counter
from .search_utils import (
  CACHE_DIR,
  DEFAULT_SEARCH_LIMIT,
  DOCMAP_PATH,
  INDEX_PATH,
  TERM_FREQUENCIES_PATH,
  DOCS_LENGTHS_PATH,
  load_movies,
  load_stop_words,
)

import string
from nltk.stem import PorterStemmer

import os
import pickle


def bm25_search_cmd(query: str, limit: int) -> list[dict]:
  idx = InvertedIndex()
  idx.load()
  
  return idx.bm25_search(query, limit)

def bm25tf_cmd(doc_id: int, term: str, k1: float, b: float) -> float:
  idx = InvertedIndex()
  idx.load()
  
  return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25idf_cmd(term) -> float:
  idx = InvertedIndex()
  idx.load()
  
  return idx.get_bm25_idf(term)

def tf_idf_cmd(doc_id: int, term: str) -> float:
  idx = InvertedIndex()
  idx.load()
  
  tf = idx.get_tf(doc_id, term)
  idf = idx.get_idf(term)
  
  return tf * idf

def inverse_document_frequency_cmd(term: str) -> float:
  idx = InvertedIndex()
  idx.load()
  
  return idx.get_idf(term)

# Look for term frequency at certain doc_id
def term_frequency_cmd(doc_id: int, term: str) -> int:
  idx = InvertedIndex()
  idx.load()
  # stop_words = load_stop_words()
  
  return idx.get_tf(doc_id, term)

# inverted index based search
def search_cmd(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
  idx = InvertedIndex()
  idx.load()
  stop_words = load_stop_words()
  
  query_tokens = tokenize_text(query, stop_words)
  
  results = []
  for t in query_tokens:
    docs = idx.get_documents(t)
    results.extend(docs)
    if len(results) > limit:
      break
  
  return results[:limit]

def build_cmd() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
  

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
    self.index: dict[str, set[int]] = defaultdict(set)
    self.docmap: dict[int, object] = {}
    self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
    self.doc_lengths: dict[int, int] = {}

  def __add_document(self, doc_id: int, text: str, stop_words: list[str]) -> None:
    tokens = tokenize_text(text, stop_words)
    for t in set(tokens):
      self.index[t].add(doc_id)
    
    # count frequency
    self.term_frequencies[doc_id].update(tokens)
    self.doc_lengths[doc_id] = len(tokens)

  def get_documents(self, term: str) -> list[object]:
    results = []
    doc_ids = self.index.get(term, set())
    for doc_id in doc_ids:
      doc = self.docmap.get(doc_id)
      if doc:
        results.append(doc)

    # sort results by ID (optional)
    results.sort(key=lambda d: d["id"])
    return results
  
  def __get_avg_doc_length(self) -> float:
    if len(self.doc_lengths) == 0:
      return 0.0
    
    lengths = self.doc_lengths.values()
    total_sum = sum(lengths)
    n = len(lengths)
    return total_sum / n

  
  def get_tf(self, doc_id, term):
    stop_words = load_stop_words()
    tokens = tokenize_text(term, stop_words)
    
    if len(tokens) == 0:
      return 0
    
    if len(tokens) > 1:
      raise ValueError(f"Error at get_tf(): term has too many tokens.")
      
    t = tokens[0]
    return self.term_frequencies[doc_id][t]
  
  
  def get_idf(self, term) -> float:
    stop_words = load_stop_words()
    tokens = tokenize_text(term, stop_words)
    
    doc_count = len(self.docmap)
    term_doc_count = 0
    if len(tokens) == 1:
      t = tokens[0]
      doc_ids_set = self.index.get(t)
      if (doc_ids_set):
        term_doc_count = len(doc_ids_set)
    
    return math.log((doc_count + 1) / (term_doc_count + 1))
  
  def get_bm25_idf(self, term: str) -> float:
    N = len(self.docmap)
    
    stop_words = load_stop_words()
    tokens = tokenize_text(term, stop_words)
    if len(tokens) != 1:
      raise ValueError(f"Error at get_bm25_idf(): term has not a single token")
    
    doc_ids = self.index.get(tokens[0])
    df = 0
    if doc_ids:
      df = len(doc_ids)
      
    return math.log((N - df + 0.5) / (df + 0.5) + 1)
  
  
  def get_bm25_tf(self, doc_id: int, term: str, k1: float, b: float) -> float:
    tf = self.get_tf(doc_id, term)
    
    avg_doc_length = self.__get_avg_doc_length()
    doc_length = self.doc_lengths[doc_id]
    
    # Length normalization factor
    length_norm = 1 - b + b * (doc_length /avg_doc_length)

    # Apply to term frequency
    bm25tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    return bm25tf
  
  def bm25(self, doc_id, term):
    bm25idf= self.get_bm25_idf(term)
    bm25tf = self.get_bm25_tf(doc_id, term, 1.5, 0.75)
    
    return bm25idf * bm25tf
  
  
  def bm25_search(self, query, limit):
    stop_words = load_stop_words()
    q_tokens = tokenize_text(query, stop_words)
    
    scores = {}
    for doc_id in self.docmap:
      score = 0
      for t in q_tokens:
        score += self.bm25(doc_id, t)
      scores[doc_id] = score
    
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    top_results = sorted_docs[:limit]
    
    enriched_results = [
      {"id": doc_id, "score": score, "movie": self.docmap[doc_id]}
      for doc_id, score in top_results
    ]

    return enriched_results

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
      
    with open(TERM_FREQUENCIES_PATH, "wb") as f:
      pickle.dump(self.term_frequencies, f)
      
    with open(DOCS_LENGTHS_PATH, "wb") as f:
      pickle.dump(self.doc_lengths, f)

  def load(self) -> None:
    """Load index, docmap, term_frequencies from disk if they exist."""
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
    
    if os.path.exists(TERM_FREQUENCIES_PATH):
      with open(TERM_FREQUENCIES_PATH, "rb") as f:
        self.term_frequencies = pickle.load(f)
    else:
      raise ValueError(f"Loading failed: '{TERM_FREQUENCIES_PATH}' is missing.")

    if os.path.exists(DOCS_LENGTHS_PATH):
      with open(DOCS_LENGTHS_PATH, "rb") as f:
        self.doc_lengths = pickle.load(f)
    else:
      raise ValueError(f"Loading failed: '{DOCS_LENGTHS_PATH}' is missing.")
    
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

def tokenize_text(text: str, stop_words: list[str]) -> list[str]:
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
  tokens = text.split()
  
  # remove empty strings "", " ", and stop words like: "the", "in", etc
  tokens = [s for s in tokens if s.strip() != "" and s not in stop_words]
  
  # reduce each token to its root (stemmed form)
  stemmer = PorterStemmer()
  tokens = [stemmer.stem(t) for t in tokens]
  
  return tokens


