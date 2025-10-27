from .search_utils import (
  DEFAULT_SEARCH_LIMIT,
  load_movies,
  load_stop_words,
)

import string


def search_cmd(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
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
  
  # remove empty strings "", " ", etc
  tokens = [s for s in tokens if s.strip() != "" and s not in stops_words]
  
  return tokens


