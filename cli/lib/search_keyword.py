from .search_utils import (
  DEFAULT_SEARCH_LIMIT,
  load_movies,
)

import string


def search_cmd(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
  movies = load_movies()
  results = []
  for movie in movies:
    preprocessed_query = preprocess_text(query)
    preprocessed_title = preprocess_text(movie["title"])
    if preprocessed_query in preprocessed_title:
      results.append(movie)
      if len(results) >= limit:
          break
  return results


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