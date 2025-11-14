import json
import os

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75

# Define project-level paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT,"data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOCS_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")

EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


def load_movies() -> list[dict]:
  try:
    with open(DATA_PATH, "r") as f:
      data = json.load(f)
    return data["movies"]
  except FileNotFoundError:
      print(f"Error: The file '{DATA_PATH}' was not found.")
  except Exception as e:
      print(f"An error occurred: {e}")

def load_stop_words() -> list[str]:
  try:
    with open(STOP_WORDS_PATH, "r") as f:
      lines = f.readlines()    
    # remove newline characters from each line
    lines = [line.strip() for line in lines]
    
    return lines
  except FileNotFoundError:
    print(f"Error: The file '{STOP_WORDS_PATH}' was not found.")
  except Exception as e:
    print(f"An error occurred: {e}")
