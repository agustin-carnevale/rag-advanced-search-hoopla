import json
import os

DEFAULT_SEARCH_LIMIT = 5

# Define project-level paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT,"data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")



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
