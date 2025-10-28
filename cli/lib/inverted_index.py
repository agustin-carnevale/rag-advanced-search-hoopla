import os
import pickle
from cli.lib.search_keyword import tokenize_text
from cli.lib.search_utils import load_movies, load_stop_words

# Define project-level paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")


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

        if os.path.exists(DOCMAP_PATH):
            with open(DOCMAP_PATH, "rb") as f:
                self.docmap = pickle.load(f)
