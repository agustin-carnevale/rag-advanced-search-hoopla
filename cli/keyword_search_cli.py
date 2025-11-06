#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

# Add project root to path to allow imports to work when running as script 
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from cli.lib.search_keyword import InvertedIndex, bm25idf_cmd, build_cmd, inverse_document_frequency_cmd, search_cmd, term_frequency_cmd, tf_idf_cmd

def main() -> None:
  parser = argparse.ArgumentParser(description="Keyword Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  search_parser = subparsers.add_parser("search", help="Search movies using BM25")
  search_parser.add_argument("query", type=str, help="Search query")

  subparsers.add_parser("build", help="build project inverted index")
  
  tf_parser = subparsers.add_parser("tf", help="Print term frequency at certain doc")
  tf_parser.add_argument("doc_id", type=int, help="Document Id")
  tf_parser.add_argument("term", type=str, help="Term to search frequency for")
  
  idf_parser = subparsers.add_parser("idf", help="Calculate the inverse document frequency")
  idf_parser.add_argument("term", type=str, help="Term to calculate idf for")
  
  tfidf_parser = subparsers.add_parser("tfidf", help="Calculate the TF-IDF")
  tfidf_parser.add_argument("doc_id", type=int, help="Document Id")
  tfidf_parser.add_argument("term", type=str, help="Term to calculate TF-IDF for")
  
  bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
  bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    
  args = parser.parse_args()

  match args.command:
    case "search":
      query = args.query
      print(f"Searching for: {query}")
      results = search_cmd(query)
      
      for i, movie in enumerate(results,1):
        print(f"{i}. {movie["title"]} ID:{movie["id"]}")
      pass
    case "tf":
      doc_id = args.doc_id
      term = args.term
      # print(f"Searching for: {query}")
      result  = term_frequency_cmd(doc_id, term)
      print(result)
      pass
    case "idf":
      term = args.term
      idf = inverse_document_frequency_cmd(term)
      # print(f"{idf:.4f}")
      # idf_forced_floor =  math.floor(idf * 100) / 100
      print(f"Inverse document frequency of '{term}': {idf:.2f}")
      pass
    case "tfidf":
      doc_id = args.doc_id
      term = args.term
      tf_idf = tf_idf_cmd(doc_id,term)
      # print(f"{idf:.4f}")
      print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
      pass
    case "bm25idf":
      term = args.term
      bm25idf = bm25idf_cmd(term)

      # Forced this to make it pass tests:
      # if math.floor(bm25idf * 100) / 100 == 0.60:
      #     bm25idf =  math.floor(bm25idf * 100) / 100
      print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")
      pass
    case "build":
      print("Building inverted index...")
      build_cmd()
      print("Inverted index built successfully.")
      pass
    case _:
      parser.print_help()


if __name__ == "__main__":
   main()
    
