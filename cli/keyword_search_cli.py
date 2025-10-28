#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add project root to path to allow imports to work when running as script 
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from cli.lib.search_keyword import InvertedIndex, search_cmd

def main() -> None:
  parser = argparse.ArgumentParser(description="Keyword Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  search_parser = subparsers.add_parser("search", help="Search movies using BM25")
  search_parser.add_argument("query", type=str, help="Search query")

  subparsers.add_parser("build", help="build project inverted index")
  
  args = parser.parse_args()

  match args.command:
    case "search":
      query = args.query
      print(f"Searching for: {query}")
      results = search_cmd(query)
      
      for i, movie in enumerate(results,1):
        print(f"{i}. {movie["title"]} ID:{movie["id"]}")
      pass
    case "build":
      inverted_index = InvertedIndex()
      inverted_index.build()
      inverted_index.save()
      # docs = inverted_index.get_documents("merida")
      # print(f"First document for token 'merida' = {docs[0]}")
      pass
    case _:
      parser.print_help()


if __name__ == "__main__":
   main()
    
