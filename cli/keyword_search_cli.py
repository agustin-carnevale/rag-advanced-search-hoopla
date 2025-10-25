#!/usr/bin/env python3

import argparse
import json

from lib.search_keyword import search_cmd

def main() -> None:
  parser = argparse.ArgumentParser(description="Keyword Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  search_parser = subparsers.add_parser("search", help="Search movies using BM25")
  search_parser.add_argument("query", type=str, help="Search query")

  args = parser.parse_args()

  match args.command:
    case "search":
      query = args.query
      print(f"Searching for: {query}")
      results = search_cmd(query)
      
      for i, movie in enumerate(results,1):
        print(f"{i}. {movie["title"]}")
      pass
    case _:
      parser.print_help()


if __name__ == "__main__":
   main()
    
