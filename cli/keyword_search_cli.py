#!/usr/bin/env python3

import argparse
import json


def search_cmd(query: str) -> list[str]:
  with open('data/movies.json', 'r') as f:
    data = json.load(f)
    
  # print(type(data["movies"]))
  movies = data["movies"]
  
  results = []
  
  for movie in movies:
    title = movie["title"]
    if query in title:
      results.append(title)
      if len(results) == 5:
        break
  
  return results
      
        
    


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
            
            for i, movie in enumerate(results):
              print(f"{i+1}. {movie}")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
    
