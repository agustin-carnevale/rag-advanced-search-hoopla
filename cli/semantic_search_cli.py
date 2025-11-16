#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add project root to path to allow imports to work when running as script 
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))


from cli.lib.semantic_search import chunk_text, embed_query_text, embed_text, search_query, verify_embeddings, verify_model

def main():
  parser = argparse.ArgumentParser(description="Semantic Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  subparsers.add_parser("verify", help="Verifies model is loaded")
 
  embed_text_parser = subparsers.add_parser("embed_text", help="Created embedding for the given text")
  embed_text_parser.add_argument("text", type=str, help="Input text for the embedding")
 
  subparsers.add_parser("verify_embeddings", help="Verifies embeddings exist if not creates them")
 
  embedquery_parser = subparsers.add_parser("embedquery", help="Created embedding for the given query")
  embedquery_parser.add_argument("query", type=str, help="Input query to embed")
  
  search_parser = subparsers.add_parser("search", help="Search among all the documents/movies")
  search_parser.add_argument("query", type=str, help="Input query to search for")
  search_parser.add_argument(
    "--limit",
    type=int,
    default=5,
    help="Number of results to show (default: 5)"
  )
  
  chunk_parser = subparsers.add_parser("chunk", help="Split long text into smaller pieces for embedding")
  chunk_parser.add_argument("text", type=str, help="Text to chunk")
  chunk_parser.add_argument(
    "--chunk-size",
    type=int,
    default=200,
    help="Size in characters of the chunk (default: 200)"
)
  

  args = parser.parse_args()

  match args.command:
    case "verify":
      verify_model()
      pass
    
    case "embed_text":
      text = args.text
      embed_text(text)
      pass
    
    case "verify_embeddings":
      verify_embeddings()
      pass
    
    case "embedquery":
      query = args.query
      embed_query_text(query)
      pass
    
    case "search":
      query = args.query
      limit = args.limit
      search_query(query, limit)
      pass
    
    case "chunk":
      text = args.text
      chunk_size = args.chunk_size
      chunk_text(text, chunk_size)
      pass
    
    case _:
      parser.print_help()

if __name__ == "__main__":
    main()