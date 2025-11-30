#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add project root to path to allow imports to work when running as script 
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from cli.lib.hybrid_search import normalize_cmd,  weighted_search_cmd

def main() -> None:
  parser = argparse.ArgumentParser(description="Hybrid Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  
  normalize_parser = subparsers.add_parser("normalize", help="Normalizes scores using the min-max normalization")
  normalize_parser.add_argument('inputs', metavar='FLOAT', type=float,
    nargs='+',  # Requires one or more inputs
    help='A list of scores to normalize'
  )

  weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted search combining both keyword and semantic.")
  weighted_search_parser.add_argument("query", type=str, help="Input query to search for")
  weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Dynamically control the weighting between the two scores (default: 0.5)")
  weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to show (default: 5)")
  
  args = parser.parse_args()

  match args.command:
    case "normalize":
      inputs = args.inputs
      normalize_cmd(inputs)
      pass
    case "weighted-search":
      query = args.query
      alpha = args.alpha
      limit = args.limit
      weighted_search_cmd(query, alpha, limit)
      pass
    case _:
      parser.print_help()
      

if __name__ == "__main__":
  main()