#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add project root to path to allow imports to work when running as script 
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from cli.lib.hybrid_search import normalize_scores

def main() -> None:
  parser = argparse.ArgumentParser(description="Hybrid Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  
  normalize_parser = subparsers.add_parser("normalize", help="Normalizes scores using the min-max normalization")
  normalize_parser.add_argument(
    'inputs',
    metavar='FLOAT',
    type=float,
    nargs='+',  # Requires one or more inputs
    help='A list of scores to normalize'
)


  args = parser.parse_args()

  match args.command:
    case "normalize":
      inputs = args.inputs
      normalize_scores(inputs)
      pass
    case _:
      parser.print_help()
      

if __name__ == "__main__":
  main()