#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add project root to path to allow imports to work when running as script 
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))


from cli.lib.semantic_search import verify_model

def main():
  parser = argparse.ArgumentParser(description="Semantic Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  subparsers.add_parser("verify", help="Verifies model is loaded")
 
  args = parser.parse_args()

  match args.command:
    case "verify":
      verify_model()
      pass
    case _:
      parser.print_help()

if __name__ == "__main__":
    main()