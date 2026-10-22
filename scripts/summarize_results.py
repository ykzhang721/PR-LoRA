#!/usr/bin/env python3
"""Print the main table directly from saved per-seed measurements."""

import argparse
import json
from pathlib import Path

from pr_lora.statistics import summarize_main

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True, help="Path to generated per-seed experiment results JSON")
args = parser.parse_args()
summarize_main(json.loads(args.input.read_text()))
