#!/usr/bin/env python3
"""Download the seven public behavior datasets from a pinned upstream revision."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "84fcc677e52e1902d696c32cd1a6b663e70d3993"
FILES = {
    "sycophancy": "sycophancy/sycophancy_on_nlp_survey.jsonl",
    "sycophancy_phil": "sycophancy/sycophancy_on_philpapers2020.jsonl",
    "power_seeking": "advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl",
    "survival": "advanced-ai-risk/human_generated_evals/survival-instinct.jsonl",
    "myopia": "advanced-ai-risk/human_generated_evals/myopic-reward.jsonl",
    "corrigibility": "advanced-ai-risk/human_generated_evals/corrigible-less-HHH.jsonl",
    "coordination": "advanced-ai-risk/human_generated_evals/coordinate-other-ais.jsonl",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--behaviors", nargs="+", choices=list(FILES), default=list(FILES))
    parser.add_argument(
        "--original-names", action="store_true", help="Use upstream filenames for paper scripts"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checksums = json.loads(
        (Path(__file__).resolve().parents[1] / "data/checksums.json").read_text()
    )
    manifest = {
        "repository": "anthropics/evals",
        "revision": REVISION,
        "license": "CC-BY-4.0",
        "files": {},
    }
    for behavior in args.behaviors:
        source = FILES[behavior]
        url = f"https://raw.githubusercontent.com/anthropics/evals/{REVISION}/{source}"
        print(f"Downloading {behavior} ...", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != checksums[behavior]:
            raise ValueError(f"Checksum mismatch for {behavior}; no output written")
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        name = Path(source).name if args.original_names else f"{behavior}.jsonl"
        target = args.output_dir / name
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError(
                f"Existing {target} differs from upstream; choose a fresh directory"
            )
        target.write_bytes(payload)
        manifest["files"][behavior] = {
            "file": name,
            "url": url,
            "rows": len(rows),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        print(f"{behavior}: {len(rows)} rows -> {target}", flush=True)
    (args.output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
