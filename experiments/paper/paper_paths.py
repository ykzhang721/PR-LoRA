"""All public paper scripts read/write under one configurable workspace."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("PR_LORA_WORKDIR", Path(__file__).resolve().parents[2])).resolve()
