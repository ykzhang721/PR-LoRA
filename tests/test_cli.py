import json
import os
from pathlib import Path
import subprocess
import sys

from pr_lora.data import write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_local_cpu_pipeline(tiny_model, tokenizer, records, tmp_path):
    model_dir = tmp_path / "model"
    tiny_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    data = tmp_path / "input.jsonl"
    write_jsonl(records * 10, data)
    split = tmp_path / "split"
    directions = tmp_path / "directions.pt"
    adapter = tmp_path / "adapter"
    output = tmp_path / "scores.json"
    commands = [
        ["split_caa.py", "--input", data, "--output-dir", split, "--train-size", "4"],
        [
            "extract_directions.py",
            "--model",
            model_dir,
            "--input",
            split / "direction.jsonl",
            "--output",
            directions,
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ],
        [
            "train_pr_lora.py",
            "--model",
            model_dir,
            "--train-data",
            split / "train.jsonl",
            "--directions",
            directions,
            "--output",
            adapter,
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--steps",
            "2",
            "--max-length",
            "32",
        ],
        [
            "evaluate_behavior.py",
            "--model",
            model_dir,
            "--data",
            split / "eval.jsonl",
            "--adapter",
            adapter,
            "--output",
            output,
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ],
    ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    for script, *args in commands:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    score = json.loads(output.read_text())
    assert score["num_items"] == 6
    assert 0 <= score["undesirable_behavior_score"] <= 1
    config = json.loads((adapter / "training_config.json").read_text())
    assert len(config["training_history"]) == 2
    assert len(config["initialized_modules"]) == 4
