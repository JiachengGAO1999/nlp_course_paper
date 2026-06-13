import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import run_inference
from src.io_utils import load_yaml


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--variants", default="data/layer1/variants/smoke_variants.jsonl")
    parser.add_argument("--run-dir", default="runs/layer1_smoke_qwen3_8b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    result = run_inference(config, args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
