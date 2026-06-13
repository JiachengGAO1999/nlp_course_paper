import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.formal_selection import select_formal_set
from src.io_utils import load_yaml


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dialogues", default="data/layer1/dialogues/formal_pool_dialogues.jsonl")
    parser.add_argument("--results", default="runs/layer1_formal_pool_full_history_gate_20260613/generations.parsed.jsonl")
    parser.add_argument("--output", default="data/layer1/dialogues/formal_selected_dialogues.jsonl")
    parser.add_argument("--audit", default="data/layer1/audits/formal_selection_audit.json")
    parser.add_argument("--target-n", type=int, default=None)
    parser.add_argument("--tries", type=int, default=5000)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    audit = select_formal_set(config, args)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
