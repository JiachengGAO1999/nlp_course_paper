import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hard_smoke_generator import hard_samples, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--out", default="data/hard_smoke_samples.jsonl")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    samples = hard_samples()
    write_jsonl(args.out, samples)

    summary = {
        "samples": len(samples),
        "sample_path": args.out,
        "conditions_ready": ["full_history", "oracle_fact_state_summary"],
        "needs_llm_compression": ["llm_generated_summary", "hybrid_summary_recent"],
    }
    Path("runs/hard_smoke_v2_local").mkdir(parents=True, exist_ok=True)
    with open("runs/hard_smoke_v2_local/generation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
