import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diagnostic_generator import generate_smoke, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--variants", default="data/smoke_variants.jsonl")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    samples, variants = generate_smoke(config)
    write_jsonl(config["paths"]["smoke_samples"], samples)
    write_jsonl(args.variants, variants)

    summary = {
        "samples": len(samples),
        "variants": len(variants),
        "sample_path": config["paths"]["smoke_samples"],
        "variant_path": args.variants,
    }
    Path("runs/smoke_local").mkdir(parents=True, exist_ok=True)
    with open("runs/smoke_local/generation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
