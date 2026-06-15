"""Compute embedding-based compression quality metrics for a formal run.

Usage:
    python scripts/08_compute_embedding_metrics.py \\
        --run-dir runs/layer1_scale500_qwen3_8b_budget800_20260614
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedding_metrics import compute_all


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="Path to a completed formal run directory")
    parser.add_argument("--output-dir", default=None,
                        help="Metrics output dir (default: <run-dir>/formal/metrics)")
    parser.add_argument("--dialogue-path", default=None)
    parser.add_argument("--variant-path", default=None)
    parser.add_argument("--generation-path", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    formal_dir = run_dir / "formal"

    dialogue_path = Path(args.dialogue_path or formal_dir / "formal_selected_dialogues.jsonl")
    variant_path = Path(args.variant_path or formal_dir / "formal_variants.jsonl")
    generation_path = Path(args.generation_path or formal_dir / "inference" / "generations.parsed.jsonl")
    output_dir = Path(args.output_dir or formal_dir / "metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "embedding_metrics.jsonl"
    summary_path = output_dir / "embedding_metrics_summary.json"

    for label, path in [
        ("dialogues", dialogue_path),
        ("variants", variant_path),
        ("generations", generation_path),
    ]:
        if not path.exists():
            print("ERROR: {} file not found: {}".format(label, path))
            return 1

    result = compute_all(
        str(dialogue_path),
        str(variant_path),
        str(generation_path),
        str(output_path),
        str(summary_path),
    )
    print("Done: {} rows".format(result["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
