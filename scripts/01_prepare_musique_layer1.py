import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import load_yaml
from src.musique_sampling import prepare_musique_layer1


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--split", default="dev", choices=["dev"])
    parser.add_argument("--min-hops", type=int, default=None)
    parser.add_argument("--max-hops", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    audit = prepare_musique_layer1(
        config,
        num_samples=args.num_samples,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
