import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.formal_pool import prepare_formal_pool
from src.io_utils import load_yaml


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--preview", default=None)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    audit = prepare_formal_pool(config, args)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
