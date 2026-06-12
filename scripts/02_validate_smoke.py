import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FORBIDDEN_FACT_STATE_PHRASES = [
    "final answer",
    "therefore choose",
    "only option",
    "only valid",
    "choose a",
    "choose b",
    "choose c",
    "choose d",
]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_sample(sample, config):
    errors = []
    if sample["gold_answer"] not in sample["options"]:
        errors.append("gold answer missing from options")

    diagnostics = sample.get("option_diagnostics") or {}
    for option in sample["options"]:
        entry = diagnostics.get(option)
        if not entry:
            errors.append(f"missing option_diagnostics for {option}")
            continue
        if "is_gold" not in entry:
            errors.append(f"missing is_gold for {option}")
        if entry.get("is_gold"):
            if "satisfies" not in entry:
                errors.append(f"gold option {option} missing satisfies")
        else:
            if "error_type" not in entry:
                errors.append(f"wrong option {option} missing error_type")
            if not ("violated_constraint" in entry or "failure_reason" in entry):
                errors.append(f"wrong option {option} missing failure reason")
        if "linked_evidence" not in entry:
            errors.append(f"option {option} missing linked_evidence")

    positions = {item["position"] for item in sample["required_evidence"]}
    if len(positions) < 2:
        errors.append("required evidence lacks position diversity")

    variants = sample["compression_variants"]
    budget_min = config["budgets"]["compressed_history_budget_tolerance"]["min"]
    budget_max = config["budgets"]["compressed_history_budget_tolerance"]["max"]
    for name, variant in variants.items():
        tokens = variant["history_tokens"]
        if name == "full_history":
            continue
        if tokens > budget_max:
            errors.append(f"{name} exceeds compressed-history max: {tokens}")
        if tokens < 150:
            errors.append(f"{name} is suspiciously short: {tokens}")

    fact_state = variants["oracle_fact_state_summary"]["text"].lower()
    for phrase in FORBIDDEN_FACT_STATE_PHRASES:
        if phrase in fact_state:
            errors.append(f"fact-state summary contains forbidden phrase: {phrase}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--samples", default="data/smoke_samples.jsonl")
    parser.add_argument("--out", default="runs/smoke_local/validation_summary.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    samples = read_jsonl(args.samples)
    all_errors = {}
    phenomena = Counter()
    token_stats = defaultdict(list)
    for sample in samples:
        phenomena[sample["phenomenon"]] += 1
        errors = validate_sample(sample, config)
        if errors:
            all_errors[sample["id"]] = errors
        for condition, variant in sample["compression_variants"].items():
            token_stats[condition].append(variant["history_tokens"])

    summary = {
        "num_samples": len(samples),
        "phenomena": dict(phenomena),
        "errors": all_errors,
        "token_stats": {
            condition: {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }
            for condition, values in token_stats.items()
        },
        "passed": not all_errors,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
