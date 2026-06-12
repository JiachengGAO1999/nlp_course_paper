import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


FORBIDDEN_ORACLE_PHRASES = [
    "therefore choose",
    "only option",
    "only valid",
    "final answer",
]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_sample(sample, config):
    errors = []
    if sample["gold_answer"] not in sample["options"]:
        errors.append("gold answer missing from options")
    if sample.get("assistant_inference_error_present") != any(
        msg.get("assistant_behavior") == "incorrect_inference"
        for msg in sample["dialogue_history"]
        if msg["role"] == "assistant"
    ):
        errors.append("assistant_inference_error_present mismatch")

    diagnostics = sample["option_diagnostics"]
    evidence_ids = {ev["evidence_id"] for ev in sample["required_evidence"]}
    for option in sample["options"]:
        entry = diagnostics.get(option)
        if not entry:
            errors.append(f"missing option diagnostics for {option}")
            continue
        linked = set(entry.get("linked_evidence", []))
        missing_links = linked - evidence_ids
        if missing_links:
            errors.append(f"{option} links unknown evidence ids: {sorted(missing_links)}")
        if entry.get("is_gold"):
            if "satisfies" not in entry:
                errors.append(f"gold option {option} missing satisfies")
        else:
            if "error_type" not in entry:
                errors.append(f"wrong option {option} missing error_type")
            if not (entry.get("failure_reason") or entry.get("violated_constraint")):
                errors.append(f"wrong option {option} missing reason")

    required_conditions = config["difficulty_calibration"]["hard_smoke_v2_conditions"]
    for condition in required_conditions:
        if condition not in sample["compression_variants"]:
            errors.append(f"missing compression variant: {condition}")

    oracle = sample["compression_variants"].get("oracle_fact_state_summary", {}).get("text", "").lower()
    for phrase in FORBIDDEN_ORACLE_PHRASES:
        if phrase in oracle:
            errors.append(f"oracle fact-state contains forbidden phrase: {phrase}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--samples", default="data/hard_smoke_samples.with_llm.jsonl")
    parser.add_argument("--out", default="runs/hard_smoke_v2_local/validation_summary.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    samples = read_jsonl(args.samples)
    errors = {}
    token_stats = defaultdict(list)
    phenomena = Counter()
    difficulty = Counter()
    for sample in samples:
        phenomena[sample["phenomenon"]] += 1
        difficulty.update(sample["difficulty_modes"])
        sample_errors = validate_sample(sample, config)
        if sample_errors:
            errors[sample["id"]] = sample_errors
        for condition, variant in sample["compression_variants"].items():
            token_stats[condition].append(variant["history_tokens"])

    summary = {
        "num_samples": len(samples),
        "phenomena": dict(phenomena),
        "difficulty_modes": dict(difficulty),
        "errors": errors,
        "token_stats": {
            key: {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }
            for key, values in token_stats.items()
        },
        "passed": not errors,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
