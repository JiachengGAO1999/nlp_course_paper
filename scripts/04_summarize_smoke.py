import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", default="runs/smoke_qwen3_8b_budget_gpu7/generations.parsed.jsonl")
    parser.add_argument("--out", default="runs/smoke_qwen3_8b_budget_gpu7/smoke_breakdown.json")
    args = parser.parse_args()

    rows = list(read_jsonl(args.parsed))
    by_condition = defaultdict(list)
    by_phenomenon = defaultdict(list)
    errors = []
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_phenomenon[row["phenomenon"]].append(row)
        if not row["is_correct"]:
            errors.append(
                {
                    "sample_id": row["sample_id"],
                    "phenomenon": row["phenomenon"],
                    "condition": row["condition"],
                    "gold_answer": row["gold_answer"],
                    "parsed_answer": row["parsed_answer"],
                    "error_type": row["error_type"],
                }
            )

    def summarize(grouped):
        out = {}
        for key, group in grouped.items():
            total = len(group)
            correct = sum(1 for row in group if row["is_correct"])
            parsed = sum(1 for row in group if row["parsed_answer"])
            out[key] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else None,
                "parse_rate": parsed / total if total else None,
                "error_types": dict(Counter(row["error_type"] for row in group if row["error_type"])),
            }
        return out

    summary = {
        "total": len(rows),
        "correct": sum(1 for row in rows if row["is_correct"]),
        "accuracy": sum(1 for row in rows if row["is_correct"]) / len(rows) if rows else None,
        "parse_rate": sum(1 for row in rows if row["parsed_answer"]) / len(rows) if rows else None,
        "by_condition": summarize(by_condition),
        "by_phenomenon": summarize(by_phenomenon),
        "errors": errors,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
