import argparse
import json
import re
import sys
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ANSWER_RE = re.compile(r"Final Answer:\s*([A-D])", re.IGNORECASE)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_answer(text):
    match = ANSWER_RE.search(text or "")
    return match.group(1).upper() if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--variants", default="data/smoke_variants.jsonl")
    parser.add_argument("--run-dir", default="runs/smoke_qwen3_8b_budget_gpu7")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.snapshot.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    model_cfg = config["model"]
    url = model_cfg["base_url"].rstrip("/") + "/chat/completions"
    raw_path = run_dir / "generations.raw.jsonl"
    parsed_path = run_dir / "generations.parsed.jsonl"

    rows = list(read_jsonl(args.variants))
    parsed_rows = []
    with open(raw_path, "w", encoding="utf-8") as raw_handle:
        for row in rows:
            payload = {
                "model": model_cfg["model"],
                "messages": [{"role": "user", "content": row["input_text"]}],
                "temperature": model_cfg["temperature"],
                "top_p": model_cfg["top_p"],
                "max_tokens": model_cfg["max_tokens"],
            }
            payload.update(model_cfg.get("extra_body") or {})
            response = requests.post(url, json=payload, timeout=model_cfg.get("timeout_seconds", 1800))
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            parsed_answer = parse_answer(content)
            diagnostics = row["option_diagnostics"].get(parsed_answer) if parsed_answer else None
            out = {
                "sample_id": row["sample_id"],
                "phenomenon": row["phenomenon"],
                "condition": row["condition"],
                "model": model_cfg["model"],
                "history_tokens": row["history_tokens"],
                "gold_answer": row["gold_answer"],
                "parsed_answer": parsed_answer,
                "is_correct": parsed_answer == row["gold_answer"],
                "error_type": None if parsed_answer == row["gold_answer"] else (diagnostics or {}).get("error_type"),
                "response_reasoning": message.get("reasoning"),
                "response_content": content,
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage"),
            }
            raw_handle.write(json.dumps({"request": row, "response": data}, ensure_ascii=False) + "\n")
            parsed_rows.append(out)

    with open(parsed_path, "w", encoding="utf-8") as parsed_handle:
        for row in parsed_rows:
            parsed_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(parsed_rows)
    correct = sum(1 for row in parsed_rows if row["is_correct"])
    parse_ok = sum(1 for row in parsed_rows if row["parsed_answer"])
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "parse_rate": parse_ok / total if total else None,
    }
    with open(run_dir / "smoke_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
