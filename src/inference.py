import re
from collections import Counter, defaultdict
from pathlib import Path

from src.io_utils import ensure_dir, load_yaml, read_jsonl, write_json, write_jsonl
from src.llm_client import chat_with_retries


ANSWER_RE = re.compile(r"Final Answer:\s*([A-D])", re.IGNORECASE)


def parse_answer(text):
    match = ANSWER_RE.search(text or "")
    return match.group(1).upper() if match else None


def run_answer_model(row, config, args):
    model_cfg = config["model"]
    extra_body = dict(model_cfg.get("extra_body") or {})
    content, _, attempts, response_message = chat_with_retries(
        base_url=args.base_url or model_cfg["base_url"],
        model=args.model or model_cfg["model"],
        messages=[{"role": "user", "content": row["input_text"]}],
        temperature=float(model_cfg.get("temperature", 0.0)),
        top_p=float(model_cfg.get("top_p", 1.0)),
        max_tokens=int(args.max_tokens or model_cfg.get("max_tokens", 2048)),
        timeout_seconds=int(args.timeout_seconds or model_cfg.get("timeout_seconds", 1800)),
        retries=int(args.retries),
        enable_thinking=bool(extra_body.get("chat_template_kwargs", {}).get("enable_thinking", False)),
        extra_body=extra_body,
        return_message=True,
    )
    parsed = parse_answer(content)
    reasoning = (
        response_message.get("reasoning")
        or response_message.get("reasoning_content")
        or response_message.get("reasoning_text")
    )
    return {
        "source_id": row["source_id"],
        "split": row.get("split"),
        "condition": row["condition"],
        "gold": row["gold"],
        "parsed_answer": parsed,
        "is_correct": parsed == row["gold"],
        "question": row["question"],
        "options": row["options"],
        "answer": row.get("answer"),
        "hop_count": row.get("hop_count"),
        "dialogue_profile": row.get("dialogue_profile"),
        "critical_evidence_in_recent_turn": row.get("critical_evidence_in_recent_turn"),
        "history_tokens": row.get("history_tokens"),
        "input_tokens_proxy": row.get("input_tokens_proxy"),
        "response_content": content,
        "response_reasoning": reasoning,
        "response_message": response_message,
        "attempts": attempts,
        "parse_ok": parsed is not None,
    }


def summarize_results(rows):
    def summarize_group(group):
        total = len(group)
        correct = sum(1 for row in group if row["is_correct"])
        parsed = sum(1 for row in group if row["parse_ok"])
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else None,
            "parse_rate": parsed / total if total else None,
            "parsed_answers": dict(Counter(row["parsed_answer"] for row in group)),
        }

    by_condition = defaultdict(list)
    by_profile = defaultdict(list)
    full_history_correct = {
        row["source_id"]: row["is_correct"]
        for row in rows
        if row["condition"] == "full_history"
    }
    errors = []
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_profile[row.get("dialogue_profile")].append(row)
        if not row["is_correct"]:
            if row["condition"] == "full_history" or full_history_correct.get(row["source_id"]) is False:
                audit_label = "full_history_failure_requires_audit"
            else:
                audit_label = "compression_failure_candidate"
            errors.append(
                {
                    "source_id": row["source_id"],
                    "condition": row["condition"],
                    "gold": row["gold"],
                    "parsed_answer": row["parsed_answer"],
                    "question": row["question"],
                    "dialogue_profile": row.get("dialogue_profile"),
                    "critical_evidence_in_recent_turn": row.get("critical_evidence_in_recent_turn"),
                    "failure_audit_label": audit_label,
                }
            )
    return {
        "overall": summarize_group(rows),
        "by_condition": {key: summarize_group(group) for key, group in sorted(by_condition.items())},
        "by_profile": {str(key): summarize_group(group) for key, group in sorted(by_profile.items())},
        "errors": errors,
    }


def run_inference(config, args):
    variants_path = Path(args.variants)
    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)

    write_json(run_dir / "config.snapshot.json", config)
    rows = read_jsonl(variants_path)
    parsed_rows = []
    raw_rows = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['source_id']} {row['condition']}", flush=True)
        result = run_answer_model(row, config, args)
        parsed_rows.append(result)
        raw_rows.append({"request": row, "parsed": result})

    parsed_path = run_dir / "generations.parsed.jsonl"
    raw_path = run_dir / "generations.raw.jsonl"
    summary_path = run_dir / "summary.json"
    write_jsonl(parsed_path, parsed_rows)
    write_jsonl(raw_path, raw_rows)
    write_json(summary_path, summarize_results(parsed_rows))
    return {"parsed": str(parsed_path), "raw": str(raw_path), "summary": str(summary_path)}
