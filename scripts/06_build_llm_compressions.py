import argparse
import json
import sys
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hard_smoke_generator import HARD_CONDITIONS, build_input_rows, write_jsonl
from src.text_utils import count_tokens, render_dialogue


SUMMARY_PROMPT = """You are compressing a multi-turn dialogue history for later use in answering
a multiple-choice question. The compressed history must fit within approximately
{target_tokens} tokens.

Preserve:
- All stated facts, constraints, exclusions, and preferences.
- The latest state for every entity or decision (updates override earlier values).
- Which constraints are hard requirements and which are soft preferences.
- Candidate entities, plans, alternatives, and attributes mentioned in the dialogue.
- The source turn for each piece of retained information.

Do NOT:
- Solve the final question or compare candidates against it.
- Use or infer final multiple-choice option labels unless already present in the dialogue.
- State which option is correct or eliminate any option.
- Rank options or make a final recommendation.
- Add any information not present in the original dialogue.

Original dialogue:
{dialogue_text}

Compressed history (under {target_tokens} tokens):
"""


HYBRID_PROMPT = """You are compressing older dialogue history. A recent turn will be kept verbatim
separately, so you only need to summarize the older turns.

Summarize the older dialogue history below within approximately {summary_tokens} tokens.
Preserve facts, constraints, exclusions, preferences, state updates, and candidate
attributes. Mark which information is a hard constraint, soft preference, or outdated.
Do not solve the final question or compare candidates.

Older dialogue history:
{older_history_text}

Compressed older history (under {summary_tokens} tokens):
"""


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chat(config, text):
    cfg = config["summarizer"]
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": text}],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "max_tokens": cfg["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": bool(cfg.get("thinking_enabled", False))},
    }
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    response = requests.post(url, json=payload, timeout=cfg.get("timeout_seconds", 1800))
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0].get("message") or {}
    return message.get("content") or "", data.get("usage")


def group_turns(messages):
    grouped = []
    seen = []
    for msg in messages:
        if msg["turn_id"] not in seen:
            seen.append(msg["turn_id"])
    for turn_id in seen:
        grouped.append([msg for msg in messages if msg["turn_id"] == turn_id])
    return grouped


def compression_quality(sample, text, condition, usage=None):
    retained = []
    missing = []
    lower = text.lower()
    for ev in sample["required_evidence"]:
        span = ev["span"].lower()
        if span and span in lower:
            retained.append(ev["evidence_id"])
        else:
            missing.append(ev["evidence_id"])
    total = len(sample["required_evidence"])
    return {
        "required_evidence_retention": {
            "retained": retained,
            "missing": missing,
            "retention_rate": len(retained) / total if total else None,
            "method": "span_matching_pre_audit_hint",
        },
        "answerability": {
            "answerable": not missing,
            "label": "answerable" if not missing else "requires_manual_audit",
            "reason": "LLM summaries may paraphrase; hard smoke v2 requires full manual audit.",
        },
        "hallucination_check": {
            "unsupported_facts": [],
            "hallucinated_fact_count": None,
            "method": "manual_audit_required",
        },
        "manual_audit_required": True,
        "condition": condition,
        "summarizer_usage": usage,
    }


def add_llm_variants(sample, config):
    budget = config["budgets"]["compressed_history_budget_tokens"]
    dialogue = render_dialogue(sample["dialogue_history"])
    summary_text, summary_usage = chat(
        config, SUMMARY_PROMPT.format(target_tokens=budget, dialogue_text=dialogue)
    )
    sample["compression_variants"]["llm_generated_summary"] = {
        "text": summary_text,
        "history_tokens": count_tokens(summary_text),
        "budget_constrained": True,
        "summarizer_model": config["summarizer"]["model"],
        "compression_prompt": "docs/generator_spec.md §4.1",
        "compression_quality": compression_quality(sample, summary_text, "llm_generated_summary", summary_usage),
    }

    turns = group_turns(sample["dialogue_history"])
    recent = turns[-1]
    older = [msg for group in turns[:-1] for msg in group]
    recent_text = render_dialogue(recent)
    if count_tokens(recent_text) > 250:
        recent = [msg for msg in recent if msg["role"] == "user"]
        recent_text = render_dialogue(recent)
    summary_tokens = max(250, budget - min(250, count_tokens(recent_text)))
    older_text = render_dialogue(older)
    older_summary, hybrid_usage = chat(
        config, HYBRID_PROMPT.format(summary_tokens=summary_tokens, older_history_text=older_text)
    )
    hybrid_text = older_summary.strip() + "\n\n---\nRecent turn — verbatim:\n" + recent_text
    sample["compression_variants"]["hybrid_summary_recent"] = {
        "text": hybrid_text,
        "history_tokens": count_tokens(hybrid_text),
        "budget_constrained": True,
        "summarizer_model": config["summarizer"]["model"],
        "summary_scope": "older_history",
        "recent_turns_kept": sorted({msg["turn_id"] for msg in recent}),
        "compression_prompt": "docs/generator_spec.md §4.2",
        "compression_quality": compression_quality(sample, hybrid_text, "hybrid_summary_recent", hybrid_usage),
    }
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--samples", default="data/hard_smoke_samples.jsonl")
    parser.add_argument("--out-samples", default="data/hard_smoke_samples.with_llm.jsonl")
    parser.add_argument("--out-variants", default="data/hard_smoke_variants.jsonl")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    samples = [add_llm_variants(sample, config) for sample in read_jsonl(args.samples)]
    rows = build_input_rows(samples, config, conditions=config["difficulty_calibration"]["hard_smoke_v2_conditions"])
    write_jsonl(args.out_samples, samples)
    write_jsonl(args.out_variants, rows)
    summary = {
        "samples": len(samples),
        "variants": len(rows),
        "out_samples": args.out_samples,
        "out_variants": args.out_variants,
    }
    Path("runs/hard_smoke_v2_local").mkdir(parents=True, exist_ok=True)
    with open("runs/hard_smoke_v2_local/compression_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
