from pathlib import Path

from src.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from src.llm_client import chat_with_retries
from src.text_utils import count_tokens, render_dialogue, render_question


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
- State which option is correct or eliminate any option.
- Rank options or make a final recommendation.
- Add any information not present in the original dialogue.

Original dialogue:
{dialogue_text}

Compressed history (under {target_tokens} tokens):
"""


def condition_names(config, args=None):
    if args is not None and getattr(args, "conditions", None):
        return [name.strip() for name in args.conditions.split(",") if name.strip()]
    return list(config["conditions"]["main"])


def group_complete_turns(messages):
    turns = []
    for idx in range(0, len(messages), 2):
        turns.append(messages[idx : idx + 2])
    return turns


def flatten(turn_groups):
    return [msg for group in turn_groups for msg in group]


def compression_prompt(dialogue_text, target_tokens):
    return SUMMARY_PROMPT.format(dialogue_text=dialogue_text, target_tokens=target_tokens)


def summarize_dialogue(dialogue_text, target_tokens, config, args):
    summarizer = config["summarizer"]
    content, _, attempts = chat_with_retries(
        base_url=args.base_url or summarizer["base_url"],
        model=args.model or summarizer["model"],
        messages=[{"role": "user", "content": compression_prompt(dialogue_text, target_tokens)}],
        temperature=float(summarizer.get("temperature", 0.0)),
        top_p=float(summarizer.get("top_p", 1.0)),
        max_tokens=int(args.max_tokens or summarizer.get("max_tokens", 1024)),
        timeout_seconds=int(args.timeout_seconds or summarizer.get("timeout_seconds", 1800)),
        retries=int(args.retries),
        enable_thinking=bool(summarizer.get("thinking_enabled", False)),
    )
    return content.strip(), attempts


def build_answer_input(row, condition, history_text, history_tokens):
    question_text = render_question(row["question"], row["options"])
    prompt = (
        "Dialogue context:\n"
        f"{history_text}\n\n"
        "Question:\n"
        f"{question_text}\n\n"
        "Choose the best option. Provide the final answer in the exact format:\n"
        "Final Answer: <A/B/C/D>\n\n"
        "Then give a brief explanation mentioning the key evidence from the context."
    )
    return {
        "source_id": row["source_id"],
        "split": row.get("split"),
        "condition": condition,
        "gold": row["gold"],
        "question": row["question"],
        "options": row["options"],
        "answer": row.get("answer"),
        "hop_count": row.get("hop_count"),
        "dialogue_profile": row.get("dialogue_profile"),
        "critical_evidence_in_recent_turn": row.get("critical_evidence_in_recent_turn"),
        "dialogue_token_count_proxy": row.get("dialogue_token_count_proxy"),
        "history_text": history_text,
        "history_tokens": history_tokens,
        "input_text": prompt,
        "input_tokens_proxy": count_tokens(prompt),
        "evidence_turn_map": row.get("evidence_turn_map"),
        "compression_audit": {},
    }


def full_history_variant(row):
    history_text = render_dialogue(row["dialogue_messages"])
    return build_answer_input(row, "full_history", history_text, count_tokens(history_text))


def one_shot_variant(row, config, args):
    budget = int(config["budgets"]["compressed_history_budget_tokens"])
    dialogue_text = render_dialogue(row["dialogue_messages"])
    summary, attempts = summarize_dialogue(dialogue_text, budget, config, args)
    variant = build_answer_input(row, "one_shot_summary", summary, count_tokens(summary))
    variant["compression_audit"] = {
        "summary_scope": "full_history",
        "target_tokens": budget,
        "summarizer_attempts": attempts,
    }
    return variant


def hybrid_variant(row, config, args):
    budget = int(config["budgets"]["compressed_history_budget_tokens"])
    hybrid_cfg = config["compression"]["hybrid_budget"]
    recent_turns = int(config["compression"].get("hybrid_recent_turns", 1))
    turn_groups = group_complete_turns(row["dialogue_messages"])
    recent_groups = turn_groups[-recent_turns:]
    older_groups = turn_groups[:-recent_turns]

    recent_messages = flatten(recent_groups)
    recent_text = render_dialogue(recent_messages)
    recent_tokens = count_tokens(recent_text)
    if recent_tokens > int(hybrid_cfg.get("recent_turn_overflow_threshold", 250)):
        recent_messages = [msg for msg in recent_messages if msg["role"] == "user"]
        recent_text = render_dialogue(recent_messages)
        recent_tokens = count_tokens(recent_text)

    summary_budget = min(
        int(hybrid_cfg.get("summary_max_tokens", 400)),
        max(100, budget - min(recent_tokens, int(hybrid_cfg.get("recent_turn_max_tokens", 200)))),
    )
    older_text = render_dialogue(flatten(older_groups))
    older_summary, attempts = summarize_dialogue(older_text, summary_budget, config, args)
    history_text = (
        "Compressed older dialogue:\n"
        f"{older_summary}\n\n"
        "Recent turn kept verbatim:\n"
        f"{recent_text}"
    )
    variant = build_answer_input(row, "hybrid_summary_recent", history_text, count_tokens(history_text))
    variant["compression_audit"] = {
        "summary_scope": "older_history",
        "summary_target_tokens": summary_budget,
        "recent_turns_kept": recent_turns,
        "recent_tokens_proxy": recent_tokens,
        "summarizer_attempts": attempts,
    }
    return variant


def build_variants_for_row(row, config, args):
    builders = {
        "full_history": lambda: full_history_variant(row),
        "one_shot_summary": lambda: one_shot_variant(row, config, args),
        "hybrid_summary_recent": lambda: hybrid_variant(row, config, args),
    }
    return [builders[name]() for name in condition_names(config, args)]


def summarize_variants(rows):
    by_condition = {}
    for row in rows:
        condition = row["condition"]
        by_condition.setdefault(condition, []).append(row["history_tokens"])
    return {
        "num_variants": len(rows),
        "by_condition": {
            condition: {
                "count": len(tokens),
                "min_history_tokens": min(tokens),
                "mean_history_tokens": sum(tokens) / len(tokens),
                "max_history_tokens": max(tokens),
            }
            for condition, tokens in sorted(by_condition.items())
        },
    }


def build_compression_variants(config, args):
    data_dir = Path(config["paths"].get("data_dir", "data"))
    dialogue_dir = Path(config["paths"].get("dialogue_dir", data_dir / "layer1" / "dialogues"))
    output_dir = Path(args.output_dir or data_dir / "layer1" / "variants")
    audit_dir = Path(config["paths"].get("audit_dir", data_dir / "layer1" / "audits"))
    input_path = Path(args.input or dialogue_dir / f"{args.split}_dialogues.jsonl")
    output_path = Path(args.output or output_dir / f"{args.split}_variants.jsonl")
    audit_path = Path(args.audit or audit_dir / f"{args.split}_variant_audit.json")

    rows = read_jsonl(input_path)
    variants = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['source_id']}", flush=True)
        variants.extend(build_variants_for_row(row, config, args))

    ensure_dir(output_path.parent)
    write_jsonl(output_path, variants)
    write_json(audit_path, summarize_variants(variants))
    return {"output": str(output_path), "audit": str(audit_path)}
