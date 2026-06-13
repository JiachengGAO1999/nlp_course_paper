import json
import re
from pathlib import Path

from src.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from src.llm_client import chat_with_retries, extract_json
from src.stable import stable_rng


LABELS = ["A", "B", "C", "D"]


def normalize_text(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", text)


def evidence_text(item):
    chunks = []
    for ev in item.get("required_evidence") or []:
        chunks.append(
            f"Step {ev['step']}: {ev.get('subquestion')} -> {ev.get('subanswer')}\n"
            f"Title: {ev.get('title')}\n"
            f"Evidence: {ev.get('paragraph_text')}"
        )
    return "\n\n".join(chunks)


def build_prompt(item):
    aliases = item.get("answer_aliases") or []
    return f"""You are creating multiple-choice distractors for a benchmark QA item.

Task:
Generate exactly three plausible but incorrect answer options.

Rules:
- The gold answer is correct. Do not include it or any equivalent alias as a distractor.
- Distractors must have the same semantic type as the gold answer when possible.
- Distractors should be plausible enough for a multiple-choice benchmark.
- Distractors must NOT be supported by the evidence.
- Do not solve the question for a test-taker. This is data construction.
- Return only valid JSON. No markdown.

JSON schema:
{{
  "answer_type": "short semantic type, e.g. person/place/date/organization/work",
  "distractors": [
    {{"text": "...", "rationale": "why it is plausible but wrong"}},
    {{"text": "...", "rationale": "why it is plausible but wrong"}},
    {{"text": "...", "rationale": "why it is plausible but wrong"}}
  ]
}}

Question:
{item["question"]}

Gold answer:
{item["answer"]}

Gold answer aliases:
{json.dumps(aliases, ensure_ascii=False)}

Required evidence:
{evidence_text(item)}
"""


def parse_distractors(content):
    parsed = extract_json(content)
    distractors = parsed.get("distractors") or []
    if len(distractors) != 3:
        raise ValueError(f"Expected 3 distractors, got {len(distractors)}")
    return parsed


def generate_distractors(item, *, base_url, model, temperature, max_tokens, timeout_seconds, retries):
    messages = [
        {
            "role": "system",
            "content": "You generate high-quality, auditable multiple-choice distractors. You obey the requested JSON schema exactly.",
        },
        {"role": "user", "content": build_prompt(item)},
    ]
    content, parsed, attempts = chat_with_retries(
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        retries=retries,
        parser=parse_distractors,
    )
    return {
        "answer_type": parsed.get("answer_type"),
        "distractors": parsed["distractors"],
        "raw_generation": content,
        "generation_attempts": attempts,
    }


def build_options(item, generated, seed):
    rng = stable_rng(seed, item["source_id"])
    choices = [{"text": item["answer"], "is_gold": True, "source": "benchmark_gold"}]
    for distractor in generated["distractors"]:
        choices.append(
            {
                "text": str(distractor.get("text", "")).strip(),
                "is_gold": False,
                "source": "llm_generated",
                "rationale": distractor.get("rationale"),
            }
        )
    rng.shuffle(choices)

    options = {}
    distractors = []
    gold_label = None
    for label, choice in zip(LABELS, choices):
        options[label] = choice["text"]
        if choice["is_gold"]:
            gold_label = label
        else:
            distractors.append(
                {
                    "label": label,
                    "text": choice["text"],
                    "source": choice["source"],
                    "rationale": choice.get("rationale"),
                }
            )
    return options, gold_label, distractors


def audit_mc_item(item):
    issues = []
    warnings = []
    options = item.get("options") or {}
    answer = normalize_text(item.get("answer"))
    aliases = {normalize_text(alias) for alias in item.get("answer_aliases") or []}
    aliases.add(answer)
    normalized_options = {label: normalize_text(text) for label, text in options.items()}

    if sorted(options) != LABELS:
        issues.append("option_labels_not_abcd")
    if item.get("gold") not in LABELS:
        issues.append("gold_label_invalid")
    if any(not text for text in normalized_options.values()):
        issues.append("empty_option")
    if len(set(normalized_options.values())) != len(normalized_options):
        issues.append("duplicate_option_text")

    gold_matches = [label for label, text in normalized_options.items() if text in aliases]
    if len(gold_matches) != 1:
        issues.append("gold_or_alias_not_unique")
    elif gold_matches[0] != item.get("gold"):
        issues.append("gold_label_mismatch")

    evidence = normalize_text(evidence_text(item))
    for label, text in normalized_options.items():
        if label != item.get("gold") and text in aliases:
            issues.append(f"distractor_{label}_matches_alias")
        if label != item.get("gold") and text and text in evidence:
            warnings.append(f"distractor_{label}_appears_in_required_evidence")
    return issues, warnings


def convert_item(item, generated, seed):
    options, gold_label, distractors = build_options(item, generated, seed)
    converted = dict(item)
    converted.update(
        {
            "answer_type": generated.get("answer_type"),
            "options": options,
            "gold": gold_label,
            "distractors": distractors,
            "mc_conversion_status": "converted",
            "mc_generation_attempts": generated["generation_attempts"],
            "mc_raw_generation": generated["raw_generation"],
        }
    )
    issues, warnings = audit_mc_item(converted)
    converted["mc_audit"] = {
        "status": "pass" if not issues else "requires_manual_review",
        "issues": issues,
        "warnings": warnings,
    }
    return converted


def write_preview(path, rows):
    lines = ["# MC Conversion Preview", ""]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {idx}. {row['source_id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Gold: {row['gold']} ({row['answer']})",
                "",
                f"Answer type: {row.get('answer_type')}",
                "",
            ]
        )
        for label in LABELS:
            marker = " [gold]" if label == row["gold"] else ""
            lines.append(f"- {label}. {row['options'][label]}{marker}")
        lines.append("")
        lines.append(f"Audit: {row['mc_audit']['status']} {row['mc_audit']['issues']}")
        if row["mc_audit"].get("warnings"):
            lines.append(f"Warnings: {row['mc_audit']['warnings']}")
        lines.append("")
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
        f.write("\n")


def summarize(rows):
    statuses = {}
    issues = {}
    warnings = {}
    for row in rows:
        status = row["mc_audit"]["status"]
        statuses[status] = statuses.get(status, 0) + 1
        for issue in row["mc_audit"]["issues"]:
            issues[issue] = issues.get(issue, 0) + 1
        for warning in row["mc_audit"].get("warnings", []):
            warnings[warning] = warnings.get(warning, 0) + 1
    return {
        "num_items": len(rows),
        "audit_status_counts": dict(sorted(statuses.items())),
        "audit_issue_counts": dict(sorted(issues.items())),
        "audit_warning_counts": dict(sorted(warnings.items())),
        "source_ids": [row["source_id"] for row in rows],
    }


def default_input_for_split(config, split):
    data_dir = Path(config["paths"].get("data_dir", "data"))
    return {
        "smoke": config["paths"]["smoke_samples"],
        "pilot": config["paths"]["pilot_samples"],
        "formal": config["paths"]["formal_samples"],
        "formal_pool": config["paths"].get("formal_pool_samples", str(data_dir / "layer1" / "splits" / "formal_pool.jsonl")),
        "spares": config["paths"].get("layer1_spares", str(data_dir / "layer1" / "splits" / "spares.jsonl")),
    }[split]


def convert_musique_to_mc(config, args):
    seed = int(config["experiment"].get("random_seed", 42))
    data_dir = Path(config["paths"].get("data_dir", "data"))
    mc_dir = Path(config["paths"].get("mc_dir", data_dir / "layer1" / "mc"))
    audit_dir = Path(config["paths"].get("audit_dir", data_dir / "layer1" / "audits"))
    preview_dir = Path(config["paths"].get("preview_dir", data_dir / "layer1" / "previews"))

    input_path = Path(args.input or default_input_for_split(config, args.split))
    output_path = Path(args.output or mc_dir / f"{args.split}_mc.jsonl")
    rows = read_jsonl(input_path)

    converted = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['source_id']}", flush=True)
        generated = generate_distractors(
            row,
            base_url=args.base_url or config["summarizer"]["base_url"],
            model=args.model or config["summarizer"]["model"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        )
        converted.append(convert_item(row, generated, seed))

    audit_path = audit_dir / f"{args.split}_mc_audit.json"
    preview_path = preview_dir / f"{args.split}_mc_preview.md"
    write_jsonl(output_path, converted)
    write_json(audit_path, summarize(converted))
    write_preview(preview_path, converted)
    return {"output": str(output_path), "audit": str(audit_path)}
