import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


LABELS = ["A", "B", "C", "D"]


def load_yaml(path):
    if yaml is None:
        raise RuntimeError("PyYAML is required to read configs/experiment.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_text(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", text)
    return text


def stable_rng(seed, source_id):
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


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


def post_chat_completion(base_url, model, messages, temperature, max_tokens, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def extract_json(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def generate_distractors(item, config, args):
    system = (
        "You generate high-quality, auditable multiple-choice distractors. "
        "You obey the requested JSON schema exactly."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_prompt(item)},
    ]
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            content = post_chat_completion(
                args.base_url or config["summarizer"]["base_url"],
                args.model or config["summarizer"]["model"],
                messages,
                args.temperature,
                args.max_tokens,
                args.timeout_seconds,
            )
            parsed = extract_json(content)
            distractors = parsed.get("distractors") or []
            if len(distractors) != 3:
                raise ValueError(f"Expected 3 distractors, got {len(distractors)}")
            return {
                "answer_type": parsed.get("answer_type"),
                "distractors": distractors,
                "raw_generation": content,
                "generation_attempts": attempt,
            }
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Failed to generate distractors for {item['source_id']}: {last_error}")


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
        lines.append(f"## {idx}. {row['source_id']}")
        lines.append("")
        lines.append(f"Question: {row['question']}")
        lines.append("")
        lines.append(f"Gold: {row['gold']} ({row['answer']})")
        lines.append("")
        lines.append(f"Answer type: {row.get('answer_type')}")
        lines.append("")
        for label in LABELS:
            marker = " [gold]" if label == row["gold"] else ""
            lines.append(f"- {label}. {row['options'][label]}{marker}")
        lines.append("")
        lines.append(f"Audit: {row['mc_audit']['status']} {row['mc_audit']['issues']}")
        if row["mc_audit"].get("warnings"):
            lines.append(f"Warnings: {row['mc_audit']['warnings']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
        f.write("\n")


def summarize(rows):
    statuses = {}
    issues = {}
    for row in rows:
        status = row["mc_audit"]["status"]
        statuses[status] = statuses.get(status, 0) + 1
        for issue in row["mc_audit"]["issues"]:
            issues[issue] = issues.get(issue, 0) + 1
    warnings = {}
    for row in rows:
        for warning in row["mc_audit"].get("warnings", []):
            warnings[warning] = warnings.get(warning, 0) + 1
    return {
        "num_items": len(rows),
        "audit_status_counts": dict(sorted(statuses.items())),
        "audit_issue_counts": dict(sorted(issues.items())),
        "audit_warning_counts": dict(sorted(warnings.items())),
        "source_ids": [row["source_id"] for row in rows],
    }


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", default="smoke", choices=["smoke", "pilot", "formal", "spares"])
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    seed = int(config["experiment"].get("random_seed", 42))
    data_dir = Path(config["paths"].get("data_dir", "data"))
    mc_dir = Path(config["paths"].get("mc_dir", data_dir / "layer1" / "mc"))
    audit_dir = Path(config["paths"].get("audit_dir", data_dir / "layer1" / "audits"))
    preview_dir = Path(config["paths"].get("preview_dir", data_dir / "layer1" / "previews"))
    default_inputs = {
        "smoke": config["paths"]["smoke_samples"],
        "pilot": config["paths"]["pilot_samples"],
        "formal": config["paths"]["formal_samples"],
        "spares": config["paths"].get("layer1_spares", str(data_dir / "layer1" / "splits" / "spares.jsonl")),
    }
    input_path = Path(args.input or default_inputs[args.split])
    output_path = Path(args.output or mc_dir / f"{args.split}_mc.jsonl")
    audit_path = audit_dir / f"{args.split}_mc_audit.json"
    preview_path = preview_dir / f"{args.split}_mc_preview.md"

    rows = read_jsonl(input_path)
    converted = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['source_id']}", flush=True)
        generated = generate_distractors(row, config, args)
        converted.append(convert_item(row, generated, seed))

    write_jsonl(output_path, converted)
    write_json(audit_path, summarize(converted))
    write_preview(preview_path, converted)
    print(json.dumps({"output": str(output_path), "audit": str(audit_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
