import re
from pathlib import Path

from src.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from src.llm_client import chat_with_retries
from src.stable import stable_rng
from src.text_utils import count_tokens, render_dialogue


PROFILE_NAMES = ["far_early", "far_middle", "cross_turn", "late"]
FORBIDDEN_DIALOGUE_PHRASES = [
    "subquestion",
    "short answer",
    "red herring",
    "directly answers",
    "answers the subquestion",
    "aligns with the subquestion",
]


def stable_profile(seed, source_id, profiles):
    rng = stable_rng(seed, source_id)
    return profiles[rng.randrange(len(profiles))]


def expand_profile_allocation(allocation):
    profiles = []
    for profile, count in allocation.items():
        profiles.extend([profile] * int(count))
    return profiles


def assign_profiles(rows, config, split, seed):
    profiles = config.get("layer1", {}).get("evidence_position_profiles") or PROFILE_NAMES
    split_allocations = config.get("layer1", {}).get("split_profile_allocation") or {}
    allocation = split_allocations.get(split)
    if not allocation:
        return {row["source_id"]: stable_profile(seed, row["source_id"], profiles) for row in rows}

    assigned = expand_profile_allocation(allocation)
    if len(assigned) != len(rows):
        raise ValueError(
            f"Profile allocation for split '{split}' has {len(assigned)} items, "
            f"but input contains {len(rows)} rows."
        )
    unknown = sorted(set(assigned) - set(profiles))
    if unknown:
        raise ValueError(f"Unknown profile(s) in split '{split}': {unknown}")

    rng = stable_rng(seed, f"{split}:profile_allocation")
    rng.shuffle(assigned)
    return {row["source_id"]: profile for row, profile in zip(rows, assigned)}


def words(text):
    return re.findall(r"\S+", str(text or "").strip())


def truncate_words(text, max_words):
    parts = words(text)
    if len(parts) <= max_words:
        return str(text or "").strip()
    return " ".join(parts[:max_words]).rstrip(" ,.;:") + " ..."


def evidence_turns(profile, hop_count, total_turns):
    if profile == "far_early":
        return list(range(1, hop_count + 1))
    if profile == "far_middle":
        start = max(2, (total_turns - hop_count) // 2 + 1)
        return list(range(start, start + hop_count))
    if profile == "late":
        return list(range(total_turns - hop_count + 1, total_turns + 1))
    if hop_count == 1:
        return [total_turns // 2]

    positions = []
    for idx in range(hop_count):
        pos = round(1 + idx * (total_turns - 1) / (hop_count - 1))
        while pos in positions and pos < total_turns:
            pos += 1
        positions.append(int(pos))
    return positions


def build_evidence_turn(ev, turn_id):
    snippet = truncate_words(ev.get("paragraph_text"), 280)
    return (
        f"I found another source entry while collecting background. "
        f"The entry titled \"{ev.get('title')}\" says: {snippet}\n\n"
        f"For now, I am keeping this as a potentially relevant source note. "
        f"Please consider what it seems to establish and how it might connect to "
        f"earlier notes, but keep any conclusion tentative until the rest of the "
        f"material is available."
    )


def build_filler_turn(item, turn_id, filler_idx, profile):
    distractors = item.get("distractors") or []
    distractor = distractors[filler_idx % len(distractors)] if distractors else {}
    text = distractor.get("text", "a related but probably irrelevant item")
    templates = [
        (
            f"Side note {turn_id}: I also ran into \"{text}\" while looking through "
            f"nearby material. It may simply be related background, so I do not want "
            f"to treat it as an answer or conclusion. Please keep it in mind lightly "
            f"and wait for more context."
        ),
        (
            f"Another nearby item I saw was \"{text}\". It sounds connected to the "
            f"topic, but I have not checked whether it belongs to the main chain. "
            f"For now, treat it as a loose lead rather than a resolved point."
        ),
        (
            f"I also have a possible stale lead: \"{text}\". It might be useful "
            f"context, but I am not sure it is current or relevant. Please treat "
            f"it as uncertain for now."
        ),
        (
            f"One more bit of context before I add the next source: \"{text}\" came up "
            f"in the surrounding material. I am not asking you to decide anything from "
            f"it yet; just note that it was mentioned."
        ),
    ]
    content = templates[filler_idx % len(templates)]
    if profile == "cross_turn" and int(item.get("hop_count", 0)) <= 3:
        content += (
            " I am still gathering pieces from different parts of the material, so "
            "please do not draw a conclusion from this item alone."
        )
    return content


def build_user_turns(item, profile, total_turns):
    evidence = item.get("required_evidence") or []
    ev_positions = evidence_turns(profile, len(evidence), total_turns)
    ev_by_turn = {turn: ev for turn, ev in zip(ev_positions, evidence)}
    turns = []
    filler_idx = 0
    for turn_id in range(1, total_turns + 1):
        if turn_id in ev_by_turn:
            ev = ev_by_turn[turn_id]
            turns.append({"turn_id": turn_id, "kind": "evidence", "evidence_steps": [ev["step"]], "content": build_evidence_turn(ev, turn_id)})
        else:
            turns.append({"turn_id": turn_id, "kind": "distractor_or_context", "evidence_steps": [], "content": build_filler_turn(item, turn_id, filler_idx, profile)})
            filler_idx += 1
    return turns, {str(ev["step"]): turn for turn, ev in zip(ev_positions, evidence)}


def critical_evidence_in_recent_turn(evidence_turn_map, total_turns, recent_turns):
    first_recent_turn = total_turns - recent_turns + 1
    return any(int(turn) >= first_recent_turn for turn in evidence_turn_map.values())


def forbidden_phrase_hits(text):
    lowered = str(text or "").lower()
    return [phrase for phrase in FORBIDDEN_DIALOGUE_PHRASES if phrase in lowered]


def assistant_messages(history):
    system = (
        "You are a neutral assistant in a source-note collection dialogue. "
        "The user is gradually collecting source snippets and nearby leads. "
        "Respond naturally with brief intermediate reasoning about what the current "
        "note seems to establish, how it might relate to earlier notes, or what "
        "remains uncertain. You may form tentative local interpretations. "
        "Do not claim to know the hidden final question, do not invent answer "
        "options, and do not use a final-answer format. "
        "Do not use the words subquestion, short answer, or red herring. "
        "Do not say that a note directly answers a reasoning step, and do not "
        "label any hidden benchmark decomposition or reasoning chain. "
        "When a lead seems irrelevant, call it uncertain, tangential, or a loose "
        "lead instead of using task-analysis labels. "
        "Do not turn the conversation into formal notes, triples, numbered chains, "
        "or labels such as Status/Relation/Arrival order. "
        "Do not repeat long evidence spans. "
        "Keep the reply to 2-4 concise sentences."
    )
    return [{"role": "system", "content": system}] + history


def generate_assistant_reply(history, *, base_url, model, temperature, max_tokens, timeout_seconds, retries):
    content, _, _ = chat_with_retries(
        base_url=base_url,
        model=model,
        messages=assistant_messages(history),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    return content.strip()


def build_dialogue(item, config, args, seed, profile=None):
    profiles = config.get("layer1", {}).get("evidence_position_profiles") or PROFILE_NAMES
    profile = profile or stable_profile(seed, item["source_id"], profiles)
    total_turns = args.turns or int(config["data"]["history_turns"].get("max", 8))
    user_turns, evidence_turn_map = build_user_turns(item, profile, total_turns)

    messages = []
    exchanges = []
    for user_turn in user_turns:
        messages.append({"role": "user", "content": user_turn["content"]})
        assistant = generate_assistant_reply(
            messages,
            base_url=args.base_url or config["summarizer"]["base_url"],
            model=args.model or config["summarizer"]["model"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        )
        retry_idx = 0
        while forbidden_phrase_hits(assistant) and retry_idx < 2:
            retry_idx += 1
            print(
                f"  retrying assistant turn {user_turn['turn_id']} after forbidden phrase: "
                f"{forbidden_phrase_hits(assistant)}",
                flush=True,
            )
            assistant = generate_assistant_reply(
                messages,
                base_url=args.base_url or config["summarizer"]["base_url"],
                model=args.model or config["summarizer"]["model"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
            )
        messages.append({"role": "assistant", "content": assistant})
        exchanges.append(
            {
                "turn_id": user_turn["turn_id"],
                "user_kind": user_turn["kind"],
                "evidence_steps": user_turn["evidence_steps"],
                "user": user_turn["content"],
                "assistant": assistant,
            }
        )

    rendered = render_dialogue(messages)
    token_count = count_tokens(rendered)
    token_range = config["data"].get("full_history_target_tokens") or {}
    issues = []
    warnings = []
    if token_range.get("enabled"):
        min_target = int(token_range["min"])
        max_target = int(token_range["max"])
        if token_count < min_target:
            warnings.append("full_history_below_observation_range")
        if token_count > max_target:
            warnings.append("full_history_above_observation_range")
    if "Final Answer:" in rendered:
        issues.append("contains_final_answer_marker")
    for phrase in forbidden_phrase_hits(rendered):
        issues.append(f"contains_forbidden_phrase:{phrase}")

    recent_turns = int(config.get("compression", {}).get("hybrid_recent_turns", 1))
    has_recent_evidence = critical_evidence_in_recent_turn(evidence_turn_map, total_turns, recent_turns)

    row = dict(item)
    row.update(
        {
            "dialogue_generation_status": "generated",
            "dialogue_profile": profile,
            "dialogue_turn_count": total_turns,
            "evidence_turn_map": evidence_turn_map,
            "critical_evidence_in_recent_turn": has_recent_evidence,
            "dialogue_messages": messages,
            "dialogue_exchanges": exchanges,
            "dialogue_token_count_proxy": token_count,
            "dialogue_audit": {
                "status": "pass" if not issues else "requires_review",
                "issues": issues,
                "warnings": warnings,
            },
        }
    )
    return row


def write_preview(path, rows):
    lines = ["# Dialogue Generation Preview", ""]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {idx}. {row['source_id']}",
                "",
                f"Profile: {row['dialogue_profile']} | turns: {row['dialogue_turn_count']} | tokens(proxy): {row['dialogue_token_count_proxy']} | audit: {row['dialogue_audit']}",
                "",
                f"Question held out: {row['question']}",
                "",
                f"Evidence turn map: {row['evidence_turn_map']}",
                "",
            ]
        )
        for msg in row["dialogue_messages"][:6]:
            content = msg["content"].replace("\n", " ")
            lines.append(f"**{msg['role']}**: {truncate_words(content, 70)}")
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
    token_counts = []
    profiles = {}
    recent_evidence = {True: 0, False: 0}
    for row in rows:
        status = row["dialogue_audit"]["status"]
        statuses[status] = statuses.get(status, 0) + 1
        profiles[row["dialogue_profile"]] = profiles.get(row["dialogue_profile"], 0) + 1
        recent_evidence[bool(row.get("critical_evidence_in_recent_turn"))] += 1
        token_counts.append(row["dialogue_token_count_proxy"])
        for issue in row["dialogue_audit"]["issues"]:
            issues[issue] = issues.get(issue, 0) + 1
        for warning in row["dialogue_audit"].get("warnings", []):
            warnings[warning] = warnings.get(warning, 0) + 1
    return {
        "num_items": len(rows),
        "audit_status_counts": dict(sorted(statuses.items())),
        "audit_issue_counts": dict(sorted(issues.items())),
        "audit_warning_counts": dict(sorted(warnings.items())),
        "profile_counts": dict(sorted(profiles.items())),
        "critical_evidence_in_recent_turn_counts": {
            "false": recent_evidence[False],
            "true": recent_evidence[True],
        },
        "token_count_proxy": {
            "min": min(token_counts) if token_counts else None,
            "mean": sum(token_counts) / len(token_counts) if token_counts else None,
            "max": max(token_counts) if token_counts else None,
        },
        "source_ids": [row["source_id"] for row in rows],
    }


def generate_dialogues(config, args):
    data_dir = Path(config["paths"].get("data_dir", "data"))
    mc_dir = Path(config["paths"].get("mc_dir", data_dir / "layer1" / "mc"))
    dialogue_dir = Path(config["paths"].get("dialogue_dir", data_dir / "layer1" / "dialogues"))
    audit_dir = Path(config["paths"].get("audit_dir", data_dir / "layer1" / "audits"))
    preview_dir = Path(config["paths"].get("preview_dir", data_dir / "layer1" / "previews"))

    input_path = Path(args.input or mc_dir / f"{args.split}_mc.jsonl")
    output_path = Path(args.output or dialogue_dir / f"{args.split}_dialogues.jsonl")
    rows = read_jsonl(input_path)

    seed = int(config["experiment"].get("random_seed", 42))
    assigned_profiles = assign_profiles(rows, config, args.split, seed)
    generated = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['source_id']}", flush=True)
        generated.append(build_dialogue(row, config, args, seed, assigned_profiles[row["source_id"]]))

    audit_path = Path(args.audit or audit_dir / f"{args.split}_dialogue_audit.json")
    preview_path = Path(args.preview or preview_dir / f"{args.split}_dialogue_preview.md")
    write_jsonl(output_path, generated)
    write_json(audit_path, summarize(generated))
    write_preview(preview_path, generated)
    return {"output": str(output_path), "audit": str(audit_path)}
