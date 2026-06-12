import re
from pathlib import Path

from src.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from src.llm_client import chat_with_retries
from src.stable import stable_rng
from src.text_utils import count_tokens, render_dialogue


PROFILE_NAMES = ["far_early", "far_middle", "cross_turn", "late"]


def stable_profile(seed, source_id, profiles):
    rng = stable_rng(seed, source_id)
    return profiles[rng.randrange(len(profiles))]


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
        f"I am organizing source notes, and this is note {turn_id}. "
        f"The entry titled \"{ev.get('title')}\" says: {snippet}\n\n"
        f"For my notes, I am treating this as: {ev.get('subquestion')} -> "
        f"{ev.get('subanswer')}. Please keep it as a factual note for later, "
        f"but do not make a final decision yet. If this later combines with other "
        f"notes, keep the source title and the exact relation separate from any "
        f"loose background material. Also preserve whether this note arrived before "
        f"or after other notes, since recency may matter when the history is compressed."
    )


def build_filler_turn(item, turn_id, filler_idx, profile):
    distractors = item.get("distractors") or []
    distractor = distractors[filler_idx % len(distractors)] if distractors else {}
    text = distractor.get("text", "a related but probably irrelevant item")
    templates = [
        (
            f"Side note {turn_id}: I also saw \"{text}\" while collecting nearby "
            f"material. I am not treating it as a core fact yet, because I have not "
            f"connected it to a direct source note in this chain. Please keep it as "
            f"background only. The important distinction for later is that hard facts "
            f"should come from source notes with titles and explicit relations, while "
            f"this kind of nearby item should remain tentative unless a later note "
            f"links it directly. Please also preserve the order in which these notes "
            f"arrive, because later compression should know whether a point was early "
            f"background or later source-backed evidence. Do not collapse this note "
            f"into a conclusion; just keep its status clear."
        ),
        (
            f"Another housekeeping note: some sources use nearby names, places, or dates "
            f"that look tempting, including \"{text}\". I do not want to over-weight it "
            f"unless a later source ties it directly to the main chain. For now, keep it "
            f"separate from the core facts. If it appears again, please preserve that it "
            f"was only a loose contextual lead at this point, not an established answer "
            f"or a resolved conclusion. The safest representation is to keep the item "
            f"visible but clearly marked as unsupported background. The ordering matters "
            f"because this may be separated from later source notes during compression."
        ),
        (
            f"I am also tracking a possible stale lead: \"{text}\". It might be useful "
            f"context, but it should not override the more direct source notes. If later "
            f"notes conflict with this, the later direct evidence should take priority. "
            f"Please remember the state distinction: current source-backed facts outrank "
            f"older or weaker context, and unsupported leads should not be merged into "
            f"the factual chain. This should remain true even if the loose lead sounds "
            f"semantically close to a later source note. Please keep the uncertainty "
            f"attached to the item itself."
        ),
        (
            f"Before the next source note, I want to mark the distinction between hard "
            f"facts and loose context. Hard facts should come from quoted source notes. "
            f"Loose context such as \"{text}\" should remain tentative unless directly "
            f"supported. Please keep that separation explicit, because I will later need "
            f"to know which details were grounded in the quoted sources and which were "
            f"just nearby material from the collection process. Preserving that boundary "
            f"is more important than making the notes sound tidy. The later summary "
            f"should not erase this distinction."
        ),
    ]
    content = templates[filler_idx % len(templates)]
    if profile == "cross_turn" and int(item.get("hop_count", 0)) <= 3:
        content += (
            " Since the source-backed notes are spread apart in this thread, please "
            "keep this middle context visible but clearly lower priority than direct "
            "evidence. The separation between early facts, middle background, and late "
            "facts should remain recoverable after compression."
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


def assistant_messages(history):
    system = (
        "You are a helpful assistant in a note-organization conversation. "
        "Keep responses concise and natural. Do not make final decisions, do not answer "
        "a hidden multiple-choice question, and do not rank options. Acknowledge the "
        "current note, separate hard facts from tentative context, and wait for more."
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


def build_dialogue(item, config, args, seed):
    profiles = config.get("layer1", {}).get("evidence_position_profiles") or PROFILE_NAMES
    profile = stable_profile(seed, item["source_id"], profiles)
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
    min_target = int(config["data"]["full_history_target_tokens"]["min"])
    max_target = int(config["data"]["full_history_target_tokens"]["max"])
    issues = []
    if token_count < min_target:
        issues.append("full_history_below_target")
    if token_count > max_target:
        issues.append("full_history_above_target")
    if "Final Answer:" in rendered:
        issues.append("contains_final_answer_marker")

    row = dict(item)
    row.update(
        {
            "dialogue_generation_status": "generated",
            "dialogue_profile": profile,
            "dialogue_turn_count": total_turns,
            "evidence_turn_map": evidence_turn_map,
            "dialogue_messages": messages,
            "dialogue_exchanges": exchanges,
            "dialogue_token_count_proxy": token_count,
            "dialogue_audit": {"status": "pass" if not issues else "requires_review", "issues": issues},
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
    token_counts = []
    profiles = {}
    for row in rows:
        status = row["dialogue_audit"]["status"]
        statuses[status] = statuses.get(status, 0) + 1
        profiles[row["dialogue_profile"]] = profiles.get(row["dialogue_profile"], 0) + 1
        token_counts.append(row["dialogue_token_count_proxy"])
        for issue in row["dialogue_audit"]["issues"]:
            issues[issue] = issues.get(issue, 0) + 1
    return {
        "num_items": len(rows),
        "audit_status_counts": dict(sorted(statuses.items())),
        "audit_issue_counts": dict(sorted(issues.items())),
        "profile_counts": dict(sorted(profiles.items())),
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
    generated = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['source_id']}", flush=True)
        generated.append(build_dialogue(row, config, args, seed))

    audit_path = audit_dir / f"{args.split}_dialogue_audit.json"
    preview_path = preview_dir / f"{args.split}_dialogue_preview.md"
    write_jsonl(output_path, generated)
    write_json(audit_path, summarize(generated))
    write_preview(preview_path, generated)
    return {"output": str(output_path), "audit": str(audit_path)}
