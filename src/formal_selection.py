import random
from pathlib import Path

from src.io_utils import read_jsonl, write_json, write_jsonl


def normalize_int_key_map(mapping):
    return {int(k): int(v) for k, v in (mapping or {}).items()}


def passing_full_history_ids(results):
    return {
        row["source_id"]
        for row in results
        if row["condition"] == "full_history" and row.get("is_correct")
    }


def token_bin(token_count, tertiles):
    if token_count <= tertiles[0]:
        return "short"
    if token_count <= tertiles[1]:
        return "medium"
    return "long"


def score_selection(rows, hop_target, profile_target):
    hop_counts = {}
    profile_counts = {}
    recent_counts = {True: 0, False: 0}
    for row in rows:
        hop_counts[row["hop_count"]] = hop_counts.get(row["hop_count"], 0) + 1
        profile_counts[row["dialogue_profile"]] = profile_counts.get(row["dialogue_profile"], 0) + 1
        recent_counts[bool(row.get("critical_evidence_in_recent_turn"))] += 1
    hop_penalty = sum(abs(hop_counts.get(k, 0) - v) for k, v in hop_target.items())
    profile_penalty = sum(abs(profile_counts.get(k, 0) - v) for k, v in profile_target.items())
    recent_penalty = abs(recent_counts[True] - recent_counts[False])
    return hop_penalty * 100 + profile_penalty * 10 + recent_penalty


def try_select(candidates, hop_target, profile_target, seed):
    rng = random.Random(seed)
    rows = list(candidates)
    rng.shuffle(rows)
    hop_remaining = dict(hop_target)
    profile_remaining = dict(profile_target)
    selected = []
    for row in rows:
        hop = row["hop_count"]
        profile = row["dialogue_profile"]
        if hop_remaining.get(hop, 0) <= 0:
            continue
        if profile_remaining.get(profile, 0) <= 0:
            continue
        selected.append(row)
        hop_remaining[hop] -= 1
        profile_remaining[profile] -= 1
        if sum(hop_remaining.values()) == 0 and sum(profile_remaining.values()) == 0:
            break
    return selected


def select_formal_set(config, args):
    target_n = int(args.target_n or config["data"].get("formal_num_samples", 40))
    hop_target = normalize_int_key_map((config.get("layer1") or {}).get("formal_hop_allocation") or {2: 8, 3: 20, 4: 12})
    profile_target = (config.get("layer1") or {}).get("formal_profile_allocation") or {
        "far_early": 10,
        "far_middle": 10,
        "cross_turn": 10,
        "late": 10,
    }
    if sum(hop_target.values()) != target_n or sum(profile_target.values()) != target_n:
        raise ValueError("Formal hop/profile targets must both sum to target_n")

    dialogues = read_jsonl(args.dialogues)
    results = read_jsonl(args.results)
    pass_ids = passing_full_history_ids(results)
    candidates = [row for row in dialogues if row["source_id"] in pass_ids]
    if len(candidates) < target_n:
        raise ValueError(f"Only {len(candidates)} full-history-pass candidates for target {target_n}")

    best = []
    best_score = None
    seed = int(config["experiment"].get("random_seed", 42))
    for offset in range(int(args.tries)):
        selected = try_select(candidates, hop_target, profile_target, seed + offset)
        score = score_selection(selected, hop_target, profile_target)
        if best_score is None or score < best_score or (score == best_score and len(selected) > len(best)):
            best = selected
            best_score = score
        if len(selected) == target_n and score == 0:
            break
    if len(best) != target_n:
        raise ValueError(f"Could not select {target_n} rows under targets; best={len(best)} score={best_score}")

    tokens = sorted(row["dialogue_token_count_proxy"] for row in best)
    tertiles = (tokens[len(tokens) // 3], tokens[(2 * len(tokens)) // 3])
    for row in best:
        row["split"] = "formal"
        row["formal_selection"] = {
            "selected_from": "formal_pool",
            "full_history_gate": "pass",
            "token_bin": token_bin(row["dialogue_token_count_proxy"], tertiles),
        }

    output_path = Path(args.output)
    audit_path = Path(args.audit)
    write_jsonl(output_path, best)
    audit = {
        "target_n": target_n,
        "formal_selected_count": len(best),
        "full_history_pass_candidates": len(candidates),
        "full_history_fail_candidates": len(dialogues) - len(candidates),
        "hop_counts": {str(k): sum(1 for row in best if row["hop_count"] == k) for k in sorted(hop_target)},
        "profile_counts": {
            profile: sum(1 for row in best if row["dialogue_profile"] == profile)
            for profile in sorted(profile_target)
        },
        "critical_evidence_in_recent_turn_counts": {
            "false": sum(1 for row in best if not row.get("critical_evidence_in_recent_turn")),
            "true": sum(1 for row in best if row.get("critical_evidence_in_recent_turn")),
        },
        "token_bins": {
            name: sum(1 for row in best if row["formal_selection"]["token_bin"] == name)
            for name in ["short", "medium", "long"]
        },
        "source_ids": [row["source_id"] for row in best],
    }
    write_json(audit_path, audit)
    return audit
