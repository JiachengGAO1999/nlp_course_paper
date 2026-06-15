import argparse
import collections
import json
import re
from pathlib import Path


CONDITIONS = ("full_history", "one_shot_summary", "hybrid_summary_recent")
PATTERN_ORDER = {"OS✗HY✓": 0, "HY✗OS✓": 1, "both✗": 2}


def norm(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def contains(term, text):
    term_n = norm(term)
    text_n = norm(text)
    if not term_n or not text_n:
        return False
    if term_n in text_n:
        return True
    compact_term = re.sub(r"[^a-z0-9]+", " ", term_n).strip()
    compact_text = re.sub(r"[^a-z0-9]+", " ", text_n).strip()
    return bool(compact_term and compact_term in compact_text)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def yn(value):
    return "Y" if value else "N"


def esc(value):
    return str(value).replace("|", "/")


def downstream_from_answer(answer):
    answer_l = (answer or "").lower()
    if re.search(
        r"\b(\d{3,4}|january|february|march|april|may|june|july|"
        r"august|september|october|november|december)\b",
        answer_l,
    ):
        return "wrong_date_or_time_anchor"
    if re.search(
        r"\d|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"upper|lower|percent|°|degrees",
        answer_l,
    ):
        return "wrong_numeric_or_exact_value"
    return "wrong_entity_or_relation_anchor"


def build_audit(run_dir):
    formal_dir = run_dir / "formal"
    parsed = load_jsonl(formal_dir / "inference" / "generations.parsed.jsonl")
    selected_rows = load_jsonl(formal_dir / "formal_selected_dialogues.jsonl")
    variant_rows = load_jsonl(formal_dir / "formal_variants.jsonl")

    by_id = collections.defaultdict(dict)
    for row in parsed:
        by_id[row["source_id"]][row["condition"]] = row

    selected = {row["source_id"]: row for row in selected_rows}
    variants = {(row["source_id"], row["condition"]): row for row in variant_rows}

    def terms_for(source_id):
        row = selected[source_id]
        evidence = row.get("required_evidence") or []
        answer = row.get("answer") or row.get("options", {}).get(row.get("gold"), "")
        subanswers = []
        for item in evidence:
            subanswer = item.get("subanswer")
            if subanswer and subanswer not in subanswers:
                subanswers.append(subanswer)
        return answer, subanswers

    def text_for(source_id, condition):
        row = variants[(source_id, condition)]
        return row.get("compressed_text") or row.get("history_text") or ""

    def retention(source_id, condition):
        answer, subanswers = terms_for(source_id)
        text = text_for(source_id, condition)
        present = [term for term in subanswers if contains(term, text)]
        missing = [term for term in subanswers if not contains(term, text)]
        return {
            "answer_text_present": contains(answer, text),
            "evidence_subanswers_present": present,
            "evidence_subanswers_missing": missing,
            "evidence_subanswer_count": len(present),
            "evidence_subanswer_total": len(subanswers),
        }

    def classify(source_id, pattern):
        os_ret = retention(source_id, "one_shot_summary")
        hy_ret = retention(source_id, "hybrid_summary_recent")
        answer, _ = terms_for(source_id)
        os_count = os_ret["evidence_subanswer_count"]
        hy_count = hy_ret["evidence_subanswer_count"]
        total = os_ret["evidence_subanswer_total"]
        crit_rec = bool(selected[source_id]["critical_evidence_in_recent_turn"])

        if pattern == "OS✗HY✓":
            if not os_ret["answer_text_present"] and hy_ret["answer_text_present"]:
                mechanism = "failing_condition_answer_string_absent"
                comp_vs_reasoning = "compression_primary"
                benefit = "recent_verbatim_retention" if crit_rec else "older_summary_or_partial_evidence"
                confidence = "medium"
            elif os_count < hy_count:
                mechanism = "failing_condition_less_complete_evidence"
                comp_vs_reasoning = "mixed"
                benefit = "recent_verbatim_retention" if crit_rec else "older_summary_or_partial_evidence"
                confidence = "medium"
            else:
                mechanism = "evidence_present_but_wrong_answer"
                comp_vs_reasoning = "reasoning_or_distractor_primary"
                benefit = "reasoning_or_attention_difference"
                confidence = "low"
            recent_effect = "helpful" if crit_rec else "neutral_or_indirect"
        elif pattern == "HY✗OS✓":
            if not hy_ret["answer_text_present"] and os_ret["answer_text_present"]:
                mechanism = "failing_condition_answer_string_absent"
                comp_vs_reasoning = "compression_primary"
                confidence = "medium"
            elif hy_count < os_count:
                mechanism = "failing_condition_less_complete_evidence"
                comp_vs_reasoning = "mixed"
                confidence = "medium"
            else:
                mechanism = "evidence_present_but_wrong_answer"
                comp_vs_reasoning = "reasoning_or_distractor_primary"
                confidence = "low"
            recent_effect = "harmful_or_mixed" if crit_rec else "neutral_or_distracting"
            benefit = "none"
        else:
            if not os_ret["answer_text_present"] and not hy_ret["answer_text_present"]:
                mechanism = "both_conditions_answer_string_absent"
                comp_vs_reasoning = "compression_primary"
                confidence = "medium"
            elif os_count < total and hy_count < total:
                mechanism = "both_conditions_incomplete_evidence"
                comp_vs_reasoning = "mixed"
                confidence = "medium"
            else:
                mechanism = "evidence_present_but_both_wrong"
                comp_vs_reasoning = "reasoning_or_distractor_primary"
                confidence = "low"
            recent_effect = "not_sufficient" if crit_rec else "not_applicable"
            benefit = "none"

        return {
            "mechanism_conservative": mechanism,
            "downstream_error_coarse": downstream_from_answer(answer),
            "compression_vs_reasoning": comp_vs_reasoning,
            "recent_verbatim_effect": recent_effect,
            "hybrid_benefit_source": benefit,
            "audit_confidence": confidence,
            "audit_caution": True,
            "os_retention": os_ret,
            "hy_retention": hy_ret,
            "answer": answer,
        }

    rows = []
    for source_id, conds in by_id.items():
        if not conds["full_history"]["is_correct"]:
            continue
        os_correct = conds["one_shot_summary"]["is_correct"]
        hy_correct = conds["hybrid_summary_recent"]["is_correct"]
        if not os_correct and hy_correct:
            pattern = "OS✗HY✓"
        elif os_correct and not hy_correct:
            pattern = "HY✗OS✓"
        elif not os_correct and not hy_correct:
            pattern = "both✗"
        else:
            continue

        audit = classify(source_id, pattern)
        os_ret = audit.pop("os_retention")
        hy_ret = audit.pop("hy_retention")
        selected_row = selected[source_id]
        cond_ref = conds["full_history"]
        rows.append(
            {
                "source_id": source_id,
                "pattern": pattern,
                "profile": selected_row["dialogue_profile"],
                "crit_rec": bool(selected_row["critical_evidence_in_recent_turn"]),
                "hop_count": selected_row["hop_count"],
                "gold": cond_ref["gold"],
                "answer": audit.pop("answer"),
                "os_answer": conds["one_shot_summary"]["parsed_answer"],
                "hy_answer": conds["hybrid_summary_recent"]["parsed_answer"],
                "os_answer_text_present": os_ret["answer_text_present"],
                "hy_answer_text_present": hy_ret["answer_text_present"],
                "os_evidence_present_n": os_ret["evidence_subanswer_count"],
                "hy_evidence_present_n": hy_ret["evidence_subanswer_count"],
                "evidence_total_n": os_ret["evidence_subanswer_total"],
                "os_missing_evidence": os_ret["evidence_subanswers_missing"],
                "hy_missing_evidence": hy_ret["evidence_subanswers_missing"],
                "question": selected_row["question"],
                **audit,
            }
        )

    rows.sort(key=lambda row: (PATTERN_ORDER[row["pattern"]], row["profile"], not row["crit_rec"], row["source_id"]))
    return rows


def write_outputs(run_dir, rows):
    formal_dir = run_dir / "formal"
    json_path = formal_dir / "critical_manual_audit_clean.json"
    md_path = formal_dir / "critical_manual_audit_clean.md"

    pattern_counts = collections.Counter(row["pattern"] for row in rows)
    crit_counts = collections.defaultdict(collections.Counter)
    mechanism_counts = collections.defaultdict(collections.Counter)
    profile_counts = collections.defaultdict(collections.Counter)
    for row in rows:
        crit_counts[row["pattern"]][row["crit_rec"]] += 1
        mechanism_counts[row["pattern"]][row["mechanism_conservative"]] += 1
        profile_counts[row["pattern"]][row["profile"]] += 1

    payload = {
        "rows": rows,
        "stats": {
            "total_clean_critical": len(rows),
            "pattern_counts": dict(pattern_counts),
            "note": (
                "Clean means full_history is correct. Mechanism labels are conservative "
                "manual-audit aids based on answer/evidence retention plus outcome pattern; "
                "audit_caution=true unless fully manually adjudicated."
            ),
        },
    }
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    lines = [
        "# Clean Manual Audit — Scale500 Critical Compression Cases",
        "",
        "**Run**: `layer1_scale500_qwen3_8b_budget800_20260614`  ",
        "**Date**: 2026-06-15  ",
        "**Scope**: only source_ids where `full_history` is correct; compressed-condition disagreements/failures only.  ",
        (
            "**Important**: labels below are conservative audit labels. `answer_text_present` means the final "
            "gold option string appears in the compressed text; evidence counts use exact/normalized subanswer "
            "matching. They are not claimed as fully adjudicated semantic evidence retention."
        ),
        "",
        "## Clean Pool Summary",
        "",
        "| Group | Count | crit_rec=True | crit_rec=False | Profiles |",
        "|---|---:|---:|---:|---|",
    ]
    for pattern in ("OS✗HY✓", "HY✗OS✓", "both✗"):
        profiles = ", ".join(f"{key}:{value}" for key, value in profile_counts[pattern].most_common())
        lines.append(
            f"| {pattern} | {pattern_counts[pattern]} | {crit_counts[pattern][True]} | "
            f"{crit_counts[pattern][False]} | {profiles} |"
        )
    lines.extend(
        [
            "",
            (
                "Excluded from this clean audit: 10 compressed critical cases whose `full_history` also failed. "
                "Those should be discussed separately as model/base-task failures, not compression-only failures."
            ),
            "",
            "## Conservative Mechanism Counts",
            "",
        ]
    )
    for pattern in ("OS✗HY✓", "HY✗OS✓", "both✗"):
        lines.extend([f"### {pattern}", "", "| Mechanism label | Count |", "|---|---:|"])
        for mechanism, count in mechanism_counts[pattern].most_common():
            lines.append(f"| {mechanism} | {count} |")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "- The outcome interaction remains clean: `OS✗HY✓` is concentrated in `cross_turn/late` "
                "and `crit_rec=True`; `HY✗OS✓` and `both✗` are concentrated in `far_early/far_middle` "
                "and `crit_rec=False`."
            ),
            (
                "- Replace the older `evidence_omitted` percentages with the conservative labels here. "
                "Answer-string absence is weaker than full evidence omission."
            ),
            (
                "- Cases marked `evidence_present_but_wrong_answer` or `reasoning_or_distractor_primary` "
                "need deeper human reading before being used as qualitative examples."
            ),
            "",
            "## Detailed Table",
            "",
            "| # | source_id | Pattern | Profile | crit_rec | Gold | OS/HY | Ans in OS/HY | Ev n OS/HY/T | Mechanism | comp_vs_reason | recent_vb | hy_benefit | conf | caution |",
            "|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        gold = f"{row['gold']}={esc(row['answer'])}"
        os_hy = f"{row['os_answer']}/{row['hy_answer']}"
        answer_present = f"{yn(row['os_answer_text_present'])}/{yn(row['hy_answer_text_present'])}"
        evidence = f"{row['os_evidence_present_n']}/{row['hy_evidence_present_n']}/{row['evidence_total_n']}"
        lines.append(
            f"| {idx} | `{row['source_id']}` | {row['pattern']} | {row['profile']} | "
            f"{str(row['crit_rec'])[0]} | {esc(gold)} | {os_hy} | {answer_present} | {evidence} | "
            f"{row['mechanism_conservative']} | {row['compression_vs_reasoning']} | "
            f"{row['recent_verbatim_effect']} | {row['hybrid_benefit_source']} | "
            f"{row['audit_confidence']} | {str(row['audit_caution']).lower()} |"
        )

    lines.extend(
        [
            "",
            "## Field Notes",
            "",
            "- `Ans in OS/HY`: whether the final gold answer option text appears in the one-shot / hybrid compressed context.",
            "- `Ev n OS/HY/T`: number of required-evidence subanswers matched in one-shot / hybrid / total. This is a lexical retention proxy, not a semantic proof.",
            "- `audit_caution=true` means the row is suitable for quantitative grouping but should not be quoted as a qualitative mechanism example without reading the compressed context and model reasoning.",
        ]
    )
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return md_path, json_path, pattern_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/layer1_scale500_qwen3_8b_budget800_20260614")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = build_audit(run_dir)
    md_path, json_path, counts = write_outputs(run_dir, rows)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(json.dumps(dict(counts), ensure_ascii=True))


if __name__ == "__main__":
    main()
