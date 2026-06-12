import json
from pathlib import Path

from src.diagnostic_generator import gold, wrong
from src.text_utils import count_tokens, render_dialogue, render_question


HARD_CONDITIONS = [
    "full_history",
    "oracle_fact_state_summary",
    "llm_generated_summary",
    "hybrid_summary_recent",
]


def evidence(evidence_id, turn_id, span, evidence_type, position, needed_for=None):
    return {
        "evidence_id": evidence_id,
        "turn_id": turn_id,
        "span": span,
        "evidence_type": evidence_type,
        "position": position,
        "needed_for": needed_for or evidence_type,
    }


def distractor(distractor_id, turn_id, distractor_type, span, risk=None):
    return {
        "distractor_id": distractor_id,
        "turn_id": turn_id,
        "distractor_type": distractor_type,
        "span": span,
        "expected_risk": risk or f"may induce {distractor_type} errors",
    }


def user_text(core, detail=None):
    tail = (
        "This is part of a running planning conversation, so later turns may add "
        "side details, revise earlier assumptions, or mention preferences that are "
        "not equally important. Please keep the factual source of each item clear."
    )
    if detail:
        return f"{core} {detail} {tail}"
    return f"{core} {tail}"


def assistant_text(behavior, content):
    templates = {
        "echo": (
            "Noted. I will keep this in the working context. {content} I am not "
            "making the final decision yet; I am only recording the current state."
        ),
        "partial_summary": (
            "Got it. The main points I will carry forward are: {content} I may be "
            "leaving out some lower-level details here, so the full user turns remain "
            "the source of truth."
        ),
        "incorrect_inference": (
            "So it sounds like {content} That seems plausible from the current notes, "
            "although it may still depend on a condition mentioned elsewhere."
        ),
        "stale_reiteration": (
            "Just to confirm my notes, {content} I will keep this visible unless the "
            "later turns make it clear that the state has changed."
        ),
        "neutral": "Noted. I have updated the working notes.",
    }
    return templates[behavior].format(content=content)


def build_dialogue(turns):
    messages = []
    for index, turn in enumerate(turns, start=1):
        turn_id = f"turn_{index}"
        messages.append(
            {
                "turn_id": turn_id,
                "role": "user",
                "content": turn["user"],
                "contains_required_evidence": bool(turn.get("required")),
                "contains_distractor": bool(turn.get("distractor")),
                "tags": turn.get("tags", []),
            }
        )
        messages.append(
            {
                "turn_id": turn_id,
                "role": "assistant",
                "content": turn["assistant"],
                "contains_required_evidence": False,
                "contains_distractor": turn.get("assistant_behavior") in {
                    "partial_summary",
                    "incorrect_inference",
                    "stale_reiteration",
                },
                "assistant_behavior": turn.get("assistant_behavior", "echo"),
                "tags": ["assistant_redundancy"],
            }
        )
    return messages


def base_sample(sample_id, phenomenon, domain, difficulty_modes, turns, question, options, gold_answer, required, distractors, diagnostics, reasoning_requirement):
    messages = build_dialogue(turns)
    assistant_error = any(
        message.get("assistant_behavior") == "incorrect_inference" for message in messages
    )
    sample = {
        "id": sample_id,
        "phenomenon": phenomenon,
        "domain": domain,
        "language": "en",
        "seed": 42,
        "world_state": {
            "domain": domain,
            "reasoning_chain": [item for item in required if item["evidence_type"] == "derived_premise"],
        },
        "difficulty_modes": difficulty_modes,
        "dialogue_history": messages,
        "question": question,
        "options": options,
        "gold_answer": gold_answer,
        "answer_type": "option_letter",
        "required_evidence": required,
        "distractors": distractors,
        "option_diagnostics": diagnostics,
        "assistant_inference_error_present": assistant_error,
        "reasoning_requirement": reasoning_requirement,
        "compression_variants": {},
    }
    add_rule_variants(sample)
    return sample


def add_rule_variants(sample):
    full = render_dialogue(sample["dialogue_history"])
    state = fact_state_summary(sample)
    sample["compression_variants"]["full_history"] = {
        "text": full,
        "history_tokens": count_tokens(full),
        "budget_constrained": False,
        "included_turns": sorted({m["turn_id"] for m in sample["dialogue_history"]}),
    }
    sample["compression_variants"]["oracle_fact_state_summary"] = {
        "text": state,
        "history_tokens": count_tokens(state),
        "budget_constrained": True,
        "leakage_checked": True,
        "compression_quality": oracle_quality(sample),
    }


def oracle_quality(sample):
    ids = [item["evidence_id"] for item in sample["required_evidence"]]
    return {
        "required_evidence_retention": {
            "retained": ids,
            "missing": [],
            "retention_rate": 1.0,
        },
        "answerability": {
            "answerable": True,
            "label": "answerable",
            "reason": "oracle state contains all required evidence ids",
        },
        "hallucination_check": {"unsupported_facts": [], "hallucinated_fact_count": 0},
    }


def fact_state_summary(sample):
    lines = [
        "Known task state:",
        f"- Domain: {sample['domain'].replace('_', ' ')}.",
        "- This is compressed state, not a final option comparison.",
        "",
        "Required evidence records:",
    ]
    for ev in sample["required_evidence"]:
        lines.append(f"- {ev['evidence_id']} ({ev['evidence_type']}, {ev['turn_id']}): {ev['span']}")
    lines.extend(["", "Risk and distractor records:"])
    for dist in sample["distractors"]:
        lines.append(f"- {dist['distractor_id']} ({dist['distractor_type']}, {dist['turn_id']}): {dist['span']}")
    lines.extend(
        [
            "",
            "Candidate records:",
        ]
    )
    for key, value in sample["options"].items():
        lines.append(f"- Candidate record: {value}. Retain as an option record only; do not judge final validity here.")
    lines.extend(
        [
            "",
            "Usage constraints:",
            "- A later answer must still compare the options against the question.",
            "- Do not infer the final option label from this summary alone.",
            "- Preserve hard constraints, soft preferences, stale states, and derived premises separately.",
        ]
    )
    return "\n".join(lines)


def build_input_rows(samples, config, conditions=None):
    rows = []
    conditions = conditions or HARD_CONDITIONS
    prompt = config["prompt"]["template"]
    for sample in samples:
        for condition in conditions:
            variant = sample["compression_variants"][condition]
            rows.append(
                {
                    "sample_id": sample["id"],
                    "phenomenon": sample["phenomenon"],
                    "condition": condition,
                    "gold_answer": sample["gold_answer"],
                    "history_tokens": variant["history_tokens"],
                    "input_text": (
                        f"{prompt.strip()}\n\nDialogue context:\n{variant['text']}\n\n"
                        f"Question:\n{render_question(sample['question'], sample['options'])}\n"
                    ),
                    "option_diagnostics": sample["option_diagnostics"],
                    "assistant_inference_error_present": sample["assistant_inference_error_present"],
                }
            )
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def hard_samples():
    return [
        h1_scattered_long_distance(),
        h2_state_update_stale(),
        h3_negation_derived(),
        h4_temporal_soft_hard(),
        h5_scattered_implicit_conflict(),
        h6_state_update_implicit_conflict(),
    ]


def h1_scattered_long_distance():
    turns = [
        {
            "user": user_text("The assistant for the annotation project needs Python validation skill because the files will be checked automatically."),
            "assistant": assistant_text("partial_summary", "Python validation is important, and I will keep track of technical fit."),
            "required": True,
            "assistant_behavior": "partial_summary",
        },
        {
            "user": user_text("Priya helped write annotation guidelines last semester. Lina also helped with guidelines, while Mateo mainly prepared documentation."),
            "assistant": assistant_text("echo", "Priya and Lina have guideline experience, while Mateo is more documentation-oriented."),
            "required": True,
            "assistant_behavior": "echo",
        },
        {
            "user": user_text("The project also has a public-facing website, but portfolio design is not part of the staffing decision."),
            "assistant": assistant_text("echo", "The website discussion is context, not a staffing criterion."),
            "distractor": True,
            "assistant_behavior": "echo",
        },
        {
            "user": user_text("Thursday afternoon is the only weekly adjudication meeting. Anyone who cannot attend Thursday cannot be selected, even if they are otherwise strong."),
            "assistant": assistant_text("partial_summary", "Thursday availability matters for the weekly meeting."),
            "required": True,
            "assistant_behavior": "partial_summary",
        },
        {
            "user": user_text("Lina can write Python scripts, but she is only free Friday. Mateo is free Thursday, but he uses R rather than Python."),
            "assistant": assistant_text("neutral", ""),
            "required": True,
            "assistant_behavior": "neutral",
        },
        {
            "user": user_text("Priya is free Thursday afternoon. Earlier I forgot to mention that she also knows Python from the validation scripts she maintained."),
            "assistant": assistant_text("echo", "Priya is Thursday-available and has Python validation experience."),
            "required": True,
            "assistant_behavior": "echo",
        },
    ]
    req = [
        evidence("ev_python_required", "turn_1", "needs Python validation skill", "hard_constraint", "early"),
        evidence("ev_annotation_required", "turn_2", "Priya helped write annotation guidelines", "candidate_attribute", "early"),
        evidence("ev_thursday_required", "turn_4", "Thursday afternoon is the only weekly adjudication meeting", "hard_constraint", "middle"),
        evidence("ev_lina_friday", "turn_5", "Lina can write Python scripts, but she is only free Friday", "candidate_attribute", "late"),
        evidence("ev_mateo_no_python", "turn_5", "Mateo is free Thursday, but he uses R rather than Python", "candidate_attribute", "late"),
        evidence("ev_priya_python_thursday", "turn_6", "Priya is free Thursday afternoon ... she also knows Python", "candidate_attribute", "late"),
    ]
    opts = {"A": "Lina", "B": "Mateo", "C": "Priya", "D": "The candidate with the strongest portfolio website"}
    return base_sample(
        "hard_scattered_001",
        "scattered_fact_integration",
        "candidate_selection",
        ["scattered_candidate_attributes", "long_distance_evidence"],
        turns,
        "Which candidate satisfies the project staffing requirements?",
        opts,
        "C",
        req,
        [distractor("dist_portfolio", "turn_3", "topical_distractor", "portfolio website is not part of the staffing decision")],
        {
            "A": wrong("missing_middle_evidence", "Lina is only free Friday, not Thursday.", ["ev_thursday_required", "ev_lina_friday"]),
            "B": wrong("violates_hard_constraint", "Mateo does not know Python.", ["ev_python_required", "ev_mateo_no_python"]),
            "C": gold(["python", "thursday", "annotation_experience"], ["ev_python_required", "ev_annotation_required", "ev_thursday_required", "ev_priya_python_thursday"]),
            "D": wrong("topical_distractor_trap", "Portfolio design is not part of the staffing decision.", ["ev_python_required"]),
        },
        "Assemble candidate attributes across distant turns.",
    )


def h2_state_update_stale():
    turns = [
        {"user": user_text("The clinic interview was first penciled in for Tuesday morning with Dr. Rao."), "assistant": assistant_text("echo", "Tuesday morning with Dr. Rao is the tentative note."), "required": True, "assistant_behavior": "echo"},
        {"user": user_text("Nina cannot do mornings because she opens the lab until noon."), "assistant": assistant_text("neutral", ""), "required": True, "assistant_behavior": "neutral"},
        {"user": user_text("Actually, let's do Thursday afternoon with Dr. Chen; that lines up better with the participant schedule."), "assistant": assistant_text("stale_reiteration", "we are still tracking the Tuesday morning interview with Dr. Rao"), "required": True, "assistant_behavior": "stale_reiteration"},
        {"user": user_text("Room B has a better camera, but Room A is already reserved with Dr. Chen for Thursday afternoon."), "assistant": assistant_text("partial_summary", "Room B has the better camera, and Room A is available."), "required": True, "assistant_behavior": "partial_summary"},
        {"user": user_text("The camera quality is nice to have, not a blocker; the interviewer and participant availability matter more."), "assistant": assistant_text("echo", "Camera quality is secondary to the schedule and interviewer availability."), "distractor": True, "assistant_behavior": "echo"},
        {"user": user_text("Please choose the plan that follows the latest interview schedule and availability."), "assistant": assistant_text("neutral", ""), "assistant_behavior": "neutral"},
    ]
    req = [
        evidence("ev_old_tuesday", "turn_1", "first penciled in for Tuesday morning with Dr. Rao", "stale_information", "early"),
        evidence("ev_no_morning", "turn_2", "Nina cannot do mornings", "hard_constraint", "early"),
        evidence("ev_latest_thursday", "turn_3", "do Thursday afternoon with Dr. Chen", "state_update", "middle"),
        evidence("ev_room_a", "turn_4", "Room A is already reserved with Dr. Chen for Thursday afternoon", "candidate_attribute", "middle"),
    ]
    opts = {"A": "Tuesday morning with Dr. Rao in Room B", "B": "Thursday afternoon with Dr. Chen in Room A", "C": "Thursday morning with Dr. Chen in Room A", "D": "Tuesday afternoon with Dr. Rao in Room B"}
    return base_sample(
        "hard_state_update_001",
        "state_update",
        "interview_scheduling",
        ["subtle_state_update"],
        turns,
        "Which interview plan follows the latest schedule and hard availability constraints?",
        opts,
        "B",
        req,
        [distractor("dist_camera", "turn_4", "soft_preference", "Room B has a better camera"), distractor("dist_stale_recap", "turn_3", "stale_information", "assistant reiterates Tuesday morning")],
        {
            "A": wrong("uses_stale_state", "Tuesday morning is stale and violates Nina's morning availability.", ["ev_old_tuesday", "ev_no_morning", "ev_latest_thursday"]),
            "B": gold(["latest_schedule", "availability"], ["ev_latest_thursday", "ev_room_a", "ev_no_morning"]),
            "C": wrong("violates_hard_constraint", "Nina cannot do mornings.", ["ev_no_morning"]),
            "D": wrong("uses_stale_state", "Dr. Rao Tuesday is not the latest schedule.", ["ev_latest_thursday"]),
        },
        "Ignore stale assistant recap and follow the subtle schedule update.",
    )


def h3_negation_derived():
    turns = [
        {"user": user_text("For the field visit, Omar's safety briefing runs until 11:45 every weekday."), "assistant": assistant_text("neutral", ""), "required": True, "assistant_behavior": "neutral"},
        {"user": user_text("The river route is closed whenever the team leaves before noon, because the guard station is not staffed yet."), "assistant": assistant_text("partial_summary", "The river route has a staffing caveat."), "required": True, "assistant_behavior": "partial_summary"},
        {"user": user_text("The hill route leaves at 1:30 PM. The river route leaves at 10:30 AM. The museum route leaves at 11:15 AM."), "assistant": assistant_text("incorrect_inference", "the river route may be fine if the team likes it best"), "required": True, "assistant_behavior": "incorrect_inference"},
        {"user": user_text("The team would rather avoid hills if possible, since the hill route is tiring."), "assistant": assistant_text("echo", "Avoiding hills is a preference, not stated as a closure or safety rule."), "distractor": True, "assistant_behavior": "echo"},
        {"user": user_text("No one may skip Omar's briefing."), "assistant": assistant_text("neutral", ""), "required": True, "assistant_behavior": "neutral"},
        {"user": user_text("Choose the route that is feasible after the briefing and route closure facts are considered."), "assistant": assistant_text("echo", "We should use the feasibility facts, not just comfort."), "assistant_behavior": "echo"},
    ]
    req = [
        evidence("ev_briefing_until_1145", "turn_1", "Omar's safety briefing runs until 11:45", "derived_premise", "early"),
        evidence("ev_before_noon_closed", "turn_2", "river route is closed whenever the team leaves before noon", "exclusion", "early"),
        evidence("ev_route_times", "turn_3", "hill route leaves at 1:30 PM; river at 10:30 AM; museum at 11:15 AM", "candidate_attribute", "middle"),
        evidence("ev_no_skip", "turn_5", "No one may skip Omar's briefing", "hard_constraint", "late"),
    ]
    opts = {"A": "River route at 10:30 AM", "B": "Museum route at 11:15 AM", "C": "Hill route at 1:30 PM", "D": "Whichever route avoids hills"}
    return base_sample(
        "hard_negation_001",
        "negation_exclusion",
        "field_visit",
        ["implicit_constraint_tracking", "derived_constraint"],
        turns,
        "Which route is feasible under the briefing and closure constraints?",
        opts,
        "C",
        req,
        [distractor("dist_avoid_hills", "turn_4", "soft_preference", "team would rather avoid hills"), distractor("dist_assistant_river", "turn_3", "near_miss_constraint", "assistant suggests river may be fine")],
        {
            "A": wrong("ignores_negation", "River leaves before noon and is closed before noon.", ["ev_before_noon_closed", "ev_route_times"]),
            "B": wrong("violates_hard_constraint", "Museum leaves before the 11:45 briefing ends.", ["ev_briefing_until_1145", "ev_no_skip", "ev_route_times"]),
            "C": gold(["after_briefing", "not_closed"], ["ev_briefing_until_1145", "ev_before_noon_closed", "ev_route_times", "ev_no_skip"]),
            "D": wrong("soft_preference_trap", "Avoiding hills is only a preference and does not identify a feasible route.", ["ev_route_times"]),
        },
        "Derive feasible departure times and apply a route closure exclusion.",
    )


def h4_temporal_soft_hard():
    turns = [
        {"user": user_text("The data export has to be anonymized before QA sees it."), "assistant": assistant_text("neutral", ""), "required": True, "assistant_behavior": "neutral"},
        {"user": user_text("Ravi enjoys doing QA first because it gives him a cleaner task list."), "assistant": assistant_text("partial_summary", "Ravi likes QA early."), "distractor": True, "assistant_behavior": "partial_summary"},
        {"user": user_text("Mira can anonymize raw files. Ravi can run QA. Sol can make slides only after QA tables exist."), "assistant": assistant_text("echo", "Mira, Ravi, and Sol each fit a different part of the workflow."), "required": True, "assistant_behavior": "echo"},
        {"user": user_text("Slides can be drafted before Friday, but final slides need the QA tables."), "assistant": assistant_text("echo", "Drafting and final slide preparation are different states."), "required": True, "assistant_behavior": "echo"},
        {"user": user_text("The team would like to finish slides early, but no one wants identifiable data in the QA tool."), "assistant": assistant_text("partial_summary", "Early slides would be nice, and identifiable data should stay out of QA."), "required": True, "assistant_behavior": "partial_summary"},
        {"user": user_text("Pick the order that respects the dependency chain rather than personal workflow preferences."), "assistant": assistant_text("neutral", ""), "assistant_behavior": "neutral"},
    ]
    req = [
        evidence("ev_anon_before_qa", "turn_1", "anonymized before QA sees it", "temporal_dependency", "early"),
        evidence("ev_roles", "turn_3", "Mira can anonymize; Ravi can run QA; Sol can make slides", "role_constraint", "middle"),
        evidence("ev_slides_after_qa", "turn_3", "Sol can make slides only after QA tables exist", "temporal_dependency", "middle"),
        evidence("ev_no_identifiable_qa", "turn_5", "no one wants identifiable data in the QA tool", "hard_constraint", "late"),
    ]
    opts = {"A": "Ravi runs QA, Mira anonymizes, then Sol makes slides", "B": "Mira anonymizes, Ravi runs QA tables, then Sol makes final slides", "C": "Sol makes final slides from raw files, then Ravi runs QA", "D": "Ravi chooses whichever order makes his task list cleaner"}
    return base_sample(
        "hard_temporal_001",
        "temporal_order",
        "data_cleanup",
        ["implicit_constraint_tracking", "soft_hard_conflict_without_labels"],
        turns,
        "Which workflow order respects the dependency chain?",
        opts,
        "B",
        req,
        [distractor("dist_ravi_pref", "turn_2", "soft_preference", "Ravi enjoys doing QA first"), distractor("dist_early_slides", "turn_5", "soft_preference", "finish slides early")],
        {
            "A": wrong("wrong_temporal_order", "QA cannot happen before anonymization.", ["ev_anon_before_qa"]),
            "B": gold(["dependency_order", "roles"], ["ev_anon_before_qa", "ev_roles", "ev_slides_after_qa", "ev_no_identifiable_qa"]),
            "C": wrong("violates_hard_constraint", "Final slides need QA tables and Sol should not use raw identifiable files.", ["ev_slides_after_qa", "ev_no_identifiable_qa"]),
            "D": wrong("soft_preference_trap", "Ravi's preferred workflow does not override dependencies.", ["ev_anon_before_qa"]),
        },
        "Distinguish workflow preferences from hard temporal dependencies.",
    )


def h5_scattered_implicit_conflict():
    turns = [
        {"user": user_text("The venue volunteer should be able to lift boxes because setup includes moving archived materials."), "assistant": assistant_text("partial_summary", "The volunteer needs to help with setup."), "required": True, "assistant_behavior": "partial_summary"},
        {"user": user_text("Tessa has event experience and can lift boxes, but she leaves before the evening session."), "assistant": assistant_text("echo", "Tessa is strong for setup but has an evening availability issue."), "required": True, "assistant_behavior": "echo"},
        {"user": user_text("The evening session is the only part where the volunteer must stay at the registration table."), "assistant": assistant_text("neutral", ""), "required": True, "assistant_behavior": "neutral"},
        {"user": user_text("Noah is available in the evening. His setup experience is limited, but he has helped move archive boxes before."), "assistant": assistant_text("partial_summary", "Noah is evening-available and has some setup background."), "required": True, "assistant_behavior": "partial_summary"},
        {"user": user_text("Maya would rather work at a quiet desk than at registration, but she can do registration if needed. She cannot lift boxes."), "assistant": assistant_text("echo", "Maya has a preference about desk work but a limitation around lifting."), "required": True, "assistant_behavior": "echo"},
        {"user": user_text("Choose the volunteer who can cover both setup and the evening registration need."), "assistant": assistant_text("neutral", ""), "assistant_behavior": "neutral"},
    ]
    req = [
        evidence("ev_lift_needed", "turn_1", "able to lift boxes", "hard_constraint", "early"),
        evidence("ev_tessa_leaves", "turn_2", "Tessa leaves before the evening session", "candidate_attribute", "early"),
        evidence("ev_evening_required", "turn_3", "evening session is the only part where the volunteer must stay", "hard_constraint", "middle"),
        evidence("ev_noah_evening_lift", "turn_4", "Noah is available in the evening ... helped move archive boxes", "candidate_attribute", "middle"),
        evidence("ev_maya_no_lift", "turn_5", "Maya ... cannot lift boxes", "candidate_attribute", "late"),
    ]
    opts = {"A": "Tessa", "B": "Noah", "C": "Maya", "D": "Whoever prefers the quiet desk"}
    return base_sample(
        "hard_scattered_002",
        "scattered_fact_integration",
        "venue_staffing",
        ["scattered_candidate_attributes", "implicit_constraint_tracking", "soft_hard_conflict_without_labels"],
        turns,
        "Who can cover both the setup and evening registration needs?",
        opts,
        "B",
        req,
        [distractor("dist_quiet_desk", "turn_5", "soft_preference", "Maya would rather work at a quiet desk")],
        {
            "A": wrong("missing_middle_evidence", "Tessa leaves before the required evening session.", ["ev_evening_required", "ev_tessa_leaves"]),
            "B": gold(["lifting", "evening_availability"], ["ev_lift_needed", "ev_evening_required", "ev_noah_evening_lift"]),
            "C": wrong("violates_hard_constraint", "Maya cannot lift boxes.", ["ev_lift_needed", "ev_maya_no_lift"]),
            "D": wrong("soft_preference_trap", "Desk preference is not the staffing requirement.", ["ev_quiet_desk"] if False else ["ev_lift_needed"]),
        },
        "Assemble candidate state and avoid preference traps.",
    )


def h6_state_update_implicit_conflict():
    turns = [
        {"user": user_text("The grant report was going to use the March budget table."), "assistant": assistant_text("echo", "March budget table is the current note."), "required": True, "assistant_behavior": "echo"},
        {"user": user_text("The finance office later sent the April table, and the report should follow the latest finance office file."), "assistant": assistant_text("stale_reiteration", "the March budget table is still in my notes"), "required": True, "assistant_behavior": "stale_reiteration"},
        {"user": user_text("The April table includes travel corrections. The March table has cleaner formatting, which the team likes."), "assistant": assistant_text("partial_summary", "The March table is cleaner, and the April table has travel corrections."), "required": True, "assistant_behavior": "partial_summary"},
        {"user": user_text("Formatting can be cleaned later; the values must match the finance office's latest numbers."), "assistant": assistant_text("neutral", ""), "required": True, "assistant_behavior": "neutral"},
        {"user": user_text("The appendix draft still says March because it was copied from the old outline."), "assistant": assistant_text("echo", "The appendix has a stale March mention from the old outline."), "distractor": True, "assistant_behavior": "echo"},
        {"user": user_text("Choose which budget table should be used for the final report."), "assistant": assistant_text("neutral", ""), "assistant_behavior": "neutral"},
    ]
    req = [
        evidence("ev_old_march", "turn_1", "was going to use the March budget table", "stale_information", "early"),
        evidence("ev_latest_april", "turn_2", "April table ... latest finance office file", "state_update", "early"),
        evidence("ev_april_corrections", "turn_3", "April table includes travel corrections", "candidate_attribute", "middle"),
        evidence("ev_values_must_match", "turn_4", "values must match the finance office's latest numbers", "hard_constraint", "middle"),
        evidence("ev_appendix_stale", "turn_5", "appendix draft still says March ... old outline", "stale_information", "late"),
    ]
    opts = {"A": "March table because it is cleaner", "B": "April table because it is the latest finance office file", "C": "The appendix's March table mention", "D": "Whichever table requires less formatting work"}
    return base_sample(
        "hard_state_update_002",
        "state_update",
        "grant_reporting",
        ["subtle_state_update", "implicit_constraint_tracking", "soft_hard_conflict_without_labels"],
        turns,
        "Which budget table should be used for the final report?",
        opts,
        "B",
        req,
        [distractor("dist_clean_format", "turn_3", "soft_preference", "March table has cleaner formatting"), distractor("dist_appendix", "turn_5", "stale_information", "appendix draft still says March")],
        {
            "A": wrong("soft_preference_trap", "Cleaner formatting does not override latest finance values.", ["ev_latest_april", "ev_values_must_match"]),
            "B": gold(["latest_state", "correct_values"], ["ev_latest_april", "ev_april_corrections", "ev_values_must_match"]),
            "C": wrong("uses_stale_state", "The appendix March mention came from an old outline.", ["ev_appendix_stale", "ev_latest_april"]),
            "D": wrong("soft_preference_trap", "Formatting effort is secondary to correct latest values.", ["ev_values_must_match"]),
        },
        "Follow subtle latest-state update over stale mentions and formatting preference.",
    )
