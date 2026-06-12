import json
import random
from pathlib import Path

from src.text_utils import approximate_fit, count_tokens, render_dialogue, render_question


CONDITIONS = [
    "full_history",
    "sliding_window",
    "user_only_history",
    "oracle_dialogue_summary",
    "oracle_fact_state_summary",
]


def verbose_assistant(topic, current_state, caution):
    return (
        f"Understood. I will keep track of the {topic} details as working context. "
        f"At this point, the relevant state is: {current_state}. "
        f"I will treat later corrections as updates rather than as extra alternatives. "
        f"For now, I am not making the final choice; I am only recording the constraint "
        f"so that the eventual answer can be based on the latest consistent state. "
        f"One caution is that {caution}. "
        "I will also keep separate notes for hard constraints, soft preferences, "
        "old tentative information, and unrelated details. That separation matters "
        "because a later compressed history should preserve the decision-relevant "
        "state without letting repeated confirmations or attractive side details "
        "become mistaken for requirements. I will avoid treating this local summary "
        "as a final answer, because the eventual question may require combining "
        "this turn with earlier and later turns. If the history is compressed later, "
        "the compression should preserve the source and status of this information."
    )


def build_messages(turn_specs):
    messages = []
    for idx, spec in enumerate(turn_specs, start=1):
        turn_id = f"turn_{idx}"
        messages.append(
            {
                "turn_id": turn_id,
                "role": "user",
                "content": expand_user_content(spec["user"]),
                "tags": spec.get("tags", []),
            }
        )
        messages.append(
            {
                "turn_id": turn_id,
                "role": "assistant",
                "content": spec["assistant"],
                "tags": ["assistant_redundancy"] + spec.get("assistant_tags", []),
            }
        )
    return messages


def expand_user_content(content):
    return (
        content
        + " I am including this detail as part of the running conversation rather "
        "than as a standalone document. Some pieces may become stale after later "
        "updates, and some pieces are only background context. Please preserve the "
        "distinction between hard constraints, soft preferences, tentative notes, "
        "and unrelated side information. I may refer back to this later, so the "
        "important part is not just the topic but whether the detail is current, "
        "mandatory, optional, or merely explanatory."
    )


def sample_specs():
    return [
        meeting_state_update(),
        travel_negation(),
        task_temporal_order(),
        candidate_scattered_facts(),
    ]


def meeting_state_update():
    turn_specs = [
        {
            "user": (
                "We are arranging a design review for Alice, Ben, and Cara. The first "
                "placeholder was Wednesday morning in Room 204 because Alice said she "
                "could attend then. The room has a projector and a whiteboard, which is "
                "convenient but not required for the final decision."
            ),
            "assistant": verbose_assistant(
                "meeting",
                "a tentative Wednesday morning review in Room 204",
                "room equipment is topical but should not override availability",
            ),
            "tags": ["stale_information", "topical_distractor"],
        },
        {
            "user": (
                "Ben later clarified that he cannot attend anything before noon this "
                "week because of a lab duty. He also mentioned that he prefers rooms "
                "near the elevator, but he said this is only a preference and not a hard "
                "requirement."
            ),
            "assistant": verbose_assistant(
                "meeting",
                "Ben has a hard no-before-noon constraint and a soft room-location preference",
                "soft preferences must not be treated as hard exclusions",
            ),
            "tags": ["required", "soft_preference"],
        },
        {
            "user": (
                "Cara is available on Friday afternoon and Friday evening. She is not "
                "available on Wednesday because she will be traveling. The team also "
                "discussed whether snacks should be ordered, but that decision is not "
                "part of the scheduling question."
            ),
            "assistant": verbose_assistant(
                "meeting",
                "Cara rules out Wednesday and allows Friday afternoon or evening",
                "snack planning is unrelated to selecting a valid meeting slot",
            ),
            "tags": ["required", "topical_distractor"],
        },
        {
            "user": (
                "The organizer then moved the review from Wednesday to Friday afternoon. "
                "This update replaces the old Wednesday placeholder. Please remember "
                "that Friday afternoon is the latest intended date unless another update "
                "appears."
            ),
            "assistant": verbose_assistant(
                "meeting",
                "the latest intended review time is Friday afternoon",
                "the older Wednesday placeholder is stale after this update",
            ),
            "tags": ["required", "state_update"],
        },
        {
            "user": (
                "Room 310 is available on Friday afternoon, while Room 102 is available "
                "on Saturday morning. Room 310 is farther from the elevator, but it has "
                "enough seats and satisfies the scheduling need."
            ),
            "assistant": verbose_assistant(
                "meeting",
                "Room 310 can host the Friday afternoon meeting",
                "distance from the elevator remains a soft preference issue",
            ),
            "tags": ["required", "soft_preference"],
        },
        {
            "user": (
                "There was also a suggestion to use Friday evening because it would give "
                "more preparation time. However, Ben cannot attend after 5 PM, so Friday "
                "evening should not be treated as valid."
            ),
            "assistant": verbose_assistant(
                "meeting",
                "Friday evening is a near miss because it violates Ben's hard constraint",
                "preparation convenience does not override availability",
            ),
            "tags": ["near_miss_constraint", "required"],
        },
    ]
    return {
        "id": "smoke_state_update_001",
        "phenomenon": "state_update",
        "domain": "meeting_scheduling",
        "dialogue_history": build_messages(turn_specs),
        "question": "Which meeting option satisfies the latest schedule and all hard constraints?",
        "options": {
            "A": "Wednesday morning in Room 204",
            "B": "Friday afternoon in Room 310",
            "C": "Friday evening in Room 310",
            "D": "Saturday morning in Room 102",
        },
        "gold_answer": "B",
        "required_evidence": [
            evidence("ev_ben_no_morning", "turn_2", "Ben cannot attend anything before noon.", "hard_constraint", "early"),
            evidence("ev_cara_no_wed", "turn_3", "Cara is not available on Wednesday.", "hard_constraint", "middle"),
            evidence("ev_latest_friday", "turn_4", "The review was moved from Wednesday to Friday afternoon.", "state_update", "middle"),
            evidence("ev_ben_no_evening", "turn_6", "Ben cannot attend after 5 PM.", "hard_constraint", "late"),
        ],
        "distractors": [
            distractor("dist_projector", "turn_1", "topical_distractor", "projector and whiteboard"),
            distractor("dist_wednesday", "turn_1", "stale_information", "Wednesday morning placeholder"),
            distractor("dist_elevator", "turn_2", "soft_preference", "near the elevator"),
            distractor("dist_evening", "turn_6", "near_miss_constraint", "Friday evening gives more preparation time"),
        ],
        "option_diagnostics": {
            "A": wrong("uses_stale_state", "The Wednesday placeholder was replaced and Cara cannot attend Wednesday.", ["ev_cara_no_wed", "ev_latest_friday"]),
            "B": gold(["latest_state", "all_hard_constraints"], ["ev_latest_friday", "ev_ben_no_morning", "ev_cara_no_wed", "ev_ben_no_evening"]),
            "C": wrong("violates_hard_constraint", "Ben cannot attend after 5 PM.", ["ev_ben_no_evening"]),
            "D": wrong("wrong_temporal_order", "Saturday morning is not the latest scheduled review time.", ["ev_latest_friday"]),
        },
        "reasoning_requirement": "Use the latest schedule update and eliminate options that violate attendee availability.",
    }


def travel_negation():
    turn_specs = [
        {
            "user": (
                "I am comparing weekend travel plans for Dana and Eli. At first, the "
                "coastal train looked attractive because it is scenic and inexpensive. "
                "The scenic route matters only as a preference, not as a hard constraint."
            ),
            "assistant": verbose_assistant(
                "travel plan",
                "the coastal train is an attractive but preference-based candidate",
                "scenic value should not override hard travel constraints",
            ),
            "tags": ["soft_preference"],
        },
        {
            "user": (
                "Dana must arrive before 2 PM because of a workshop check-in. Any plan "
                "arriving after 2 PM should be excluded even if it is cheaper or more "
                "comfortable."
            ),
            "assistant": verbose_assistant(
                "travel plan",
                "arrival before 2 PM is a hard constraint",
                "price and comfort are secondary after feasibility",
            ),
            "tags": ["required"],
        },
        {
            "user": (
                "Eli cannot take buses this weekend because of motion sickness. This is "
                "a hard exclusion. Eli is fine with trains or flights, and does not care "
                "much about seat class."
            ),
            "assistant": verbose_assistant(
                "travel plan",
                "bus options are excluded, while train or flight options remain possible",
                "transport mode exclusions must be preserved",
            ),
            "tags": ["required", "negation"],
        },
        {
            "user": (
                "The coastal train arrives at 3:10 PM. The express flight arrives at "
                "1:20 PM. The city bus arrives at 1:10 PM. The overnight train arrives "
                "at 8 AM but requires leaving the previous night."
            ),
            "assistant": verbose_assistant(
                "travel plan",
                "the flight and overnight train meet the arrival-time constraint, while the coastal train is late",
                "the bus arrival time alone is not enough because buses are excluded",
            ),
            "tags": ["required", "near_miss_constraint"],
        },
        {
            "user": (
                "Dana also prefers not to leave the previous night unless necessary. "
                "This is a soft preference. If another valid same-day option exists, "
                "the same-day option should be preferred."
            ),
            "assistant": verbose_assistant(
                "travel plan",
                "previous-night departure is disfavored but not impossible",
                "this preference should be used only after hard constraints are satisfied",
            ),
            "tags": ["soft_preference", "required"],
        },
        {
            "user": (
                "There was a note about hotel breakfast near the station, but breakfast "
                "is not part of the transportation decision. The final choice should be "
                "based on arrival time, transport exclusions, and the previous-night "
                "preference only if there is a valid same-day alternative."
            ),
            "assistant": verbose_assistant(
                "travel plan",
                "breakfast is a topical distractor and the transport constraints remain decisive",
                "unrelated amenities should not enter the final travel choice",
            ),
            "tags": ["topical_distractor"],
        },
    ]
    return {
        "id": "smoke_negation_001",
        "phenomenon": "negation_exclusion",
        "domain": "travel_planning",
        "dialogue_history": build_messages(turn_specs),
        "question": "Which travel plan should be selected under all hard constraints and the stated soft preference?",
        "options": {
            "A": "Take the coastal train arriving at 3:10 PM",
            "B": "Take the express flight arriving at 1:20 PM",
            "C": "Take the city bus arriving at 1:10 PM",
            "D": "Take the overnight train arriving at 8:00 AM",
        },
        "gold_answer": "B",
        "required_evidence": [
            evidence("ev_arrive_before_2", "turn_2", "Dana must arrive before 2 PM.", "hard_constraint", "early"),
            evidence("ev_no_bus", "turn_3", "Eli cannot take buses this weekend.", "exclusion", "middle"),
            evidence("ev_arrival_times", "turn_4", "The express flight arrives at 1:20 PM.", "candidate_attribute", "middle"),
            evidence("ev_no_previous_night", "turn_5", "Dana prefers not to leave the previous night if a valid same-day option exists.", "soft_preference", "late"),
        ],
        "distractors": [
            distractor("dist_scenic", "turn_1", "soft_preference", "scenic coastal train"),
            distractor("dist_bus_time", "turn_4", "near_miss_constraint", "city bus arrives before 2 PM"),
            distractor("dist_breakfast", "turn_6", "topical_distractor", "hotel breakfast"),
        ],
        "option_diagnostics": {
            "A": wrong("violates_hard_constraint", "It arrives after 2 PM.", ["ev_arrive_before_2", "ev_arrival_times"]),
            "B": gold(["arrival_before_deadline", "not_bus", "same_day"], ["ev_arrive_before_2", "ev_no_bus", "ev_arrival_times", "ev_no_previous_night"]),
            "C": wrong("ignores_negation", "It is a bus, and buses are excluded for Eli.", ["ev_no_bus"]),
            "D": wrong("soft_preference_trap", "It is valid but violates the soft preference when a same-day valid option exists.", ["ev_no_previous_night"]),
        },
        "reasoning_requirement": "Apply hard exclusions before using the soft preference as a tie-breaker.",
    }


def task_temporal_order():
    turn_specs = [
        {
            "user": (
                "We need assign a three-step data cleanup task. The raw survey export "
                "must be anonymized before anyone runs quality checks, because the QA "
                "tool should not see names or email addresses."
            ),
            "assistant": verbose_assistant(
                "task assignment",
                "anonymization must precede quality checks",
                "dependency order matters more than convenience",
            ),
            "tags": ["required", "temporal_order"],
        },
        {
            "user": (
                "Mira can anonymize data and prepare documentation. Omar can run QA "
                "checks and produce summary tables. Nia can prepare slides, but she is "
                "not authorized to access raw identifiable data."
            ),
            "assistant": verbose_assistant(
                "task assignment",
                "Mira can handle anonymization, Omar can handle QA, and Nia must avoid raw identifiable data",
                "authorization constraints must be respected",
            ),
            "tags": ["required"],
        },
        {
            "user": (
                "Originally, Omar was listed first because he is fastest with the QA "
                "tool. That ordering was just a draft and should not override the "
                "dependency that anonymization comes first."
            ),
            "assistant": verbose_assistant(
                "task assignment",
                "Omar-first is stale draft ordering, not the final dependency order",
                "speed should not reverse required task order",
            ),
            "tags": ["stale_information", "near_miss_constraint"],
        },
        {
            "user": (
                "The documentation can be written after QA, but the slides should be "
                "prepared only after the summary tables exist. The slides do not need "
                "raw data access; they only need the final table outputs."
            ),
            "assistant": verbose_assistant(
                "task assignment",
                "slides depend on summary tables, and documentation can follow QA",
                "Nia can work after outputs are safe",
            ),
            "tags": ["required", "temporal_order"],
        },
        {
            "user": (
                "There is also a preference that Mira avoid presentation work because "
                "she has another meeting. This is only a preference; the hard issues are "
                "data authorization and dependency order."
            ),
            "assistant": verbose_assistant(
                "task assignment",
                "Mira presentation avoidance is a soft preference",
                "hard dependencies remain the primary constraints",
            ),
            "tags": ["soft_preference"],
        },
        {
            "user": (
                "Please choose the assignment sequence that respects all dependencies: "
                "first anonymize, then QA summary tables, then slides. Ignore the old "
                "draft that put Omar first."
            ),
            "assistant": verbose_assistant(
                "task assignment",
                "the latest sequence is anonymize, then QA tables, then slides",
                "the older Omar-first draft is stale",
            ),
            "tags": ["required", "state_update"],
        },
    ]
    return {
        "id": "smoke_temporal_001",
        "phenomenon": "temporal_order",
        "domain": "task_assignment",
        "dialogue_history": build_messages(turn_specs),
        "question": "Which assignment sequence respects the task dependencies and access constraints?",
        "options": {
            "A": "Omar runs QA first, Mira anonymizes afterward, then Nia prepares slides",
            "B": "Mira anonymizes first, Omar runs QA and summary tables second, then Nia prepares slides",
            "C": "Nia prepares slides from the raw export, then Omar checks quality",
            "D": "Mira writes documentation first, Nia prepares slides second, and Omar runs QA last",
        },
        "gold_answer": "B",
        "required_evidence": [
            evidence("ev_anon_before_qa", "turn_1", "The export must be anonymized before QA.", "temporal_dependency", "early"),
            evidence("ev_roles", "turn_2", "Mira can anonymize; Omar can run QA; Nia cannot access raw identifiable data.", "role_constraint", "middle"),
            evidence("ev_slides_after_tables", "turn_4", "Slides should be prepared only after summary tables exist.", "temporal_dependency", "middle"),
            evidence("ev_latest_sequence", "turn_6", "The latest sequence is anonymize, QA tables, then slides.", "state_update", "late"),
        ],
        "distractors": [
            distractor("dist_omar_fast", "turn_3", "stale_information", "Omar was listed first in a draft"),
            distractor("dist_mira_preference", "turn_5", "soft_preference", "Mira avoid presentation work"),
            distractor("dist_docs", "turn_4", "topical_distractor", "documentation timing"),
        ],
        "option_diagnostics": {
            "A": wrong("wrong_temporal_order", "QA cannot happen before anonymization.", ["ev_anon_before_qa"]),
            "B": gold(["dependency_order", "role_constraints"], ["ev_anon_before_qa", "ev_roles", "ev_slides_after_tables", "ev_latest_sequence"]),
            "C": wrong("violates_hard_constraint", "Nia cannot access raw identifiable data.", ["ev_roles"]),
            "D": wrong("wrong_temporal_order", "Slides and documentation do not precede QA summary tables.", ["ev_slides_after_tables", "ev_latest_sequence"]),
        },
        "reasoning_requirement": "Respect temporal dependencies and access constraints rather than stale draft order.",
    }


def candidate_scattered_facts():
    turn_specs = [
        {
            "user": (
                "We are choosing a student assistant for a multilingual annotation "
                "project. The assistant must know Python because the annotation files "
                "need automated validation before submission."
            ),
            "assistant": verbose_assistant(
                "candidate selection",
                "Python skill is a hard requirement",
                "general interest in linguistics is not enough without the required skill",
            ),
            "tags": ["required"],
        },
        {
            "user": (
                "The assistant also must be available on Thursday afternoon for the "
                "weekly adjudication meeting. Availability on Friday is useful, but it "
                "does not replace the Thursday meeting requirement."
            ),
            "assistant": verbose_assistant(
                "candidate selection",
                "Thursday afternoon availability is required and Friday availability is secondary",
                "availability facts should not be swapped",
            ),
            "tags": ["required", "near_miss_constraint"],
        },
        {
            "user": (
                "Candidate Lina knows Python and has worked with annotation guidelines. "
                "However, she is only available on Friday afternoon this term."
            ),
            "assistant": verbose_assistant(
                "candidate selection",
                "Lina has strong skills but misses Thursday availability",
                "a near-miss candidate can satisfy most but not all constraints",
            ),
            "tags": ["near_miss_constraint"],
        },
        {
            "user": (
                "Candidate Mateo knows R but not Python. He is available on Thursday "
                "afternoon and has excellent documentation habits."
            ),
            "assistant": verbose_assistant(
                "candidate selection",
                "Mateo has the schedule but lacks Python",
                "documentation habits are useful but not a substitute for the hard skill",
            ),
            "tags": ["near_miss_constraint"],
        },
        {
            "user": (
                "Candidate Priya knows Python, is available on Thursday afternoon, and "
                "has prior annotation experience. She prefers remote work, but she can "
                "attend the weekly meeting in person if needed."
            ),
            "assistant": verbose_assistant(
                "candidate selection",
                "Priya satisfies Python, Thursday availability, and annotation experience",
                "remote preference is soft and does not block meeting attendance",
            ),
            "tags": ["required", "soft_preference"],
        },
        {
            "user": (
                "There was also a discussion about who has the nicest portfolio website, "
                "but portfolio design is not part of the decision. The required criteria "
                "remain Python, Thursday afternoon availability, and annotation experience."
            ),
            "assistant": verbose_assistant(
                "candidate selection",
                "portfolio design is irrelevant to the required criteria",
                "the answer should combine the scattered hard requirements",
            ),
            "tags": ["topical_distractor", "required"],
        },
    ]
    return {
        "id": "smoke_scattered_001",
        "phenomenon": "scattered_fact_integration",
        "domain": "candidate_selection",
        "dialogue_history": build_messages(turn_specs),
        "question": "Which candidate satisfies all required criteria for the assistant role?",
        "options": {
            "A": "Lina",
            "B": "Mateo",
            "C": "Priya",
            "D": "The candidate with the best portfolio website",
        },
        "gold_answer": "C",
        "required_evidence": [
            evidence("ev_python", "turn_1", "The assistant must know Python.", "hard_constraint", "early"),
            evidence("ev_thursday", "turn_2", "The assistant must be available Thursday afternoon.", "hard_constraint", "early"),
            evidence("ev_priya", "turn_5", "Priya knows Python, is available Thursday afternoon, and has annotation experience.", "candidate_attribute", "late"),
            evidence("ev_criteria_restated", "turn_6", "The criteria are Python, Thursday availability, and annotation experience.", "criteria", "late"),
        ],
        "distractors": [
            distractor("dist_friday", "turn_2", "near_miss_constraint", "Friday availability is useful"),
            distractor("dist_lina", "turn_3", "near_miss_constraint", "Lina knows Python but is only free Friday"),
            distractor("dist_remote", "turn_5", "soft_preference", "Priya prefers remote work"),
            distractor("dist_portfolio", "turn_6", "topical_distractor", "portfolio website"),
        ],
        "option_diagnostics": {
            "A": wrong("missing_early_evidence", "Lina is not available Thursday afternoon.", ["ev_thursday"]),
            "B": wrong("violates_hard_constraint", "Mateo does not know Python.", ["ev_python"]),
            "C": gold(["python", "thursday_availability", "annotation_experience"], ["ev_python", "ev_thursday", "ev_priya", "ev_criteria_restated"]),
            "D": wrong("topical_distractor_trap", "Portfolio design is not part of the decision.", ["ev_criteria_restated"]),
        },
        "reasoning_requirement": "Combine scattered hard criteria and candidate facts across early and late turns.",
    }


def evidence(evidence_id, turn_id, span, evidence_type, position):
    return {
        "evidence_id": evidence_id,
        "turn_id": turn_id,
        "span": span,
        "evidence_type": evidence_type,
        "position": position,
    }


def distractor(distractor_id, turn_id, distractor_type, span):
    return {
        "distractor_id": distractor_id,
        "turn_id": turn_id,
        "distractor_type": distractor_type,
        "span": span,
        "expected_risk": f"may induce {distractor_type} errors",
    }


def wrong(error_type, reason, linked_evidence):
    return {
        "is_gold": False,
        "error_type": error_type,
        "failure_reason": reason,
        "linked_evidence": linked_evidence,
    }


def gold(satisfies, linked_evidence):
    return {
        "is_gold": True,
        "satisfies": satisfies,
        "linked_evidence": linked_evidence,
    }


def dialogue_summary(sample):
    evidence_lines = "\n".join(
        f"- Evidence from {item['turn_id']} ({item['evidence_type']}): {item['span']}"
        for item in sample["required_evidence"]
    )
    distractor_lines = "\n".join(
        f"- {item['distractor_type']} in {item['turn_id']}: {item['span']}. Risk: {item['expected_risk']}."
        for item in sample["distractors"]
    )
    option_lines = "\n".join(
        f"- Candidate {key}: keep its attributes available for the later question without judging it here."
        for key in sample["options"]
    )
    timeline_lines = "\n".join(
        f"- {msg['turn_id']} {msg['role']}: {msg['content'][:180]}..."
        for msg in sample["dialogue_history"]
        if msg["role"] == "user"
    )
    return (
        "Dialogue summary:\n"
        f"The conversation concerns {sample['domain'].replace('_', ' ')}. The dialogue "
        "contains repeated assistant confirmations, tentative early notes, later "
        "updates, hard constraints, soft preferences, and topical side information. "
        "The retained summary should be read as compressed dialogue history, not as "
        "a solved decision.\n\n"
        "Preserved evidence:\n"
        f"{evidence_lines}\n\n"
        "Non-decisive or risky context:\n"
        f"{distractor_lines}\n\n"
        "Candidate availability for the later question:\n"
        f"{option_lines}\n\n"
        "Condensed dialogue timeline:\n"
        f"{timeline_lines}\n\n"
        "Compression note: later updates override stale tentative information. Hard "
        "constraints should remain stronger than soft preferences. Topical details "
        "may explain why the conversation is long, but they should not become new "
        "requirements. This dialogue summary intentionally avoids performing the "
        "final candidate comparison. It keeps source order visible so that a later "
        "answering model can distinguish early tentative notes from later updates, "
        "and it keeps distractor labels visible so that related but non-decisive "
        "information does not silently become task state."
    )


def fact_state_summary(sample):
    hard = []
    soft = []
    updates = []
    temporal = []
    notes = []
    for item in sample["required_evidence"]:
        line = f"{item['span']} (source: {item['turn_id']})"
        if item["evidence_type"] in {"hard_constraint", "exclusion", "role_constraint"}:
            hard.append(line)
        elif item["evidence_type"] in {"state_update"}:
            updates.append(line)
        elif item["evidence_type"] in {"temporal_dependency"}:
            temporal.append(line)
        elif item["evidence_type"] in {"soft_preference"}:
            soft.append(line)
        else:
            notes.append(line)
    for item in sample["distractors"]:
        if item["distractor_type"] == "soft_preference":
            soft.append(f"{item['span']} is soft or secondary (source: {item['turn_id']})")
        elif item["distractor_type"] == "stale_information":
            updates.append(f"{item['span']} is stale if contradicted by later updates (source: {item['turn_id']})")
        else:
            notes.append(f"{item['span']} is {item['distractor_type']} (source: {item['turn_id']})")
    option_records = [
        f"Candidate {key}: {value}. Retain as a candidate record without judging final validity."
        for key, value in sample["options"].items()
    ]
    source_coverage = [
        f"{item['evidence_id']} from {item['turn_id']} is required evidence of type {item['evidence_type']}."
        for item in sample["required_evidence"]
    ]
    risk_coverage = [
        f"{item['distractor_id']} from {item['turn_id']} is labeled {item['distractor_type']}."
        for item in sample["distractors"]
    ]
    return sectioned(
        {
            "Known facts": notes,
            "Current hard constraints": hard,
            "Soft preferences": soft,
            "State updates": updates,
            "Temporal relations": temporal,
            "Candidate records": option_records,
            "Source coverage": source_coverage,
            "Risk labels": risk_coverage,
            "Usage notes": [
                "Candidate attributes are retained only as state, not as a completed decision.",
                "Use this state only as compressed history for the later question.",
                "A later model response must still compare the candidates against the question.",
            ],
            "Compression audit": [
                "Hard constraints, exclusions, and latest updates are preserved as separate state fields.",
                "Soft preferences are labeled so that they do not override hard constraints.",
                "Stale or topical context is retained only when it explains a possible failure mode.",
                "Assistant confirmations are collapsed into state notes to reduce conversational redundancy.",
            ],
        }
    )


def sectioned(parts):
    lines = []
    for title, items in parts.items():
        lines.append(f"{title}:")
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- None explicitly recorded.")
        lines.append("")
    return "\n".join(lines).strip()


def add_variants(sample, budget):
    messages = sample["dialogue_history"]
    full = render_dialogue(messages)
    turns = group_turns(messages)
    kept_turns, sliding = approximate_fit(turns, budget, render_turn_groups)
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    kept_users, user_only = approximate_fit(user_messages, budget, render_dialogue)
    dialogue = dialogue_summary(sample)
    state = fact_state_summary(sample)
    variants = {
        "full_history": variant(full, False, [msg["turn_id"] for msg in messages]),
        "sliding_window": variant(sliding, True, sorted({msg["turn_id"] for group in kept_turns for msg in group})),
        "user_only_history": variant(user_only, True, [msg["turn_id"] for msg in kept_users]),
        "oracle_dialogue_summary": variant(dialogue, True, []),
        "oracle_fact_state_summary": variant(state, True, []),
    }
    sample["compression_variants"] = variants
    return sample


def group_turns(messages):
    grouped = []
    by_turn = {}
    for msg in messages:
        by_turn.setdefault(msg["turn_id"], []).append(msg)
    for turn_id in sorted(by_turn, key=lambda value: int(value.split("_")[1])):
        grouped.append(by_turn[turn_id])
    return grouped


def render_turn_groups(groups):
    return render_dialogue([msg for group in groups for msg in group])


def variant(text, constrained, included_turns):
    return {
        "text": text,
        "history_tokens": count_tokens(text),
        "budget_constrained": constrained,
        "included_turns": included_turns,
    }


def build_model_input(prompt_template, sample, condition):
    history = sample["compression_variants"][condition]["text"]
    question = render_question(sample["question"], sample["options"])
    return (
        f"{prompt_template.strip()}\n\n"
        f"Dialogue context:\n{history}\n\n"
        f"Question:\n{question}\n"
    )


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_smoke(config):
    random.seed(config["experiment"]["random_seed"])
    budget = config["budgets"]["compressed_history_budget_tokens"]
    samples = [add_variants(spec, budget) for spec in sample_specs()]
    variant_rows = []
    prompt = config["prompt"]["template"]
    for sample in samples:
        for condition in CONDITIONS:
            variant_rows.append(
                {
                    "sample_id": sample["id"],
                    "phenomenon": sample["phenomenon"],
                    "condition": condition,
                    "gold_answer": sample["gold_answer"],
                    "history_tokens": sample["compression_variants"][condition]["history_tokens"],
                    "input_text": build_model_input(prompt, sample, condition),
                    "option_diagnostics": sample["option_diagnostics"],
                }
            )
    return samples, variant_rows
