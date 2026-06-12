# Data Schema

The data should be generated from structured world state, not hand-written as
free-form dialogue. The generator should produce both the structured sample and
all rendered history variants.

## Sample Object

```json
{
  "id": "state_update_003",
  "phenomenon": "state_update",
  "domain": "meeting_scheduling",
  "language": "en",
  "seed": 42,
  "world_state": {},
  "dialogue_history": [],
  "question": "",
  "options": {},
  "gold_answer": "B",
  "answer_type": "option_letter",
  "required_evidence": [],
  "distractors": [],
  "option_diagnostics": {},
  "compression_quality": {},
  "reasoning_requirement": "",
  "compression_variants": {}
}
```

## Dialogue Turn

```json
{
  "turn_id": "turn_3",
  "role": "user",
  "content": "...",
  "contains_required_evidence": true,
  "contains_distractor": false,
  "tags": ["state_update"]
}
```

A logical turn may contain both user and assistant messages. For budget-aware
windowing, complete dialogue turns should be preserved when possible.

## Required Evidence

```json
{
  "evidence_id": "ev_latest_date",
  "turn_id": "turn_5",
  "span": "The review was later moved from Wednesday to Friday afternoon.",
  "evidence_type": "state_update",
  "position": "middle",
  "needed_for": "latest meeting date"
}
```

Required evidence should vary in position:

- `early`
- `middle`
- `late`
- `cross_turn`

## Distractor

```json
{
  "distractor_id": "dist_projector",
  "turn_id": "turn_3",
  "distractor_type": "topical_distractor",
  "span": "The team also asked whether the room has a projector.",
  "expected_risk": "may distract from the scheduling constraint"
}
```

Allowed `distractor_type` values:

- `topical_distractor`
- `stale_information`
- `assistant_redundancy`
- `near_miss_constraint`
- `soft_preference`

## Options

```json
{
  "A": "Wednesday morning in Room 204",
  "B": "Friday afternoon in Room 310",
  "C": "Friday evening in Room 310",
  "D": "Saturday morning in Room 102"
}
```

## Option Diagnostics

Every option must have a diagnostic entry.

```json
{
  "A": {
    "is_gold": false,
    "error_type": "uses_stale_state",
    "violated_constraint": "The meeting was moved from Wednesday to Friday."
  },
  "B": {
    "is_gold": true,
    "satisfies": [
      "latest_date",
      "all_required_attendees_available",
      "within_budget"
    ]
  },
  "C": {
    "is_gold": false,
    "error_type": "violates_hard_constraint",
    "violated_constraint": "Ben cannot attend after 5 PM."
  },
  "D": {
    "is_gold": false,
    "error_type": "soft_preference_trap",
    "violated_constraint": "Quiet room is preferred but does not override availability."
  }
}
```

Required fields:

- all options:
  - `is_gold`;
  - `linked_evidence`;
- wrong options:
  - `error_type`;
  - either `violated_constraint` or `failure_reason`;
- gold option:
  - `satisfies`.

Recommended `error_type` values:

- `uses_stale_state`
- `missing_early_evidence`
- `missing_middle_evidence`
- `missing_late_evidence`
- `violates_hard_constraint`
- `ignores_negation`
- `wrong_temporal_order`
- `soft_preference_trap`
- `topical_distractor_trap`
- `near_miss_constraint`

## Compression Variants

```json
{
  "full_history": {
    "text": "...",
    "history_tokens": 2100,
    "budget_constrained": false
  },
  "sliding_window": {
    "text": "...",
    "history_tokens": 620,
    "budget_constrained": true,
    "included_turns": ["turn_5", "turn_6", "turn_7"]
  },
  "user_only_history": {
    "text": "...",
    "history_tokens": 570,
    "budget_constrained": true,
    "included_turns": ["turn_2", "turn_4", "turn_5", "turn_7"]
  },
  "oracle_dialogue_summary": {
    "text": "...",
    "history_tokens": 650,
    "budget_constrained": true
  },
  "llm_generated_summary": {
    "text": "...",
    "history_tokens": 610,
    "budget_constrained": true,
    "summarizer_model": "qwen3-8b-budget",
    "compression_quality": {}
  },
  "hybrid_summary_recent": {
    "text": "...",
    "history_tokens": 630,
    "budget_constrained": true,
    "summarizer_model": "qwen3-8b-budget",
    "summary_scope": "older_history",
    "recent_turns_kept": ["turn_6", "turn_7"],
    "compression_quality": {}
  },
  "oracle_fact_state_summary": {
    "text": "...",
    "history_tokens": 530,
    "budget_constrained": true,
    "leakage_checked": true
  }
}
```

## Compression Quality

Each compressed artifact should have quality annotations or automatically
computed checks.

```json
{
  "required_evidence_retention": {
    "retained": ["ev_latest_date", "ev_availability"],
    "missing": ["ev_no_evening"],
    "retention_rate": 0.67
  },
  "answerability": {
    "answerable": false,
    "label": "insufficient_context_for_answer",
    "reason": "candidate attributes needed for the gold answer are missing"
  },
  "constraint_preservation": {
    "hard_constraints_preserved": true,
    "soft_preferences_preserved_as_soft": true,
    "exclusions_preserved": false
  },
  "stale_state_handling": {
    "stale_information_marked": true,
    "stale_information_promoted_to_current": false
  },
  "hallucination_check": {
    "unsupported_facts": [],
    "hallucinated_fact_count": 0
  }
}
```

These fields are especially important for `llm_generated_summary` and
`hybrid_summary_recent`, because their downstream errors should be connected to
compression artifacts rather than treated only as answer-model failures.

## Fact-State Summary Format

Recommended rendering:

```text
Known facts:
- ...

Current hard constraints:
- ...

Soft preferences:
- ...

State updates:
- ...

Temporal relations:
- ...

Candidate notes:
- ...
```

Forbidden:

- final option label;
- final answer text;
- "therefore choose ...";
- any sentence that directly solves the target question.
- "only option X is valid" or equivalent;
- final A/B/C/D comparison;
- eliminating all wrong options;
- any conclusion that should be derived only when answering the target question.

## Model Output Record

```json
{
  "sample_id": "state_update_003",
  "condition": "oracle_fact_state_summary",
  "model": "qwen3-8b-budget",
  "prompt_tokens": 0,
  "history_tokens": 0,
  "response_reasoning": "...",
  "response_content": "Final Answer: B\nExplanation: ...",
  "parsed_answer": "B",
  "gold_answer": "B",
  "is_correct": true,
  "error_type": null,
  "finish_reason": "stop",
  "usage": {}
}
```
