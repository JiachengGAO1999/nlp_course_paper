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
  "difficulty_modes": ["subtle_state_update", "implicit_constraint_tracking"],
  "dialogue_history": [],
  "question": "",
  "options": {},
  "gold_answer": "B",
  "answer_type": "option_letter",
  "required_evidence": [],
  "distractors": [],
  "option_diagnostics": {},
  "assistant_inference_error_present": false,
  "reasoning_requirement": "",
  "compression_variants": {}
}
```

New fields:

- `difficulty_modes`: list of difficulty mode labels applied to this sample.
  Values from the set defined in `docs/generator_spec.md`.
- `world_state`: structured representation from which the dialogue, evidence,
  distractors, and gold answer are deterministically generated. Must include
  `reasoning_chain` when `derived_constraint` is in `difficulty_modes`.
- `assistant_inference_error_present`: sample-level flag set to `true` if any
  assistant turn uses `assistant_behavior: incorrect_inference`.

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

For assistant turns, an additional `assistant_behavior` field is required:

```json
{
  "turn_id": "turn_3",
  "role": "assistant",
  "content": "...",
  "contains_required_evidence": false,
  "contains_distractor": true,
  "assistant_behavior": "partial_summary",
  "tags": ["assistant_redundancy"]
}
```

Allowed `assistant_behavior` values:
- `echo` — redundant confirmation of user content
- `partial_summary` — selective summary omitting some constraints
- `incorrect_inference` — nearly-correct inference missing one condition
- `stale_reiteration` — reiteration of an already-updated state
- `neutral` — brief 1-2 sentence acknowledgment

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
    "violated_constraint": "The meeting was moved from Wednesday to Friday.",
    "linked_evidence": ["ev_latest_date"]
  },
  "B": {
    "is_gold": true,
    "satisfies": [
      "latest_date",
      "all_required_attendees_available",
      "within_budget"
    ],
    "linked_evidence": ["ev_latest_date", "ev_availability"]
  },
  "C": {
    "is_gold": false,
    "error_type": "violates_hard_constraint",
    "violated_constraint": "Ben cannot attend after 5 PM.",
    "linked_evidence": ["ev_ben_no_evening"]
  },
  "D": {
    "is_gold": false,
    "error_type": "soft_preference_trap",
    "violated_constraint": "Quiet room is preferred but does not override availability.",
    "linked_evidence": ["ev_quiet_room_preference"]
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
    "included_turns": ["turn_5", "turn_6", "turn_7"],
    "compression_quality": {}
  },
  "user_only_history": {
    "text": "...",
    "history_tokens": 570,
    "budget_constrained": true,
    "included_turns": ["turn_2", "turn_4", "turn_5", "turn_7"],
    "compression_quality": {}
  },
  "oracle_dialogue_summary": {
    "text": "...",
    "history_tokens": 650,
    "budget_constrained": true,
    "leakage_checked": true,
    "compression_quality": {}
  },
  "llm_generated_summary": {
    "text": "...",
    "history_tokens": 610,
    "budget_constrained": true,
    "summarizer_model": "qwen3-8b-budget",
    "compression_prompt": "docs/generator_spec.md §4.1",
    "compression_quality": {}
  },
  "hybrid_summary_recent": {
    "text": "...",
    "history_tokens": 630,
    "budget_constrained": true,
    "summarizer_model": "qwen3-8b-budget",
    "summary_scope": "older_history",
    "recent_turns_kept": ["turn_6"],
    "compression_prompt": "docs/generator_spec.md §4.2",
    "compression_quality": {}
  },
  "oracle_fact_state_summary": {
    "text": "...",
    "history_tokens": 530,
    "budget_constrained": true,
    "leakage_checked": true,
    "compression_quality": {}
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

For oracle variants (`oracle_fact_state_summary`, `oracle_dialogue_summary`),
`required_evidence_retention` is always 1.0 (the generator reads labels directly)
and `hallucinated_fact_count` is always 0. These fields are still computed and
stored for fair comparison with LLM-based variants.

## World State

The `world_state` object is the source of truth from which the dialogue,
evidence, distractors, and gold answer are generated. For samples using
`derived_constraint`, it must include a `reasoning_chain`:

```json
{
  "domain": "meeting_scheduling",
  "entities": ["Alice", "Ben", "Cara"],
  "constraints": [],
  "updates": [],
  "candidates": {},
  "reasoning_chain": {
    "target_conclusion": "Ben can attend Friday afternoon",
    "premises": [
      {
        "premise_id": "prem_lab_ends_noon",
        "statement": "Ben's lab duty ends at 12:00 noon",
        "turn": 1,
        "surface_form": "Ben told us he's stuck in the lab until noon every day this week."
      },
      {
        "premise_id": "prem_friday_slot",
        "statement": "Friday afternoon slots are 2:00-5:00 PM",
        "turn": 3,
        "surface_form": "The only open slots Friday are the afternoon block, 2 to 5."
      }
    ]
  }
}
```

The generator renders only the `surface_form` into the dialogue. The premises and
target conclusion are used to validate that the gold answer remains derivable.

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
  "assistant_inference_error_present": false,
  "finish_reason": "stop",
  "usage": {}
}
```

New field:
- `assistant_inference_error_present`: `true` if the original dialogue history
  for this sample contains at least one `assistant_behavior: incorrect_inference`
  turn. Used to separate compression failures from assistant contamination
  failures in the analysis (see `docs/generator_spec.md` §8).
