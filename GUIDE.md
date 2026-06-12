# NLP Course Paper Guide

## Project Positioning

This course-paper project studies:

> Budget-aware dialogue history compression for multi-turn reasoning QA.

The project is aligned with the broader research direction:

> Multi-turn reasoning reliability and inference-time context/state control for LLMs.

The course paper should not be framed as a generic prompt-format comparison or a
general benchmark proposal. It is a small-scale controlled synthetic diagnostic
testbed for studying how different history compression strategies affect later
reasoning reliability when prior dialogue history must be compressed.

Working title:

> 对话历史压缩对多轮推理问答可靠性的影响：一项小规模可控实验

## Core Research Question

When the full multi-turn dialogue history cannot be retained under context
budget pressure, which compressed history representation best preserves
downstream multi-turn reasoning QA reliability under a fixed reasoning budget?

The main comparison is not "which history format does the model prefer". The
main comparison is:

> Under a similar compressed-history budget, which compression strategy preserves
> useful task state, avoids misleading stale context, and maintains answer
> accuracy at lower input cost?

## Fixed Design Decisions

- Main task: English multiple-choice multi-turn lightweight reasoning QA.
- Paper language: Chinese.
- Data type: rule-based controlled synthetic diagnostic testbed.
- Dataset role: diagnostic testbed, not a general benchmark.
- Core logic: generated from structured world state by deterministic rules.
- LLM role: optional surface paraphrase only; it must not decide gold answers,
  required evidence, option diagnostics, or distractor labels.
- Main model: Qwen3-8B served by vLLM with Qwen3 reasoning parser.
- Main inference setting: thinking enabled with fixed budgeted thinking.
- Main answer format: `Final Answer: <A/B/C/D>` plus brief evidence-based
  explanation.
- No step-by-step or explicit verification instruction in the answer prompt.
- Main experiment: 40 samples, 4 phenomena, 5 history conditions.
- Formal run should happen only after smoke and pilot checks are healthy.

## Budgets

The project distinguishes three budget concepts.

| Budget | Object Controlled | Role in This Project |
| --- | --- | --- |
| Overall input/context budget | instruction + history + question + options | Real-system motivation |
| Compressed-history budget | retained dialogue-history representation | Shared constraint for compressed conditions |
| Thinking budget | Qwen3 reasoning output | Fixed output-side control |

The compressed-history budget is not the physical model context length. It is
the maximum token budget allocated to the retained history portion after
compression.

Main planned setting:

- Full History target length: 1,500-2,500 tokens.
- Compressed-history budget: approximately 600 tokens.
- Acceptable compressed-history range: 500-800 tokens.
- Optional budget sweep: 300 / 600 / 900 tokens on 10 representative samples.
- Thinking budget: 512 or 1024 tokens; start with 512 for compatibility with the
  existing budgeted-thinking validation setup.

Full History is the uncompressed quality/cost upper bound and is not constrained
by the compressed-history budget.

## Main Conditions

1. `full_history`
   - Complete dialogue history.
   - No compressed-history budget.
   - Serves as uncompressed quality/cost upper bound.

2. `sliding_window`
   - Budget-aware dynamic retention.
   - Keeps the most recent complete turns that fit within the compressed-history
     budget.
   - Does not truncate individual messages.

3. `user_only_history`
   - Keeps only user messages within the compressed-history budget.
   - Tests whether assistant responses mainly add useful state or redundant cost.

4. `oracle_dialogue_summary`
   - Rule-generated natural-language summary of the dialogue history.
   - Constrained by compressed-history budget.
   - Preserves relevant facts, updates, constraints, and exclusions without
     solving the final question.

5. `oracle_fact_state_summary`
   - Rule-generated semi-structured task-state compression.
   - Constrained by compressed-history budget.
   - Preserves current facts, latest states, hard constraints, soft preferences,
     exclusions, and temporal relations.
   - Must not include the final reasoning conclusion or leak the correct option.

Optional supplementary condition:

- `llm_generated_summary`
  - Used only as a supplementary realism check.
  - Not part of the main causal comparison, because it mixes compression strategy
    with summarizer quality.

## Diagnostic Phenomena

Main dataset target:

- 40 samples total.
- 4 phenomena x 10 samples each.
- Each sample has 6-8 history turns.
- Each sample has 2-4 required evidence items.
- Non-required content should be diagnostic distractors, not random filler.

Phenomena:

1. `scattered_fact_integration`
   - Required evidence is distributed across turns and must be combined.

2. `state_update`
   - A stale state is introduced and later updated or overwritten.

3. `negation_exclusion`
   - Correct reasoning depends on "cannot", "no longer", "unless", exclusion,
     or hard negative constraints.

4. `temporal_order`
   - Correct reasoning depends on latest state, event order, or dependency order.

## Diagnostic Distractors

Distractors should be generated and labeled. They should increase context
pressure and induce interpretable failure modes without making the gold answer
ambiguous.

Allowed distractor types:

- `topical_distractor`: related topic, not answer-determining.
- `stale_information`: old state later overwritten.
- `assistant_redundancy`: confirmation, restatement, or local summary that
  increases input length.
- `near_miss_constraint`: plausible option or fact that nearly satisfies the
  conditions but fails one key constraint.
- `soft_preference`: preference that should not override hard constraints.

Each multiple-choice option should have an `option_diagnostics` entry explaining
why it is correct or which failure mode it represents.

Minimum `option_diagnostics` schema:

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
    "satisfies": ["latest_state", "hard_constraint"],
    "linked_evidence": ["ev_latest_date", "ev_availability"]
  }
}
```

For wrong options, `error_type` and either `violated_constraint` or
`failure_reason` are required. For the gold option, `satisfies` and
`linked_evidence` are required.

Required evidence placement must vary. The generator should distribute required
evidence across early, middle, and late turns and should avoid placing all
required evidence for a sample in the same relative position. Formal data should
include cross-turn evidence chains where the answer depends on evidence from
more than one part of the dialogue.

## Surface Realization Diversity

The generator should avoid producing near-identical surface forms across
samples. It should vary names, dates, option order, evidence positions,
assistant confirmation styles, constraint wording, and distractor placement.
Each domain or phenomenon should use multiple surface templates where possible.

## Experiment Progression

Follow the same style as the existing research project: document first, run
small, freeze config, then scale.

1. Design freeze draft
   - Keep this guide, proposal, experiment design, schema, and config aligned.

2. Minimal closed loop
   - Generate 2-4 samples.
   - Build all history variants.
   - Run local validation scripts.
   - If server access is needed, run only a tiny inference smoke.

3. Smoke test
   - Check gold answers.
   - Check option diagnostics.
   - Check compressed-history budget.
   - Check Fact-State Summary does not leak final answer.
   - Check answer prompt and parser.
   - Check Qwen3 reasoning/content logging.

4. Pilot / preflight
   - Generate 8-12 samples.
   - Include all 4 phenomena.
   - Run all 5 main conditions.
   - Inspect accuracy, token counts, parse rate, and error attribution.
   - Enter formal only if the pilot acceptance criteria below are met.

5. Formal run
   - Freeze seed, config, prompt, generator, scoring, and analysis scripts.
   - Generate 40 formal samples.
   - Run 40 samples x 5 conditions.
   - Save config snapshot and all artifacts under a fresh run directory.

6. Supplementary checks
   - Optional 10-sample budget sweep: 300 / 600 / 900 tokens.
   - Optional LLM-generated summary condition.
   - Optional LLM-paraphrased or more natural histories.
   - Optional larger thinking budget sanity check.

## Artifact Policy

Each run should save:

- config snapshot;
- generated structured samples;
- rendered history variants;
- raw generations;
- parsed generations;
- scoring output;
- token statistics;
- error attribution summary;
- manual audit notes if any.

Do not mix:

- smoke outputs with pilot/formal outputs;
- budgeted-thinking outputs with long-context open-budget outputs;
- different prompts;
- different model settings;
- different generated dataset seeds.

If the dataset or generator is changed after a pilot, create a new dataset/run
version instead of silently editing old artifacts.

## Remote Command Protocol

When running commands from Windows PowerShell against the remote Linux server,
avoid complex inline `ssh` commands. In particular, avoid nested quotes, `$()`,
long pipelines, redirections, and `bash -lc` one-liners unless the command is
trivially simple. Windows PowerShell, local sandbox wrappers, and remote bash
quoting interact poorly and can produce misleading failures.

For recurring remote tasks, prefer stable helper scripts under `~/bin`, such as
GPU/process inspection utilities. Keep these helpers small and project-agnostic
when possible.

For one-off remote tasks, create a uniquely named temporary script under `/tmp`,
execute it, and remove it immediately after execution. Use a cleanup trap inside
the script when practical.

Rules:

- do not leave temporary diagnostic scripts in the project directory;
- do not leave one-off scripts in `$HOME` unless explicitly requested;
- put reusable remote helpers in `~/bin`;
- put formal experiment outputs under the project `runs/` directory;
- keep `/tmp` scripts short-lived and self-cleaning;
- prefer script upload + simple `ssh` execution over fragile inline remote
  shell programs.

## Pilot Acceptance Criteria

The project may move from pilot to formal only if:

- final-answer parse rate is at least 95%;
- compressed variants are mostly within 500-800 history tokens;
- Full History averages within the 1,500-2,500 token target range;
- no inspected Fact-State Summary leaks the final answer or performs final
  option comparison;
- no inspected sample has an ambiguous gold answer;
- every diagnostic phenomenon appears in the pilot;
- `option_diagnostics` is present for every option and supports automatic error
  attribution;
- Qwen3 `reasoning` and `content` fields are both logged correctly.

If a criterion fails, fix the generator, prompt, parser, or config and create a
new pilot version before formal inference.

## Fact-State Summary Leakage Rules

`oracle_fact_state_summary` is a state compression, not an oracle solver. It may
preserve facts, latest states, hard constraints, soft preferences, exclusions,
candidate attributes, and temporal relations. It must not perform the final
target-question comparison.

Forbidden:

- mentioning the correct option label;
- saying "only option B is valid" or equivalent;
- saying "therefore choose ...";
- ranking A/B/C/D by final validity;
- eliminating all wrong options in the summary;
- stating any conclusion that can only be derived by answering the target
  question.

## Current Status

- 2026-06-10: Design grilling completed.
- 2026-06-10: Project guide and design documents initialized.

## Next Steps

1. Implement the rule-based synthetic diagnostic generator.
2. Generate 2-4 smoke samples.
3. Implement variant builder and token counting.
4. Implement parser/scorer for `Final Answer: <A/B/C/D>`.
5. Run local smoke checks before any formal GPU inference.
