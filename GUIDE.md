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

The main comparison is not "which history format does the model prefer" or
"does deleting old evidence hurt performance". The main comparison is:

> Under a similar compressed-history budget, which practical or oracle
> compression strategy preserves answer-critical evidence, avoids misleading
> stale context, and maintains answer accuracy at lower input cost?

## Fixed Design Decisions

- Main task: English multiple-choice multi-turn lightweight reasoning QA.
- Paper language: Chinese.
- Data type: rule-based controlled synthetic diagnostic testbed.
- Dataset role: diagnostic testbed, not a general benchmark.
- Core logic: generated from structured world state by deterministic rules.
- LLM role in data generation: optional surface paraphrase only; it must not
  decide gold answers, required evidence, option diagnostics, or distractor
  labels.
- LLM role in compression: used for `llm_generated_summary` and
  `hybrid_summary_recent` conditions, where summary quality is part of the
  evaluated compression pipeline.
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

2. `user_only_history`
   - Keeps only user messages within the compressed-history budget.
   - Tests whether assistant responses mainly add useful state or redundant cost.

3. `oracle_fact_state_summary`
   - Rule-generated semi-structured task-state compression.
   - Constrained by compressed-history budget.
   - Preserves current facts, latest states, hard constraints, soft preferences,
     exclusions, and temporal relations.
   - Must not include the final reasoning conclusion or leak the correct option.
   - Serves as the rule-based compression upper bound.

4. `llm_generated_summary`
   - LLM-generated one-shot summary of the full dialogue history under the same
     compressed-history budget.
   - Represents a practical generic agent compression strategy.
   - Evaluates whether the summarizer retains required evidence, preserves
     hard/soft constraint distinctions, and avoids hallucinating task state.
   - Summarizer runs with thinking disabled, temperature 0.0, max_tokens 1024
     (see `docs/generator_spec.md` §6 for full inference settings).

5. `hybrid_summary_recent`
   - LLM-generated summary of older history plus the most recent 1 complete turn
     kept verbatim.
   - Budget allocation: summary ≤400 tokens, recent turn ≤200 tokens.
   - If the most recent turn exceeds 250 tokens (e.g., unusually long assistant
     message), keep only the most recent user message verbatim and allocate the
     remainder to the summary.
   - Represents a common practical pattern: compressed long-term history plus
     raw recent context.
   - Tests whether preserving recent turns mitigates summary omissions or
     update/recency failures.

Supplementary / diagnostic conditions:

- `sliding_window`
  - Naive recency baseline.
  - Useful for showing the lower bound of deletion-based context management, but
    not a core comparison because failures can be trivial evidence loss.

- `oracle_dialogue_summary`
  - Rule-generated natural-language oracle summary.
  - May be used in pilot to test whether it adds information beyond
    `oracle_fact_state_summary`; if redundant, omit from formal main analysis.

## Compression Quality Metrics

Final answer accuracy is not enough. Each compressed representation should also
be evaluated as an intermediate artifact.

Track:

- required evidence retention: how many `required_evidence` items are preserved
  in the compressed history;
- hard/soft constraint preservation: whether hard constraints, exclusions, and
  soft preferences are kept distinct;
- stale-state handling: whether outdated information is removed, marked stale,
  or incorrectly preserved as current;
- hallucinated fact count: whether the compressed history introduces facts not
  present in the original dialogue;
- answerability after compression: whether the compressed history still contains
  enough information to identify the gold answer.

If a compressed variant lacks the evidence needed to answer the question, the
result should be labeled `insufficient_context_for_answer` rather than treated
as a pure reasoning failure.

For hard smoke v2, all LLM-generated compression artifacts are manually audited:

```text
6 hard-smoke samples × 2 LLM compression conditions = 12 artifacts
```

For these artifacts, automatic span/overlap checks are only hints because LLM
summaries may paraphrase evidence. Final evidence-retention, hallucination,
stale-state, and answerability labels come from manual inspection.

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

## Benchmark-Inspired Difficulty Design

The synthetic generator should remain controlled, but its difficulty should not
be invented from scratch. It should borrow failure modes observed in multi-turn
and long-context evaluation work, without directly importing those benchmarks as
the main dataset.

Use benchmark-inspired patterns as design motifs, not as unverified numerical
claims. Do not cite specific benchmark scores in the course paper unless the
original source has been checked.

Difficulty modes to inject:

- `implicit_constraint_tracking`
  - Introduce an early constraint once, then stop repeating it.
  - Tests whether compression preserves constraints that are no longer locally
    salient.

- `derived_constraint`
  - Make a required constraint usable only after one or two inference steps.
  - Example: "Ben's lab duty ends at noon" plus "Friday afternoon is 2-5 PM"
    implies Friday afternoon satisfies Ben's availability.

- `long_distance_evidence`
  - Place answer-critical evidence far from the final question, with redundant
    assistant confirmations or topical context in between.
  - Use multiple distance profiles rather than always placing all evidence at
    the beginning.

- `soft_hard_conflict_without_labels`
  - Avoid explicitly labeling every item as "hard" or "soft".
  - Let the wording imply priority, so summaries can misclassify preferences as
    constraints or constraints as preferences.

- `subtle_state_update`
  - Replace or revise an earlier state using natural wording rather than
    explicit "this replaces the old value" phrasing every time.

- `scattered_candidate_attributes`
  - Distribute candidate attributes across turns so the final answer requires
    assembling entity-level state, not matching a single sentence.

Full implementation details — including template strategies for each difficulty
mode, assistant behavior specifications, and the 40-sample phenomenon × difficulty
× assistant × evidence-position allocation table — are in
[docs/generator_spec.md](docs/generator_spec.md). The generator spec also defines
compression prompt templates for `llm_generated_summary` and
`hybrid_summary_recent`.

The goal is not to make Full History fail catastrophically. A healthy hard
subset should be challenging enough to avoid ceiling effects while preserving a
meaningful uncompressed upper bound.

Target difficulty for hard smoke v2 (6 samples):

- `full_history`: 5/6 correct is ideal; 4/6 is acceptable for a deliberately
  hard subset but may be too difficult for the main formal set; 6/6 correct
  suggests possible ceiling (check explanation quality and summary-condition
  differentiation before proceeding);
- `oracle_fact_state_summary`: should be close to Full History if compression is
  faithful;
- `llm_generated_summary` and `hybrid_summary_recent`: should reveal whether
  practical compression loses or distorts answer-critical state;
- if Full History remains 6/6 with no cross-condition differentiation,
  increase generator difficulty before pilot;
- if Full History falls to 3/6 or below, separate those samples into a hard subset
  rather than using them as the main formal distribution.

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

4. Hard smoke v2
   - Generate about 6 benchmark-inspired hard samples.
   - Include the difficulty modes above.
   - First run only `full_history`, `oracle_fact_state_summary`,
     `llm_generated_summary`, and `hybrid_summary_recent`.
   - Use this step to calibrate difficulty before expanding pilot.

5. Pilot / preflight
   - Generate 8-12 samples.
   - Include all 4 phenomena.
   - Run all 5 main conditions.
   - Inspect accuracy, token counts, parse rate, and error attribution.
   - Enter formal only if the pilot acceptance criteria below are met.

6. Formal run
   - Freeze seed, config, prompt, generator, scoring, and analysis scripts.
   - Generate 40 formal samples.
   - Run 40 samples x 5 conditions.
   - Save config snapshot and all artifacts under a fresh run directory.

7. Supplementary checks
   - Optional 10-sample budget sweep: 300 / 600 / 900 tokens.
   - Optional `sliding_window` naive baseline.
   - Optional `oracle_dialogue_summary` redundancy check.
   - Optional `llm_progressive_summary` condition.
   - Optional LLM-paraphrased or more natural histories.
   - Optional larger thinking budget sanity check.

## Artifact Policy

Each run should save:

- config snapshot;
- generated structured samples;
- rendered history variants;
- compressed artifact quality annotations;
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

Use three levels of remote scripts:

1. One-off diagnostic or maintenance scripts should not enter git. Create a
   uniquely named temporary script under `/tmp`, execute it, and remove it
   immediately after execution. Use a cleanup trap inside the script when
   practical.

2. Personal recurring server helpers should not enter this project by default.
   Put them under `~/bin`, for example GPU/process inspection helpers. Keep
   them small and project-agnostic when possible.

3. Project remote scripts should enter git only when they are stable,
   repeatedly used, and necessary for reproducing the project workflow. Avoid
   committing one-off server maintenance scripts.

Rules:

- do not leave temporary diagnostic scripts in the project directory;
- do not leave one-off scripts in `$HOME` unless explicitly requested;
- put reusable remote helpers in `~/bin`;
- put only stable project-specific remote entry points in `scripts/remote/`;
- put formal experiment outputs under the project `runs/` directory;
- keep `/tmp` scripts short-lived and self-cleaning;
- prefer script upload + simple `ssh` execution over fragile inline remote
  shell programs.

## Pilot Acceptance Criteria

The project may move from pilot to formal only if:

- final-answer parse rate is at least 95%;
- hard smoke v2 has been inspected and does not show a severe ceiling effect;
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
- 2026-06-12: Basic smoke (4 samples) generated, locally validated, and inferred
  on Qwen3-8B. Results: 95% overall accuracy. Full history 100%, oracle
  compression 100%, sliding window 75% (1 trivial evidence-loss error). Smoke
  revealed ceiling effect risk and over-redundant assistant messages.
- 2026-06-12: Experimental conditions redesigned (5+3 structure, benchmark-inspired
  difficulty modes, compression quality metrics). Generator specification written.

## Next Steps

1. Implement hard smoke v2 generator with the 6 difficulty modes and assistant
   behavior matrix defined in `docs/generator_spec.md`.
2. Generate 6 hard smoke v2 samples covering all 4 phenomena.
3. Add `llm_generated_summary` and `hybrid_summary_recent` variant builders with
   compression quality checks.
4. Run hard smoke v2 inference and calibrate difficulty before expanding.
5. Only then build the 8-12 sample pilot set.
6. Run pilot, inspect accuracy and compression quality at 12-sample scale.
7. If pilot passes acceptance criteria, freeze config and generate 40 formal samples.
