# Experiment Design

## Objective

Measure how dialogue-history compression strategies affect multi-turn reasoning
QA reliability under compressed-history budget pressure and a fixed Qwen3
thinking budget.

## Main Hypothesis

Under the same compressed-history budget, task-state-oriented or well-formed
hybrid compression should preserve answer-critical evidence and constraint
structure better than generic one-shot LLM summaries, while naive recency-only
retention can fail through trivial evidence deletion.

## Variables

Independent variable:

- History compression strategy.

Main conditions:

- `full_history`
- `user_only_history`
- `oracle_fact_state_summary`
- `llm_generated_summary`
- `hybrid_summary_recent`

Supplementary / diagnostic conditions:

- `sliding_window`
- `oracle_dialogue_summary`
- `llm_progressive_summary`

Controlled constraints:

- same model;
- same decoding settings;
- same answer prompt;
- same question/options;
- same compressed-history budget for compressed conditions;
- same fixed thinking budget;
- same generated sample seed for formal run.

Dependent variables:

- final-answer accuracy;
- error type;
- answerability after compression;
- required evidence retention;
- hard/soft constraint preservation;
- stale-state handling;
- hallucinated fact count;
- input token count;
- compressed-history token count;
- compression ratio;
- reasoning output length;
- content output length;
- parse success rate;
- evidence mention quality from explanation, if manually audited.

## Dataset

The dataset is a controlled synthetic diagnostic testbed.

Formal target:

- 40 samples total;
- 10 samples per phenomenon;
- 6-8 dialogue turns per sample;
- full history target length: 1,500-2,500 tokens;
- compressed-history budget: approximately 600 tokens, acceptable 500-800;
- English prompts and data;
- Chinese course-paper writing.

Phenomena:

- `scattered_fact_integration`
- `state_update`
- `negation_exclusion`
- `temporal_order`

Required evidence positions should vary across samples:

- early;
- middle;
- late;
- cross-turn.

This prevents Sliding Window failure from being mechanically predetermined.
The generator should avoid placing all required evidence for a sample in the
same relative position.

Surface realization should also vary across samples. The generator should vary
names, dates, option order, evidence positions, assistant confirmation styles,
constraint wording, and distractor placement. Each domain or phenomenon should
use multiple surface templates where possible.

## Diagnostic Options

Each sample uses 3-4 answer options. Incorrect options should be near-miss
diagnostic distractors, not random wrong answers.

Possible option error types:

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

Each option must have an `option_diagnostics` entry.

Minimum fields:

- wrong option: `is_gold`, `error_type`, `linked_evidence`, and either
  `violated_constraint` or `failure_reason`;
- gold option: `is_gold`, `satisfies`, and `linked_evidence`.

## Compression Strategy Definitions

### Full History

Render every dialogue turn in order. This is the uncompressed upper bound and is
not restricted by the compressed-history budget.

### User-only History

Render only user messages under the compressed-history budget. This condition
tests whether assistant responses mainly add useful state or redundant cost.

### Oracle Fact-State Summary

Generate a semi-structured state table from structured fields. It should
preserve latest task state, facts, hard constraints, soft preferences,
exclusions, temporal relations, and unresolved candidate information. It must
not include the final option choice or a final reasoning conclusion.

It also must not silently solve the final question. In particular, it must not
say which option is the only valid one, rank A/B/C/D by final validity, eliminate
all wrong options, or state a conclusion that requires answering the target
question.

### LLM-generated Summary

Ask an LLM to compress the full dialogue history into a summary within the
compressed-history budget. The compression prompt should request preservation of
facts, updates, constraints, exclusions, soft preferences, and candidate
attributes, but it must forbid solving the final question.

This condition represents a practical generic agent compression strategy. Its
quality should be evaluated through evidence retention, hallucination,
constraint preservation, stale-state handling, and downstream QA accuracy.

### Hybrid Summary + Recent Turns

Summarize older dialogue history with an LLM and keep the most recent 1-2 turns
verbatim, within the compressed-history budget. This represents the common
pattern of compressed long-term context plus raw recent context.

### Sliding Window

Supplementary naive baseline only.

Starting from the latest complete turn, include as many complete dialogue turns
as fit within the compressed-history budget. Do not truncate individual
messages.

### Oracle Dialogue Summary

Supplementary diagnostic oracle only.

Generate a natural-language summary from structured state and event fields. It
should preserve relevant facts, updates, exclusions, hard constraints, soft
preferences, and temporal relations. It should not include the final answer or
solve the target question.

### Progressive LLM Summary

Optional supplementary condition. Compress history incrementally during the
dialogue, for example after turn 3 and again after turn 6. This tests cumulative
summary error, but should not enter the main experiment until the one-shot and
hybrid conditions are stable.

## Prompt

Use one fixed prompt across all history conditions.

```text
You will answer a multiple-choice question based on the provided dialogue context.

Choose the best option. Provide the final answer in the exact format:
Final Answer: <A/B/C/D>

Then give a brief explanation mentioning the key evidence from the context.
```

Avoid:

- "Think step by step";
- "Carefully verify every constraint";
- any instruction that explicitly changes verification behavior.

## Model Setting

Primary setting:

- model: Qwen3-8B;
- served model name: `qwen3-8b-budget` or equivalent;
- serving: vLLM OpenAI-compatible server;
- reasoning parser: Qwen3;
- temperature: 0.0;
- top_p: 1.0;
- max_tokens: 2048;
- thinking enabled;
- thinking_token_budget: 512 to start.

Optional supplementary setting:

- larger thinking budget or no explicit thinking budget on a small subset.

## Evaluation

Automatic:

- parse final answer after `Final Answer:`;
- score against gold option;
- attribute errors through `option_diagnostics`;
- label `insufficient_context_for_answer` when a compressed artifact lacks
  required evidence for the gold answer;
- compute required evidence retention for each compressed artifact;
- check hard/soft constraint preservation;
- check stale-state handling;
- count hallucinated facts if the summary introduces unsupported content;
- compute token counts and compression ratios;
- compute output lengths.

Manual audit:

- sample 10-20% of generated data;
- check gold answer consistency;
- check no answer leakage in summaries;
- check whether explanations cite correct evidence;
- check ambiguous or unnatural samples.

## Run Plan

1. Smoke
   - 2-4 samples;
   - all 5 history conditions;
   - local generation and validation;
   - optional tiny vLLM inference.

2. Pilot
   - 8-12 samples;
   - all 4 phenomena;
   - inspect parse rate, budget fit, accuracy range, and error attribution.
   - acceptance criteria:
     - final-answer parse rate >= 95%;
     - compressed variants mostly within 500-800 history tokens;
     - Full History averages within 1,500-2,500 tokens;
     - no inspected Fact-State Summary leaks or solves the final answer;
     - no ambiguous gold answer;
     - every phenomenon appears in the pilot;
     - Qwen3 reasoning/content logging works.

3. Formal
   - freeze seed/config/generator/prompt/parser/scorer;
   - 40 samples x 5 conditions;
   - save all artifacts to a fresh run directory.

4. Supplementary
   - optional 10-sample 300/600/900 budget sweep;
   - optional sliding-window naive baseline;
   - optional oracle dialogue summary redundancy check;
   - optional progressive LLM summary;
   - optional LLM paraphrase / naturalness check.
