# Keep Recent or Keep Relevant?

Repository entry point. For the authoritative project framing, fixed decisions,
status, and next actions, see [GUIDE.md](GUIDE.md).

Position-dependent trade-offs in prompt-based compression of multi-turn
reasoning histories.

Course paper project: Under a fixed compressed-history budget, does retaining
recent turns verbatim improve compressed multi-turn reasoning histories, or does
it create a position-dependent trade-off between preserving recent evidence and
losing older evidence?

## Quick Start (planned)

```bash
# 0. Prepare Layer 1 MuSiQue candidates
python scripts/01_prepare_musique_layer1.py --config configs/experiment.yaml

# 1. Convert MuSiQue open-answer QA to multiple choice
python scripts/02_convert_musique_to_mc.py --config configs/experiment.yaml --split smoke

# 2. Generate dialogues from benchmark QA items
python scripts/03_generate_dialogues.py --config configs/experiment.yaml

# 3. Build compression variants
python scripts/04_build_compressions.py --config configs/experiment.yaml

# 4. Run inference
python scripts/05_run_inference.py --config configs/experiment.yaml

# 5. Score and summarize
python scripts/06_summarize_results.py
```

## Current Framing

The comparison between one-shot and hybrid compression is used to diagnose a
common recency-based history-compression heuristic, not to claim that one
architecture is globally better:

- `full_history` checks whether the original dialogue is answerable.
- `one_shot_summary` is a global compaction baseline: all turns are summarized
  together under the shared prompt.
- `hybrid_summary_recent` represents the common heuristic of compressing older
  history while keeping the most recent turn verbatim.

The core finding from the 100-sample formal run is not an aggregate architecture
main effect: `one_shot_summary` scores 85/100 and `hybrid_summary_recent` scores
86/100. The useful signal appears after stratifying by whether answer-critical
evidence falls in the recent-turn window:

| Evidence placement | One-shot | Hybrid | Hybrid gap |
| --- | ---: | ---: | ---: |
| Critical evidence in recent window | 43/50 | 50/50 | +14pp |
| Critical evidence outside recent window | 42/50 | 36/50 | -12pp |

Recent-turn retention is therefore not a free lunch. It protects recent
answer-critical evidence, but it also reallocates a fixed compression budget
away from older history. Aggregate accuracy hides this sign-reversing trade-off.

Manual annotation is used as secondary mechanism evidence: it explains whether
critical failures involve evidence omission, distractor overweighting,
recent-distractor interference, or downstream reasoning error.

## Conditions

| Condition | Architecture |
|---|---|
| `full_history` | No compression (upper bound) |
| `one_shot_summary` | All turns → one compression call |
| `hybrid_summary_recent` | Older turns compressed + recent turn verbatim |

All compressed conditions share the same compression prompt template. The
architecture comparison is therefore a controlled way to diagnose how a
recency-biased budget allocation preserves, drops, or overweights evidence.

## Project Structure

```
configs/          Experiment configuration
data/             Generated data (not committed)
docs/             Design documents
runs/             Experiment outputs (not committed)
scripts/          Pipeline scripts
src/              Shared utilities
```

`data/` is reserved for reusable benchmark/source data. Concrete experiment
artifacts live under a single run directory:

```
runs/<run_id>/
  pool/       Candidate pool, MC conversion, generated dialogues, audits
  gate/       Full-history gate variants and inference results
  formal/     Selected formal dialogues, variants, final inference results
  logs/       Pipeline logs
```

See [docs/artifact_layout.md](docs/artifact_layout.md) for the full layout.

## Git Remotes

This repository uses explicit synchronization for the server and GitHub remotes:

```bash
git push origin main
git push github main
```

`origin` is the server repository, and `github` is the GitHub sharing copy.

## Current Run

The current main run is:

```text
runs/layer1_scale100_qwen3_8b_budget800_20260614/
```

It contains 100 selected formal items, 300 final inferences, and a 22-case manual
critical-error annotation. The next research step is to write the result
narrative around the position-dependent recency trade-off and treat manual
annotation as mechanism analysis.

## Design Documents

- [GUIDE.md](GUIDE.md) — project command document; current framing, fixed decisions, status, and next actions.
- [docs/research_framework.md](docs/research_framework.md) — concise shareable research narrative for collaborators.
- [docs/experiment_design.md](docs/experiment_design.md) — detailed experimental protocol, variables, prompts, and evaluation plan.
- [docs/artifact_layout.md](docs/artifact_layout.md) — run/data artifact layout and naming rules.
