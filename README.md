# Compression Architecture for Multi-Turn Reasoning QA

Course paper project: When prompt-based self-compression is applied to
evidence-bearing multi-turn histories, do downstream reasoning failures come
mainly from evidence position or from interference between answer-critical
evidence and competing context?

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

The comparison between one-shot and hybrid compression is used as a diagnostic
intervention, not as a simple architecture leaderboard:

- `full_history` checks whether the original dialogue is answerable.
- `one_shot_summary` exposes failure modes of prompt-based self-compression.
- `hybrid_summary_recent` tests what changes when recent context is retained
  verbatim instead of compressed.

The current manual audit focuses on whether failures are better explained by
evidence omission, distractor overweighting, relation-structure blur, exact
value collapse, or harmful recent-distractor retention.

## Conditions

| Condition | Architecture |
|---|---|
| `full_history` | No compression (upper bound) |
| `one_shot_summary` | All turns → one compression call |
| `hybrid_summary_recent` | Older turns compressed + recent turn verbatim |

All compressed conditions share the same compression prompt template. The
architecture comparison is therefore a controlled way to diagnose what
compression preserves, drops, or overweights.

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

## Scale-Up Plan

The next run should follow the same pipeline and stop after inference/automatic
summaries, before any new manual mechanism annotation. For example, on the
server:

```bash
RUN_DATE=20260614 \
RUN_ID=layer1_scale80_qwen3_8b_budget800_20260614 \
FORMAL_POOL_SIZE=160 \
FORMAL_TARGET_N=80 \
bash scripts/remote/run_formal_pipeline.sh
```

This generates a larger pool, runs the full-history gate, selects 80 formal
items with scaled hop/profile targets, builds the three variants, and runs final
inference. Manual annotation is a separate follow-up step.

## Design Documents

- [docs/research_framework.md](docs/research_framework.md) — concise shareable research framework.
- [GUIDE.md](GUIDE.md) — full project guide, decisions, and status.
- [docs/experiment_design.md](docs/experiment_design.md) — detailed experiment design.
- [docs/artifact_layout.md](docs/artifact_layout.md) — run/data artifact layout.
