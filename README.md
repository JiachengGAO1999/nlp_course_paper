# Compression Architecture for Multi-Turn Reasoning QA

Course paper project: How do different prompt-based self-compression architectures
affect downstream multi-turn reasoning QA reliability?

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

## Conditions

| Condition | Architecture |
|---|---|
| `full_history` | No compression (upper bound) |
| `one_shot_summary` | All turns → one compression call |
| `hybrid_summary_recent` | Older turns compressed + recent turn verbatim |

All compressed conditions share the same compression prompt template.
The independent variable is compression architecture, not prompt phrasing.

## Project Structure

```
configs/          Experiment configuration
data/             Generated data (not committed)
docs/             Design documents
runs/             Experiment outputs (not committed)
scripts/          Pipeline scripts
src/              Shared utilities
```

Generated Layer 1 artifacts are organized as:

```
data/raw/                 Downloaded benchmark source files
data/layer1/splits/       MuSiQue candidate, smoke, pilot, formal, spares splits
data/layer1/mc/           Multiple-choice converted samples
data/layer1/dialogues/    Generated multi-turn dialogue histories
data/layer1/audits/       Sampling and MC conversion audit summaries
data/layer1/previews/     Human-readable inspection previews
```

## Design Documents

- [GUIDE.md](GUIDE.md) — full project guide, decisions, and status.
- [docs/experiment_design.md](docs/experiment_design.md) — detailed experiment design.
