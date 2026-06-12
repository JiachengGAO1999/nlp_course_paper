# Compression Architecture for Multi-Turn Reasoning QA

Course paper project: How do different prompt-based self-compression architectures
affect downstream multi-turn reasoning QA reliability?

## Quick Start

```bash
# 1. Generate dialogues from benchmark QA items
python scripts/01_generate_dialogues.py --config configs/experiment.yaml

# 2. Build compression variants
python scripts/02_build_compressions.py --config configs/experiment.yaml

# 3. Run inference
python scripts/03_run_inference.py --config configs/experiment.yaml

# 4. Score and summarize
python scripts/04_summarize_results.py
```

## Conditions

| Condition | Architecture |
|---|---|
| `full_history` | No compression (upper bound) |
| `one_shot_summary` | All turns → one compression call |
| `hybrid_summary_recent` | Older turns compressed + recent turn verbatim |
| `user_only_summary` | User messages only → compression call |

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

## Design Documents

- [GUIDE.md](GUIDE.md) — full project guide, decisions, and status.
- [docs/experiment_design.md](docs/experiment_design.md) — detailed experiment design.
- [docs/data_schema.md](docs/data_schema.md) — data format specification.
