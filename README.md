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

## Design Documents

- [docs/research_framework.md](docs/research_framework.md) — concise shareable research framework.
- [GUIDE.md](GUIDE.md) — full project guide, decisions, and status.
- [docs/experiment_design.md](docs/experiment_design.md) — detailed experiment design.
- [docs/artifact_layout.md](docs/artifact_layout.md) — run/data artifact layout.
