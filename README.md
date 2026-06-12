# NLP Course Paper

This repository contains a small controlled experiment for an NLP course paper:

> Budget-aware dialogue history compression for multi-turn reasoning QA.

The project studies how different compressed representations of prior dialogue
history affect Qwen3-8B's later multiple-choice reasoning reliability under a
fixed thinking budget.

Start with [GUIDE.md](GUIDE.md). It records the project framing, fixed design
decisions, experiment progression, and artifact policy.

## Planned Layout

- `GUIDE.md`: operational guide and current project state.
- `docs/proposal.md`: course-paper proposal in Chinese.
- `docs/experiment_design.md`: experiment design and variables.
- `docs/data_schema.md`: synthetic diagnostic testbed schema.
- `configs/experiment.yaml`: first main experiment configuration.
- `src/`: reusable code.
- `scripts/`: command-line entry points.
- `data/`: generated datasets and variants.
- `runs/`: run outputs and analysis artifacts.
