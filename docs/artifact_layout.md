# Artifact Layout

This project separates reusable benchmark data from experiment-run artifacts.

## Top-Level Rule

- `data/`: reusable external or benchmark-derived source data only.
- `runs/<run_id>/`: every generated artifact for one concrete experiment run.

Do not store MC conversions, generated dialogues, compression variants,
inference outputs, or run-specific audits under `data/`.

## `data/`

```text
data/
  raw/
    musique_ans_v1.0_dev.jsonl
```

`data/raw/` may contain downloaded benchmark files or immutable local caches.
These files are not committed.

## Formal Run Layout

Example:

```text
runs/layer1_formal_qwen3_8b_budget800_20260613/
  config.snapshot.yaml
  logs/
    formal_pipeline_20260613.log

  pool/
    formal_pool_samples.jsonl
    formal_pool_sampling_audit.json
    formal_pool_sampling_preview.md
    formal_pool_mc.jsonl
    formal_pool_mc_audit.json
    formal_pool_mc_preview.md
    formal_pool_dialogues.jsonl
    formal_pool_dialogue_audit.json
    formal_pool_dialogue_preview.md

  gate/
    formal_pool_full_history_variants.jsonl
    formal_pool_full_history_variant_audit.json
    inference/
      config.snapshot.json
      generations.raw.jsonl
      generations.parsed.jsonl
      summary.json

  formal/
    formal_selected_dialogues.jsonl
    formal_selection_audit.json
    formal_variants.jsonl
    formal_variant_audit.json
    inference/
      config.snapshot.json
      generations.raw.jsonl
      generations.parsed.jsonl
      summary.json
    11_critical_annotation.md  # optional manual mechanism audit
    11_critical_audit.md       # optional rendered audit packet
```

Manual annotation files are optional and should be added only after automatic
inference results have been inspected and a critical subset has been selected.
Scale-up runs should stop at `formal/inference/` before this step.

## Naming

Run IDs should be stable and descriptive:

```text
layer1_<phase>_<model>_budget<tokens>_<YYYYMMDD>
```

Examples:

```text
layer1_smoke_qwen3_8b_budget800_20260613
layer1_pilot_qwen3_8b_budget800_20260613
layer1_formal_qwen3_8b_budget800_20260613
```

## Legacy Artifacts

Older Layer 1 artifacts from early development have been moved under
`runs/legacy_layer1_artifacts_20260613/`. New pipeline runs should use the
run-root layout above. `data/` should contain only reusable benchmark/source
data such as `data/raw/`.
