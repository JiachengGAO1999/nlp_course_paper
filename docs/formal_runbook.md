# Formal Runbook

Role: reproducible operating procedure for remote formal experiments. This file
is the shared memory for humans and agents who need to launch, monitor, pull,
or verify a formal run.

For research design, see `docs/experiment_design.md`. For artifact layout, see
`docs/artifact_layout.md`.

## Server Contract

Default remote project path:

```bash
/data/gaojc/projects/nlp_course_paper
```

Default SSH host from the Windows workstation:

```bash
sjtu-a800
```

Formal runs use:

```bash
scripts/remote/run_formal_pipeline.sh
```

The remote script creates one complete run directory:

```text
runs/<RUN_ID>/
  config.snapshot.yaml
  config.effective.yaml
  pool/
  gate/
  formal/
  logs/
```

`runs/` is ignored by git. Share concrete run artifacts by copying or zipping
the run directory, not by committing it.

## Run Naming

Use stable descriptive run IDs:

```text
layer1_scale<N>_<model>_budget<tokens>_<YYYYMMDD>
```

Current examples:

```text
layer1_scale100_qwen3_8b_budget800_20260614
layer1_scale500_qwen3_8b_budget800_20260614
layer1_scale100_gemma4_e4b_budget800_20260615
layer1_scale500_gemma4_e4b_budget800_20260615
```

## Pre-Flight Checks

From Windows PowerShell:

```powershell
ssh sjtu-a800 "cd /data/gaojc/projects/nlp_course_paper && pwd && ls -la scripts/remote/run_formal_pipeline.sh scripts/make_effective_config.py configs/experiment.yaml"
```

Check GPU and vLLM service:

```powershell
ssh sjtu-a800 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader && ss -ltnp | grep ':8004' || true"
ssh sjtu-a800 "curl -s http://127.0.0.1:8004/v1/models | head -c 500"
```

Check that the target run directory does not already contain a real run:

```powershell
ssh sjtu-a800 "cd /data/gaojc/projects/nlp_course_paper && test -e runs/<RUN_ID> && echo EXISTS || echo ABSENT"
```

Do not overwrite an existing run directory unless you have explicitly decided
that the earlier run is disposable.

## Launch Pattern

Prefer a temporary shell script on the server over complex inline `ssh`
commands. This avoids broken quoting for JSON environment variables.

The pipeline has eight stages:

1. Prepare formal pool.
2. Convert formal pool to MC.
3. Generate formal-pool dialogues.
4. Build full-history gate variants.
5. Run full-history gate.
6. Select formal set.
7. Build formal variants.
8. Run formal inference.

The key environment variables are:

| Variable | Meaning |
| --- | --- |
| `RUN_ID` | Stable run name |
| `RUN_DIR` | Usually `runs/${RUN_ID}` |
| `FORMAL_POOL_SIZE` | Oversized pool before full-history gate |
| `FORMAL_TARGET_N` | Final formal sample size |
| `MODEL` | Served model name exposed by vLLM |
| `BASE_URL` | OpenAI-compatible vLLM endpoint |
| `MODEL_EXTRA_BODY_JSON` | Model-specific chat extra body |
| `SERVER_VERSION` | Human-readable server/model provenance |

## Qwen3-8B 500-Case Run

Assumes a Qwen3-8B vLLM server is already running on port 8004 as
`qwen3-8b-budget`, with the Qwen3 reasoning parser and thinking budget support.

On the server:

```bash
cd /data/gaojc/projects/nlp_course_paper
mkdir -p logs

export RUN_ID="layer1_scale500_qwen3_8b_budget800_YYYYMMDD"
export RUN_DIR="runs/${RUN_ID}"
export FORMAL_POOL_SIZE="800"
export FORMAL_TARGET_N="500"
export MODEL="qwen3-8b-budget"
export BASE_URL="http://127.0.0.1:8004/v1"
export MODEL_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":true},"thinking_token_budget":512}'
export SERVER_VERSION="vllm; qwen3-8b; gpu7; port=8004; reasoning_parser=qwen3; max_model_len=8192; thinking_token_budget=512"

LOG="logs/${RUN_ID}.driver.log"
PID="logs/${RUN_ID}.driver.pid"
nohup bash scripts/remote/run_formal_pipeline.sh > "$LOG" 2>&1 &
echo $! > "$PID"
cat "$PID"
```

Use the actual date in `RUN_ID`.

## Gemma4 E4B 500-Case Run

Assumes a Gemma4 E4B vLLM server is already running on GPU7 / port 8004 as
`gemma4-e4b-it`, with the Gemma reasoning parser.

On the server:

```bash
cd /data/gaojc/projects/nlp_course_paper
mkdir -p logs

export RUN_ID="layer1_scale500_gemma4_e4b_budget800_YYYYMMDD"
export RUN_DIR="runs/${RUN_ID}"
export FORMAL_POOL_SIZE="800"
export FORMAL_TARGET_N="500"
export MODEL="gemma4-e4b-it"
export BASE_URL="http://127.0.0.1:8004/v1"
export MODEL_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":true}}'
export SERVER_VERSION="vllm 0.19.1; gemma-4-E4B-it; gpu7; port=8004; reasoning_parser=gemma4; max_model_len=8192"

LOG="logs/${RUN_ID}.driver.log"
PID="logs/${RUN_ID}.driver.pid"
nohup bash scripts/remote/run_formal_pipeline.sh > "$LOG" 2>&1 &
echo $! > "$PID"
cat "$PID"
```

Important: Gemma should not receive Qwen's `thinking_token_budget` field. Use
only:

```json
{"chat_template_kwargs":{"enable_thinking":true}}
```

## Monitor

Replace `<RUN_ID>` and `<PID>`:

```powershell
ssh sjtu-a800 "ps -p <PID> -o pid,ppid,stat,etime,cmd"
ssh sjtu-a800 "cd /data/gaojc/projects/nlp_course_paper && tail -f logs/<RUN_ID>.driver.log"
```

Check stage progress without following:

```powershell
ssh sjtu-a800 "cd /data/gaojc/projects/nlp_course_paper && tail -60 logs/<RUN_ID>.driver.log"
```

Check final summary:

```powershell
ssh sjtu-a800 "cd /data/gaojc/projects/nlp_course_paper && cat runs/<RUN_ID>/formal/inference/summary.json"
```

## Pull To Local

From the local repository root on Windows:

```powershell
scp -r sjtu-a800:/data/gaojc/projects/nlp_course_paper/runs/<RUN_ID> runs\
```

If the local directory already exists, inspect it before re-copying.

## Verify Local Artifacts

Check summary:

```powershell
Get-Content runs\<RUN_ID>\formal\inference\summary.json
```

Check row counts and reasoning capture:

```powershell
python -c "import json,pathlib; p=pathlib.Path('runs/<RUN_ID>/formal/inference/generations.parsed.jsonl'); rows=[json.loads(l) for l in p.open(encoding='utf-8')]; print('rows',len(rows)); print('reasoning_nonempty',sum(bool(r.get('response_reasoning')) for r in rows)); print('content_nonempty',sum(bool(r.get('response_content')) for r in rows)); print('raw_message',sum(bool(r.get('response_message')) for r in rows)); print('conditions',{c:sum(r.get('condition')==c for r in rows) for c in sorted(set(r.get('condition') for r in rows))})"
```

Expected formal row count is:

```text
FORMAL_TARGET_N * 3
```

For a 500-case run, expect 1500 parsed rows and 500 rows for each condition.

## Output Files To Inspect

Primary evaluation:

```text
runs/<RUN_ID>/formal/inference/summary.json
runs/<RUN_ID>/formal/inference/generations.parsed.jsonl
```

Raw model responses:

```text
runs/<RUN_ID>/formal/inference/generations.raw.jsonl
```

Effective configuration:

```text
runs/<RUN_ID>/config.effective.yaml
runs/<RUN_ID>/formal/inference/config.snapshot.json
```

Selection and token audits:

```text
runs/<RUN_ID>/formal/formal_selection_audit.json
runs/<RUN_ID>/formal/formal_variant_audit.json
```

Gate results:

```text
runs/<RUN_ID>/gate/inference/summary.json
```

Embedding metrics (run after inference completes):

```powershell
ssh sjtu-a800 "cd /data/gaojc/projects/nlp_course_paper && \
  /data/gaojc/mamba/envs/cotx/bin/python scripts/08_compute_embedding_metrics.py \
  --run-dir runs/<RUN_ID> \
  > runs/<RUN_ID>/logs/embedding_metrics.log 2>&1 &"
```

Check embedding metrics summary:

```powershell
ssh sjtu-a800 "cd /data/gaojc/projects/nlp_course_paper && \
  cat runs/<RUN_ID>/formal/metrics/embedding_metrics_summary.json"
```

## Current Reference Runs

Use these as comparison anchors:

| Run | Model | Formal N | Status |
| --- | --- | ---: | --- |
| `layer1_scale100_qwen3_8b_budget800_20260614` | Qwen3-8B | 100 | Complete |
| `layer1_scale500_qwen3_8b_budget800_20260614` | Qwen3-8B | 500 | Complete |
| `layer1_scale100_gemma4_e4b_budget800_20260615` | Gemma4 E4B | 100 | Complete |
| `layer1_scale500_gemma4_e4b_budget800_20260615` | Gemma4 E4B | 500 | Complete |

Update this table when a run completes or is abandoned.
