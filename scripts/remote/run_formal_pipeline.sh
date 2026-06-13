#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/gaojc/projects/nlp_course_paper}"
PYTHON="${PYTHON:-/data/gaojc/mamba/envs/cotx/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8004/v1}"
MODEL="${MODEL:-qwen3-8b-budget}"
RUN_DATE="${RUN_DATE:-20260613}"

cd "$PROJECT_DIR"
mkdir -p logs

echo "[formal] project=$PROJECT_DIR"
echo "[formal] python=$PYTHON"
echo "[formal] model=$MODEL base_url=$BASE_URL"
echo "[formal] started=$(date -Is)"

echo "[formal] 1/8 prepare formal pool"
"$PYTHON" scripts/07_prepare_formal_pool.py --config configs/experiment.yaml

echo "[formal] 2/8 convert formal pool to MC"
"$PYTHON" scripts/02_convert_musique_to_mc.py \
  --config configs/experiment.yaml \
  --split formal_pool \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --retries 3

echo "[formal] 3/8 generate formal pool dialogues"
"$PYTHON" scripts/03_generate_dialogues.py \
  --config configs/experiment.yaml \
  --split formal_pool \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --retries 3

echo "[formal] 4/8 build full-history gate variants"
"$PYTHON" scripts/04_build_compressions.py \
  --config configs/experiment.yaml \
  --split formal_pool \
  --conditions full_history \
  --output data/layer1/variants/formal_pool_full_history_variants.jsonl
cp data/layer1/audits/formal_pool_variant_audit.json data/layer1/audits/formal_pool_full_history_variant_audit.json

echo "[formal] 5/8 run full-history gate"
"$PYTHON" scripts/05_run_inference.py \
  --config configs/experiment.yaml \
  --variants data/layer1/variants/formal_pool_full_history_variants.jsonl \
  --run-dir "runs/layer1_formal_pool_full_history_gate_${RUN_DATE}" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout-seconds 600 \
  --retries 3

echo "[formal] 6/8 select formal set"
"$PYTHON" scripts/06_select_formal_set.py \
  --config configs/experiment.yaml \
  --dialogues data/layer1/dialogues/formal_pool_dialogues.jsonl \
  --results "runs/layer1_formal_pool_full_history_gate_${RUN_DATE}/generations.parsed.jsonl" \
  --output data/layer1/dialogues/formal_selected_dialogues.jsonl \
  --audit data/layer1/audits/formal_selection_audit.json

echo "[formal] 7/8 build formal variants"
"$PYTHON" scripts/04_build_compressions.py \
  --config configs/experiment.yaml \
  --split formal \
  --input data/layer1/dialogues/formal_selected_dialogues.jsonl \
  --output data/layer1/variants/formal_variants.jsonl \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --retries 3
cp data/layer1/audits/formal_variant_audit.json data/layer1/audits/formal_selected_variant_audit.json

echo "[formal] 8/8 run formal inference"
"$PYTHON" scripts/05_run_inference.py \
  --config configs/experiment.yaml \
  --variants data/layer1/variants/formal_variants.jsonl \
  --run-dir "runs/layer1_formal_qwen3_8b_budget800_${RUN_DATE}" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout-seconds 600 \
  --retries 3

echo "[formal] finished=$(date -Is)"
