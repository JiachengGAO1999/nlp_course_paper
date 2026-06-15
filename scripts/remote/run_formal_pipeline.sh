#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/gaojc/projects/nlp_course_paper}"
PYTHON="${PYTHON:-/data/gaojc/mamba/envs/cotx/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8004/v1}"
MODEL="${MODEL:-qwen3-8b-budget}"
RUN_DATE="${RUN_DATE:-20260613}"
RUN_ID="${RUN_ID:-layer1_formal_qwen3_8b_budget800_${RUN_DATE}}"
RUN_DIR="${RUN_DIR:-runs/${RUN_ID}}"
FORMAL_POOL_SIZE="${FORMAL_POOL_SIZE:-}"
FORMAL_TARGET_N="${FORMAL_TARGET_N:-}"
FORMAL_POOL_HOP_ALLOCATION_JSON="${FORMAL_POOL_HOP_ALLOCATION_JSON:-}"
FORMAL_HOP_ALLOCATION_JSON="${FORMAL_HOP_ALLOCATION_JSON:-}"
FORMAL_PROFILE_ALLOCATION_JSON="${FORMAL_PROFILE_ALLOCATION_JSON:-}"
MODEL_EXTRA_BODY_JSON="${MODEL_EXTRA_BODY_JSON:-}"
SERVER_VERSION="${SERVER_VERSION:-}"

cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR"/{logs,pool,formal,gate}
EFFECTIVE_CONFIG="$RUN_DIR/config.effective.yaml"

echo "[formal] project=$PROJECT_DIR"
echo "[formal] python=$PYTHON"
echo "[formal] model=$MODEL base_url=$BASE_URL"
echo "[formal] run_dir=$RUN_DIR"
echo "[formal] formal_pool_size=${FORMAL_POOL_SIZE:-config-default}"
echo "[formal] formal_target_n=${FORMAL_TARGET_N:-config-default}"
echo "[formal] started=$(date -Is)"

cp configs/experiment.yaml "$RUN_DIR/config.snapshot.yaml"
EFFECTIVE_ARGS=()
if [[ -n "$FORMAL_POOL_SIZE" ]]; then
  EFFECTIVE_ARGS+=(--formal-pool-size "$FORMAL_POOL_SIZE")
fi
if [[ -n "$FORMAL_TARGET_N" ]]; then
  EFFECTIVE_ARGS+=(--formal-target-n "$FORMAL_TARGET_N")
fi
if [[ -n "$FORMAL_POOL_HOP_ALLOCATION_JSON" ]]; then
  EFFECTIVE_ARGS+=(--formal-pool-hop-allocation-json "$FORMAL_POOL_HOP_ALLOCATION_JSON")
fi
if [[ -n "$FORMAL_HOP_ALLOCATION_JSON" ]]; then
  EFFECTIVE_ARGS+=(--formal-hop-allocation-json "$FORMAL_HOP_ALLOCATION_JSON")
fi
if [[ -n "$FORMAL_PROFILE_ALLOCATION_JSON" ]]; then
  EFFECTIVE_ARGS+=(--formal-profile-allocation-json "$FORMAL_PROFILE_ALLOCATION_JSON")
fi
EFFECTIVE_ARGS+=(--model "$MODEL")
EFFECTIVE_ARGS+=(--base-url "$BASE_URL")
if [[ -n "$SERVER_VERSION" ]]; then
  EFFECTIVE_ARGS+=(--server-version "$SERVER_VERSION")
fi
if [[ -n "$MODEL_EXTRA_BODY_JSON" ]]; then
  EFFECTIVE_ARGS+=(--model-extra-body-json "$MODEL_EXTRA_BODY_JSON")
fi
"$PYTHON" scripts/make_effective_config.py \
  --config configs/experiment.yaml \
  --output "$EFFECTIVE_CONFIG" \
  "${EFFECTIVE_ARGS[@]}"

echo "[formal] 1/8 prepare formal pool"
POOL_ARGS=()
if [[ -n "$FORMAL_POOL_SIZE" ]]; then
  POOL_ARGS+=(--pool-size "$FORMAL_POOL_SIZE")
fi
if [[ -n "$FORMAL_POOL_HOP_ALLOCATION_JSON" ]]; then
  POOL_ARGS+=(--hop-allocation-json "$FORMAL_POOL_HOP_ALLOCATION_JSON")
fi
"$PYTHON" scripts/07_prepare_formal_pool.py \
  --config "$EFFECTIVE_CONFIG" \
  --output "$RUN_DIR/pool/formal_pool_samples.jsonl" \
  --audit "$RUN_DIR/pool/formal_pool_sampling_audit.json" \
  --preview "$RUN_DIR/pool/formal_pool_sampling_preview.md" \
  "${POOL_ARGS[@]}"

echo "[formal] 2/8 convert formal pool to MC"
"$PYTHON" scripts/02_convert_musique_to_mc.py \
  --config "$EFFECTIVE_CONFIG" \
  --split formal_pool \
  --input "$RUN_DIR/pool/formal_pool_samples.jsonl" \
  --output "$RUN_DIR/pool/formal_pool_mc.jsonl" \
  --audit "$RUN_DIR/pool/formal_pool_mc_audit.json" \
  --preview "$RUN_DIR/pool/formal_pool_mc_preview.md" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --retries 3

echo "[formal] 3/8 generate formal pool dialogues"
"$PYTHON" scripts/03_generate_dialogues.py \
  --config "$EFFECTIVE_CONFIG" \
  --split formal_pool \
  --input "$RUN_DIR/pool/formal_pool_mc.jsonl" \
  --output "$RUN_DIR/pool/formal_pool_dialogues.jsonl" \
  --audit "$RUN_DIR/pool/formal_pool_dialogue_audit.json" \
  --preview "$RUN_DIR/pool/formal_pool_dialogue_preview.md" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --retries 3

echo "[formal] 4/8 build full-history gate variants"
"$PYTHON" scripts/04_build_compressions.py \
  --config "$EFFECTIVE_CONFIG" \
  --split formal_pool \
  --conditions full_history \
  --input "$RUN_DIR/pool/formal_pool_dialogues.jsonl" \
  --output "$RUN_DIR/gate/formal_pool_full_history_variants.jsonl" \
  --audit "$RUN_DIR/gate/formal_pool_full_history_variant_audit.json"

echo "[formal] 5/8 run full-history gate"
"$PYTHON" scripts/05_run_inference.py \
  --config "$EFFECTIVE_CONFIG" \
  --variants "$RUN_DIR/gate/formal_pool_full_history_variants.jsonl" \
  --run-dir "$RUN_DIR/gate/inference" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout-seconds 600 \
  --retries 3

echo "[formal] 6/8 select formal set"
SELECT_ARGS=()
if [[ -n "$FORMAL_TARGET_N" ]]; then
  SELECT_ARGS+=(--target-n "$FORMAL_TARGET_N")
fi
if [[ -n "$FORMAL_HOP_ALLOCATION_JSON" ]]; then
  SELECT_ARGS+=(--hop-allocation-json "$FORMAL_HOP_ALLOCATION_JSON")
fi
if [[ -n "$FORMAL_PROFILE_ALLOCATION_JSON" ]]; then
  SELECT_ARGS+=(--profile-allocation-json "$FORMAL_PROFILE_ALLOCATION_JSON")
fi
"$PYTHON" scripts/06_select_formal_set.py \
  --config "$EFFECTIVE_CONFIG" \
  --dialogues "$RUN_DIR/pool/formal_pool_dialogues.jsonl" \
  --results "$RUN_DIR/gate/inference/generations.parsed.jsonl" \
  --output "$RUN_DIR/formal/formal_selected_dialogues.jsonl" \
  --audit "$RUN_DIR/formal/formal_selection_audit.json" \
  "${SELECT_ARGS[@]}"

echo "[formal] 7/8 build formal variants"
"$PYTHON" scripts/04_build_compressions.py \
  --config "$EFFECTIVE_CONFIG" \
  --split formal \
  --input "$RUN_DIR/formal/formal_selected_dialogues.jsonl" \
  --output "$RUN_DIR/formal/formal_variants.jsonl" \
  --audit "$RUN_DIR/formal/formal_variant_audit.json" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout-seconds 300 \
  --retries 3

echo "[formal] 8/8 run formal inference"
"$PYTHON" scripts/05_run_inference.py \
  --config "$EFFECTIVE_CONFIG" \
  --variants "$RUN_DIR/formal/formal_variants.jsonl" \
  --run-dir "$RUN_DIR/formal/inference" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout-seconds 600 \
  --retries 3

echo "[formal] finished=$(date -Is)"
