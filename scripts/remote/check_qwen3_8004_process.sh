#!/usr/bin/env bash
set -euo pipefail

pid="${1:-2974130}"

echo "pid: ${pid}"
echo "cmdline:"
tr '\0' ' ' < "/proc/${pid}/cmdline"
echo

echo "env:"
tr '\0' '\n' < "/proc/${pid}/environ" | grep -E 'CUDA_VISIBLE_DEVICES|VLLM|CUDA' || true

echo "gpu process:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader | grep "${pid}" || true

echo "models:"
curl -s --max-time 5 http://127.0.0.1:8004/v1/models
echo
