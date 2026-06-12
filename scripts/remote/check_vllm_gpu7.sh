#!/usr/bin/env bash
set -euo pipefail

echo "host: $(hostname)"
echo "gpu:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader | sed -n '1,8p'

echo "port 8004:"
if command -v ss >/dev/null 2>&1; then
  ss -ltnp 2>/dev/null | grep 8004 || true
else
  netstat -ltnp 2>/dev/null | grep 8004 || true
fi

echo "vllm processes:"
ps -ef | grep -E 'vllm|api_server' | grep -v grep || true

echo "models endpoint:"
python - <<'PY'
import json
import urllib.request

url = "http://127.0.0.1:8004/v1/models"
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8", errors="replace")
        print(response.status)
        print(body[:1000])
except Exception as exc:
    print(type(exc).__name__, exc)
PY
