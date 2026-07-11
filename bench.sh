#!/usr/bin/env bash
set -euo pipefail

# Measure generation speed of the llama-server on $PORT (default 8080).
# Sends a code-generation prompt (MTP acceptance is highest on code) and
# reports the server-side tokens/sec from the response timings.

PORT="${PORT:-8080}"
N_PREDICT="${N_PREDICT:-512}"
RUNS="${RUNS:-3}"

PROMPT='Write a complete Python implementation of a binary search tree with insert, delete, search, and in-order traversal methods. Include docstrings.'

echo "Benchmarking http://localhost:$PORT ($RUNS runs, $N_PREDICT tokens each)..."
total=0
for i in $(seq 1 "$RUNS"); do
    tps=$(curl -s "http://localhost:$PORT/completion" \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\": \"$PROMPT\", \"n_predict\": $N_PREDICT, \"temperature\": 0}" \
        | python3 -c 'import json,sys; print(round(json.load(sys.stdin)["timings"]["predicted_per_second"], 1))')
    echo "  run $i: $tps tok/s"
    total=$(python3 -c "print($total + $tps)")
done
python3 -c "print(f'Average: {$total/$RUNS:.1f} tok/s')"
