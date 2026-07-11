#!/usr/bin/env bash
set -euo pipefail

# Concurrent MTP example: Gemma 4 12B + co-trained MTP
# Optimized for 4 concurrent sessions on 16GB VRAM.
#
# Usage:
#   ./run_concurrent.sh            # 4 concurrent sessions (32k ctx)
#   ./run_concurrent.sh --no-mtp   # baseline, for comparison
#   ./run_concurrent.sh -c 65536   # override context size

LOCAL_LLM_DIR="$(cd "$(dirname "$0")/../local-llm" && pwd)"
LLAMA_BIN_DIR="$LOCAL_LLM_DIR/llama.cpp/build/bin"
MODEL="$LOCAL_LLM_DIR/models/gemma-4-12b-it-UD-Q5_K_XL.gguf"
DRAFT_MODEL="$LOCAL_LLM_DIR/models/gemma-4-12b-it-Q6_K_M-MTP.gguf"

PORT="${PORT:-8080}"
# Reduced from 262k to 32k to allow 4 sessions in 16GB VRAM.
CTX_SIZE=32768
KV_TYPE=q8_0    
WITH_MTP=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-mtp) WITH_MTP=false; shift ;;
        -c|--ctx-size) CTX_SIZE="$2"; shift 2 ;;
        --kv) KV_TYPE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1 (supported: --no-mtp, -c NUM, --kv TYPE, --port NUM)"; exit 1 ;;
    esac
done

for f in "$LLAMA_BIN_DIR/llama-server" "$MODEL"; do
    [[ -e "$f" ]] || { echo "Error: not found: $f"; exit 1; }
done

# At 32k, we can use larger batches for better prompt throughput.
if [[ "$CTX_SIZE" -gt 131072 ]]; then
    BATCH_SIZE=2048; UBATCH_SIZE=512
else
    BATCH_SIZE=4096; UBATCH_SIZE=2048
fi

CMD=(
    "$LLAMA_BIN_DIR/llama-server"
    --model "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --ctx-size "$CTX_SIZE"
    --n-gpu-layers 99
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --temp 1.0
    --top-p 0.95
    --top-k 64
    --parallel 4
    --threads 4
    --flash-attn on
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
)

if $WITH_MTP; then
    [[ -f "$DRAFT_MODEL" ]] || { echo "Error: MTP drafter not found: $DRAFT_MODEL"; exit 1; }
    CMD+=(
        --model-draft "$DRAFT_MODEL"
        --spec-type draft-mtp
        --spec-draft-n-max 4
    )
    echo ">> Gemma 4 12B Concurrent MTP, ctx $CTX_SIZE, kv $KV_TYPE, port $PORT, parallel 4"
else
    echo ">> Gemma 4 12B Baseline Concurrent, ctx $CTX_SIZE, kv $KV_TYPE, port $PORT, parallel 4"
fi

export LD_LIBRARY_PATH="$LLAMA_BIN_DIR:${LD_LIBRARY_PATH:-}"
exec "${CMD[@]}"
