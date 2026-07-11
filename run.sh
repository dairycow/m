#!/usr/bin/env bash
set -euo pipefail

# Minimal MTP (multi-token prediction) example: Gemma 4 12B + co-trained MTP
# drafter on llama.cpp. Reuses the llama.cpp build and models from
# ../local-llm — nothing is duplicated here.
#
# Usage:
#   ./run.sh            # MTP enabled (default)
#   ./run.sh --no-mtp   # baseline, for comparison
#   ./run.sh -c 65536   # override context size

LOCAL_LLM_DIR="$(cd "$(dirname "$0")/../local-llm" && pwd)"
LLAMA_BIN_DIR="$LOCAL_LLM_DIR/llama.cpp/build/bin"
MODEL="$LOCAL_LLM_DIR/models/gemma-4-12b-it-UD-Q5_K_XL.gguf"
DRAFT_MODEL="$LOCAL_LLM_DIR/models/gemma-4-12b-it-Q6_K_M-MTP.gguf"

PORT="${PORT:-8080}"
CTX_SIZE=262144
KV_TYPE=q8_0    # f16 is ~4% faster but leaves <1GB VRAM headroom at 131k.
                # Never use q4_0 for V: it halves MTP throughput (166 -> 87 tok/s).
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

# Above 131k the ubatch-2048 compute buffer (~1.8GB) no longer fits and
# llama.cpp silently spills to host memory, dropping generation to ~57 tok/s.
# Smaller batches keep everything on-GPU; only prompt processing slows down.
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
    --parallel 1
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
    echo ">> Gemma 4 12B with MTP (draft n-max 4), ctx $CTX_SIZE, kv $KV_TYPE, port $PORT"
else
    echo ">> Gemma 4 12B baseline (no MTP), ctx $CTX_SIZE, kv $KV_TYPE, port $PORT"
fi

export LD_LIBRARY_PATH="$LLAMA_BIN_DIR:${LD_LIBRARY_PATH:-}"
exec "${CMD[@]}"
