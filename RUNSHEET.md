# Runsheet — Gemma 4 12B + MTP server

## Locations

| What | Where |
|---|---|
| Start script | `~/Projects/m/run.sh` |
| Benchmark script | `~/Projects/m/bench.sh` |
| llama-server binary | `~/Projects/local-llm/llama.cpp/build/bin/llama-server` (b9569) |
| Target model | `~/Projects/local-llm/models/gemma-4-12b-it-UD-Q5_K_XL.gguf` |
| MTP drafter | `~/Projects/local-llm/models/gemma-4-12b-it-Q6_K_M-MTP.gguf` |
| Endpoint | `http://localhost:8080` (OpenAI-compatible + `/completion`) |

## Start

```bash
cd ~/Projects/m

./run.sh                            # foreground: 131k ctx, q8_0 KV, MTP — 160 tok/s
./run.sh -c 262144                  # full 262k ctx — 158 tok/s
./run.sh --no-mtp                   # baseline without MTP (54 tok/s)
./run.sh --kv f16                   # +4% speed, ~1.2 GB more VRAM

# background with log
./run.sh > server.log 2>&1 &
```

Model load takes ~5-10 s. Ready when `/health` returns ok.

## Stop

```bash
pkill -f llama-server
```

## Status / verify

```bash
curl -s http://localhost:8080/health          # {"status":"ok"} when ready
pgrep -af llama-server                        # process running?
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # VRAM (expect ~14 GB)
./bench.sh                                    # tok/s (expect ~155-165)
```

## Health checks after config changes

1. **Speed sanity:** `./bench.sh` should report 155+ tok/s. If it reads
   ~55-90 tok/s, something regressed — see below.
2. **MTP working:** server log should show `draft acceptance = 0.7-0.8`
   on code generation. Near zero → drafter/KV mismatch.
3. **No silent spill:** `grep "failed to allocate" server.log` must be
   empty. One hit means the compute buffer went to host RAM (generation
   collapses to ~57 tok/s with no other warning).

## Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| ~57 tok/s, log has `failed to allocate CUDA0 buffer` | ubatch too big for ctx size; buffer spilled to host | ctx > 131k needs batch/ubatch 2048/512 (run.sh does this automatically) |
| ~87 tok/s | `q4_0` V-cache (old local-llm default) | use q8_0 or f16 for `--cache-type-v` |
| `unknown model architecture: 'gemma4_assistant'` | drafter GGUF converted for a different runtime (underscore arch) | use a `gemma4-assistant` (hyphen) drafter, e.g. cloudnathan5/gemma-4-12b-it-MTP-GGUF |
| `error while loading shared libraries: libllama-server-impl.so` | binary run without lib path | `run.sh` sets `LD_LIBRARY_PATH`; for manual runs: `LD_LIBRARY_PATH=~/Projects/local-llm/llama.cpp/build/bin` |
| OOM on load | desktop VRAM use grew (Firefox etc.) | drop to `-c 131072`, or free GPU memory |

## Connect a harness

OpenCode: `~/Projects/local-llm/scripts/sync_opencode_config.py` writes the
config pointing at `localhost:8080`. Any OpenAI-compatible client works:
base URL `http://localhost:8080/v1`, any non-empty API key.

## Reference numbers (RTX 4070 Ti SUPER 16 GB, 2026-07-11)

- Generation: 160 tok/s @ 131k ctx, 158 @ 262k, ~173 @ 32k f16 KV
- Prompt processing: ~2,900 tok/s at ubatch 512 (ctx > 131k); cold 100k
  fill ≈ 35 s, then incremental via prompt cache
- Draft acceptance on code: ~79-82%
- VRAM: ~12.4 GB server total @ 131k (14.1 GB shown incl. ~1.7 GB desktop)
