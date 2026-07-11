# Gemma 4 12B + MTP on llama.cpp — minimal example

Multi-token prediction (MTP) speeds up generation by letting a small,
co-trained drafter model propose several tokens per step, which the target
model then verifies in a single forward pass. Unlike generic speculative
decoding, Gemma 4's drafters were trained alongside the base model, so
acceptance rates (and speedups) are much higher — roughly 2x on dense models.

llama.cpp gained MTP support in build b9549
([PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)).
The build in `../local-llm/llama.cpp` is b9569, so it's covered.

## What this uses (all already on disk, shared with `../local-llm`)

| Piece | Path |
|---|---|
| llama.cpp (b9569, CUDA) | `../local-llm/llama.cpp/build/bin/llama-server` |
| Target model (Q5_K_XL, 8.6 GB) | `../local-llm/models/gemma-4-12b-it-UD-Q5_K_XL.gguf` |
| MTP drafter (Q6_K_M, 0.35 GB) | `../local-llm/models/gemma-4-12b-it-Q6_K_M-MTP.gguf` |

The drafter comes from
[cloudnathan5/gemma-4-12b-it-MTP-GGUF](https://huggingface.co/cloudnathan5/gemma-4-12b-it-MTP-GGUF)
— the non-QAT `it` assistant, matching the non-QAT target quant.

> **Gotcha:** drafter GGUFs for Gemma 4 exist in two incompatible metadata
> formats. llama.cpp mainline wants arch `gemma4-assistant` (hyphen, with
> `nextn_predict_layers` metadata). Files with arch `gemma4_assistant`
> (underscore, `requires_target_arch` metadata — e.g. the `cortexist` repo)
> were converted for a different runtime and fail to load with
> `unknown model architecture`. Check before downloading:
> `curl -sL -r 0-4095 <resolve-url> | grep -aoE 'gemma4[_-]assistant'`

## Run

Day-to-day operations (start/stop/status/troubleshooting): see [RUNSHEET.md](RUNSHEET.md).

```bash
./run.sh              # server on :8080 with MTP
./run.sh --no-mtp     # baseline for comparison
./bench.sh            # measure tok/s against the running server
```

The MTP magic is just three flags on `llama-server`:

```
--model-draft <drafter.gguf> --spec-type draft-mtp --spec-draft-n-max 4
```

## Results (RTX 4070 Ti SUPER 16 GB, ctx 32768)

Measured 2026-07-11, 512-token code generation, 3-run averages:

| Config | Ctx | KV cache | tok/s | VRAM used* |
|---|---|---|---|---|
| baseline (no MTP) | 32k | f16 | 54.1 | — |
| MTP | 32k | f16 | 172.9 | 12.6 GB |
| MTP | 131k | f16 | 166.4 | 15.1 GB |
| **MTP (default)** | **131k** | **q8_0/q8_0** | **160.2** | **13.9 GB** |
| MTP (`-c 262144`) | 262k | q8_0/q8_0 | 158.2 | 14.1 GB |
| MTP | 131k | q8_0/q4_0 | 86.6 | 12.2 GB |

*includes ~1.7 GB of desktop use on this 16 GB card.

Draft acceptance on code was ~79-82% in every config. Speed is identical at
temp 0 and at the Gemma-recommended sampling (temp 1.0, top-p 0.95, top-k 64).

## Notes

- **KV cache defaults to q8_0/q8_0.** The reported "quantized KV breaks MTP
  acceptance" bug does NOT affect build b9569 (acceptance stayed ~80% in
  our tests). But **q4_0 on the V cache halves MTP throughput** (166 → 87
  tok/s) — the `local-llm` scripts use `--cache-type-v q4_0`; don't copy
  that here. `--kv f16` buys ~4% more speed at the cost of ~1.2 GB VRAM.
- **Full 262k training context works** (`./run.sh -c 262144`) at ~158 tok/s.
  Above 131k, `run.sh` automatically drops batch/ubatch from 4096/2048 to
  2048/512: the big-ubatch compute buffer (~1.8 GB) no longer fits, and
  llama.cpp's auto-fit silently spills it to host memory, collapsing
  generation to ~57 tok/s. The smaller ubatch only costs prompt-processing
  speed (~2,900 tok/s, i.e. ~35 s to cold-fill 100k tokens; cached
  incrementally on later harness turns).
- MTP speedup is workload-dependent: highest on code and structured output
  (predictable tokens → high acceptance), lower on free-form prose.
- The server log prints draft acceptance stats — healthy is ~55-70%.
- To use from OpenCode, the existing `../local-llm/scripts/sync_opencode_config.py`
  already points OpenCode at `localhost:8080`.
