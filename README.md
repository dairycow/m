# m — a minimal, ultra-fast coding agent

`m` is a terminal coding agent in the spirit of [pi](https://pi.dev)
(minimal core, file-based extensibility, no baked-in bloat) with the
performance envelope of [jcode](https://github.com/1jehuang/jcode)
(instant Rust TUI). It is built to drive the local
**Gemma 4 12B + MTP** llama-server in this repo ([SERVER.md](SERVER.md),
[RUNSHEET.md](RUNSHEET.md)) — but any OpenAI-compatible endpoint works.

## Measured (this machine, RTX 4070 Ti SUPER)

| metric | m | reference |
|---|---|---|
| cold start (`m --version`) | **0.3 ms** | jcode 14 ms, node harnesses 200 ms+ |
| TUI time-to-first-frame | **0.4 ms** | jcode 14.0 ms |
| RSS at first frame | **6.8 MB** | jcode 27.8 MB |
| binary size | 5.3 MB (4.2 MB static musl) | — |
| generation (local Gemma 4 MTP) | 145–160 tok/s, 73–82 % draft acceptance | 54 tok/s without MTP |

Reproduce: `M_PERF=1 m` prints first-frame time and RSS on exit; the
startup number is a 100-run median of `m --version`.

## Design

- **Four tools** — `read`, `write`, `edit`, `bash` (+ a lazy `skill` tool
  only when skills exist). System prompt + tool schemas ≈ 700 tokens.
- **No async runtime.** Threads and channels. The HTTP/SSE client is
  hand-rolled over `TcpStream` (rustls for https), which buys two things
  packaged clients can't do: cancellation that actually aborts server-side
  generation (socket shutdown), and calm survival of 30 s+ first-token
  latency on large prompt refills.
- **Prompt-cache discipline.** Stable system prompt, append-only history —
  so llama-server's KV prefix cache makes multi-turn latency near-zero
  (watch `cache` in the status bar).
- **Local-server telemetry** in the status bar: ctx fill %, live tok/s,
  MTP draft-acceptance %, cached tokens. No other harness shows these.
- **Sessions** are append-only JSONL under `~/.local/share/m/sessions/`,
  resumable (`m -r`, `ctrl+r` picker), faithful even across crashes.
- **Small-model armor**, tuned on real SWE-bench trajectories: runaway
  responses are discarded and retried with a corrective nudge instead of
  poisoning the context, and a loop breaker intercepts the third identical
  tool call so the agent steps back instead of spinning.
- **YOLO by default**, like pi. You are the permission system.

## Usage

```bash
m                       # interactive TUI (this directory = the project)
m -r                    # resume the latest session here
m -p "fix the failing test"          # headless: stream answer to stdout
m -p --json "..."       # headless with a JSONL event stream
m --temp 0 --max-turns 40 -p "..."   # bench-style determinism
```

Keys: `enter` send · `shift/alt+enter` newline · `esc` cancel ·
`ctrl+c ×2` quit · `ctrl+o`/`ctrl+t` expand tool/thinking · `ctrl+r`
sessions · `pgup/pgdn`/wheel scroll. Type `/` for commands
(`/new /resume /compact /skills /help /quit`). Typing while the agent runs
queues steering input for the next turn.

## Configuration

Zero-config default: `http://localhost:8080`, any model. Profiles in
`~/.config/m/config.toml`:

```toml
default_profile = "local"

[profiles.local]
base_url = "http://localhost:8080"
api_key = "none"
model = "gemma-4-12b"

[profiles.or]
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-or-..."
model = "qwen/qwen3-coder"
temperature = 0.2      # optional; omitted = server default
```

Select with `m -m or` or `M_PROFILE=or`.

## Extensibility (all files, no plugins)

- **AGENTS.md** — loaded hierarchically: `~/.config/m/AGENTS.md`, then every
  `AGENTS.md`/`CLAUDE.md` from `/` down to the working directory.
- **Skills** — Claude Code-compatible `SKILL.md` folders discovered in
  `~/.claude/skills`, `~/.config/m/skills`, `.claude/skills`, `.m/skills`.
  Only name+description enter the system prompt; the model loads a skill
  on demand with the `skill` tool.
- **Slash commands** — markdown templates in `~/.config/m/commands/*.md`
  and `.m/commands/*.md`; `$ARGUMENTS` is substituted, frontmatter
  `description:` shows in the completion menu.

## SWE-bench

`m-bench` runs SWE-bench Lite instances in the official per-instance
docker images against the local server, and the official harness scores
the patches. See [bench/RESULTS.md](bench/RESULTS.md).

**Result (2026-07-12): 11/30 resolved (36.7%)** on a reproducible
stratified 30-instance slice of SWE-bench Lite — entirely on the local
Gemma 4 12B MTP server (temp 0, ≤40 turns, 4096-token responses).
20/30 produced a patch; mean 21.9 turns and ~2 minutes of agent time per
instance; 1h01m total.

```bash
# one-time
cargo build --release
cargo build -p m-tui --release --target x86_64-unknown-linux-musl --no-default-features
./target/release/m-bench fetch                      # dataset → bench/dataset.json
./target/release/m-bench pick -n 30 > bench/instances.txt

# run (server must be up: ./run.sh)
./target/release/m-bench run --instances bench/instances.txt --out bench/runs/v1

# score with the official harness
bench/.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path bench/runs/v1/predictions.jsonl \
  --run_id m-v1 --max_workers 4

# report (merges the eval JSON the harness writes)
./target/release/m-bench report --run bench/runs/v1 --eval m-gemma4-12b-mtp.m-v1.json
```

The instance list is a deterministic stratified slice (all 300 Lite ids
sorted, every 10th) — reproducible by anyone. To keep scaffold tuning
honest, that slice is treated as the *dev set* (its trajectories get mined
for generic failure modes) while `m-bench pick -n 30 --offset 5` yields a
disjoint *held-out* slice on which changes are actually judged. Only
behavioral fixes are allowed — nothing instance- or repo-specific.

## Build

```bash
cargo build --release          # target/release/m, m-bench
cargo test && cargo clippy     # 14 tests, zero warnings
```

Workspace: `crates/m-core` (agent loop, provider, tools, sessions, config,
context/skills — no UI deps), `crates/m-tui` (the `m` binary: TUI +
headless modes), `crates/m-bench` (SWE-bench runner). TLS is a cargo
feature; the musl bench build disables it (containers only talk plain
http to localhost) so the static binary needs no C toolchain at all.

## Development

Architecture, the invariants behind the speed (socket ownership,
prompt-cache discipline, faithful session logs), how to add tools or slash
commands, the bench anti-overfitting protocol, and the release checklist
live in [DEVELOPMENT.md](DEVELOPMENT.md) — including how to hack on m
using m itself.
