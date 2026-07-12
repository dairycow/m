# m — developer guide

How the harness works inside, the invariants that make it fast, and how to
extend it without breaking them. User-facing docs live in [README.md](README.md);
the local model server is documented in [SERVER.md](SERVER.md) / [RUNSHEET.md](RUNSHEET.md).

## Layout

```
crates/
  m-core/            # everything except UI — depends only on serde/toml/dirs/similar (+rustls behind `tls`)
    src/http.rs      #   hand-rolled HTTP/1.1 + SSE over TcpStream/rustls
    src/provider.rs  #   OpenAI-compatible chat: streaming, tool calls, llama.cpp telemetry
    src/agent.rs     #   the loop: model → tools → repeat; nudges, loop breaker, context guard
    src/tools.rs     #   read / write / edit / bash (+ lazy skill tool)
    src/session.rs   #   append-only JSONL sessions
    src/config.rs    #   TOML profiles; zero-config default = localhost:8080
    src/context.rs   #   AGENTS.md hierarchy, SKILL.md discovery, command templates
    src/prompt.rs    #   the <1k-token system prompt
  m-tui/             # the `m` binary
    src/main.rs      #   arg parsing, print/JSON modes, env assembly (build_env)
    src/tui/mod.rs   #   app state, event loop, cells, overlays, agent thread
    src/tui/md.rs    #   streaming markdown → width-wrapped styled lines
    src/tui/hl.rs    #   lazy syntect (loads on a background thread)
    src/tui/input.rs #   multi-line editor, history, kill ring
    src/tui/theme.rs #   the one dark theme
  m-bench/           # SWE-bench Lite runner (fetch / pick / run / report)
```

## Build & test

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release             # host binaries: m, m-bench
cargo test                        # 28 unit tests, incl. the scripted agent loop
cargo clippy --all-targets        # kept at zero warnings

# static agent for SWE-bench containers — no TLS, no C toolchain needed
cargo build -p m-tui --release --target x86_64-unknown-linux-musl --no-default-features
```

Perf check (budgets: <20ms first frame, <30MB RSS — currently ~0.5ms / ~7MB):

```bash
M_PERF=1 ./target/release/m       # prints "first frame Xms · rss YMB" on exit
```

TUI testing without a human: drive it in tmux.

```bash
tmux new-session -d -s t -x 100 -y 30 './target/release/m'
tmux send-keys -t t "your prompt" Enter && sleep 3
tmux capture-pane -t t -p         # assert on the rendered frame
```

## The invariants (read before changing anything)

**No async runtime.** Concurrency is threads + `std::sync::mpsc`: the TUI
main thread renders; one agent thread runs the loop; the SSE read blocks on
its own socket. If a change seems to need tokio, restructure it around a
blocking thread instead.

**We own the socket** (`http.rs`). The HTTP client is hand-rolled precisely so
that: (a) cancellation is a `shutdown()` — llama-server sees the disconnect
and aborts generation immediately, freeing the single slot; (b) reads use a
200ms timeout polled against the cancel flag, which survives 30s+ of silence
during a 100k-token prompt refill. Any replacement client must preserve both,
including WouldBlock-retry mid-chunk in the chunked decoder.

**Prompt-cache discipline** (`agent.rs`). The system prompt is built once per
process and never mutated; conversation history is append-only on the wire.
llama-server reuses the KV prefix, so multi-turn latency stays near zero
(watch `cache N` in the status bar). Anything that rewrites earlier messages
(e.g. the context guard) trades one cache refill for survival — do it rarely
and deliberately.

**The session file is a faithful log** (`session.rs`). Context clipping
applies to the in-memory copy only; resume must always reconstruct the
conversation as the model saw it. (Length-runaway responses are the one
deliberate exception: they are discarded before ever being pushed, so
neither the wire nor the file carries them — resending a runaway reliably
re-triggers it.)

**Tool results are model-visible; `detail` is not** (`tools.rs`). The diff
shown in the TUI rides on `ToolOutput::detail` and never reaches the model —
don't merge those channels; context is expensive at 131k.

**The system prompt stays under ~1k tokens** with tool schemas. New
instructions must earn their tokens; project-specific guidance belongs in
AGENTS.md, model-loadable knowledge in skills.

## Agent-loop mechanics (`agent.rs::run_loop`)

Per iteration: build wire messages (system + history, reasoning stripped) →
`stream_chat` (one transport retry) → telemetry event → execute tool calls →
inject queued steering input → repeat. Exit when a response has no tool calls.

The loop talks to the model through the `ChatProvider` trait (`provider.rs`;
`Http` is the production impl). That seam is what makes the loop testable:
`agent.rs::tests` drives every layer below with scripted completions — no
server, no GPU, milliseconds per test. Tune a robustness layer by writing
the unit test first; the bench run is then confirmation, not exploration.

Robustness layers, all tuned on SWE-bench trajectories (see bench/):

- **Length nudges**: a `finish_reason == "length"` response with no tool call
  is a runaway (small models loop hard at temp 0 — one turn once produced 6MB
  of reasoning). The truncated message is *discarded* (never pushed to the
  wire or the session), a corrective user message is appended, and the turn
  retries; up to 5 times, then `StopReason::Length`.
- **Repeat detection** (not a blocker): tool calls are FNV-fingerprinted
  (name+args); identical reruns with identical output get an escalating
  note appended to the result. The seen-set clears on any successful
  `write`/`edit`, so "edit → rerun test" cycles are never punished.
  **History**: v2 *blocked* the third identical call — the held-out A/B
  (heldout-v1 5/30 resolved vs heldout-v2 3/30, patches 18 vs 13) showed
  blocking makes temp-0 loops stickier: a refused call leaves the context
  frozen, which is a fixed point, and one instance the old scaffold
  resolved (django-13933) spent 33/40 turns looping on the refusal.
  Executing the redundant call keeps the context evolving. If loops return,
  the next candidate is a temperature bump on the retry turn (resample out
  of the attractor), not harder blocking.
- **Context guard**: past 85% of the context window (true size probed from
  `/props` on a background thread), old tool outputs are clipped in memory,
  keeping the last 8 messages intact.
- **Cancellation**: `Agent::cancel` (AtomicBool) is checked between turns, in
  the SSE read loop, and in `bash` polling; every issued tool call still gets
  a tool-result message so the next request stays well-formed.

## Adding a tool

1. Schema + description in `tools.rs::specs()` — terse, the model pays for it.
2. Implementation `fn my_tool(args: &Value, ...) -> Result<ToolOutput>`, match
   arm in `execute()`. Errors go back to the model as text (`ToolOutput::err`)
   so it can self-correct; only truly exceptional failures should abort.
3. One-line display summary in `m-tui/src/main.rs::summarize_args`.
4. Unit test next to the others in `tools.rs`.

Ask first whether a skill, an AGENTS.md note, or bash can do the job — the
core stays at read/write/edit/bash on purpose (pi's lesson: that's enough).

## The TUI (`m-tui/src/tui/`)

Rendering is cell-based: each transcript entry is a `Cell` with a version
counter and a per-width line cache, so a streaming turn re-renders only the
tail cell. Draws happen only when dirty, coalescing all pending agent events
into one frame, capped at ~60fps. The agent thread communicates via two
channels (`AgentCmd` in, `UiMsg` out); never touch agent state from the render
thread except through the shared atomics (`cancel`, `ctx_limit`) and the
steering queue.

To add a slash command: built-ins live in `SLASH_COMMANDS` + `run_slash()`;
anything user-facing and optional should instead be a markdown template in
`~/.config/m/commands/` — zero code.

## m-bench

Per instance: pull `swebench/sweb.eval.x86_64.<id with __ → _1776_>` → start
container with `--network host` (agent reaches the host llama-server on
localhost) → `docker cp` the static musl `m` in → run `m -p --json
--max-turns 40 --max-tokens 4096 --temp 0` with the issue prompt, trajectory
written to a file *inside* the container and copied out (host-side capture
clips at 48KB — that bug cost a day) → if the diff is empty after a clean
exit, one resumed 15-turn "you changed nothing" pass → `git diff` becomes the
prediction.

Scoring is always the official harness (`bench/.venv`):

```bash
bench/.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path bench/runs/<name>/predictions.jsonl \
  --run_id <name> --max_workers 4
./target/release/m-bench report --run bench/runs/<name> --eval m-gemma4-12b-mtp.<name>.json
```

**Anti-overfitting protocol.** `pick -n 30` (offset 0) is the *dev* slice —
mine its trajectories for generic failure modes. `pick -n 30 --offset 5` is
*held-out* — scaffold changes are judged only on it, and only behavioral fixes
(loop handling, budgets, prompt discipline) are allowed; nothing
instance- or repo-specific. One instance ≈ 3.3%, so treat <±7% as noise.
Slice difficulty varies a lot: the same binary scored 36.7% on dev and
16.7% on held-out. Always A/B on the *same* slice before crediting or
blaming a scaffold change (build the old binary from git in a worktree and
pass it to `m-bench run --bin`). Held-out slices wear out as you make
decisions against them — retire to a fresh offset after a couple of uses
(offset 5 has now judged two decisions; use a fresh offset next).

Triage — the table that motivated every scaffold change so far — is a
subcommand:

```bash
./target/release/m-bench triage --run bench/runs/<name>   # or tb/runs/<ts>
```

It reports per instance: turns, repeated identical tool calls, length
nudges, edit errors, tool errors, and stop reason, worst offenders first.

## Terminal-Bench

Terminal-Bench measures the thesis directly (arbitrary terminal work, not
just SWE patches) and is wired up in [tb/](tb/README.md). The adapter runs
`m -p --json` teed to `/logs/trajectory.jsonl` in each task container, so
`m-bench triage` reads tb runs and SWE-bench runs identically, and
`tb/pick.py` gives the same deterministic dev/held-out discipline (43/43
tasks, alternating over the sorted id list). The same anti-overfitting
rules apply: mine dev, judge on held-out, behavioral fixes only.

## Working on m with m

The self-hosting loop works today: run `m` in this repo (AGENTS.md context
included), let it edit its own source, then `cargo build --release` and
restart — sessions are JSONL on disk, so `m -r` resumes the conversation in
the new binary. Keep `cargo test` + a tmux TUI smoke as the gate before
adopting a self-produced change.

## Release checklist

1. `cargo test` green, `cargo clippy --all-targets` zero warnings.
2. `M_PERF=1 m` within budget (<20ms / <30MB; expect ~0.5ms / ~7MB).
3. tmux TUI smoke: prompt → streamed answer → Esc cancel → `/resume`.
4. Headless smoke: `m -p "write and run a hello-world script"` in a temp dir.
5. Bench sanity when the loop changed: 2-instance smoke + official scoring.
