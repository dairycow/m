# Terminal-Bench × m

[Terminal-Bench](https://github.com/laude-institute/terminal-bench) measures
exactly what m claims: an agent doing arbitrary computer work through a
terminal. This directory wires m in as an *installed agent*: the static musl
binary from the GitHub release is dropped into each task container and talks
to the llama-server on the host.

## One-time setup

```bash
uv venv --python 3.12 tb/.venv
uv pip install --python tb/.venv/bin/python terminal-bench

# the tasks live on a dataset branch of the terminal-bench repo
git clone --depth 1 -b dataset/terminal-bench-core/v0.1.x \
    https://github.com/laude-institute/terminal-bench tb/tb-core
```

## Run

```bash
cd tb   # llama-server must be up: ../run.sh

# one task as a smoke test
.venv/bin/tb run --agent-import-path m_agent:MAgent \
    --dataset-path tb-core/tasks --task-id hello-world --output-path runs

# the dev slice (43 tasks; the other 43 are held out — see below)
.venv/bin/tb run --agent-import-path m_agent:MAgent \
    --dataset-path tb-core/tasks --output-path runs \
    $(./pick.py dev | sed 's/^/--task-id /')
```

The adapter auto-detects the host IP (containers reach the host because
llama-server binds 0.0.0.0); override with `M_SERVER_URL=http://<ip>:8080`.

The agent runs with `--json`, teed to `/logs/trajectory.jsonl` in each task
container — the same event-stream trajectory the SWE-bench runner produces,
so one tool reads both:

```bash
../target/release/m-bench triage --run runs/<timestamp>
```

## Anti-overfitting protocol

Same rules as SWE-bench (DEVELOPMENT.md): `pick.py dev` is the slice whose
trajectories get mined for generic failure modes; scaffold changes are
judged only on `pick.py heldout` (the disjoint other half), and only
behavioral fixes are allowed — nothing task-specific. The split is
deterministic (task ids sorted, alternating), so anyone can reproduce it.
