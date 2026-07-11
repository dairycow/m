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

# the full core set
.venv/bin/tb run --agent-import-path m_agent:MAgent \
    --dataset-path tb-core/tasks --output-path runs
```

The adapter auto-detects the host IP (containers reach the host because
llama-server binds 0.0.0.0); override with `M_SERVER_URL=http://<ip>:8080`.
Agent output is teed to the mounted agent-logs dir of each run.
