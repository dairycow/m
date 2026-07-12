#!/usr/bin/env python3
"""Deterministic dev/held-out split of the tb-core task set.

Mirrors `m-bench pick`: all task ids sorted, even indices -> dev, odd ->
held-out. Reproducible by anyone from the dataset checkout. Mine dev
trajectories for generic failure modes; judge scaffold changes only on
held-out, and only behavioral fixes — nothing task-specific (see the
anti-overfitting protocol in DEVELOPMENT.md).

    .venv/bin/tb run --agent-import-path m_agent:MAgent \\
        --dataset-path tb-core/tasks --output-path runs \\
        $(./pick.py dev | sed 's/^/--task-id /')
"""

import sys
from pathlib import Path


def task_ids(tasks_dir: Path) -> list[str]:
    return sorted(p.name for p in tasks_dir.iterdir() if (p / "task.yaml").exists())


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in ("dev", "heldout"):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    tasks = task_ids(Path(__file__).parent / "tb-core/tasks")
    if not tasks:
        print("pick.py: no tasks found (clone tb-core first, see README.md)", file=sys.stderr)
        return 1
    keep = 0 if which == "dev" else 1
    for i, task in enumerate(tasks):
        if i % 2 == keep:
            print(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
