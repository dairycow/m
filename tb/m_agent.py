"""Terminal-Bench adapter for m (https://github.com/dairycow/m).

m is installed *inside* the task container (static musl binary from the
GitHub release) and pointed at the llama-server running on the host, whose
IP is auto-detected here and passed in via the environment.

Run from this directory:

    .venv/bin/tb run \
        --agent-import-path m_agent:MAgent \
        --dataset-path tb-core/tasks \
        --task-id hello-world \
        --output-path runs

Override the server with M_SERVER_URL=http://<ip>:<port>.
"""

import os
import shlex
import socket
import subprocess
import urllib.request
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand

M_RELEASE_URL = (
    "https://github.com/dairycow/m/releases/download/v0.1.0/m-x86_64-linux-musl"
)


def _host_ip() -> str:
    """The host IP task containers can reach (llama-server binds 0.0.0.0)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets sent; just picks the route
        return s.getsockname()[0]
    finally:
        s.close()


class MAgent(AbstractInstalledAgent):
    @staticmethod
    def name() -> str:
        return "m"

    def __init__(self, *args, **kwargs):
        # tb passes model_name through; m's model comes from its config.
        kwargs.pop("model_name", None)
        super().__init__(*args, **kwargs)
        self._server_url = os.environ.get(
            "M_SERVER_URL", f"http://{_host_ip()}:8080"
        )

    @property
    def _env(self) -> dict[str, str]:
        return {"M_SERVER_URL": self._server_url}

    @property
    def _install_agent_script_path(self) -> Path:
        return Path(__file__).parent / "m-setup.sh"

    def _binary_path(self) -> Path:
        """Static musl m: $M_BINARY > local build > cached release download.

        The task images ship without curl/wget, so the binary is copied in
        from the host rather than downloaded inside the container.
        """
        if env := os.environ.get("M_BINARY"):
            return Path(env)
        local = (
            Path(__file__).parent.parent
            / "target/x86_64-unknown-linux-musl/release/m"
        )
        if local.exists():
            return local
        cached = Path.home() / ".cache/m-tb/m-x86_64-linux-musl"
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(M_RELEASE_URL, cached)
        return cached

    def perform_task(self, instruction, session, logging_dir=None):
        session.copy_to_container(
            self._binary_path(),
            container_dir="/usr/local/bin",
            container_filename="m",
        )
        return super().perform_task(instruction, session, logging_dir)

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        quoted = shlex.quote(instruction)
        return [
            TerminalCommand(
                command=(
                    f"m -p --max-turns 40 --max-tokens 4096 --temp 0 {quoted} "
                    f"2>&1 | tee {self.CONTAINER_AGENT_LOGS_PATH}/m.log"
                ),
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            ),
        ]
