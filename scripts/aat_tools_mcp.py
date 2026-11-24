"""Cell B tool builder: MCP stdio connections to the team's hardened
Smart Grid MCP servers.

Returns connected MCPServerStdio objects. The AaT runner passes them
to Agent(mcp_servers=...) and calls .cleanup() on each in a finally
block — dangling stdio subprocesses after a Slurm trial would leak.

Cell A uses in-process callables for the same tool set; see
aat_tools_direct.py. The test_direct_and_mcp_tool_schemas_match test
(Task 5) asserts the two paths expose identical tool surfaces.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, List

from agents.mcp import MCPServerStdio
from mcp.types import CallToolResult, TextContent

SERVER_MODULES: list[tuple[str, str]] = [
    ("iot", "mcp_servers/iot_server/server.py"),
    ("fmsr", "mcp_servers/fmsr_server/server.py"),
    ("tsfm", "mcp_servers/tsfm_server/server.py"),
    ("wo", "mcp_servers/wo_server/server.py"),
]

SERVER_UV_DEPS = [
    "mcp[cli]==1.27.0",
    "pandas",
    "numpy",
]


def _client_timeout_seconds(value: str | None = None) -> float:
    from scripts.gcp_resume_state import normalize_aat_client_timeout_seconds

    raw = (
        os.environ.get("AAT_MCP_CLIENT_TIMEOUT_SECONDS", "30")
        if value is None
        else value
    )
    return normalize_aat_client_timeout_seconds(raw)


def _server_launch_mode() -> str:
    mode = os.environ.get("AAT_MCP_SERVER_LAUNCH_MODE", "python").strip().lower()
    if mode not in {"python", "uv"}:
        raise ValueError(
            f"AAT_MCP_SERVER_LAUNCH_MODE must be either 'python' or 'uv', got {mode!r}"
        )
    return mode


def _server_params(repo_root: Path, abs_path: Path) -> dict[str, object]:
    bootstrap_path = repo_root / "scripts" / "aat_mcp_server_bootstrap.py"
    if not bootstrap_path.exists():
        raise FileNotFoundError(f"AaT MCP server bootstrap missing: {bootstrap_path}")

    launch_mode = _server_launch_mode()
    env = {"PYTHONUNBUFFERED": "1", "AAT_MCP_REPO_ROOT": str(repo_root)}

    if launch_mode == "uv":
        uv_python = os.environ.get("AAT_UV_PYTHON", "").strip()
        python_args = ["--python", uv_python] if uv_python else []
        return {
            "command": "uv",
            "args": [
                "run",
                *python_args,
                *(arg for dep in SERVER_UV_DEPS for arg in ("--with", dep)),
                "python",
                "-u",
                str(bootstrap_path),
                str(abs_path),
            ],
            "cwd": str(repo_root),
            "env": env,
        }

    server_python = os.environ.get("AAT_MCP_SERVER_PYTHON", "").strip()
    if not server_python:
        raise ValueError(
            "AAT_MCP_SERVER_LAUNCH_MODE=python requires AAT_MCP_SERVER_PYTHON. "
            "scripts/run_experiment.sh sets this to .venv-insomnia/bin/python "
            "for Insomnia LAUNCH_VLLM=1 runs; use "
            "AAT_MCP_SERVER_LAUNCH_MODE=uv for local uv-managed launches."
        )
    python_path = Path(server_python)
    if not python_path.exists():
        raise FileNotFoundError(f"AAT_MCP_SERVER_PYTHON not found: {python_path}")
    return {
        "command": str(python_path),
        "args": ["-u", str(bootstrap_path), str(abs_path)],
        "cwd": str(repo_root),
        "env": env,
    }


class _EmptyContentSafeMCPServerStdio(MCPServerStdio):
    """MCPServerStdio that renders a zero-item tool result as textual JSON.

    Why this exists (Cell B/C sibling of the Cell A empty-result defect):

    ``MCPUtil.invoke_mcp_tool`` only uses ``result.structuredContent`` when
    ``server.use_structured_content`` is true. That flag defaults to ``False``
    (``agents/mcp/server.py``) and this repo never sets it, so every MCP call
    here takes the other branch, which builds a list from ``result.content``.
    When a tool legitimately returns nothing that list is empty, and because it
    is not length 1 the SDK hands ``[]`` straight to
    ``ItemHelpers._convert_tool_output``. There ``all([])`` is vacuously true,
    so the empty list is emitted as a bare ``[]`` — not a valid ``role: tool``
    content value, and rejected outright by some OpenAI-compatible providers.
    Measured against this repo's own fmsr server: a ``search_failure_modes``
    query that matches nothing returns ``len(result.content) == 0`` with
    ``structuredContent == {"result": []}``.

    Two things this deliberately does NOT do:

    * It does not set ``use_structured_content=True``. That flag applies to
      every call, and would collapse a multi-item result (or an image) into one
      JSON blob, changing the tool contract and breaking the Cell A/B tool-surface
      parity that Experiment 1's transport-overhead comparison depends on.
    * It does not touch non-empty results at all. Single items, multi-item lists,
      images, and non-text content keep the SDK's own structured handling.

    Only the empty case is rewritten, and it is rewritten to the same textual
    JSON Cell A produces for an empty return. Cell A carries that textual JSON
    directly, while MCP carries the same textual JSON in typed text content; the
    payload text matches without claiming identical SDK content shapes.
    """

    @staticmethod
    def _empty_result_text(structured: Any) -> str:
        """Render what an empty content list actually meant, as JSON text.

        FastMCP wraps a non-dict return value as ``{"result": <value>}``, so when
        that shape is present the tool's real return value is recoverable exactly
        — no guessing. Otherwise fall back to an empty JSON array, which is what
        "no content items" means on this tool surface and what Cell A emits for
        the same situation.
        """
        if isinstance(structured, dict) and "result" in structured:
            return json.dumps(structured["result"])
        if structured:
            return json.dumps(structured)
        return "[]"

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        result = await super().call_tool(tool_name, arguments, meta=meta)
        # An error result already carries its own content; leave it alone so the
        # SDK's failure pipeline still sees what the server actually said.
        if result.content or getattr(result, "isError", False):
            return result
        text = self._empty_result_text(getattr(result, "structuredContent", None))
        return result.model_copy(
            update={"content": [TextContent(type="text", text=text)]}
        )


async def build_mcp_servers(repo_root: Path) -> List[MCPServerStdio]:
    """Return a list of connected MCPServerStdio objects.

    On failure mid-way through, cleans up already-connected servers
    before re-raising, so callers don't have to handle partial state.
    """
    connected: List[MCPServerStdio] = []
    try:
        client_timeout = _client_timeout_seconds()
        print(
            f"AaT MCP client initialize timeout: {client_timeout:g}s",
            file=sys.stderr,
        )
        print(
            f"AaT MCP server launch mode: {_server_launch_mode()}",
            file=sys.stderr,
        )
        for name, rel in SERVER_MODULES:
            abs_path = repo_root / rel
            if not abs_path.exists():
                raise FileNotFoundError(
                    f"MCP server module missing: {abs_path} "
                    f"(expected under the shared team checkout)"
                )
            # Use an explicit uv dependency envelope: the AaT runner itself runs
            # from a temporary uv env, which may not include server data deps.
            # cache_tools_list=True avoids a list_tools round-trip per turn.
            srv = _EmptyContentSafeMCPServerStdio(
                name=name,
                params=_server_params(repo_root, abs_path),
                cache_tools_list=True,
                client_session_timeout_seconds=client_timeout,
            )
            command_line = [srv.params.command, *srv.params.args]
            print(
                f"Connecting MCP server {name}: {shlex.join(command_line)}",
                file=sys.stderr,
            )
            await srv.connect()
            print(f"Connected MCP server {name}", file=sys.stderr)
            connected.append(srv)
        return connected
    except Exception:
        for srv in connected:
            try:
                await srv.cleanup()
            except (asyncio.CancelledError, Exception):
                # asyncio.CancelledError is a BaseException in Py3.8+, not an
                # Exception; widen the catch so partial-failure cleanup can't
                # mask the original connection error with a teardown leak.
                pass
        raise
