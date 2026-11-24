"""Tests for the Cell B MCP tool builder."""

from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest


def test_mcp_client_timeout_default(monkeypatch):
    from scripts.aat_tools_mcp import _client_timeout_seconds

    monkeypatch.delenv("AAT_MCP_CLIENT_TIMEOUT_SECONDS", raising=False)

    assert _client_timeout_seconds() == 30


def test_mcp_client_timeout_env(monkeypatch):
    from scripts.aat_tools_mcp import _client_timeout_seconds

    monkeypatch.setenv("AAT_MCP_CLIENT_TIMEOUT_SECONDS", "45.5")

    assert _client_timeout_seconds() == 45.5


def test_mcp_client_timeout_rejects_invalid_env(monkeypatch):
    from scripts.aat_tools_mcp import _client_timeout_seconds

    monkeypatch.setenv("AAT_MCP_CLIENT_TIMEOUT_SECONDS", "nope")

    with pytest.raises(ValueError, match="must be numeric"):
        _client_timeout_seconds()


def test_server_launch_mode_rejects_invalid_env(monkeypatch):
    from scripts.aat_tools_mcp import _server_launch_mode

    monkeypatch.setenv("AAT_MCP_SERVER_LAUNCH_MODE", "bogus")

    with pytest.raises(ValueError, match="must be either"):
        _server_launch_mode()


def test_server_params_python_uses_bootstrap(monkeypatch):
    from scripts.aat_tools_mcp import _server_params

    repo_root = pathlib.Path.cwd()
    server_path = repo_root / "mcp_servers/iot_server/server.py"
    monkeypatch.setenv("AAT_MCP_SERVER_LAUNCH_MODE", "python")
    monkeypatch.setenv("AAT_MCP_SERVER_PYTHON", sys.executable)

    params = _server_params(repo_root, server_path)

    assert params["command"] == sys.executable
    assert params["args"][:2] == [
        "-u",
        str(repo_root / "scripts/aat_mcp_server_bootstrap.py"),
    ]
    assert params["args"][-1] == str(server_path)
    assert params["cwd"] == str(repo_root)
    assert params["env"] == {
        "PYTHONUNBUFFERED": "1",
        "AAT_MCP_REPO_ROOT": str(repo_root),
    }


def test_server_params_python_requires_server_python(monkeypatch):
    from scripts.aat_tools_mcp import _server_params

    repo_root = pathlib.Path.cwd()
    server_path = repo_root / "mcp_servers/iot_server/server.py"
    monkeypatch.setenv("AAT_MCP_SERVER_LAUNCH_MODE", "python")
    monkeypatch.delenv("AAT_MCP_SERVER_PYTHON", raising=False)

    with pytest.raises(ValueError, match="requires AAT_MCP_SERVER_PYTHON"):
        _server_params(repo_root, server_path)


def test_server_params_uv_ignores_server_python(monkeypatch):
    from scripts.aat_tools_mcp import _server_params

    repo_root = pathlib.Path.cwd()
    server_path = repo_root / "mcp_servers/iot_server/server.py"
    monkeypatch.setenv("AAT_MCP_SERVER_LAUNCH_MODE", "uv")
    monkeypatch.setenv("AAT_MCP_SERVER_PYTHON", "/usr/bin/python3")
    monkeypatch.setenv("AAT_UV_PYTHON", sys.executable)

    params = _server_params(repo_root, server_path)

    assert params["command"] == "uv"
    assert params["args"][:4] == ["run", "--python", sys.executable, "--with"]
    assert "python" in params["args"]
    assert "-u" in params["args"]
    assert str(repo_root / "scripts/aat_mcp_server_bootstrap.py") in params["args"]
    assert params["args"][-1] == str(server_path)
    assert params["env"] == {
        "PYTHONUNBUFFERED": "1",
        "AAT_MCP_REPO_ROOT": str(repo_root),
    }


# ---------------------------------------------------------------------------
# Cell B/C sibling of the Cell A empty-result defect.
#
# MCPUtil.invoke_mcp_tool uses result.structuredContent only when
# server.use_structured_content is true. It defaults to False and this repo
# never sets it, so every MCP call builds a list from result.content. A tool
# that legitimately returns nothing yields an empty list, which is not length 1,
# so the SDK passes [] to _convert_tool_output -> all([]) is vacuously true ->
# a bare [] is emitted as role:tool content. Measured against this repo's fmsr
# server: a no-match search returns len(result.content) == 0 with
# structuredContent == {"result": []}.
# ---------------------------------------------------------------------------


def _call_tool_result(content, structured=None, is_error=False):
    from mcp.types import CallToolResult

    return CallToolResult(
        content=content, structuredContent=structured, isError=is_error
    )


def _patched_server(monkeypatch, result):
    """An _EmptyContentSafeMCPServerStdio whose super().call_tool returns result."""
    from agents.mcp import MCPServerStdio

    from scripts.aat_tools_mcp import _EmptyContentSafeMCPServerStdio

    srv = _EmptyContentSafeMCPServerStdio.__new__(_EmptyContentSafeMCPServerStdio)

    async def fake_call_tool(self, tool_name, arguments, meta=None):
        return result

    monkeypatch.setattr(MCPServerStdio, "call_tool", fake_call_tool, raising=True)
    return srv


def test_mcp_empty_content_becomes_textual_json(monkeypatch):
    """The exact shape this repo's fmsr server returns for a no-match query."""
    result = _call_tool_result([], structured={"result": []})
    srv = _patched_server(monkeypatch, result)

    out = asyncio.run(srv.call_tool("search_failure_modes", {"query": "Hydrogen"}))

    assert len(out.content) == 1
    assert out.content[0].type == "text"
    assert out.content[0].text == "[]"


def test_mcp_empty_content_survives_sdk_conversion_as_a_string(monkeypatch):
    """End-to-end through the vendor conversion the defect actually lives in."""
    from openai.types.responses import ResponseFunctionToolCall

    from agents.items import ItemHelpers
    from agents.models.chatcmpl_converter import Converter

    result = _call_tool_result([], structured={"result": []})
    srv = _patched_server(monkeypatch, result)
    out = asyncio.run(srv.call_tool("search_failure_modes", {"query": "Hydrogen"}))

    # Mirror MCPUtil.invoke_mcp_tool's non-structured branch exactly.
    tool_output_list = [
        {"type": "text", "text": item.text}
        for item in out.content
        if item.type == "text"
    ]
    tool_output = (
        tool_output_list[0] if len(tool_output_list) == 1 else tool_output_list
    )

    call = ResponseFunctionToolCall(
        call_id="call_sfm",
        name="search_failure_modes",
        arguments="{}",
        type="function_call",
    )
    item = ItemHelpers.tool_call_output_item(call, tool_output)
    messages = Converter.items_to_messages([call.model_dump(), item])
    content = [m for m in messages if m.get("role") == "tool"][0]["content"]

    assert content != [], "bare empty list reached role:tool content"
    assert content, "role:tool content must not be empty"


def test_mcp_nonempty_content_is_left_alone(monkeypatch):
    """Non-empty results keep the SDK's own structured handling, untouched."""
    from mcp.types import TextContent

    items = [
        TextContent(type="text", text='{"failure_mode_id": "FM-002"}'),
        TextContent(type="text", text='{"failure_mode_id": "FM-003"}'),
    ]
    result = _call_tool_result(items, structured={"result": [{"a": 1}, {"b": 2}]})
    srv = _patched_server(monkeypatch, result)

    out = asyncio.run(srv.call_tool("search_failure_modes", {"query": "thermal"}))

    # The non-empty branch returns the server's own result object untouched.
    assert out is result
    assert len(out.content) == 2
    assert [c.text for c in out.content] == [i.text for i in items]


def test_mcp_single_and_image_content_semantics_preserved(monkeypatch):
    """A scalar text item and an image item must not be rewritten or flattened."""
    from mcp.types import ImageContent, TextContent

    scalar_result = _call_tool_result([TextContent(type="text", text="57")])
    srv = _patched_server(monkeypatch, scalar_result)
    out = asyncio.run(srv.call_tool("get_rul", {"transformer_id": "T-016"}))
    assert out is scalar_result
    assert len(out.content) == 1
    assert out.content[0].type == "text"
    assert out.content[0].text == "57"

    image_result = _call_tool_result(
        [ImageContent(type="image", data="aGk=", mimeType="image/png")]
    )
    srv = _patched_server(monkeypatch, image_result)
    out = asyncio.run(srv.call_tool("plot", {}))
    # Image content must stay image content — never flattened to text.
    assert out is image_result
    assert out.content[0].type == "image"
    assert out.content[0].mimeType == "image/png"


def test_mcp_error_result_with_empty_content_is_not_rewritten(monkeypatch):
    """An error result keeps whatever the server said, for the failure pipeline."""
    result = _call_tool_result([], structured=None, is_error=True)
    srv = _patched_server(monkeypatch, result)

    out = asyncio.run(srv.call_tool("boom", {}))

    assert out.content == []
    assert out.isError is True


def test_mcp_empty_result_text_unwraps_fastmcp_result_envelope():
    """FastMCP wraps non-dict returns as {"result": value} — recover it exactly."""
    from scripts.aat_tools_mcp import _EmptyContentSafeMCPServerStdio as S

    assert S._empty_result_text({"result": []}) == "[]"
    assert S._empty_result_text({"result": {}}) == "{}"
    assert S._empty_result_text({"result": None}) == "null"
    assert S._empty_result_text({"other": 1}) == '{"other": 1}'
    assert S._empty_result_text(None) == "[]"
