"""Tests for the Cell A tool builder.

The builder wraps every entry in mcp_servers.direct_adapter as an
agents SDK function_tool, preserving the MCP-visible name and description.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from types import SimpleNamespace

import pandas as pd
import pytest


def test_build_direct_tools_returns_one_per_registry_entry() -> None:
    from mcp_servers import direct_adapter
    from scripts.aat_tools_direct import build_direct_tools

    tools = build_direct_tools()
    registry = direct_adapter.get_tools()

    assert len(tools) == len(registry)


def test_build_direct_tools_uses_mcp_visible_names() -> None:
    from mcp_servers import direct_adapter
    from scripts.aat_tools_direct import build_direct_tools

    tools = build_direct_tools()
    registry_names = {
        spec.name.rsplit(".", 1)[-1] for spec in direct_adapter.get_tools()
    }
    wrapped_names = {getattr(tool, "name", None) for tool in tools}

    assert wrapped_names == registry_names, (
        f"Missing or renamed tools.\n"
        f"Only in registry: {registry_names - wrapped_names}\n"
        f"Only in wrapped:  {wrapped_names - registry_names}"
    )


def test_build_direct_tools_rejects_duplicate_visible_names(monkeypatch) -> None:
    from scripts import aat_tools_direct

    def first_tool() -> None:
        pass

    def second_tool() -> None:
        pass

    monkeypatch.setattr(
        aat_tools_direct.direct_adapter,
        "get_tools",
        lambda: [
            SimpleNamespace(name="iot.duplicate", fn=first_tool, doc="first"),
            SimpleNamespace(name="fmsr.duplicate", fn=second_tool, doc="second"),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate Agent-visible AaT tool name"):
        aat_tools_direct.build_direct_tools()


@pytest.mark.slow
def test_direct_and_mcp_tool_schemas_match(monkeypatch):
    """Fairness-contract enforcer for Experiment 1.

    The Cell A tool surface (in-process callables) and the Cell B tool
    surface (MCP stdio) must expose the same tool names and parameter
    requiredness. If this test
    fails, (Cell B - Cell A) is no longer a clean measurement of MCP
    transport overhead — there's an additional tool-surface delta.

    Marked @pytest.mark.slow because it launches 4 MCP stdio subprocesses.
    """
    from scripts.aat_tools_mcp import build_mcp_servers

    from scripts.aat_tools_direct import build_direct_tools

    monkeypatch.setenv("AAT_MCP_SERVER_LAUNCH_MODE", "uv")
    monkeypatch.delenv("AAT_MCP_SERVER_PYTHON", raising=False)

    def schema_contract(schema: dict) -> dict[str, bool]:
        properties = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])
        return {name: name in required for name in properties}

    direct_tools = build_direct_tools()
    direct_names = {getattr(tool, "name", None) for tool in direct_tools}
    direct_schemas = {
        getattr(tool, "name", None): schema_contract(tool.params_json_schema)
        for tool in direct_tools
    }

    async def collect_mcp_schemas() -> tuple[set[str], dict[str, dict[str, bool]]]:
        servers = await build_mcp_servers(pathlib.Path.cwd())
        try:
            names: set[str] = set()
            schemas: dict[str, dict[str, bool]] = {}
            for srv in servers:
                tools_result = await srv.list_tools()
                for t in tools_result:
                    names.add(t.name)
                    schemas[t.name] = schema_contract(t.inputSchema)
            return names, schemas
        finally:
            for srv in servers:
                try:
                    await srv.cleanup()
                except (asyncio.CancelledError, Exception):
                    # asyncio.CancelledError is a BaseException in Py3.8+, not
                    # an Exception. The openai-agents MCP stdio teardown can
                    # leak a benign CancelledError from the anyio subprocess
                    # wait() — absorb it here so the parity assertion still runs.
                    pass

    mcp_names, mcp_schemas = asyncio.run(collect_mcp_schemas())

    assert direct_names == mcp_names, (
        f"Cell A and Cell B tool surfaces diverge.\n"
        f"Only in direct_adapter: {direct_names - mcp_names}\n"
        f"Only in MCP stdio:      {mcp_names - direct_names}\n"
        f"If tanisha's MCP server hardening renamed or added tools, "
        f"resync mcp_servers/direct_adapter.py."
    )
    assert direct_schemas == mcp_schemas


# ---------------------------------------------------------------------------
# Tool results reach the model as valid textual JSON.
#
# openai-agents 0.14.5 converts a tool return in items.py::_convert_tool_output
# with `if all(maybe_converted_output_list)` over a per-element comprehension.
# all([]) is vacuously True, so an EMPTY list took the structured-content
# branch and was emitted as a bare [] — not a valid role:tool content value,
# and rejected outright by at least one hosted provider. Every other structured
# return fell through to str(output), i.e. Python repr rather than JSON.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ([], "[]"),
        ({}, "{}"),
        ([{"failure_mode_id": "FM-001"}], '[{"failure_mode_id": "FM-001"}]'),
        ({"rul_days": 57}, '{"rul_days": 57}'),
    ],
)
def test_serialize_tool_result_emits_textual_json(value, expected) -> None:
    from scripts.aat_tools_direct import _serialize_tool_result

    out = _serialize_tool_result(value)

    assert isinstance(out, str)
    assert out == expected


def test_serialize_tool_result_passes_strings_through_unchanged() -> None:
    """A string is already textual; re-encoding would add quotes and escapes."""
    from scripts.aat_tools_direct import _serialize_tool_result

    assert _serialize_tool_result("already text") == "already text"


def test_serialize_tool_result_preserves_information() -> None:
    """Encoding must not lose or alter the payload — round-trip it."""
    import json

    from scripts.aat_tools_direct import _serialize_tool_result

    payload = {
        "transformer_id": "T-012",
        "dissolved_h2_ppm": 41.98,
        "candidates": [{"failure_mode_id": "FM-002", "severity": "medium"}],
        "empty": [],
    }

    assert json.loads(_serialize_tool_result(payload)) == payload


def test_serialize_tool_result_normalizes_nested_nonfinite_missing_unit_values() -> (
    None
):
    """Missing-unit payloads must remain valid JSON when values are non-finite."""
    import json
    import math

    from scripts.aat_tools_direct import _serialize_tool_result

    payload = {
        "sensor_id": "oil_temp_c",
        "unit": None,
        "reading": {
            "value": math.nan,
            "limits": [math.inf, -math.inf, 42.0],
        },
        "history": [{"value": math.nan}, {"value": 7.5}],
    }

    encoded = _serialize_tool_result(payload)
    assert json.loads(encoded) == {
        "sensor_id": "oil_temp_c",
        "unit": None,
        "reading": {"value": None, "limits": [None, None, 42.0]},
        "history": [{"value": None}, {"value": 7.5}],
    }
    assert all(token not in encoded for token in ("NaN", "Infinity", "-Infinity"))


def test_build_direct_tools_wrapper_keeps_schema_and_serializes(monkeypatch) -> None:
    """Wrapping must not disturb the schema function_tool derives."""
    from scripts import aat_tools_direct
    from scripts.aat_tools_direct import build_direct_tools

    def search_failure_modes(query: str) -> list:
        """Search failure modes."""
        return []

    spec = SimpleNamespace(
        name="fmsr.search_failure_modes",
        fn=search_failure_modes,
        doc="Search failure modes.",
    )
    monkeypatch.setattr(aat_tools_direct.direct_adapter, "get_tools", lambda: [spec])

    (tool,) = build_direct_tools()

    assert tool.name == "search_failure_modes"
    properties = tool.params_json_schema["properties"]
    assert set(properties) == {"query"}
    assert properties["query"]["type"] == "string"


def test_tsfm_direct_sdk_preserves_raw_anomaly_types_against_mcp(monkeypatch) -> None:
    """The real SDK tool context must not hide coercible raw Cell A arguments."""
    from agents.tool_context import ToolContext
    from mcp.server.fastmcp.exceptions import ToolError

    import mcp_servers.tsfm_server.server as tsfm_server
    from scripts.aat_tools_direct import build_direct_tools

    readings = pd.DataFrame(
        {
            "transformer_id": ["T-synthetic"] * 3,
            "sensor_id": ["sensor"] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00",
                    "2024-01-01T01:00:00",
                    "2024-01-01T02:00:00",
                ]
            ),
            "value": [1.0, 1.0, 10.0],
        }
    )
    monkeypatch.setattr(tsfm_server, "_get_readings", lambda: readings)
    sdk_tool = next(
        tool for tool in build_direct_tools() if tool.name == "detect_anomalies"
    )

    def invoke_sdk(arguments: dict) -> dict:
        raw_arguments = json.dumps(arguments)
        context = ToolContext(
            context=None,
            tool_name=sdk_tool.name,
            tool_call_id="call-tsfm-validation",
            tool_arguments=raw_arguments,
        )
        result = asyncio.run(sdk_tool.on_invoke_tool(context, raw_arguments))
        return json.loads(result)

    base_arguments = {
        "transformer_id": "T-synthetic",
        "sensor_id": "sensor",
    }
    for parameter, invalid_value in (
        ("window_size", True),
        ("window_size", "2"),
        ("z_threshold", True),
        ("z_threshold", "0.5"),
    ):
        arguments = {**base_arguments, parameter: invalid_value}
        direct_result = invoke_sdk(arguments)
        assert "error" in direct_result
        with pytest.raises(ToolError):
            asyncio.run(tsfm_server.mcp.call_tool("detect_anomalies", arguments))

    valid_arguments = {
        **base_arguments,
        "window_size": 2,
        "z_threshold": 0.5,
    }
    direct_result = invoke_sdk(valid_arguments)
    mcp_result = asyncio.run(
        tsfm_server.mcp.call_tool("detect_anomalies", valid_arguments)
    )
    assert "error" not in direct_result
    assert (
        json.loads(mcp_result[0].text)["anomaly_count"]
        == direct_result["anomaly_count"]
    )


def test_empty_tool_result_becomes_a_valid_role_tool_message() -> None:
    """End-to-end through the vendor conversion, with a mocked completion.

    This is the exact boundary the bug lives on: the SDK turns a tool return
    into a ``role: tool`` chat message. Nothing here touches the network.
    """
    from openai.types.responses import ResponseFunctionToolCall

    from agents.items import ItemHelpers
    from agents.models.chatcmpl_converter import Converter
    from scripts.aat_tools_direct import _serialize_tool_result

    call = ResponseFunctionToolCall(
        call_id="call_sfm",
        name="search_failure_modes",
        arguments='{"query": "Hydrogen"}',
        type="function_call",
    )
    # A real no-match result from a list-returning tool.
    item = ItemHelpers.tool_call_output_item(call, _serialize_tool_result([]))

    # items_to_messages consumes input-item dicts, not the pydantic call object.
    messages = Converter.items_to_messages([call.model_dump(), item])
    tool_messages = [m for m in messages if m.get("role") == "tool"]

    assert len(tool_messages) == 1
    content = tool_messages[0]["content"]
    assert isinstance(content, str)
    assert content == "[]"


def test_unserialized_empty_list_would_produce_invalid_content() -> None:
    """Pin the upstream defect this wrapper exists to work around.

    If a future openai-agents fixes ``all([])``, this test fails and the
    wrapper can be reconsidered — it is a canary, not a requirement.
    """
    from openai.types.responses import ResponseFunctionToolCall

    from agents.items import ItemHelpers
    from agents.models.chatcmpl_converter import Converter

    call = ResponseFunctionToolCall(
        call_id="call_sfm",
        name="search_failure_modes",
        arguments="{}",
        type="function_call",
    )
    item = ItemHelpers.tool_call_output_item(call, [])  # raw, unserialized

    messages = Converter.items_to_messages([call.model_dump(), item])
    content = [m for m in messages if m.get("role") == "tool"][0]["content"]

    assert content == [], (
        "upstream openai-agents appears to have fixed the vacuous all([]) "
        "branch; revisit _serialize_tool_result"
    )
