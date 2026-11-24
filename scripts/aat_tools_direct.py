"""Cell A tool builder: in-process Python callables from
mcp_servers.direct_adapter, wrapped as OpenAI Agents SDK function_tools.

Cell B uses MCPServerStdio for the same underlying functions; see
aat_tools_mcp.py. Keeping the name/description set identical between
this module and the MCP stdio path is the fairness contract for
Experiment 1 (issue #25) — (Cell B latency) - (Cell A latency) is
pure MCP transport overhead only if the tool surface matches.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
from typing import Any, Callable, List

from agents import function_tool
from agents.tool_context import ToolContext
from mcp_servers import direct_adapter


def _normalize_nonfinite(value: Any) -> Any:
    """Replace non-finite floats in nested tool payloads with JSON null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _normalize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_nonfinite(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_nonfinite(item) for item in value)
    return value


def _serialize_tool_result(value: Any) -> Any:
    """Render a tool's return value as valid textual JSON.

    Two reasons this is not left to the SDK:

    1. ``openai-agents==0.14.5`` converts a tool return in
       ``agents/items.py::_convert_tool_output`` with
       ``if all(maybe_converted_output_list)`` over a per-element list
       comprehension. ``all([])`` is vacuously ``True``, so an **empty** list
       takes the structured-content branch and is emitted as a bare ``[]``.
       That is not a valid ``role: tool`` content value for OpenAI-compatible
       chat completions, and at least one hosted provider rejects the whole
       request with ``content must be a string``. A tool that legitimately finds
       nothing would therefore kill the trajectory.
    2. Every other structured return falls through to ``str(output)``, i.e.
       Python ``repr`` (``{'a': Timestamp('...')}``) rather than JSON, which the
       model then has to read as pseudo-JSON.

    Encoding here fixes both without patching vendor internals. ``str`` returns
    pass through untouched: they are already textual, and re-encoding one would
    add quotes and escapes that change what the model sees.
    """
    if isinstance(value, str):
        return value
    # Reuse the runner's encoder so Timestamp/date/numpy scalars serialize the
    # same way they do in trajectory artifacts. Imported lazily to keep this
    # tool builder independent of runner import order.
    from scripts.aat_runner import _json_default

    return json.dumps(
        _normalize_nonfinite(value), default=_json_default, allow_nan=False
    )


def _json_result(
    fn: Callable[..., Any], *, preserve_raw_arguments: bool = False
) -> Callable[..., Any]:
    """Wrap a registry callable so its result is textual JSON.

    ``functools.wraps`` keeps ``__wrapped__`` set, so ``inspect.signature`` —
    and therefore the schema ``function_tool`` derives — still resolves to the
    original callable. Arguments are forwarded untouched: this changes how a
    result is rendered, never what the model asked for.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _serialize_tool_result(fn(*args, **kwargs))

    if preserve_raw_arguments:

        @functools.wraps(fn)
        def contextual_wrapper(
            context: ToolContext[Any], *args: Any, **kwargs: Any
        ) -> Any:
            raw_arguments = json.loads(context.tool_arguments)
            bound = inspect.signature(fn).bind(*args, **kwargs)
            for argument_name in ("window_size", "z_threshold"):
                if argument_name in raw_arguments:
                    bound.arguments[argument_name] = raw_arguments[argument_name]
            return _serialize_tool_result(fn(*bound.args, **bound.kwargs))

        original_signature = inspect.signature(fn)
        context_parameter = inspect.Parameter(
            "context",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=ToolContext[Any],
        )
        contextual_wrapper.__signature__ = original_signature.replace(
            parameters=(context_parameter, *original_signature.parameters.values())
        )
        contextual_wrapper.__annotations__ = {
            "context": ToolContext[Any],
            **fn.__annotations__,
        }
        return contextual_wrapper

    return wrapper


def _agent_visible_name(registry_name: str) -> str:
    """Return the tool name the model should see.

    The direct adapter keeps domain-qualified names internally
    (``iot.list_assets``), but MCP stdio exposes bare names
    (``list_assets``). Cell A must match Cell B's model-visible names for
    the transport-overhead comparison to stay fair.
    """
    return registry_name.rsplit(".", 1)[-1]


def build_direct_tools() -> List[Any]:
    """Return a list of function_tool objects, one per entry in the
    direct_adapter registry.
    """
    wrapped: List[Any] = []
    seen_names: dict[str, str] = {}
    for spec in direct_adapter.get_tools():
        callable_fn: Callable[..., Any] = _json_result(
            spec.fn, preserve_raw_arguments=spec.name == "tsfm.detect_anomalies"
        )
        tool_name = _agent_visible_name(spec.name)
        if tool_name in seen_names:
            raise ValueError(
                "Duplicate Agent-visible AaT tool name after stripping domain "
                f"prefix: {tool_name!r} from {seen_names[tool_name]!r} and "
                f"{spec.name!r}"
            )
        seen_names[tool_name] = spec.name
        wrapped.append(
            function_tool(
                callable_fn,
                name_override=tool_name,
                description_override=spec.doc or f"Direct call to {tool_name}",
                strict_mode=False,
            )
        )
    return wrapped
