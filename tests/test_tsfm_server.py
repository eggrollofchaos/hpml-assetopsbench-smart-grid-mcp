"""
Tests for the TSFM MCP server (issue #10).

Covers all four tools:
  - get_rul: known transformer, missing transformer, interpretation thresholds
  - forecast_rul: horizon capping at 365 days, known transformer, missing transformer
  - detect_anomalies: happy path, missing transformer/sensor
  - trend_analysis: happy path, time window, insufficient data window
"""

import asyncio
from io import StringIO
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mcp.server.fastmcp.exceptions import ToolError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_servers.tsfm_server.server import (
    _parse_time_bound,
    detect_anomalies,
    forecast_rul,
    get_rul,
    mcp,
    trend_analysis,
)

# ---------------------------------------------------------------------------
# get_rul
# ---------------------------------------------------------------------------


def test_get_rul_known():
    result = get_rul("T-001")
    assert "error" not in result
    for key in (
        "transformer_id",
        "as_of_date",
        "rul_days",
        "health_index",
        "fdd_category",
        "interpretation",
    ):
        assert key in result, f"Missing field: {key}"
    assert result["transformer_id"] == "T-001"
    assert isinstance(result["rul_days"], int)
    assert isinstance(result["health_index"], float)


def test_get_rul_missing():
    result = get_rul("T-NONEXISTENT")
    assert "error" in result


def test_get_rul_interpretation_healthy():
    result = get_rul("T-001")
    assert "error" not in result
    rul = result["rul_days"]
    interp = result["interpretation"]
    if rul >= 730:
        assert "Healthy" in interp
    elif rul >= 180:
        assert "Aging" in interp
    elif rul >= 30:
        assert "Degraded" in interp
    else:
        assert "Critical" in interp


# ---------------------------------------------------------------------------
# forecast_rul
# ---------------------------------------------------------------------------


def test_forecast_rul_known():
    result = forecast_rul("T-001", horizon_days=30)
    assert "error" not in result
    for key in (
        "transformer_id",
        "current_rul_days",
        "forecast_date",
        "projected_rul_days",
        "projected_health_index",
        "confidence",
        "method",
    ):
        assert key in result, f"Missing field: {key}"
    assert result["projected_rul_days"] <= result["current_rul_days"]


def test_forecast_rul_missing():
    result = forecast_rul("T-NONEXISTENT", horizon_days=30)
    assert "error" in result


def test_forecast_rul_horizon_out_of_range_returns_error():
    # horizon_days > 365 must return an error (not silently clamp)
    result = forecast_rul("T-001", horizon_days=9999)
    assert "error" in result


def test_forecast_rul_zero_floor():
    # Projected RUL must never go below zero
    result = forecast_rul("T-001", horizon_days=365)
    assert result["projected_rul_days"] >= 0


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------


def test_detect_anomalies_happy_path():
    result = detect_anomalies("T-018", "winding_temp_top_c")
    assert "error" not in result
    for key in (
        "transformer_id",
        "sensor_id",
        "total_readings",
        "anomaly_count",
        "anomaly_rate_pct",
        "anomalies",
    ):
        assert key in result, f"Missing field: {key}"
    assert isinstance(result["anomalies"], list)
    assert result["anomaly_count"] >= 0
    # Capped at 50
    assert len(result["anomalies"]) <= 50


def test_detect_anomalies_missing_transformer():
    result = detect_anomalies("T-NONEXISTENT", "winding_temp_top_c")
    assert "error" in result


def test_detect_anomalies_missing_sensor():
    result = detect_anomalies("T-001", "sensor_does_not_exist")
    assert "error" in result


def test_detect_anomalies_high_threshold_returns_fewer_anomalies():
    low_thresh = detect_anomalies("T-018", "winding_temp_top_c", z_threshold=1.0)
    high_thresh = detect_anomalies("T-018", "winding_temp_top_c", z_threshold=10.0)
    assert "error" not in low_thresh
    assert "error" not in high_thresh
    assert low_thresh["anomaly_count"] >= high_thresh["anomaly_count"]


@pytest.mark.parametrize(
    ("window_size", "z_threshold"),
    [
        (0, 3.0),
        (-1, 3.0),
        (True, 3.0),
        (24, 0),
        (24, -1.0),
        (24, float("nan")),
        (24, float("inf")),
        (24, "3.0"),
    ],
)
def test_detect_anomalies_rejects_invalid_parameters(window_size, z_threshold):
    result = detect_anomalies(
        "T-018", "winding_temp_top_c", window_size=window_size, z_threshold=z_threshold
    )
    assert "error" in result


def test_detect_anomalies_accepts_valid_parameter_control():
    result = detect_anomalies(
        "T-018", "winding_temp_top_c", window_size=12, z_threshold=3.0
    )
    assert "error" not in result
    assert result["window_size"] == 12
    assert result["z_threshold"] == 3.0


@pytest.mark.parametrize(
    ("window_size", "z_threshold", "message"),
    [
        (10**100, 3.0, "window_size"),
        (24, 10**400, "z_threshold"),
    ],
)
def test_detect_anomalies_rejects_unrepresentable_parameters(
    window_size, z_threshold, message
):
    result = detect_anomalies(
        "T-018", "winding_temp_top_c", window_size=window_size, z_threshold=z_threshold
    )
    assert message in result["error"]


def test_detect_anomalies_accepts_numpy_numeric_controls():
    result = detect_anomalies(
        "T-018",
        "winding_temp_top_c",
        window_size=np.int64(12),
        z_threshold=np.float64(3.0),
    )
    assert "error" not in result
    assert result["window_size"] == 12
    assert result["z_threshold"] == 3.0


def test_detect_anomalies_accepts_window_larger_than_series():
    full_series = detect_anomalies(
        "T-018", "winding_temp_top_c", window_size=24, z_threshold=3.0
    )
    result = detect_anomalies(
        "T-018",
        "winding_temp_top_c",
        window_size=full_series["total_readings"] + 1,
        z_threshold=3.0,
    )
    assert "error" not in result
    assert result["window_size"] == full_series["total_readings"] + 1


def test_detect_anomalies_mcp_schema_is_stable_and_rejects_coercion():
    async def invoke(arguments):
        return await mcp.call_tool("detect_anomalies", arguments)

    tools = asyncio.run(mcp.list_tools())
    schema = next(tool.inputSchema for tool in tools if tool.name == "detect_anomalies")
    assert schema["properties"]["window_size"]["type"] == "integer"
    assert schema["properties"]["z_threshold"]["type"] == "number"
    assert schema["properties"]["window_size"]["default"] == 24
    assert schema["properties"]["z_threshold"]["default"] == 3.0
    assert "window_size" not in schema.get("required", [])
    assert "z_threshold" not in schema.get("required", [])
    trend_schema = next(
        tool.inputSchema for tool in tools if tool.name == "trend_analysis"
    )
    for argument in ("start_time", "end_time"):
        assert trend_schema["properties"][argument]["anyOf"] == [
            {"type": "string"},
            {"type": "null"},
        ]
        assert trend_schema["properties"][argument]["default"] is None
        assert argument not in trend_schema.get("required", [])

    base = {"transformer_id": "T-018", "sensor_id": "winding_temp_top_c"}
    for key, bad_value in (
        ("window_size", True),
        ("window_size", "3"),
        ("z_threshold", True),
        ("z_threshold", "3.0"),
    ):
        with pytest.raises(ToolError):
            asyncio.run(invoke({**base, key: bad_value}))

    for args in (
        {**base, "window_size": 12, "z_threshold": 3.0},
        {**base, "window_size": 12, "z_threshold": 3},
    ):
        result = asyncio.run(invoke(args))
        assert result and not getattr(result[0], "isError", False)

    for args in (
        {**base, "window_size": sys.maxsize + 1, "z_threshold": 3.0},
        {**base, "window_size": 12, "z_threshold": 10**400},
    ):
        try:
            result = asyncio.run(invoke(args))
        except ToolError:
            continue
        assert result and (
            getattr(result[0], "isError", False) or '"error"' in result[0].text
        )

    for argument in ("start_time", "end_time"):
        try:
            result = asyncio.run(
                mcp.call_tool(
                    "trend_analysis",
                    {
                        "transformer_id": "T-018",
                        "sensor_id": "winding_temp_top_c",
                        argument: 1704153600000000000,
                    },
                )
            )
        except ToolError:
            continue
        assert result and (
            getattr(result[0], "isError", False) or '"error"' in result[0].text
        )
        textual_result = asyncio.run(
            mcp.call_tool(
                "trend_analysis",
                {
                    "transformer_id": "T-018",
                    "sensor_id": "winding_temp_top_c",
                    argument: "01/02/2024",
                },
            )
        )
        assert '"error"' in textual_result[0].text
        assert "valid ISO-8601" in textual_result[0].text


@pytest.mark.parametrize("bound", [1704153600000000000, True, "2024", "01/02/2024"])
@pytest.mark.parametrize("argument", ["start_time", "end_time"])
def test_trend_analysis_rejects_non_iso_or_non_string_bounds(bound, argument):
    result = trend_analysis("T-018", "winding_temp_top_c", **{argument: bound})
    assert result == {"error": f"{argument} must be a valid ISO-8601 datetime."}


def test_trend_analysis_matches_aware_bounds_and_aware_readings(monkeypatch):
    import mcp_servers.tsfm_server.server as server

    stored = pd.DataFrame(
        {
            "transformer_id": ["T-synthetic"] * 3,
            "sensor_id": ["s"] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "value": [1.0, 2.0, 4.0],
        }
    )
    monkeypatch.setattr(server, "_get_readings", lambda: stored)

    unbounded_aware_result = trend_analysis("T-synthetic", "s")
    aware_result = trend_analysis(
        "T-synthetic",
        "s",
        start_time="2023-12-31T19:00:00-05:00",
        end_time="2024-01-03T00:00:00Z",
    )

    assert aware_result["num_readings"] == unbounded_aware_result["num_readings"] == 3
    assert aware_result["slope_per_day"] == unbounded_aware_result["slope_per_day"]
    assert str(stored["timestamp"].dtype) == "datetime64[ns, UTC]"

    naive = stored.copy()
    naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
    monkeypatch.setattr(server, "_get_readings", lambda: naive)
    naive_result = trend_analysis(
        "T-synthetic",
        "s",
        start_time="2024-01-01T00:00:00",
        end_time="2024-01-03T00:00:00",
    )
    equivalent_offset_result = trend_analysis(
        "T-synthetic",
        "s",
        start_time="2023-12-31T19:00:00-05:00",
        end_time="2024-01-03T00:00:00Z",
    )
    assert naive_result["num_readings"] == equivalent_offset_result["num_readings"] == 3
    assert naive_result["slope_per_day"] == equivalent_offset_result["slope_per_day"]
    assert naive_result["slope_per_day"] == aware_result["slope_per_day"]
    assert naive_result["start_time"] == "2024-01-01 00:00:00"
    assert str(naive["timestamp"].dtype) == "datetime64[ns]"


def test_trend_analysis_accepts_iso_date_and_z_bounds():
    result = trend_analysis(
        "T-018",
        "winding_temp_top_c",
        start_time="2024-01-01",
        end_time="2024-12-31T23:59:59Z",
    )
    assert "error" not in result
    assert result["num_readings"] >= 2


@pytest.mark.parametrize(
    "bound",
    [
        "2024-02-03T04:05:06.123456789Z",
        "2024-02-03t04:05:06.123456789Z",
        "2024-02-03 04:05:06.123456789+05:30",
    ],
)
def test_parse_time_bound_preserves_nanosecond_precision(bound):
    parsed, error = _parse_time_bound(bound, "start_time")
    assert error is None
    assert parsed is not None
    assert parsed.nanosecond == 789


@pytest.mark.parametrize(
    "bound",
    [
        "2024-02-03T04:05:06.1234567890Z",
        "2024-02-03 04:05:06,1234567890+05:30",
    ],
)
def test_parse_time_bound_rejects_precision_beyond_nanoseconds(bound):
    parsed, error = _parse_time_bound(bound, "end_time")
    assert parsed is None
    assert error == {"error": "end_time must be a valid ISO-8601 datetime."}


def test_trend_analysis_includes_fractional_second_endpoint_direct_and_mcp(
    monkeypatch,
):
    import mcp_servers.tsfm_server.server as server

    readings = pd.DataFrame(
        {
            "transformer_id": ["T-ns", "T-ns"],
            "sensor_id": ["s", "s"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00.123456000Z",
                    "2024-01-01T00:00:00.123456789Z",
                ],
                utc=True,
            ),
            "value": [1.0, 2.0],
        }
    )
    monkeypatch.setattr(server, "_get_readings", lambda: readings)
    start = "2024-01-01T00:00:00.123456000Z"
    end = "2024-01-01T00:00:00.123456789Z"

    direct_result = trend_analysis("T-ns", "s", start_time=start, end_time=end)
    mcp_result = asyncio.run(
        mcp.call_tool(
            "trend_analysis",
            {
                "transformer_id": "T-ns",
                "sensor_id": "s",
                "start_time": start,
                "end_time": end,
            },
        )
    )

    assert direct_result["num_readings"] == 2
    assert json.loads(mcp_result[0].text)["num_readings"] == 2
    assert str(readings["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_trend_analysis_rejects_subnanosecond_bounds_without_moving_them():
    result = trend_analysis(
        "T-018",
        "winding_temp_top_c",
        start_time="2024-01-01T00:00:00.1234567899Z",
    )
    assert result == {"error": "start_time must be a valid ISO-8601 datetime."}
    mcp_result = asyncio.run(
        mcp.call_tool(
            "trend_analysis",
            {
                "transformer_id": "T-018",
                "sensor_id": "winding_temp_top_c",
                "end_time": "2024-01-01T00:00:00.1234567899Z",
            },
        )
    )
    assert "valid ISO-8601" in mcp_result[0].text


def test_trend_analysis_preserves_aware_timestamp_output_representation(monkeypatch):
    import mcp_servers.tsfm_server.server as server

    readings = pd.DataFrame(
        {
            "transformer_id": ["T-offset"] * 2,
            "sensor_id": ["s"] * 2,
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00+05:30",
                    "2024-01-02T00:00:00+05:30",
                ]
            ),
            "value": [1.0, 2.0],
        }
    )
    monkeypatch.setattr(server, "_get_readings", lambda: readings)

    result = trend_analysis("T-offset", "s")

    assert result["start_time"] == "2024-01-01 00:00:00+05:30"
    assert result["end_time"] == "2024-01-02 00:00:00+05:30"
    assert str(readings["timestamp"].dtype) == "datetime64[ns, UTC+05:30]"


def test_trend_analysis_handles_mixed_offsets_from_csv_direct_and_mcp(monkeypatch):
    import mcp_servers.tsfm_server.server as server

    csv_data = StringIO(
        "transformer_id,timestamp,sensor_id,value\n"
        "T-dst,2024-03-10T01:30:00-05:00,s,1\n"
        "T-dst,2024-03-10T03:30:00-04:00,s,3\n"
    )
    readings = pd.read_csv(csv_data, parse_dates=["timestamp"])
    assert readings["timestamp"].dtype == object
    assert all(isinstance(value, pd.Timestamp) for value in readings["timestamp"])
    original_timestamps = readings["timestamp"].copy(deep=True)
    monkeypatch.setattr(server, "_get_readings", lambda: readings)
    arguments = {
        "transformer_id": "T-dst",
        "sensor_id": "s",
        "start_time": "2024-03-10T06:30:00Z",
        "end_time": "2024-03-10T03:30:00-04:00",
    }

    direct_result = trend_analysis(**arguments)
    mcp_result = asyncio.run(mcp.call_tool("trend_analysis", arguments))
    mcp_result = json.loads(mcp_result[0].text)

    for result in (direct_result, mcp_result):
        assert result["num_readings"] == 2
        assert result["slope_per_day"] == 48.0
        assert result["start_time"] == "2024-03-10 01:30:00-05:00"
        assert result["end_time"] == "2024-03-10 03:30:00-04:00"
    assert readings["timestamp"].equals(original_timestamps)


# ---------------------------------------------------------------------------
# trend_analysis
# ---------------------------------------------------------------------------


def test_trend_analysis_happy_path():
    result = trend_analysis("T-018", "winding_temp_top_c")
    assert "error" not in result
    for key in (
        "transformer_id",
        "sensor_id",
        "num_readings",
        "start_time",
        "end_time",
        "mean_value",
        "min_value",
        "max_value",
        "slope_per_day",
        "direction",
        "r_squared",
    ):
        assert key in result, f"Missing field: {key}"
    assert result["direction"] in ("increasing", "decreasing", "stable")


def test_trend_analysis_missing_transformer():
    result = trend_analysis("T-NONEXISTENT", "winding_temp_top_c")
    assert "error" in result


def test_trend_analysis_missing_sensor():
    result = trend_analysis("T-001", "sensor_does_not_exist")
    assert "error" in result


def test_trend_analysis_with_tight_window_too_small():
    # A window so tight it contains < 2 readings should return an error
    result = trend_analysis(
        "T-018",
        "winding_temp_top_c",
        start_time="2099-01-01T00:00:00",
        end_time="2099-01-01T01:00:00",
    )
    assert "error" in result


@pytest.mark.parametrize("bound", ["not-a-date", ""])
def test_trend_analysis_rejects_invalid_date_bounds(bound):
    result = trend_analysis("T-018", "winding_temp_top_c", start_time=bound)
    assert result == {"error": "start_time must be a valid ISO-8601 datetime."}


def test_trend_analysis_rejects_reversed_date_bounds():
    result = trend_analysis(
        "T-018",
        "winding_temp_top_c",
        start_time="2024-01-02",
        end_time="2024-01-01",
    )
    assert result == {"error": "start_time must be before or equal to end_time."}


def test_trend_analysis_normalizes_timezone_aware_bounds():
    result = trend_analysis(
        "T-018",
        "winding_temp_top_c",
        start_time="2023-12-31T19:00:00-05:00",
        end_time="2025-01-01T00:00:00+00:00",
    )
    assert "error" not in result
    assert result["num_readings"] >= 2
