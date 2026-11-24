"""
TSFM MCP Server — Time-Series Forecasting and anomaly detection for Smart Grid transformers.

TSFM = Time-Series Foundation Model.  In the full project this server will call
a fine-tuned TSFM (e.g., Chronos or a Llama-based forecaster served via vLLM).
This skeleton implements lightweight statistical baselines so the server is
functional end-to-end for scenario testing before the model is integrated.

Baseline methods used here:
  - RUL forecast:      returns the label from rul_labels.csv + a linear projection
  - Anomaly detection: z-score over a rolling window
  - Trend analysis:    linear regression slope over a requested time period

Tools exposed to the LLM agent:
  get_rul                — current remaining useful life estimate for a transformer
  forecast_rul           — project RUL N days into the future
  detect_anomalies       — flag sensor readings that exceed a z-score threshold
  trend_analysis         — slope and direction of a sensor over a time window

Data source: data/processed/rul_labels.csv, sensor_readings.csv
"""

from __future__ import annotations

import math
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import StrictFloat, StrictInt, StrictStr

from data.constants import HI_FULL_HEALTH_DAYS
from mcp_servers.base import load_rul_labels, load_sensor_readings

mcp = FastMCP("smart-grid-tsfm")

_rul: pd.DataFrame | None = None
_readings: pd.DataFrame | None = None


def _validate_anomaly_parameters(window_size: int, z_threshold: float) -> dict | None:
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, (int, np.integer))
        or window_size <= 0
        or window_size > sys.maxsize
    ):
        return {"error": "window_size must be a positive integer."}
    if isinstance(z_threshold, bool) or not isinstance(
        z_threshold, (int, float, np.integer, np.floating)
    ):
        return {"error": "z_threshold must be a finite positive number."}
    try:
        threshold = float(z_threshold)
    except OverflowError:
        return {"error": "z_threshold must be a finite positive number."}
    if not math.isfinite(threshold) or threshold <= 0:
        return {"error": "z_threshold must be a finite positive number."}
    return None


def _parse_time_bound(value: str, name: str) -> tuple[pd.Timestamp | None, dict | None]:
    if not isinstance(value, str):
        return None, {"error": f"{name} must be a valid ISO-8601 datetime."}
    try:
        if len(value) == 10:
            parsed = pd.Timestamp(date.fromisoformat(value))
        elif len(value) > 10 and value[10] in "Tt ":
            datetime.fromisoformat(
                value[:-1] + "+00:00" if value.endswith("Z") else value
            )
            time_value = value[11:]
            offset_positions = [
                position
                for marker in ("+", "-")
                if (position := time_value.find(marker)) >= 0
            ]
            clock_value = (
                time_value[: min(offset_positions)] if offset_positions else time_value
            )
            if clock_value.endswith("Z"):
                clock_value = clock_value[:-1]
            decimal_positions = [
                position
                for marker in (".", ",")
                if (position := clock_value.find(marker)) >= 0
            ]
            if decimal_positions and len(clock_value) - min(decimal_positions) - 1 > 9:
                raise ValueError
            timestamp_value = value[:10] + "T" + value[11:]
            if timestamp_value.endswith("Z"):
                timestamp_value = timestamp_value[:-1] + "+00:00"
            timestamp_value = timestamp_value.replace(",", ".", 1)
            parsed = pd.Timestamp(timestamp_value)
        else:
            raise ValueError
    except (OverflowError, TypeError, ValueError):
        return None, {"error": f"{name} must be a valid ISO-8601 datetime."}
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed, None


def _confidence_from_history(num_points: int, horizon_days: int = 0) -> float:
    """Simple bounded baseline confidence for deterministic synthetic labels."""
    history_factor = min(0.99, 0.55 + num_points / 120)
    horizon_penalty = min(0.45, max(0, horizon_days) / 900)
    return round(max(0.1, history_factor - horizon_penalty), 3)


def _get_rul() -> pd.DataFrame:
    global _rul
    if _rul is None:
        _rul = load_rul_labels()
    return _rul


def _get_readings() -> pd.DataFrame:
    global _readings
    if _readings is None:
        _readings = load_sensor_readings()
    return _readings


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_rul(transformer_id: str) -> dict:
    """
    Return the most recent remaining useful life (RUL) estimate for a transformer.

    RUL is sourced from the FDD & RUL dataset labels and represents the number
    of days of useful operation remaining before the transformer is expected to
    require major maintenance or replacement.

    Args:
        transformer_id: Asset identifier, e.g. "T-018".

    Returns:
        Dict with keys: transformer_id, as_of_date, rul_days, health_index,
        fdd_category, interpretation.
        Returns an error dict if not found.
    """
    df = _get_rul()
    subset = df[df["transformer_id"] == transformer_id].sort_values("timestamp")
    if subset.empty:
        return {"error": f"No RUL data found for '{transformer_id}'."}

    latest = subset.iloc[-1]
    rul_days = int(latest["rul_days"])
    fdd_category = (
        int(latest["fdd_category"]) if pd.notna(latest["fdd_category"]) else None
    )
    as_of_ts = latest["timestamp"]

    if rul_days >= 730:
        interpretation = "Healthy — no immediate action required."
    elif rul_days >= 180:
        interpretation = "Aging — schedule routine inspection within 6 months."
    elif rul_days >= 30:
        interpretation = "Degraded — maintenance recommended within 30 days."
    else:
        interpretation = "Critical — immediate inspection required."

    return {
        "transformer_id": transformer_id,
        "as_of_date": str(as_of_ts.date()) if pd.notna(as_of_ts) else None,
        "rul_days": rul_days,
        "health_index": round(float(latest["health_index"]), 4),
        "fdd_category": fdd_category,
        "confidence": _confidence_from_history(len(subset)),
        "interpretation": interpretation,
    }


@mcp.tool()
def forecast_rul(transformer_id: str, horizon_days: int = 30) -> dict:
    """
    Project the RUL forward by `horizon_days` using a linear degradation model.

    This is a statistical baseline. The full project will replace this with a
    TSFM (Time-Series Foundation Model) inference call via vLLM.

    Args:
        transformer_id: Asset identifier, e.g. "T-007".
        horizon_days:   Number of days to project forward (default 30, max 365).

    Returns:
        Dict with keys: transformer_id, current_rul_days, forecast_date,
        projected_rul_days, projected_health_index, confidence, method.
    """
    df = _get_rul()
    subset = df[df["transformer_id"] == transformer_id].sort_values("timestamp")
    if subset.empty:
        return {"error": f"No RUL data found for '{transformer_id}'."}

    if horizon_days < 0 or horizon_days > 365:
        return {
            "error": "horizon_days must be between 0 and 365.",
            "provided_horizon_days": horizon_days,
        }
    latest = subset.iloc[-1]
    current_rul = int(latest["rul_days"])
    latest_ts = latest["timestamp"]

    # Linear model: assume 1 RUL-day consumed per calendar day
    projected_rul = max(0, current_rul - horizon_days)
    forecast_date = (
        pd.to_datetime(latest_ts) + pd.Timedelta(days=horizon_days)
        if pd.notna(latest_ts)
        else None
    )
    projected_hi = (
        0.0
        if current_rul <= 0
        else round(min(1.0, projected_rul / HI_FULL_HEALTH_DAYS), 4)
    )

    return {
        "transformer_id": transformer_id,
        "current_rul_days": current_rul,
        "forecast_date": (
            str(forecast_date.date()) if forecast_date is not None else None
        ),
        "projected_rul_days": projected_rul,
        "projected_health_index": projected_hi,
        "confidence": _confidence_from_history(len(subset), horizon_days),
        "method": "linear_degradation_baseline",
    }


@mcp.tool()
def detect_anomalies(
    transformer_id: str,
    sensor_id: str,
    window_size: StrictInt = 24,
    z_threshold: StrictFloat = 3.0,
) -> dict:
    """
    Detect anomalous sensor readings using a rolling z-score method.

    A reading is flagged as anomalous when it deviates more than `z_threshold`
    standard deviations from the rolling mean over `window_size` readings.

    Args:
        transformer_id: Asset identifier, e.g. "T-016".
        sensor_id:      Sensor to analyse, e.g. "dga_h2_ppm".
        window_size:    Rolling window size in readings (default 24).
        z_threshold:    Number of standard deviations to flag (default 3.0).

    Returns:
        Dict with keys: transformer_id, sensor_id, total_readings,
        anomaly_count, anomaly_rate_pct, anomalies (list of flagged readings
        with timestamp, value, z_score).
    """
    validation_error = _validate_anomaly_parameters(window_size, z_threshold)
    if validation_error is not None:
        return validation_error
    window_size = int(window_size)
    z_threshold = float(z_threshold)

    df = _get_readings()
    subset = (
        df[(df["transformer_id"] == transformer_id) & (df["sensor_id"] == sensor_id)]
        .copy()
        .sort_values("timestamp")
    )

    if subset.empty:
        return {
            "error": f"No readings for transformer='{transformer_id}' "
            f"sensor='{sensor_id}'."
        }

    vals = subset["value"].astype(float)
    rolling_mean = vals.rolling(window_size, min_periods=1).mean()
    rolling_std = vals.rolling(window_size, min_periods=1).std()
    rolling_std = rolling_std.where(
        rolling_std > 0, other=rolling_mean.abs() * 0.001 + 1e-3
    )
    z_scores = ((vals - rolling_mean) / rolling_std).abs()

    anomaly_mask = z_scores > z_threshold
    anomalies_df = subset[anomaly_mask][["timestamp", "value"]].copy()
    anomalies_df["z_score"] = z_scores[anomaly_mask].round(3).values

    anomalies = anomalies_df.head(50).to_dict(orient="records")
    # Ensure timestamps are strings
    for row in anomalies:
        row["timestamp"] = str(row["timestamp"])

    return {
        "transformer_id": transformer_id,
        "sensor_id": sensor_id,
        "window_size": window_size,
        "z_threshold": z_threshold,
        "total_readings": len(subset),
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_rate_pct": round(100 * anomaly_mask.mean(), 2),
        "anomalies": anomalies,
    }


@mcp.tool()
def trend_analysis(
    transformer_id: str,
    sensor_id: str,
    start_time: StrictStr | None = None,
    end_time: StrictStr | None = None,
) -> dict:
    """
    Compute the trend (slope) of a sensor's readings over a time window.

    Uses ordinary least squares linear regression over the selected readings.
    A positive slope means the sensor value is increasing over time.

    Args:
        transformer_id: Asset identifier, e.g. "T-012".
        sensor_id:      Sensor to analyse, e.g. "dga_c2h2_ppm".
        start_time:     ISO-8601 start of window (optional).
        end_time:       ISO-8601 end of window (optional).

    Returns:
        Dict with keys: transformer_id, sensor_id, num_readings,
        start_time, end_time, mean_value, min_value, max_value,
        slope_per_day, direction ("increasing"/"decreasing"/"stable"),
        r_squared.
    """
    start_bound = end_bound = None
    if start_time is not None:
        start_bound, error = _parse_time_bound(start_time, "start_time")
        if error is not None:
            return error
    if end_time is not None:
        end_bound, error = _parse_time_bound(end_time, "end_time")
        if error is not None:
            return error
    if start_bound is not None and end_bound is not None and start_bound > end_bound:
        return {"error": "start_time must be before or equal to end_time."}

    df = _get_readings()
    subset = df[
        (df["transformer_id"] == transformer_id) & (df["sensor_id"] == sensor_id)
    ].copy()

    if subset.empty:
        return {
            "error": f"No readings for transformer='{transformer_id}' "
            f"sensor='{sensor_id}'."
        }

    subset["timestamp"] = subset["timestamp"].map(pd.Timestamp)
    subset["_timestamp_utc"] = pd.to_datetime(
        subset["timestamp"], utc=True
    ).dt.tz_localize(None)
    if start_bound is not None:
        subset = subset[subset["_timestamp_utc"] >= start_bound]
    if end_bound is not None:
        subset = subset[subset["_timestamp_utc"] <= end_bound]

    subset = subset.sort_values("_timestamp_utc")
    if len(subset) < 2:
        return {
            "error": "Not enough readings in the specified window for trend analysis."
        }

    # Convert timestamps to numeric (days since first reading)
    t0 = subset["_timestamp_utc"].iloc[0]
    days = (subset["_timestamp_utc"] - t0).dt.total_seconds() / 86400
    vals = subset["value"].astype(float)

    # OLS: slope and R²
    coeffs = np.polyfit(days, vals, 1)
    slope = float(coeffs[0])
    y_hat = np.polyval(coeffs, days)
    ss_res = float(np.sum((vals - y_hat) ** 2))
    ss_tot = float(np.sum((vals - vals.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Direction: stable if slope < 1% of mean per day
    mean_val = float(vals.mean())
    rel_slope = abs(slope) / (mean_val + 1e-9)
    if rel_slope < 0.01:
        direction = "stable"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return {
        "transformer_id": transformer_id,
        "sensor_id": sensor_id,
        "num_readings": len(subset),
        "start_time": str(subset["timestamp"].iloc[0]),
        "end_time": str(subset["timestamp"].iloc[-1]),
        "mean_value": round(mean_val, 4),
        "min_value": round(float(vals.min()), 4),
        "max_value": round(float(vals.max()), 4),
        "slope_per_day": round(slope, 6),
        "direction": direction,
        "r_squared": round(r2, 4),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
