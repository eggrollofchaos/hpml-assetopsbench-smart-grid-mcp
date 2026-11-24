#!/usr/bin/env python3
"""Resume-state helpers for SmartGridBench GCP fallback runs.

The shell runner owns process orchestration. This helper owns the fiddly,
testable parts of resume: trial classification, latency-row dedupe, atomic
finalization, and manifest events.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _posix(path: Path | str) -> str:
    return Path(path).as_posix()


def _norm(path: Path | str) -> str:
    return os.path.normpath(_posix(path))


def _shell_bool(value: bool) -> str:
    return "1" if value else "0"


def _flag_enabled(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _shell_flag_enabled(value: str | bool | None) -> bool:
    """Mirror the wrapper's exact 1/0 switch semantics for identity fields."""
    if isinstance(value, bool):
        return value
    return str(value or "") == "1"


def _emit_shell(values: dict[str, Any]) -> None:
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            args,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        ).strip()
    except Exception:
        return None


def collect_runtime_versions() -> dict[str, Any]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = (cuda_visible or "0").split(",")[0].strip() or "0"
    versions: dict[str, Any] = {
        "vllm_version": _package_version("vllm"),
        "torch_version": _package_version("torch"),
        "cuda_visible_devices": cuda_visible,
        "nvidia_driver_version": None,
        "cuda_version": None,
        "nvidia_smi_query": None,
    }
    raw = _command_output(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=driver_version,cuda_version",
            "--format=csv,noheader",
        ]
    )
    if raw:
        versions["nvidia_smi_query"] = raw
        first_row = raw.splitlines()[0]
        parts = [part.strip() for part in first_row.split(",")]
        if parts:
            versions["nvidia_driver_version"] = parts[0] or None
        if len(parts) > 1:
            versions["cuda_version"] = parts[1] or None
    if versions["nvidia_driver_version"] is None:
        driver = _command_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        )
        if driver:
            versions["nvidia_driver_version"] = driver.splitlines()[0].strip() or None
    if versions["cuda_version"] is None:
        smi = _command_output(["nvidia-smi"])
        if smi:
            match = re.search(r"CUDA Version:\s*([0-9.]+)", smi)
            if match:
                versions["cuda_version"] = match.group(1)
    return versions


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing_json"
    try:
        if path.stat().st_size == 0:
            return None, "zero_byte_json"
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "invalid_json"
    except OSError as exc:
        return None, f"read_error:{exc}"
    if not isinstance(payload, dict):
        return None, "non_object_json"
    return payload, None


def _scenario_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"scenario is not a JSON object: {path}")
    return payload


def _scenario_basename(path: Path) -> str:
    return path.stem


def _expand_scenario_glob(scenario_glob: str) -> list[Path]:
    """Expand either one glob pattern or a shell-style list of scenario paths."""
    tokens = shlex.split(scenario_glob)
    if not tokens:
        tokens = [scenario_glob]
    paths: list[Path] = []
    seen: set[str] = set()
    for token in tokens:
        for match in sorted(glob.glob(token)):
            key = _norm(match)
            if key in seen:
                continue
            paths.append(Path(match))
            seen.add(key)
    return paths


def _step_failed(step: dict[str, Any]) -> bool:
    if step.get("success") is False:
        return True
    if step.get("error"):
        return True
    response = step.get("response")
    return isinstance(response, dict) and bool(response.get("error"))


def derive_success(data: dict[str, Any]) -> bool | None:
    raw = data.get("success")
    if isinstance(raw, bool):
        return raw
    steps = data.get("history") or data.get("trajectory") or []
    if not steps and not data.get("answer"):
        return None
    for step in steps:
        if isinstance(step, dict) and _step_failed(step):
            return False
    return bool(data.get("answer"))


def _is_infrastructure_error_payload(data: dict[str, Any]) -> bool:
    """Identify rerunnable runner exceptions without masking model failures."""
    if (
        data.get("infrastructure_error") is True
        or data.get("state") == "infrastructure_error"
    ):
        return True
    error = data.get("error")
    if "error" in data and (not isinstance(error, str) or not error.strip()):
        return True
    if _is_terminal_model_failure_payload(data):
        return False
    return bool(data.get("error"))


_LEGACY_MODEL_ERROR_MARKERS = (
    "maxturnsexceeded",
    "max turns exceeded",
    "modelbehaviorerror",
    "model behavior error",
)


def _is_terminal_model_failure_payload(data: dict[str, Any]) -> bool:
    """Recognize current and known historical terminal model failures."""
    if data.get("infrastructure_error") is True:
        return False
    error = data.get("error")
    if "error" in data and (not isinstance(error, str) or not error.strip()):
        return False
    if data.get("model_error") is True or data.get("max_turns_exhausted") is True:
        return isinstance(error, str) and bool(error.strip())
    if isinstance(error, str):
        error_class = error.partition(":")[0].strip().casefold().replace("_", " ")
        return error_class in _LEGACY_MODEL_ERROR_MARKERS
    return data.get("state") == "complete_failure" or data.get("success") is False


def _scenario_id_matches(
    payload: dict[str, Any],
    scenario: dict[str, Any],
) -> bool:
    embedded = payload.get("scenario")
    if not isinstance(embedded, dict):
        return True
    expected_id = scenario.get("id")
    actual_id = embedded.get("id")
    return not expected_id or not actual_id or expected_id == actual_id


_RUN_IDENTITY_FIELDS = (
    "config_source_sha256",
    "runner_code_sha256",
    "processed_data_sha256",
    "git_sha",
    "missing_evidence_guard",
    "missing_evidence_repair",
    "explicit_fault_risk_adjudication",
    "missing_evidence_repair_max_attempts",
    "missing_evidence_repair_max_attempts_per_target",
    "enable_self_ask",
    "plan_execute_repo_local",
    "aat_openai_agents_version",
    "aat_mcp_version",
    "aat_litellm_version",
    "aat_mcp_client_timeout_seconds",
    "runner_template_sha256",
    "server_iot_sha256",
    "server_fmsr_sha256",
    "server_tsfm_sha256",
    "server_wo_sha256",
    "gpu_type",
    "scenario_set_name",
    "scenario_set_hash",
    "trial_count",
    "experiment_cell",
    "orchestration_mode",
    "mcp_mode",
    "model_id",
    "model_provider",
    "judge_model",
    "context_window",
    "temperature",
    "max_tokens",
    "serving_stack",
    "quantization_mode",
    "launch_vllm",
    "vllm_model_path",
    "vllm_served_model_name",
    "vllm_dtype",
    "vllm_generation_config",
    "vllm_tool_call_parser",
    "vllm_enable_auto_tool_choice",
    "vllm_extra_args",
    "enable_smartgrid_servers",
    "aat_parallel_tool_calls",
    "aat_tool_choice",
    "aob_python_version",
    "aat_mcp_server_launch_mode",
    "aat_mcp_server_python_version",
    "runner_python_version",
    "runner_interpreter_identity",
    "aob_interpreter_identity",
    "aat_mcp_interpreter_identity",
    "aat_uv_interpreter_identity",
    "aob_uv_interpreter_identity",
)

_RUNNER_CODE_FILES = (
    "scripts/run_experiment.sh",
    "scripts/gcp_resume_state.py",
    "scripts/aat_runner.py",
    "scripts/aat_system_prompt.py",
    "scripts/aat_tools_direct.py",
    "scripts/aat_tools_mcp.py",
    "scripts/aat_mcp_server_bootstrap.py",
    "scripts/watsonx_env.py",
    "scripts/orchestration_utils.py",
    "scripts/plan_execute_self_ask_runner.py",
    "scripts/verified_pe_runner.py",
    "scripts/mitigation_guards.py",
    "mcp_servers/direct_adapter.py",
    "mcp_servers/base.py",
    "mcp_servers/iot_server/server.py",
    "mcp_servers/fmsr_server/server.py",
    "mcp_servers/tsfm_server/server.py",
    "mcp_servers/wo_server/server.py",
)


def _git_revision(repo_root: Path) -> str | None:
    return _command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"])


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "diff-index", "--quiet", "HEAD", "--"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    return None


def _runner_code_sha256(repo_root: Path) -> str | None:
    digest = hashlib.sha256()
    for relative in _RUNNER_CODE_FILES:
        path = repo_root / relative
        if not path.is_file():
            return None
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _processed_data_sha256(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted((repo_root / "data" / "processed").glob("*.csv")):
        digest = _sha256(path)
        if digest is None:
            raise ValueError(f"processed data is unreadable: {path}")
        hashes[path.relative_to(repo_root).as_posix()] = digest
    return hashes


def normalize_aat_client_timeout_seconds(value: str | float) -> float:
    raw = str(value).strip()
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"AAT_MCP_CLIENT_TIMEOUT_SECONDS must be numeric, got {raw!r}"
        ) from exc
    if timeout <= 0:
        raise ValueError(f"AAT_MCP_CLIENT_TIMEOUT_SECONDS must be > 0, got {timeout!r}")
    return timeout


def interpreter_identity(python_path: Path, version: str) -> dict[str, str]:
    """Bind an executable's bytes and reported runtime without host-path equality."""
    try:
        candidate = str(python_path)
        if not python_path.is_absolute() and len(python_path.parts) == 1:
            found = shutil.which(candidate)
            if found is None:
                raise FileNotFoundError(candidate)
            python_path = Path(found)
        resolved = python_path.resolve(strict=True)
        digest = _sha256(resolved)
    except OSError as exc:
        raise ValueError(
            f"selected Python interpreter is unavailable: {python_path}: {exc}"
        ) from exc
    if digest is None or not resolved.is_file():
        raise ValueError(f"selected Python interpreter is unreadable: {python_path}")
    label = version.strip()
    if not label:
        raise ValueError(
            f"selected Python interpreter version is unavailable: {python_path}"
        )
    return {"version": label, "binary_sha256": digest}


def effective_behavior_identity(
    *,
    orchestration_mode: str,
    enable_smartgrid_servers: str | bool = "1",
    missing_evidence_guard: str | bool = "0",
    missing_evidence_repair: str | bool = "0",
    explicit_fault_risk_adjudication: str | bool = "0",
    missing_evidence_repair_max_attempts: str | int = "2",
    missing_evidence_repair_max_attempts_per_target: str | int = "1",
    enable_self_ask: str | bool = "0",
    plan_execute_repo_local: str | bool = "0",
    aat_openai_agents_version: str = "0.14.5",
    aat_mcp_version: str = "1.27.0",
    aat_litellm_version: str = "1.81.13",
    aat_mcp_client_timeout_seconds: str | float = "30",
    aat_runner_template: str = "",
    hybrid_runner_template: str = "",
    verified_pe_runner_template: str = "",
    server_iot_path: str = "",
    server_fmsr_path: str = "",
    server_tsfm_path: str = "",
    server_wo_path: str = "",
) -> dict[str, Any]:
    from scripts.mitigation_guards import env_flag_enabled

    repair_enabled = env_flag_enabled(missing_evidence_repair)
    if repair_enabled:
        try:
            repair_attempts = int(missing_evidence_repair_max_attempts)
            repair_per_target = int(missing_evidence_repair_max_attempts_per_target)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "missing-evidence repair attempts must be positive integers"
            ) from exc
        if repair_attempts < 1 or repair_per_target < 1:
            raise ValueError(
                "missing-evidence repair attempts must be positive integers"
            )
    else:
        repair_attempts = 0
        repair_per_target = 0

    if orchestration_mode == "agent_as_tool":
        selected_template = aat_runner_template
    elif orchestration_mode == "verified_pe":
        selected_template = verified_pe_runner_template
    elif orchestration_mode == "hybrid":
        selected_template = hybrid_runner_template
    else:
        selected_template = ""
    template_hash = (
        hashlib.sha256(selected_template.encode("utf-8")).hexdigest()
        if selected_template
        else None
    )
    uses_aat = orchestration_mode == "agent_as_tool"
    server_hashes = {
        "server_iot_sha256": server_iot_path,
        "server_fmsr_sha256": server_fmsr_path,
        "server_tsfm_sha256": server_tsfm_path,
        "server_wo_sha256": server_wo_path,
    }
    external_template_consumes_overrides = bool(selected_template)
    native_runner_consumes_overrides = orchestration_mode == "verified_pe" or (
        orchestration_mode == "plan_execute"
        and (
            _shell_flag_enabled(enable_self_ask)
            or _shell_flag_enabled(plan_execute_repo_local)
        )
    )
    selected_server_overrides = (
        external_template_consumes_overrides or native_runner_consumes_overrides
    )
    if selected_server_overrides:
        for field, raw_path in tuple(server_hashes.items()):
            if not raw_path:
                raise ValueError(f"selected server source is unavailable: {field}")
            source = Path(raw_path).resolve()
            if not source.is_file():
                raise ValueError(f"selected server source is unreadable: {source}")
            digest = _sha256(source)
            if digest is None:
                raise ValueError(f"selected server source is unreadable: {source}")
            server_hashes[field] = digest
    else:
        server_hashes = {field: None for field in server_hashes}
    return {
        "missing_evidence_guard": env_flag_enabled(missing_evidence_guard),
        "missing_evidence_repair": repair_enabled,
        "explicit_fault_risk_adjudication": env_flag_enabled(
            explicit_fault_risk_adjudication
        ),
        "missing_evidence_repair_max_attempts": repair_attempts,
        "missing_evidence_repair_max_attempts_per_target": repair_per_target,
        "enable_self_ask": _shell_flag_enabled(enable_self_ask),
        "plan_execute_repo_local": _shell_flag_enabled(plan_execute_repo_local),
        "aat_openai_agents_version": aat_openai_agents_version if uses_aat else None,
        "aat_mcp_version": aat_mcp_version if uses_aat else None,
        "aat_litellm_version": aat_litellm_version if uses_aat else None,
        "aat_mcp_client_timeout_seconds": (
            normalize_aat_client_timeout_seconds(aat_mcp_client_timeout_seconds)
            if uses_aat
            else None
        ),
        "runner_template_sha256": template_hash,
        **server_hashes,
    }


def build_run_identity(
    *,
    config_path: Path,
    repo_root: Path,
    gpu_type: str = "unknown",
    enable_smartgrid_servers: str | bool = "1",
    scenario_set_name: str,
    scenario_set_hash: str,
    trial_count: int = 1,
    experiment_cell: str,
    orchestration_mode: str,
    mcp_mode: str,
    model_id: str,
    model_provider: str,
    judge_model: str | None,
    context_window: int,
    temperature: float,
    max_tokens: int,
    serving_stack: str = "unknown",
    quantization_mode: str = "none",
    launch_vllm: str = "0",
    vllm_model_path: str = "models/Llama-3.1-8B-Instruct",
    vllm_served_model_name: str = "Llama-3.1-8B-Instruct",
    vllm_dtype: str = "float16",
    vllm_generation_config: str = "vllm",
    vllm_tool_call_parser: str = "llama3_json",
    vllm_enable_auto_tool_choice: str = "1",
    vllm_extra_args: str = "",
    aat_parallel_tool_calls: str | bool | None = "false",
    aat_tool_choice: str | None = None,
    missing_evidence_guard: str | bool = "0",
    missing_evidence_repair: str | bool = "0",
    explicit_fault_risk_adjudication: str | bool = "0",
    missing_evidence_repair_max_attempts: str | int = "2",
    missing_evidence_repair_max_attempts_per_target: str | int = "1",
    enable_self_ask: str | bool = "0",
    plan_execute_repo_local: str | bool = "0",
    aat_openai_agents_version: str = "0.14.5",
    aat_mcp_version: str = "1.27.0",
    aat_litellm_version: str = "1.81.13",
    aat_mcp_client_timeout_seconds: str | float = "30",
    aat_runner_template: str = "",
    hybrid_runner_template: str = "",
    verified_pe_runner_template: str = "",
    server_iot_path: str = "",
    server_fmsr_path: str = "",
    server_tsfm_path: str = "",
    server_wo_path: str = "",
    aob_python_version: str = "unknown",
    aat_mcp_server_launch_mode: str = "python",
    aat_mcp_server_python_version: str = "unknown",
    runner_python_version: str = "unknown",
) -> dict[str, Any]:
    config_source_sha256 = _sha256(config_path)
    runner_code_sha256 = _runner_code_sha256(repo_root)
    git_sha = _git_revision(repo_root)
    git_dirty = _git_dirty(repo_root)
    if config_source_sha256 is None:
        raise ValueError(f"config source is unreadable: {config_path}")
    if runner_code_sha256 is None:
        raise ValueError(f"runner code is unreadable: {repo_root}")
    if git_sha is None:
        raise ValueError(f"git revision is unavailable: {repo_root}")
    if git_dirty is None:
        raise ValueError(f"git dirty state is unavailable: {repo_root}")
    if orchestration_mode == "agent_as_tool":
        from scripts.aat_runner import _parse_parallel_tool_calls

        effective_parallel_tool_calls = _parse_parallel_tool_calls(
            str(aat_parallel_tool_calls)
        )
        if aat_tool_choice not in {None, "auto", "required", "none"}:
            raise ValueError(f"invalid AAT tool-choice setting: {aat_tool_choice!r}")
        effective_tool_choice = aat_tool_choice or None
    else:
        effective_parallel_tool_calls = None
        effective_tool_choice = None
    behavior_identity = effective_behavior_identity(
        orchestration_mode=orchestration_mode,
        enable_smartgrid_servers=enable_smartgrid_servers,
        missing_evidence_guard=missing_evidence_guard,
        missing_evidence_repair=missing_evidence_repair,
        explicit_fault_risk_adjudication=explicit_fault_risk_adjudication,
        missing_evidence_repair_max_attempts=missing_evidence_repair_max_attempts,
        missing_evidence_repair_max_attempts_per_target=missing_evidence_repair_max_attempts_per_target,
        enable_self_ask=enable_self_ask,
        plan_execute_repo_local=plan_execute_repo_local,
        aat_openai_agents_version=aat_openai_agents_version,
        aat_mcp_version=aat_mcp_version,
        aat_litellm_version=aat_litellm_version,
        aat_mcp_client_timeout_seconds=aat_mcp_client_timeout_seconds,
        aat_runner_template=aat_runner_template,
        hybrid_runner_template=hybrid_runner_template,
        verified_pe_runner_template=verified_pe_runner_template,
        server_iot_path=server_iot_path,
        server_fmsr_path=server_fmsr_path,
        server_tsfm_path=server_tsfm_path,
        server_wo_path=server_wo_path,
    )
    return {
        "config_source_sha256": config_source_sha256,
        "runner_code_sha256": runner_code_sha256,
        "processed_data_sha256": _processed_data_sha256(repo_root),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "gpu_type": gpu_type.strip() or "unknown",
        "scenario_set_name": scenario_set_name,
        "scenario_set_hash": scenario_set_hash,
        "trial_count": int(trial_count),
        "experiment_cell": experiment_cell,
        "orchestration_mode": orchestration_mode,
        "mcp_mode": mcp_mode,
        "model_id": model_id,
        "model_provider": model_provider,
        "judge_model": judge_model or None,
        "context_window": int(context_window),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "serving_stack": serving_stack,
        "quantization_mode": quantization_mode,
        "launch_vllm": launch_vllm,
        "vllm_model_path": vllm_model_path,
        "vllm_served_model_name": vllm_served_model_name,
        "vllm_dtype": vllm_dtype,
        "vllm_generation_config": vllm_generation_config,
        "vllm_tool_call_parser": vllm_tool_call_parser,
        "vllm_enable_auto_tool_choice": vllm_enable_auto_tool_choice,
        "vllm_extra_args": vllm_extra_args,
        "enable_smartgrid_servers": _shell_flag_enabled(enable_smartgrid_servers),
        "aat_parallel_tool_calls": effective_parallel_tool_calls,
        "aat_tool_choice": effective_tool_choice,
        "aob_python_version": aob_python_version,
        "aat_mcp_server_launch_mode": aat_mcp_server_launch_mode,
        "aat_mcp_server_python_version": aat_mcp_server_python_version,
        "runner_python_version": runner_python_version,
        "runner_interpreter_identity": os.environ.get(
            "RUNNER_PYTHON_IDENTITY", "unresolved"
        ),
        "aob_interpreter_identity": os.environ.get("AOB_PYTHON_IDENTITY", "unresolved"),
        "aat_mcp_interpreter_identity": os.environ.get(
            "AAT_MCP_PYTHON_IDENTITY", "not_applicable"
        ),
        "aat_uv_interpreter_identity": os.environ.get(
            "AAT_UV_PYTHON_IDENTITY", "not_applicable"
        ),
        "aob_uv_interpreter_identity": os.environ.get(
            "AOB_UV_PYTHON_IDENTITY", "not_applicable"
        ),
        **behavior_identity,
    }


def _stored_run_identity(meta: dict[str, Any]) -> dict[str, Any] | None:
    identity = meta.get("run_identity")
    if not isinstance(identity, dict):
        return None
    return identity


def validate_resume_identity(
    *,
    meta_file: Path,
    run_dir: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Refuse resume when existing artifacts lack an exact immutable identity."""
    if not run_dir.exists():
        return {"validated": False, "reason": "new_run"}
    try:
        has_artifacts = next(run_dir.iterdir(), None) is not None
    except OSError as exc:
        raise ValueError(f"run directory is unreadable: {run_dir}: {exc}") from exc
    if not has_artifacts:
        return {"validated": False, "reason": "empty_run_directory"}

    gpu_type = expected.get("gpu_type")
    if not isinstance(gpu_type, str) or gpu_type.strip().casefold() in {
        "",
        "unknown",
        "unavailable",
        "none",
        "null",
    }:
        raise ValueError(
            "resume hardware identity unavailable; set GPU_TYPE to the "
            "actual benchmark execution GPU before reusing artifacts"
        )

    meta, error = _load_json_object(meta_file)
    if error is not None or meta is None:
        raise ValueError(f"resume identity unavailable: {meta_file}: {error}")
    actual = _stored_run_identity(meta)
    if actual is None:
        raise ValueError(f"resume identity missing: {meta_file}")

    missing_fields = [field for field in _RUN_IDENTITY_FIELDS if field not in actual]
    if missing_fields:
        fields = ", ".join(sorted(missing_fields))
        raise ValueError(f"resume identity missing required fields ({fields})")

    mismatches = {
        field: {"stored": actual[field], "current": expected.get(field)}
        for field in _RUN_IDENTITY_FIELDS
        if field != "git_sha"
        if actual[field] != expected.get(field)
    }
    if mismatches:
        fields = ", ".join(sorted(mismatches))
        raise ValueError(f"resume identity mismatch ({fields})")
    return {"validated": True, "reason": "exact_identity"}


def _latency_records(latency_file: Path) -> list[dict[str, Any]]:
    if not latency_file.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in latency_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _record_matches_identity(
    record: dict[str, Any],
    scenario_file: Path,
    trial_index: int,
    output_path: Path,
) -> bool:
    if int(record.get("trial_index", -1)) != int(trial_index):
        return False
    output = record.get("output_path")
    if output and _norm(output) != _norm(output_path):
        return False
    scenario = record.get("scenario_file")
    return not scenario or _norm(scenario) == _norm(scenario_file)


def _matching_latency_rows(
    records: list[dict[str, Any]],
    scenario_file: Path,
    trial_index: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if _record_matches_identity(record, scenario_file, trial_index, output_path)
    ]


def _rows_conflict(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 2:
        return False
    normalized = {
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    }
    return len(normalized) > 1


def _candidate_trial_paths(
    run_dir: Path,
    scenario_file: Path,
    trial_index: int,
    output_path: Path,
) -> list[Path]:
    run_label = f"{_scenario_basename(scenario_file)}_run{trial_index:02d}.json"
    candidates = [output_path]
    candidates.extend(sorted(run_dir.glob(f"*_{run_label}")))
    seen: set[str] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def classify_trial(
    *,
    run_dir: Path,
    scenario_file: Path,
    trial_index: int,
    output_path: Path,
    latency_file: Path,
    require_latency: bool,
) -> dict[str, Any]:
    scenario = _scenario_payload(scenario_file)
    latency_records = _latency_records(latency_file)
    incomplete_reasons: list[str] = []
    saw_candidate = False

    for candidate in _candidate_trial_paths(
        run_dir, scenario_file, trial_index, output_path
    ):
        if not candidate.exists():
            continue
        saw_candidate = True
        payload, error = _load_json_object(candidate)
        if error is not None or payload is None:
            incomplete_reasons.append(error or "invalid_json")
            continue
        if not _scenario_id_matches(payload, scenario):
            incomplete_reasons.append("scenario_identity_conflict")
            continue
        if _is_infrastructure_error_payload(payload):
            incomplete_reasons.append("infrastructure_error_payload")
            continue
        success = derive_success(payload)
        if success is None:
            incomplete_reasons.append("success_not_derivable")
            continue
        rows = _matching_latency_rows(
            latency_records, scenario_file, trial_index, candidate
        )
        if require_latency:
            if not rows:
                incomplete_reasons.append("missing_latency")
                continue
            if _rows_conflict(rows):
                incomplete_reasons.append("duplicate_conflicting_latency")
                continue
        return {
            "state": "complete_success" if success else "complete_failure",
            "complete": True,
            "success": success,
            "output_path": _posix(candidate),
            "reason": "terminal_trial",
            "latency_rows": len(rows),
        }

    stdout_candidates = [
        path
        for path in run_dir.glob(
            f"*_{_scenario_basename(scenario_file)}_run{trial_index:02d}.json.stdout"
        )
        if path.exists()
    ]
    if stdout_candidates and not saw_candidate:
        incomplete_reasons.append("dangling_stdout")

    return {
        "state": "incomplete",
        "complete": False,
        "success": False,
        "output_path": _posix(output_path),
        "reason": ",".join(sorted(set(incomplete_reasons))) or "missing_json",
        "latency_rows": 0,
    }


def validate_run_artifacts(
    *,
    run_dir: Path,
    scenario_glob: str,
    trials: int,
    run_name: str = "",
    require_latency: bool = True,
) -> dict[str, Any]:
    scenario_files = _expand_scenario_glob(scenario_glob)
    expected = len(scenario_files) * int(trials)
    complete = 0
    success = 0
    failure = 0
    missing: list[dict[str, Any]] = []
    latency_file = run_dir / "latencies.jsonl"

    if expected == 0:
        return {
            "valid": False,
            "expected": 0,
            "complete": 0,
            "success": 0,
            "failure": 0,
            "missing": [],
            "reason": "no_expected_trials",
        }

    for scenario_file in scenario_files:
        basename = _scenario_basename(scenario_file)
        for trial_index in range(1, int(trials) + 1):
            output_name = f"{basename}_run{trial_index:02d}.json"
            if run_name:
                output_name = f"{run_name}_{output_name}"
            result = classify_trial(
                run_dir=run_dir,
                scenario_file=scenario_file,
                trial_index=trial_index,
                output_path=run_dir / output_name,
                latency_file=latency_file,
                require_latency=require_latency,
            )
            if result["complete"]:
                complete += 1
                if result["success"]:
                    success += 1
                else:
                    failure += 1
                continue
            missing.append(
                {
                    "scenario_file": _posix(scenario_file),
                    "trial_index": trial_index,
                    "reason": result["reason"],
                    "output_path": result["output_path"],
                }
            )

    valid = complete == expected
    return {
        "valid": valid,
        "expected": expected,
        "complete": complete,
        "success": success,
        "failure": failure,
        "missing": missing,
        "reason": "ok" if valid else "incomplete_trajectory_artifacts",
    }


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest_event(
    *,
    manifest_file: Path,
    state: str,
    scenario_file: Path,
    trial_index: int,
    output_path: Path,
    run_name: str,
    reason: str = "",
    batch_id: str = "",
    latency_seconds: float | None = None,
    return_code: int | None = None,
    started_at: str = "",
    finished_at: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "batch_id": batch_id or None,
        "run_name": run_name,
        "event": "trial_resume_state",
        "state": state,
        "reason": reason or None,
        "scenario_file": _posix(scenario_file),
        "scenario_basename": _scenario_basename(scenario_file),
        "trial_index": int(trial_index),
        "output_path": _posix(output_path),
        "latency_seconds": latency_seconds,
        "return_code": return_code,
        "started_at": started_at or None,
        "finished_at": finished_at or _now_iso(),
        "compute_provider": os.environ.get("SMARTGRID_COMPUTE_PROVIDER"),
        "compute_zone": os.environ.get("SMARTGRID_COMPUTE_ZONE"),
        "compute_instance": os.environ.get("SMARTGRID_COMPUTE_INSTANCE"),
        "gpu_type": os.environ.get("GPU_TYPE"),
        "runtime_versions": collect_runtime_versions(),
    }
    if extra:
        payload.update(extra)
    with manifest_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def preserve_incomplete_output(output_path: Path) -> Path | None:
    if not output_path.exists() and not output_path.is_symlink():
        return None
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_path.with_name(f"{output_path.name}.incomplete.{suffix}")
    preserved = base
    index = 1
    while True:
        try:
            os.link(output_path, preserved, follow_symlinks=False)
            break
        except FileExistsError:
            preserved = base.with_name(f"{base.name}.{index}")
            index += 1
    output_path.unlink()
    return preserved


def _apply_final_answer_guards(payload: dict[str, Any]) -> None:
    missing_evidence_enabled = _flag_enabled(
        os.environ.get("ENABLE_MISSING_EVIDENCE_GUARD")
    )
    explicit_adjudication_enabled = _flag_enabled(
        os.environ.get("ENABLE_EXPLICIT_FAULT_RISK_ADJUDICATION")
    )
    if not (missing_evidence_enabled or explicit_adjudication_enabled):
        return
    from scripts.mitigation_guards import (  # type: ignore
        apply_explicit_fault_risk_adjudication,
        apply_missing_evidence_final_answer_guard,
    )

    apply_missing_evidence_final_answer_guard(
        payload,
        enabled=missing_evidence_enabled,
    )
    if "fault_risk_adjudication" not in payload:
        apply_explicit_fault_risk_adjudication(
            payload,
            enabled=explicit_adjudication_enabled,
        )


def _upsert_latency_row(
    *,
    latency_file: Path,
    scenario_file: Path,
    trial_index: int,
    output_path: Path,
    latency_seconds: float,
) -> None:
    records = [
        row
        for row in _latency_records(latency_file)
        if not _record_matches_identity(row, scenario_file, trial_index, output_path)
    ]
    records.append(
        {
            "scenario_file": _posix(scenario_file),
            "trial_index": int(trial_index),
            "latency_seconds": latency_seconds,
            "output_path": _posix(output_path),
        }
    )
    latency_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = latency_file.with_suffix(latency_file.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    tmp.replace(latency_file)


def finalize_trial(
    *,
    scenario_file: Path,
    trial_index: int,
    temp_output: Path,
    output_path: Path,
    latency_file: Path,
    manifest_file: Path,
    run_name: str,
    batch_id: str,
    start_epoch: float,
    end_epoch: float,
    return_code: int,
) -> dict[str, Any]:
    payload, error = _load_json_object(temp_output)
    if error is not None or payload is None:
        write_manifest_event(
            manifest_file=manifest_file,
            state="incomplete",
            scenario_file=scenario_file,
            trial_index=trial_index,
            output_path=output_path,
            run_name=run_name,
            reason=error or "invalid_json",
            batch_id=batch_id,
            return_code=return_code,
        )
        return {
            "state": "incomplete",
            "success": False,
            "reason": error or "invalid_json",
        }

    if int(return_code) != 0 and not (
        int(return_code) == 1 and payload.get("success") is False
    ):
        reason = f"execution_failed:{int(return_code)}"
        write_manifest_event(
            manifest_file=manifest_file,
            state="incomplete",
            scenario_file=scenario_file,
            trial_index=trial_index,
            output_path=output_path,
            run_name=run_name,
            reason=reason,
            batch_id=batch_id,
            return_code=return_code,
        )
        return {"state": "incomplete", "success": False, "reason": reason}

    if _is_infrastructure_error_payload(payload):
        write_manifest_event(
            manifest_file=manifest_file,
            state="incomplete",
            scenario_file=scenario_file,
            trial_index=trial_index,
            output_path=output_path,
            run_name=run_name,
            reason="infrastructure_error_payload",
            batch_id=batch_id,
            return_code=return_code,
        )
        return {
            "state": "incomplete",
            "success": False,
            "reason": "infrastructure_error_payload",
            "infrastructure_error": True,
        }

    scenario = _scenario_payload(scenario_file)
    payload["scenario"] = scenario
    _apply_final_answer_guards(payload)
    success = derive_success(payload)
    if success is None:
        write_manifest_event(
            manifest_file=manifest_file,
            state="incomplete",
            scenario_file=scenario_file,
            trial_index=trial_index,
            output_path=output_path,
            run_name=run_name,
            reason="success_not_derivable",
            batch_id=batch_id,
            return_code=return_code,
        )
        return {
            "state": "incomplete",
            "success": False,
            "reason": "success_not_derivable",
        }
    payload["success"] = success

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    original_hash = _sha256(output_path)
    temp_hash = _sha256(temp_output)
    temp_output.replace(output_path)
    divergent: bool | None
    if original_hash is None:
        divergent = None
    else:
        divergent = original_hash != temp_hash
    latency_seconds = float(end_epoch) - float(start_epoch)
    _upsert_latency_row(
        latency_file=latency_file,
        scenario_file=scenario_file,
        trial_index=trial_index,
        output_path=output_path,
        latency_seconds=latency_seconds,
    )
    state = "complete_success" if success else "complete_failure"
    write_manifest_event(
        manifest_file=manifest_file,
        state=state,
        scenario_file=scenario_file,
        trial_index=trial_index,
        output_path=output_path,
        run_name=run_name,
        reason="trial_executed",
        batch_id=batch_id,
        latency_seconds=latency_seconds,
        return_code=return_code,
        extra={
            "original_output_sha256": original_hash,
            "final_output_sha256": temp_hash,
            "divergent": divergent,
        },
    )
    return {"state": state, "success": success, "reason": "trial_executed"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("trial-status-shell")
    status.add_argument("--run-dir", required=True, type=Path)
    status.add_argument("--scenario-file", required=True, type=Path)
    status.add_argument("--trial-index", required=True, type=int)
    status.add_argument("--output-path", required=True, type=Path)
    status.add_argument("--latency-file", required=True, type=Path)
    status.add_argument("--require-latency", action="store_true")

    identity = sub.add_parser("validate-run-identity")
    identity.add_argument("--meta-file", required=True, type=Path)
    identity.add_argument("--run-dir", required=True, type=Path)
    identity.add_argument("--config-path", required=True, type=Path)
    identity.add_argument("--repo-root", required=True, type=Path)
    identity.add_argument("--gpu-type", required=True)
    identity.add_argument("--enable-smartgrid-servers", required=True)
    identity.add_argument("--trial-count", required=True, type=int)
    identity.add_argument("--aob-python-version", required=True)
    identity.add_argument("--aat-mcp-server-launch-mode", required=True)
    identity.add_argument("--aat-mcp-server-python-version", required=True)
    identity.add_argument("--runner-python-version", required=True)
    identity.add_argument("--scenario-set-name", required=True)
    identity.add_argument("--scenario-set-hash", required=True)
    identity.add_argument("--experiment-cell", required=True)
    identity.add_argument("--orchestration-mode", required=True)
    identity.add_argument("--mcp-mode", required=True)
    identity.add_argument("--model-id", required=True)
    identity.add_argument("--model-provider", required=True)
    identity.add_argument("--judge-model", default="")
    identity.add_argument("--context-window", required=True, type=int)
    identity.add_argument("--temperature", required=True, type=float)
    identity.add_argument("--max-tokens", required=True, type=int)
    identity.add_argument("--serving-stack", default="unknown")
    identity.add_argument("--quantization-mode", default="none")
    identity.add_argument("--launch-vllm", default="0")
    identity.add_argument("--vllm-model-path", default="models/Llama-3.1-8B-Instruct")
    identity.add_argument("--vllm-served-model-name", default="Llama-3.1-8B-Instruct")
    identity.add_argument("--vllm-dtype", default="float16")
    identity.add_argument("--vllm-generation-config", default="vllm")
    identity.add_argument("--vllm-tool-call-parser", default="llama3_json")
    identity.add_argument("--vllm-enable-auto-tool-choice", default="1")
    identity.add_argument("--vllm-extra-args", default="")
    identity.add_argument("--aat-parallel-tool-calls", default="false")
    identity.add_argument("--aat-tool-choice", default="")

    interpreter = sub.add_parser("interpreter-identity")
    interpreter.add_argument("--python", required=True, type=Path)
    interpreter.add_argument("--version", required=True)

    preserve = sub.add_parser("preserve-incomplete")
    preserve.add_argument("--output-path", required=True, type=Path)
    preserve.add_argument("--manifest-file", required=True, type=Path)
    preserve.add_argument("--scenario-file", required=True, type=Path)
    preserve.add_argument("--trial-index", required=True, type=int)
    preserve.add_argument("--run-name", required=True)
    preserve.add_argument("--batch-id", default="")
    preserve.add_argument("--reason", default="rerun_preserved_incomplete")

    finalize = sub.add_parser("finalize-trial-shell")
    finalize.add_argument("--scenario-file", required=True, type=Path)
    finalize.add_argument("--trial-index", required=True, type=int)
    finalize.add_argument("--temp-output", required=True, type=Path)
    finalize.add_argument("--output-path", required=True, type=Path)
    finalize.add_argument("--latency-file", required=True, type=Path)
    finalize.add_argument("--manifest-file", required=True, type=Path)
    finalize.add_argument("--run-name", required=True)
    finalize.add_argument("--batch-id", default="")
    finalize.add_argument("--start-epoch", required=True, type=float)
    finalize.add_argument("--end-epoch", required=True, type=float)
    finalize.add_argument("--return-code", required=True, type=int)

    event = sub.add_parser("manifest-event")
    event.add_argument("--manifest-file", required=True, type=Path)
    event.add_argument("--state", required=True)
    event.add_argument("--scenario-file", required=True, type=Path)
    event.add_argument("--trial-index", required=True, type=int)
    event.add_argument("--output-path", required=True, type=Path)
    event.add_argument("--run-name", required=True)
    event.add_argument("--reason", default="")
    event.add_argument("--batch-id", default="")

    validate = sub.add_parser("validate-run-shell")
    validate.add_argument("--run-dir", required=True, type=Path)
    validate.add_argument("--scenario-glob", required=True)
    validate.add_argument("--trials", required=True, type=int)
    validate.add_argument("--run-name", default="")
    validate.add_argument("--require-latency", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "interpreter-identity":
        try:
            payload = interpreter_identity(args.python, args.version)
        except ValueError as exc:
            print(f"gcp_resume_state: ERROR: {exc}", file=sys.stderr)
            raise SystemExit(2)
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return
    if args.command == "validate-run-identity":
        try:
            expected = build_run_identity(
                config_path=args.config_path,
                repo_root=args.repo_root,
                gpu_type=args.gpu_type,
                enable_smartgrid_servers=args.enable_smartgrid_servers,
                scenario_set_name=args.scenario_set_name,
                scenario_set_hash=args.scenario_set_hash,
                trial_count=args.trial_count,
                aob_python_version=args.aob_python_version,
                aat_mcp_server_launch_mode=args.aat_mcp_server_launch_mode,
                aat_mcp_server_python_version=args.aat_mcp_server_python_version,
                runner_python_version=args.runner_python_version,
                experiment_cell=args.experiment_cell,
                orchestration_mode=args.orchestration_mode,
                mcp_mode=args.mcp_mode,
                model_id=args.model_id,
                model_provider=args.model_provider,
                judge_model=args.judge_model,
                context_window=args.context_window,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                serving_stack=args.serving_stack,
                quantization_mode=args.quantization_mode,
                launch_vllm=args.launch_vllm,
                vllm_model_path=args.vllm_model_path,
                vllm_served_model_name=args.vllm_served_model_name,
                vllm_dtype=args.vllm_dtype,
                vllm_generation_config=args.vllm_generation_config,
                vllm_tool_call_parser=args.vllm_tool_call_parser,
                vllm_enable_auto_tool_choice=args.vllm_enable_auto_tool_choice,
                vllm_extra_args=args.vllm_extra_args,
                aat_parallel_tool_calls=args.aat_parallel_tool_calls,
                aat_tool_choice=args.aat_tool_choice or None,
                missing_evidence_guard=os.environ.get(
                    "ENABLE_MISSING_EVIDENCE_GUARD", "0"
                ),
                missing_evidence_repair=os.environ.get(
                    "ENABLE_MISSING_EVIDENCE_REPAIR", "0"
                ),
                explicit_fault_risk_adjudication=os.environ.get(
                    "ENABLE_EXPLICIT_FAULT_RISK_ADJUDICATION", "0"
                ),
                missing_evidence_repair_max_attempts=os.environ.get(
                    "MISSING_EVIDENCE_REPAIR_MAX_ATTEMPTS", "2"
                ),
                missing_evidence_repair_max_attempts_per_target=os.environ.get(
                    "MISSING_EVIDENCE_REPAIR_MAX_ATTEMPTS_PER_TARGET", "1"
                ),
                enable_self_ask=os.environ.get("ENABLE_SELF_ASK", "0"),
                plan_execute_repo_local=os.environ.get("PLAN_EXECUTE_REPO_LOCAL", "0"),
                aat_openai_agents_version=os.environ.get(
                    "AAT_OPENAI_AGENTS_VERSION", "0.14.5"
                ),
                aat_mcp_version=os.environ.get("AAT_MCP_VERSION", "1.27.0"),
                aat_litellm_version=os.environ.get("AAT_LITELLM_VERSION", "1.81.13"),
                aat_mcp_client_timeout_seconds=os.environ.get(
                    "AAT_MCP_CLIENT_TIMEOUT_SECONDS", "30"
                ),
                aat_runner_template=os.environ.get("AAT_RUNNER_TEMPLATE", ""),
                hybrid_runner_template=os.environ.get("HYBRID_RUNNER_TEMPLATE", ""),
                verified_pe_runner_template=os.environ.get(
                    "VERIFIED_PE_RUNNER_TEMPLATE", ""
                ),
                server_iot_path=os.environ.get("SERVER_IOT_PATH", ""),
                server_fmsr_path=os.environ.get("SERVER_FMSR_PATH", ""),
                server_tsfm_path=os.environ.get("SERVER_TSFM_PATH", ""),
                server_wo_path=os.environ.get("SERVER_WO_PATH", ""),
            )
            validate_resume_identity(
                meta_file=args.meta_file,
                run_dir=args.run_dir,
                expected=expected,
            )
        except ValueError as exc:
            print(f"gcp_resume_state: ERROR: {exc}", file=sys.stderr)
            raise SystemExit(2)
        return
    if args.command == "trial-status-shell":
        result = classify_trial(
            run_dir=args.run_dir,
            scenario_file=args.scenario_file,
            trial_index=args.trial_index,
            output_path=args.output_path,
            latency_file=args.latency_file,
            require_latency=args.require_latency,
        )
        _emit_shell(
            {
                "RESUME_STATE": result["state"],
                "RESUME_COMPLETE": _shell_bool(bool(result["complete"])),
                "RESUME_SUCCESS": _shell_bool(bool(result["success"])),
                "RESUME_OUTPUT_PATH": result["output_path"],
                "RESUME_REASON": result["reason"],
            }
        )
        return
    if args.command == "preserve-incomplete":
        preserved = preserve_incomplete_output(args.output_path)
        if preserved is not None:
            write_manifest_event(
                manifest_file=args.manifest_file,
                state="incomplete",
                scenario_file=args.scenario_file,
                trial_index=args.trial_index,
                output_path=preserved,
                run_name=args.run_name,
                reason=args.reason,
                batch_id=args.batch_id,
            )
        return
    if args.command == "finalize-trial-shell":
        result = finalize_trial(
            scenario_file=args.scenario_file,
            trial_index=args.trial_index,
            temp_output=args.temp_output,
            output_path=args.output_path,
            latency_file=args.latency_file,
            manifest_file=args.manifest_file,
            run_name=args.run_name,
            batch_id=args.batch_id,
            start_epoch=args.start_epoch,
            end_epoch=args.end_epoch,
            return_code=args.return_code,
        )
        _emit_shell(
            {
                "FINAL_STATE": result["state"],
                "FINAL_SUCCESS": _shell_bool(bool(result["success"])),
                "FINAL_REASON": result["reason"],
            }
        )
        if result["state"] == "incomplete":
            raise SystemExit(1)
        return
    if args.command == "manifest-event":
        write_manifest_event(
            manifest_file=args.manifest_file,
            state=args.state,
            scenario_file=args.scenario_file,
            trial_index=args.trial_index,
            output_path=args.output_path,
            run_name=args.run_name,
            reason=args.reason,
            batch_id=args.batch_id,
        )
        return
    if args.command == "validate-run-shell":
        result = validate_run_artifacts(
            run_dir=args.run_dir,
            scenario_glob=args.scenario_glob,
            trials=args.trials,
            run_name=args.run_name,
            require_latency=args.require_latency,
        )
        missing_sample = result["missing"][:5]
        _emit_shell(
            {
                "VALIDATION_PASSED": _shell_bool(bool(result["valid"])),
                "VALIDATION_EXPECTED": result["expected"],
                "VALIDATION_COMPLETE": result["complete"],
                "VALIDATION_SUCCESS": result["success"],
                "VALIDATION_FAILURE": result["failure"],
                "VALIDATION_MISSING": len(result["missing"]),
                "VALIDATION_REASON": result["reason"],
                "VALIDATION_MISSING_SAMPLE": json.dumps(
                    missing_sample, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        if not result["valid"]:
            raise SystemExit(1)
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"gcp_resume_state: ERROR: {exc}", file=sys.stderr)
        raise
