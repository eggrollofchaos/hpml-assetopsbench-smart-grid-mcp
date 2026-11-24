from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import pytest
from datetime import datetime, timezone
from unittest.mock import patch
from pathlib import Path

from scripts import gcp_resume_state


def _scenario(path: Path, scenario_id: str = "SGT-001") -> Path:
    path.write_text(
        json.dumps({"id": scenario_id, "text": "test prompt"}) + "\n",
        encoding="utf-8",
    )
    return path


def _trial(path: Path, *, success: bool | None = True, answer: str = "done") -> Path:
    payload = {"answer": answer, "history": []}
    if success is not None:
        payload["success"] = success
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _latency(
    path: Path, scenario_file: Path, trial_index: int, output_path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "scenario_file": scenario_file.as_posix(),
                    "trial_index": trial_index,
                    "latency_seconds": 1.25,
                    "output_path": output_path.as_posix(),
                }
            )
            + "\n"
        )


def _classify(
    tmp_path: Path,
    scenario_file: Path,
    output_path: Path,
    *,
    require_latency: bool = True,
) -> dict:
    return gcp_resume_state.classify_trial(
        run_dir=tmp_path,
        scenario_file=scenario_file,
        trial_index=1,
        output_path=output_path,
        latency_file=tmp_path / "latencies.jsonl",
        require_latency=require_latency,
    )


def _identity(
    tmp_path: Path,
    *,
    model_id: str = "model-a",
    scenario_hash: str = "hash-a",
    repo_root: Path | None = None,
    gpu_type: str = "Test GPU",
    enable_smartgrid_servers: str | bool = "1",
    trial_count: int = 1,
    aob_python_version: str = "Python 3.12.11",
    aat_mcp_server_launch_mode: str = "python",
    aat_mcp_server_python_version: str = "Python 3.12.11",
    runner_python_version: str = "Python 3.12.11",
    orchestration_mode: str = "hybrid",
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
    server_iot_path: str | None = None,
    server_fmsr_path: str | None = None,
    server_tsfm_path: str | None = None,
    server_wo_path: str | None = None,
) -> dict:
    config_path = tmp_path / "config.env"
    config_path.write_text("MODEL_ID=" + model_id + "\n", encoding="utf-8")
    return gcp_resume_state.build_run_identity(
        config_path=config_path,
        repo_root=repo_root or Path(__file__).resolve().parent.parent,
        gpu_type=gpu_type,
        enable_smartgrid_servers=enable_smartgrid_servers,
        aob_python_version=aob_python_version,
        aat_mcp_server_launch_mode=aat_mcp_server_launch_mode,
        aat_mcp_server_python_version=aat_mcp_server_python_version,
        runner_python_version=runner_python_version,
        aat_parallel_tool_calls=aat_parallel_tool_calls,
        aat_tool_choice=aat_tool_choice,
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
        server_iot_path=server_iot_path
        or str(
            Path(__file__).resolve().parents[1] / "mcp_servers/iot_server/server.py"
        ),
        server_fmsr_path=server_fmsr_path
        or str(
            Path(__file__).resolve().parents[1] / "mcp_servers/fmsr_server/server.py"
        ),
        server_tsfm_path=server_tsfm_path
        or str(
            Path(__file__).resolve().parents[1] / "mcp_servers/tsfm_server/server.py"
        ),
        server_wo_path=server_wo_path
        or str(Path(__file__).resolve().parents[1] / "mcp_servers/wo_server/server.py"),
        scenario_set_name="set-a",
        scenario_set_hash=scenario_hash,
        trial_count=trial_count,
        experiment_cell="Y",
        orchestration_mode=orchestration_mode,
        mcp_mode="baseline",
        model_id=model_id,
        model_provider="local",
        judge_model=None,
        context_window=32768,
        temperature=0.0,
        max_tokens=0,
    )


def test_behavior_identity_normalizes_effective_controls_and_hashes_template(
    tmp_path: Path,
) -> None:
    identity = _identity(
        tmp_path,
        orchestration_mode="hybrid",
        missing_evidence_guard=" YES ",
        missing_evidence_repair="true",
        explicit_fault_risk_adjudication="on",
        missing_evidence_repair_max_attempts=" 3 ",
        missing_evidence_repair_max_attempts_per_target="2",
        enable_self_ask="1",
        plan_execute_repo_local="1",
        hybrid_runner_template="echo SECRET_TEMPLATE_BODY",
    )
    assert identity["missing_evidence_guard"] is True
    assert identity["missing_evidence_repair"] is True
    assert identity["explicit_fault_risk_adjudication"] is True
    assert identity["missing_evidence_repair_max_attempts"] == 3
    assert identity["missing_evidence_repair_max_attempts_per_target"] == 2
    assert identity["enable_self_ask"] is True
    assert identity["plan_execute_repo_local"] is True
    assert (
        identity["runner_template_sha256"]
        == hashlib.sha256(b"echo SECRET_TEMPLATE_BODY").hexdigest()
    )
    assert "SECRET_TEMPLATE_BODY" not in json.dumps(identity)
    assert identity["aat_openai_agents_version"] is None


def test_behavior_identity_uses_zero_retries_when_repair_disabled(
    tmp_path: Path,
) -> None:
    identity = _identity(
        tmp_path,
        missing_evidence_repair="0",
        missing_evidence_repair_max_attempts="invalid-but-unused",
        missing_evidence_repair_max_attempts_per_target="0",
    )
    assert identity["missing_evidence_repair_max_attempts"] == 0
    assert identity["missing_evidence_repair_max_attempts_per_target"] == 0


def test_interpreter_identity_binds_bytes_and_version_without_host_path(
    tmp_path: Path,
) -> None:
    first = tmp_path / "python-a"
    relocated = tmp_path / "python-b"
    changed = tmp_path / "python-c"
    shim = b"#!/bin/sh\nprintf 'Python 3.12.1\\n'\n"
    first.write_bytes(shim)
    relocated.write_bytes(shim)
    changed.write_bytes(b"#!/bin/sh\nprintf 'Python 3.12.2\\n'\n")
    for path in (first, relocated, changed):
        path.chmod(0o755)

    first_identity = gcp_resume_state.interpreter_identity(
        first, subprocess.check_output([str(first), "--version"], text=True).strip()
    )
    relocated_identity = gcp_resume_state.interpreter_identity(
        relocated,
        subprocess.check_output([str(relocated), "--version"], text=True).strip(),
    )
    changed_identity = gcp_resume_state.interpreter_identity(
        changed,
        subprocess.check_output([str(changed), "--version"], text=True).strip(),
    )

    assert first_identity == relocated_identity
    assert first_identity["binary_sha256"] != changed_identity["binary_sha256"]
    assert first_identity["version"] != changed_identity["version"]


def test_resume_refuses_changed_interpreter_identity_before_reuse(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    meta_file = run_dir / "meta.json"
    with patch.dict(os.environ, {"RUNNER_PYTHON_IDENTITY": '{"binary_sha256":"one"}'}):
        original = _identity(tmp_path)
    meta_file.write_text(json.dumps({"run_identity": original}) + "\n")
    with patch.dict(os.environ, {"RUNNER_PYTHON_IDENTITY": '{"binary_sha256":"two"}'}):
        changed = _identity(tmp_path)

    try:
        gcp_resume_state.validate_resume_identity(
            meta_file=meta_file, run_dir=run_dir, expected=changed
        )
    except ValueError as exc:
        assert "runner_interpreter_identity" in str(exc)
    else:
        raise AssertionError("changed interpreter identity was accepted")


def test_behavior_identity_binds_selected_aat_runtime_versions(
    tmp_path: Path,
) -> None:
    identity = _identity(
        tmp_path,
        orchestration_mode="agent_as_tool",
        aat_openai_agents_version="0.15.0",
        aat_mcp_version="1.28.0",
        aat_litellm_version="1.82.0",
        aat_mcp_client_timeout_seconds="45.0",
        aat_runner_template="python custom_runner.py",
    )
    assert identity["aat_openai_agents_version"] == "0.15.0"
    assert identity["aat_mcp_version"] == "1.28.0"
    assert identity["aat_litellm_version"] == "1.82.0"
    assert identity["aat_mcp_client_timeout_seconds"] == 45.0
    assert (
        identity["runner_template_sha256"]
        == hashlib.sha256(b"python custom_runner.py").hexdigest()
    )


def test_resume_identity_requires_new_behavior_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "output.json").write_text("old\n", encoding="utf-8")
    identity = _identity(tmp_path)
    identity.pop("runner_template_sha256")
    meta_file = run_dir / "meta.json"
    before = json.dumps({"run_identity": identity}) + "\n"
    meta_file.write_text(before, encoding="utf-8")

    try:
        gcp_resume_state.validate_resume_identity(
            meta_file=meta_file,
            run_dir=run_dir,
            expected=_identity(tmp_path),
        )
    except ValueError as exc:
        assert "runner_template_sha256" in str(exc)
    else:
        raise AssertionError("old identity without the new field was accepted")
    assert meta_file.read_text(encoding="utf-8") == before


def test_resume_identity_keeps_git_sha_as_provenance_not_equality(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "output.json").write_text("old\n", encoding="utf-8")
    meta_file = run_dir / "meta.json"
    with patch.object(gcp_resume_state, "_git_revision", return_value="commit-a"):
        stored = _identity(tmp_path)
    meta_file.write_text(json.dumps({"run_identity": stored}) + "\n")
    with patch.object(gcp_resume_state, "_git_revision", return_value="commit-b"):
        current = _identity(tmp_path)

    assert stored["git_sha"] == "commit-a"
    assert current["git_sha"] == "commit-b"
    assert gcp_resume_state.validate_resume_identity(
        meta_file=meta_file,
        run_dir=run_dir,
        expected=current,
    ) == {"validated": True, "reason": "exact_identity"}


def test_resume_identity_rejects_each_changed_behavior_control(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "output.json").write_text("old\n", encoding="utf-8")
    meta_file = run_dir / "meta.json"
    cases = [
        ({"missing_evidence_guard": "1"}, "missing_evidence_guard"),
        (
            {"missing_evidence_repair": "1", "missing_evidence_guard": "1"},
            "missing_evidence_repair",
        ),
        (
            {"explicit_fault_risk_adjudication": "true"},
            "explicit_fault_risk_adjudication",
        ),
        (
            {
                "missing_evidence_repair": "1",
                "missing_evidence_guard": "1",
                "missing_evidence_repair_max_attempts": "3",
            },
            "missing_evidence_repair_max_attempts",
        ),
        (
            {
                "missing_evidence_repair": "1",
                "missing_evidence_guard": "1",
                "missing_evidence_repair_max_attempts_per_target": "2",
            },
            "missing_evidence_repair_max_attempts_per_target",
        ),
        ({"enable_self_ask": "1"}, "enable_self_ask"),
        ({"plan_execute_repo_local": "1"}, "plan_execute_repo_local"),
        ({"hybrid_runner_template": "echo new"}, "runner_template_sha256"),
        (
            {
                "orchestration_mode": "agent_as_tool",
                "aat_openai_agents_version": "0.15.0",
            },
            "aat_openai_agents_version",
        ),
        (
            {"orchestration_mode": "agent_as_tool", "aat_mcp_version": "1.28.0"},
            "aat_mcp_version",
        ),
        (
            {
                "orchestration_mode": "agent_as_tool",
                "aat_litellm_version": "1.82.0",
            },
            "aat_litellm_version",
        ),
        (
            {
                "orchestration_mode": "agent_as_tool",
                "aat_mcp_client_timeout_seconds": "45",
            },
            "aat_mcp_client_timeout_seconds",
        ),
    ]
    for changes, field in cases:
        stored = _identity(tmp_path, **changes)
        baseline = _identity(tmp_path)
        if changes.get("orchestration_mode") == "agent_as_tool":
            baseline = _identity(tmp_path, orchestration_mode="agent_as_tool")
        meta_file.write_text(json.dumps({"run_identity": baseline}) + "\n")
        try:
            gcp_resume_state.validate_resume_identity(
                meta_file=meta_file,
                run_dir=run_dir,
                expected=stored,
            )
        except ValueError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"changed behavior control {field} was accepted")


def test_selected_server_override_identity_hashes_bytes_not_paths(
    tmp_path: Path,
) -> None:
    server_a = tmp_path / "server-a.py"
    server_b = tmp_path / "server-b.py"
    server_changed = tmp_path / "server-changed.py"
    server_a.write_bytes(b"same server bytes")
    server_b.write_bytes(b"same server bytes")
    server_changed.write_bytes(b"changed server bytes")
    first = _identity(
        tmp_path,
        orchestration_mode="hybrid",
        hybrid_runner_template="external hybrid template",
        server_iot_path=str(server_a),
        server_fmsr_path=str(server_a),
        server_tsfm_path=str(server_a),
        server_wo_path=str(server_a),
    )
    relocated = _identity(
        tmp_path,
        orchestration_mode="hybrid",
        hybrid_runner_template="external hybrid template",
        server_iot_path=str(server_b),
        server_fmsr_path=str(server_b),
        server_tsfm_path=str(server_b),
        server_wo_path=str(server_b),
    )
    changed = _identity(
        tmp_path,
        orchestration_mode="hybrid",
        hybrid_runner_template="external hybrid template",
        server_iot_path=str(server_changed),
        server_fmsr_path=str(server_a),
        server_tsfm_path=str(server_a),
        server_wo_path=str(server_a),
    )
    inactive = _identity(
        tmp_path,
        orchestration_mode="agent_as_tool",
        server_iot_path=str(server_changed),
        server_fmsr_path=str(server_changed),
        server_tsfm_path=str(server_changed),
        server_wo_path=str(server_changed),
    )
    inactive_default = _identity(tmp_path, orchestration_mode="agent_as_tool")

    keys = [key for key in first if key.startswith("server_")]
    assert [first[key] for key in keys] == [relocated[key] for key in keys]
    assert first["server_iot_sha256"] != changed["server_iot_sha256"]
    assert all(inactive[key] is None for key in keys)
    assert [inactive[key] for key in keys] == [inactive_default[key] for key in keys]


@pytest.mark.parametrize(
    ("mode", "kwargs", "template_kwargs"),
    [
        (
            "agent_as_tool",
            {"enable_smartgrid_servers": "0"},
            {"aat_runner_template": "external AAT template"},
        ),
        (
            "verified_pe",
            {"enable_smartgrid_servers": "0"},
            {},
        ),
        (
            "plan_execute",
            {"enable_smartgrid_servers": "0", "enable_self_ask": "1"},
            {},
        ),
        (
            "plan_execute",
            {"enable_smartgrid_servers": "0", "plan_execute_repo_local": "1"},
            {},
        ),
    ],
)
def test_actual_server_consumers_bind_override_bytes_when_flag_is_disabled(
    tmp_path: Path,
    mode: str,
    kwargs: dict[str, str],
    template_kwargs: dict[str, str],
) -> None:
    server_a = tmp_path / "server-a.py"
    server_b = tmp_path / "server-b.py"
    server_changed = tmp_path / "server-changed.py"
    server_a.write_bytes(b"same selected server bytes")
    server_b.write_bytes(b"same selected server bytes")
    server_changed.write_bytes(b"different selected server bytes")

    def identity(source: Path) -> dict:
        return _identity(
            tmp_path,
            orchestration_mode=mode,
            server_iot_path=str(source),
            server_fmsr_path=str(source),
            server_tsfm_path=str(source),
            server_wo_path=str(source),
            **kwargs,
            **template_kwargs,
        )

    original = identity(server_a)
    relocated = identity(server_b)
    changed = identity(server_changed)
    fields = [key for key in original if key.startswith("server_")]
    assert all(original[field] is not None for field in fields)
    assert [original[field] for field in fields] == [
        relocated[field] for field in fields
    ]
    assert original["server_iot_sha256"] != changed["server_iot_sha256"]


def test_plan_execute_server_override_identity_matches_runtime_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.orchestration_utils import effective_server_paths

    server = tmp_path / "selected-iot.py"
    server.write_bytes(b"selected runtime server")
    monkeypatch.setenv("SERVER_IOT_PATH", str(server))
    runtime_paths = effective_server_paths([], tmp_path)
    identity = _identity(
        tmp_path,
        orchestration_mode="plan_execute",
        enable_smartgrid_servers="0",
        enable_self_ask="1",
        server_iot_path=str(server),
        server_fmsr_path=str(server),
        server_tsfm_path=str(server),
        server_wo_path=str(server),
    )
    assert runtime_paths["iot"] == server.resolve()
    assert (
        identity["server_iot_sha256"] == hashlib.sha256(server.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("mode", ["agent_as_tool", "plan_execute"])
def test_inactive_server_overrides_keep_null_identity_sentinel(
    tmp_path: Path, mode: str
) -> None:
    server = tmp_path / "override.py"
    server.write_bytes(b"not consumed by this native route")
    identity = _identity(
        tmp_path,
        orchestration_mode=mode,
        enable_smartgrid_servers="0",
        server_iot_path=str(server),
        server_fmsr_path=str(server),
        server_tsfm_path=str(server),
        server_wo_path=str(server),
    )
    assert all(
        value is None for key, value in identity.items() if key.startswith("server_")
    )


def test_runner_code_hash_binds_outcome_modules(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for relative in gcp_resume_state._RUNNER_CODE_FILES:
        source = repo_root / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    before = gcp_resume_state._runner_code_sha256(tmp_path)
    target = tmp_path / "scripts" / "mitigation_guards.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# dirty\n")

    assert before is not None
    assert gcp_resume_state._runner_code_sha256(tmp_path) != before


def test_run_identity_records_dirty_worktree_state(tmp_path: Path) -> None:
    with patch.object(gcp_resume_state, "_git_dirty", return_value=False):
        clean = _identity(tmp_path)
    with patch.object(gcp_resume_state, "_git_dirty", return_value=True):
        dirty = _identity(tmp_path)

    assert clean["git_dirty"] is False
    assert dirty["git_dirty"] is True
    assert clean["runner_code_sha256"] == dirty["runner_code_sha256"]


def test_resume_binds_shared_code_and_processed_data_across_nodes(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    original = _identity(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    meta_file = run_dir / "meta.json"
    shell = (repo_root / "scripts/run_experiment.sh").read_text()
    writer = shell.split("\"$JUDGE_MODEL\" <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(tmp_path / "written-config.json"),
            str(tmp_path / "summary.json"),
            str(meta_file),
            str(tmp_path / "config.env"),
            "test-run",
            "",
            "",
            "",
            "Y",
            "hybrid",
            "baseline",
            "1",
            "1",
            "set-a",
            "hash-a",
            "smart_grid",
            "model-a",
            "local",
            "unknown",
            "none",
            "32768",
            "0.0",
            "0",
            "",
        ],
        input=writer,
        cwd=repo_root,
        env={
            "PATH": os.environ["PATH"],
            "GPU_TYPE": "Test GPU",
            "ENABLE_SMARTGRID_SERVERS": "1",
            "SERVER_IOT_PATH": str(repo_root / "mcp_servers/iot_server/server.py"),
            "SERVER_FMSR_PATH": str(repo_root / "mcp_servers/fmsr_server/server.py"),
            "SERVER_TSFM_PATH": str(repo_root / "mcp_servers/tsfm_server/server.py"),
            "SERVER_WO_PATH": str(repo_root / "mcp_servers/wo_server/server.py"),
            "AOB_PYTHON_VERSION": original["aob_python_version"],
            "RUNNER_PYTHON_VERSION": original["runner_python_version"],
            "AAT_MCP_SERVER_LAUNCH_MODE": original["aat_mcp_server_launch_mode"],
            "AAT_MCP_SERVER_PYTHON_VERSION": original["aat_mcp_server_python_version"],
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    metadata = meta_file.read_bytes()
    stored = json.loads(metadata)["run_identity"]
    assert stored == {
        field: original[field] for field in gcp_resume_state._RUN_IDENTITY_FIELDS
    }

    new_node = tmp_path / "new-node"
    csv_paths = sorted((repo_root / "data/processed").glob("*.csv"))
    relatives = list(gcp_resume_state._RUNNER_CODE_FILES) + [
        path.relative_to(repo_root).as_posix() for path in csv_paths
    ]
    for relative in relatives:
        destination = new_node / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)

    with (
        patch.object(gcp_resume_state, "_git_revision", return_value=stored["git_sha"]),
        patch.object(gcp_resume_state, "_git_dirty", return_value=True),
    ):
        current = _identity(tmp_path, repo_root=new_node)
        assert gcp_resume_state.validate_resume_identity(
            meta_file=meta_file,
            run_dir=run_dir,
            expected=current,
        ) == {"validated": True, "reason": "exact_identity"}
        for relative, field in [
            ("mcp_servers/base.py", "runner_code_sha256"),
            *(
                (path.relative_to(repo_root).as_posix(), "processed_data_sha256")
                for path in csv_paths
            ),
        ]:
            target = new_node / relative
            before = target.read_bytes()
            target.write_bytes(before + b"\n# changed bytes\n")
            try:
                gcp_resume_state.validate_resume_identity(
                    meta_file=meta_file,
                    run_dir=run_dir,
                    expected=_identity(tmp_path, repo_root=new_node),
                )
            except ValueError as exc:
                assert field in str(exc)
            else:
                raise AssertionError(f"changed {relative} was accepted")
            finally:
                target.write_bytes(before)
    assert meta_file.read_bytes() == metadata


def test_resume_identity_rejects_changed_or_unknown_gpu(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "old.json").write_text("old evidence\n", encoding="utf-8")
    meta_file = run_dir / "meta.json"
    stored = _identity(tmp_path, gpu_type="NVIDIA A100")
    meta_file.write_text(json.dumps({"run_identity": stored}) + "\n", encoding="utf-8")

    changed = _identity(tmp_path, gpu_type="NVIDIA A6000")
    try:
        gcp_resume_state.validate_resume_identity(
            meta_file=meta_file,
            run_dir=run_dir,
            expected=changed,
        )
    except ValueError as exc:
        assert "gpu_type" in str(exc)
    else:
        raise AssertionError("changed GPU identity was accepted")

    unknown = _identity(tmp_path, gpu_type="unknown")
    try:
        gcp_resume_state.validate_resume_identity(
            meta_file=meta_file,
            run_dir=run_dir,
            expected=unknown,
        )
    except ValueError as exc:
        assert "hardware identity unavailable" in str(exc)
    else:
        raise AssertionError("unknown GPU identity was accepted")

    assert (
        meta_file.read_text(encoding="utf-8")
        == json.dumps({"run_identity": stored}) + "\n"
    )


def test_resume_identity_rejects_changed_effective_python_runtime(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "old.json").write_text("old evidence\n", encoding="utf-8")
    meta_file = run_dir / "meta.json"
    stored = _identity(
        tmp_path,
        aob_python_version="Python 3.11.9",
        aat_mcp_server_python_version="Python 3.11.9",
        runner_python_version="Python 3.11.9",
    )
    meta_file.write_text(json.dumps({"run_identity": stored}) + "\n", encoding="utf-8")

    changed = _identity(tmp_path)
    try:
        gcp_resume_state.validate_resume_identity(
            meta_file=meta_file,
            run_dir=run_dir,
            expected=changed,
        )
    except ValueError as exc:
        assert "aob_python_version" in str(exc)
    else:
        raise AssertionError("changed effective Python runtime was accepted")


def test_resume_identity_rejects_changed_effective_aat_tool_controls(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "old.json").write_text("old evidence\n", encoding="utf-8")
    stored = _identity(
        tmp_path,
        orchestration_mode="agent_as_tool",
        aat_parallel_tool_calls="false",
        aat_tool_choice="auto",
    )
    changed_parallel = _identity(
        tmp_path,
        orchestration_mode="agent_as_tool",
        aat_parallel_tool_calls="true",
        aat_tool_choice="auto",
    )
    changed_choice = _identity(
        tmp_path,
        orchestration_mode="agent_as_tool",
        aat_parallel_tool_calls="false",
        aat_tool_choice="required",
    )

    assert stored["aat_parallel_tool_calls"] is False
    assert stored["aat_tool_choice"] == "auto"
    meta_file = run_dir / "meta.json"
    meta_file.write_text(json.dumps({"run_identity": stored}) + "\n", encoding="utf-8")
    for expected, field in [
        (changed_parallel, "aat_parallel_tool_calls"),
        (changed_choice, "aat_tool_choice"),
    ]:
        try:
            gcp_resume_state.validate_resume_identity(
                meta_file=meta_file,
                run_dir=run_dir,
                expected=expected,
            )
        except ValueError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"changed {field} was accepted")


def test_run_identity_uses_exact_smartgrid_server_flag_values(tmp_path: Path) -> None:
    assert gcp_resume_state._shell_flag_enabled("0") is False
    assert gcp_resume_state._shell_flag_enabled("1") is True
    assert gcp_resume_state._shell_flag_enabled("1 ") is False

    identity = _identity(tmp_path, enable_smartgrid_servers="1 ")
    assert identity["enable_smartgrid_servers"] is False


def test_completed_success_is_terminal_with_latency(tmp_path: Path) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    output = _trial(tmp_path / "2026-05-03_Y_multi_01_run01.json", success=True)
    _latency(tmp_path / "latencies.jsonl", scenario_file, 1, output)

    result = _classify(tmp_path, scenario_file, output)

    assert result["state"] == "complete_success"
    assert result["complete"] is True
    assert result["success"] is True


def test_completed_failure_is_terminal_and_skippable(tmp_path: Path) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    output = _trial(tmp_path / "2026-05-03_Y_multi_01_run01.json", success=False)
    _latency(tmp_path / "latencies.jsonl", scenario_file, 1, output)

    result = _classify(tmp_path, scenario_file, output)

    assert result["state"] == "complete_failure"
    assert result["complete"] is True
    assert result["success"] is False


def test_preserve_incomplete_output_uses_no_clobber_unique_names(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 11, 14, 9, 30, tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_time if tz else fixed_time.replace(tzinfo=None)

    output = tmp_path / "trial.json"
    output.write_bytes(b"FIRST-SYNTHETIC-EVIDENCE\n")
    sentinel = tmp_path / "sentinel.json.incomplete.20261114T093000Z"
    sentinel.write_bytes(b"PROTECTED-SENTINEL\n")

    with patch.object(gcp_resume_state, "datetime", FixedDatetime):
        first = gcp_resume_state.preserve_incomplete_output(output)
        assert first is not None
        output.write_bytes(b"SECOND-SYNTHETIC-EVIDENCE\n")
        second = gcp_resume_state.preserve_incomplete_output(output)

        sentinel_source = tmp_path / "sentinel.json"
        sentinel_source.write_bytes(b"THIRD-SYNTHETIC-EVIDENCE\n")
        third = gcp_resume_state.preserve_incomplete_output(sentinel_source)

    assert first == tmp_path / "trial.json.incomplete.20261114T093000Z"
    assert second == tmp_path / "trial.json.incomplete.20261114T093000Z.1"
    assert first.read_bytes() == b"FIRST-SYNTHETIC-EVIDENCE\n"
    assert second is not None
    assert second.read_bytes() == b"SECOND-SYNTHETIC-EVIDENCE\n"
    assert third == tmp_path / "sentinel.json.incomplete.20261114T093000Z.1"
    assert third.read_bytes() == b"THIRD-SYNTHETIC-EVIDENCE\n"
    assert sentinel.read_bytes() == b"PROTECTED-SENTINEL\n"
    assert not output.exists()
    assert not sentinel_source.exists()


def test_infrastructure_error_payloads_are_incomplete(tmp_path: Path) -> None:
    """Runner exceptions rerun instead of becoming complete failures."""
    scenario_file = _scenario(tmp_path / "multi_01.json")
    run_dir = tmp_path / "run"
    payloads = [
        {"success": False, "error": "RuntimeError: MCP unavailable"},
        {
            "success": False,
            "error": "APIConnectionError: provider mentioned ModelBehaviorError",
        },
        {
            "success": False,
            "state": "infrastructure_error",
            "infrastructure_error": True,
            "error": "RuntimeError: MCP unavailable",
        },
    ]

    for index, payload in enumerate(payloads, start=1):
        output = run_dir / f"trial_{index}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        _latency(tmp_path / "latencies.jsonl", scenario_file, 1, output)

        result = _classify(tmp_path, scenario_file, output)

        assert result["state"] == "incomplete"
        assert result["complete"] is False
        assert "infrastructure_error_payload" in result["reason"]


def test_model_failure_payloads_are_terminal(tmp_path: Path) -> None:
    """Known SDK model failures publish as failed, skippable trials."""
    scenario_file = _scenario(tmp_path / "multi_01.json")
    run_dir = tmp_path / "run"
    payloads = [
        {
            "success": False,
            "state": "complete_failure",
            "model_error": True,
            "max_turns_exhausted": True,
            "error": "MaxTurnsExceeded: turn budget reached",
        },
        {"success": False, "error": "ModelBehaviorError: malformed model response"},
    ]

    for index, payload in enumerate(payloads, start=1):
        output = run_dir / f"trial_{index}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        _latency(tmp_path / "latencies.jsonl", scenario_file, index, output)

        result = gcp_resume_state.classify_trial(
            run_dir=tmp_path,
            scenario_file=scenario_file,
            trial_index=index,
            output_path=output,
            latency_file=tmp_path / "latencies.jsonl",
            require_latency=True,
        )

        assert result["state"] == "complete_failure"
        assert result["complete"] is True
        assert result["success"] is False


def test_finalize_trial_rejects_infrastructure_payload_before_guards(
    tmp_path: Path,
) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    temp_output = _trial(tmp_path / "run" / "trial.json.tmp", success=False)
    temp_output.write_text(
        json.dumps(
            {
                "success": False,
                "state": "infrastructure_error",
                "infrastructure_error": True,
                "error": "RuntimeError: MCP unavailable",
                "answer": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "run" / "trial.json"
    manifest_file = tmp_path / "run" / "resume_manifest.jsonl"

    result = gcp_resume_state.finalize_trial(
        scenario_file=scenario_file,
        trial_index=1,
        temp_output=temp_output,
        output_path=output,
        latency_file=tmp_path / "run" / "latencies.jsonl",
        manifest_file=manifest_file,
        run_name="resume-smoke",
        batch_id="batch-1",
        start_epoch=10.0,
        end_epoch=12.5,
        return_code=0,
    )

    assert result == {
        "state": "incomplete",
        "success": False,
        "reason": "infrastructure_error_payload",
        "infrastructure_error": True,
    }
    assert not output.exists()
    assert json.loads(manifest_file.read_text().splitlines()[-1])["reason"] == (
        "infrastructure_error_payload"
    )


def test_finalize_trial_fails_closed_when_mitigation_guards_cannot_import(
    tmp_path: Path,
) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    temp_output = _trial(tmp_path / "run" / "trial.json.tmp", success=True)
    output = tmp_path / "run" / "trial.json"
    latency_file = tmp_path / "run" / "latencies.jsonl"
    manifest_file = tmp_path / "run" / "resume_manifest.jsonl"

    with (
        patch.dict(
            os.environ,
            {
                "ENABLE_MISSING_EVIDENCE_GUARD": "1",
                "ENABLE_EXPLICIT_FAULT_RISK_ADJUDICATION": "0",
            },
        ),
        patch.dict(sys.modules, {"scripts.mitigation_guards": None}),
    ):
        try:
            gcp_resume_state.finalize_trial(
                scenario_file=scenario_file,
                trial_index=1,
                temp_output=temp_output,
                output_path=output,
                latency_file=latency_file,
                manifest_file=manifest_file,
                run_name="resume-smoke",
                batch_id="batch-1",
                start_epoch=10.0,
                end_epoch=12.5,
                return_code=0,
            )
        except ImportError:
            pass
        else:
            raise AssertionError("missing mitigation guards were silently ignored")

    assert not output.exists()
    assert temp_output.exists()
    assert not latency_file.exists()
    assert not manifest_file.exists()


def test_finalize_trial_publishes_terminal_model_failure_with_rc1(
    tmp_path: Path,
) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    temp_output = _trial(tmp_path / "run" / "trial.json.tmp", success=False)
    temp_output.write_text(
        json.dumps(
            {
                "success": False,
                "state": "complete_failure",
                "model_error": True,
                "max_turns_exhausted": True,
                "error": "MaxTurnsExceeded: turn budget reached",
                "answer": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "run" / "trial.json"
    manifest_file = tmp_path / "run" / "resume_manifest.jsonl"

    result = gcp_resume_state.finalize_trial(
        scenario_file=scenario_file,
        trial_index=1,
        temp_output=temp_output,
        output_path=output,
        latency_file=tmp_path / "run" / "latencies.jsonl",
        manifest_file=manifest_file,
        run_name="resume-smoke",
        batch_id="batch-1",
        start_epoch=10.0,
        end_epoch=12.5,
        return_code=1,
    )

    assert result == {
        "state": "complete_failure",
        "success": False,
        "reason": "trial_executed",
    }
    assert output.is_file()
    assert json.loads(output.read_text())["model_error"] is True
    assert json.loads(manifest_file.read_text().splitlines()[-1])["state"] == (
        "complete_failure"
    )


def test_resume_identity_rejects_changed_model_before_metadata_write(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    meta_file = run_dir / "meta.json"
    stored = _identity(tmp_path, model_id="model-a")
    meta_file.write_text(json.dumps({"run_identity": stored}) + "\n", encoding="utf-8")
    output = run_dir / "old.json"
    output.write_text("old evidence\n", encoding="utf-8")

    current = _identity(tmp_path, model_id="model-b")
    try:
        gcp_resume_state.validate_resume_identity(
            meta_file=meta_file,
            run_dir=run_dir,
            expected=current,
        )
    except ValueError as exc:
        assert "model_id" in str(exc)
    else:
        raise AssertionError("changed model identity was accepted")
    assert (
        meta_file.read_text(encoding="utf-8")
        == json.dumps({"run_identity": stored}) + "\n"
    )
    assert output.read_text(encoding="utf-8") == "old evidence\n"


def test_resume_identity_rejects_changed_scenario_content_hash(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    meta_file = run_dir / "meta.json"
    stored = _identity(tmp_path, scenario_hash="hash-a")
    meta_file.write_text(json.dumps({"run_identity": stored}) + "\n", encoding="utf-8")

    current = _identity(tmp_path, scenario_hash="hash-b")
    try:
        gcp_resume_state.validate_resume_identity(
            meta_file=meta_file,
            run_dir=run_dir,
            expected=current,
        )
    except ValueError as exc:
        assert "scenario_set_hash" in str(exc)
    else:
        raise AssertionError("changed scenario identity was accepted")


def test_missing_latency_keeps_valid_json_incomplete_by_default(tmp_path: Path) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    output = _trial(tmp_path / "2026-05-03_Y_multi_01_run01.json", success=True)

    result = _classify(tmp_path, scenario_file, output)

    assert result["state"] == "incomplete"
    assert "missing_latency" in result["reason"]


def test_invalid_json_is_incomplete(tmp_path: Path) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    output = tmp_path / "2026-05-03_Y_multi_01_run01.json"
    output.write_text("{not-json", encoding="utf-8")

    result = _classify(tmp_path, scenario_file, output, require_latency=False)

    assert result["state"] == "incomplete"
    assert "invalid_json" in result["reason"]


def test_legacy_stdout_next_to_terminal_json_is_not_dangling(tmp_path: Path) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    output = _trial(tmp_path / "legacy_multi_01_run01.json", success=True)
    output.with_suffix(output.suffix + ".stdout").write_text(
        "legacy log\n", encoding="utf-8"
    )
    _latency(tmp_path / "latencies.jsonl", scenario_file, 1, output)
    newer_name = tmp_path / "2026-05-04_Y_multi_01_run01.json"

    result = _classify(tmp_path, scenario_file, newer_name)

    assert result["state"] == "complete_success"
    assert result["output_path"] == output.as_posix()


def test_conflicting_duplicate_latency_rows_are_incomplete(tmp_path: Path) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    output = _trial(tmp_path / "2026-05-03_Y_multi_01_run01.json", success=True)
    rows = [
        {
            "scenario_file": scenario_file.as_posix(),
            "trial_index": 1,
            "latency_seconds": 1.0,
            "output_path": output.as_posix(),
        },
        {
            "scenario_file": scenario_file.as_posix(),
            "trial_index": 1,
            "latency_seconds": 2.0,
            "output_path": output.as_posix(),
        },
    ]
    (tmp_path / "latencies.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = _classify(tmp_path, scenario_file, output)

    assert result["state"] == "incomplete"
    assert "duplicate_conflicting_latency" in result["reason"]


def test_finalize_trial_writes_atomic_output_latency_and_manifest(
    tmp_path: Path,
) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    temp_output = _trial(tmp_path / "run" / "trial.json.tmp", success=None)
    output = tmp_path / "run" / "trial.json"
    latency_file = tmp_path / "run" / "latencies.jsonl"
    manifest_file = tmp_path / "run" / "resume_manifest.jsonl"

    result = gcp_resume_state.finalize_trial(
        scenario_file=scenario_file,
        trial_index=1,
        temp_output=temp_output,
        output_path=output,
        latency_file=latency_file,
        manifest_file=manifest_file,
        run_name="resume-smoke",
        batch_id="batch-1",
        start_epoch=10.0,
        end_epoch=12.5,
        return_code=0,
    )

    assert result["state"] == "complete_success"
    assert output.exists()
    assert not temp_output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario"]["id"] == "SGT-001"
    assert payload["success"] is True
    latency_rows = [json.loads(line) for line in latency_file.read_text().splitlines()]
    assert latency_rows == [
        {
            "latency_seconds": 2.5,
            "output_path": output.as_posix(),
            "scenario_file": scenario_file.as_posix(),
            "trial_index": 1,
        }
    ]
    manifest_rows = [
        json.loads(line)
        for line in manifest_file.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows[-1]["state"] == "complete_success"
    assert manifest_rows[-1]["batch_id"] == "batch-1"
    assert "runtime_versions" in manifest_rows[-1]


def test_finalize_trial_rejects_success_payload_when_child_failed(
    tmp_path: Path,
) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    temp_output = _trial(tmp_path / "run" / "trial.json.tmp", success=True)
    output = tmp_path / "run" / "trial.json"
    latency_file = tmp_path / "run" / "latencies.jsonl"
    manifest_file = tmp_path / "run" / "resume_manifest.jsonl"

    result = gcp_resume_state.finalize_trial(
        scenario_file=scenario_file,
        trial_index=1,
        temp_output=temp_output,
        output_path=output,
        latency_file=latency_file,
        manifest_file=manifest_file,
        run_name="resume-smoke",
        batch_id="batch-1",
        start_epoch=10.0,
        end_epoch=12.5,
        return_code=137,
    )

    assert result == {
        "state": "incomplete",
        "success": False,
        "reason": "execution_failed:137",
    }
    assert not output.exists()
    assert temp_output.exists()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8").splitlines()[-1])
    assert manifest["reason"] == "execution_failed:137"


def test_finalize_shell_returns_nonzero_for_child_failure(tmp_path: Path) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    temp_output = _trial(tmp_path / "run" / "trial.json.tmp", success=True)
    output = tmp_path / "run" / "trial.json"
    command = [
        sys.executable,
        str(Path(gcp_resume_state.__file__)),
        "finalize-trial-shell",
        "--scenario-file",
        str(scenario_file),
        "--trial-index",
        "1",
        "--temp-output",
        str(temp_output),
        "--output-path",
        str(output),
        "--latency-file",
        str(tmp_path / "run" / "latencies.jsonl"),
        "--manifest-file",
        str(tmp_path / "run" / "resume_manifest.jsonl"),
        "--run-name",
        "resume-smoke",
        "--start-epoch",
        "10",
        "--end-epoch",
        "12",
        "--return-code",
        "137",
    ]

    result = subprocess.run(command, text=True, capture_output=True, check=False)

    assert result.returncode == 1
    assert "FINAL_STATE=incomplete" in result.stdout
    assert not output.exists()


def test_trial_status_shell_malformed_scenario_is_nonzero(tmp_path: Path) -> None:
    scenario_file = tmp_path / "bad.json"
    scenario_file.write_text("not json\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(gcp_resume_state.__file__)),
            "trial-status-shell",
            "--run-dir",
            str(tmp_path),
            "--scenario-file",
            str(scenario_file),
            "--trial-index",
            "1",
            "--output-path",
            str(tmp_path / "trial.json"),
            "--latency-file",
            str(tmp_path / "latencies.jsonl"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0


def test_finalize_trial_marks_divergent_rerun(tmp_path: Path) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")
    output = tmp_path / "run" / "trial.json"
    latency_file = tmp_path / "run" / "latencies.jsonl"
    manifest_file = tmp_path / "run" / "resume_manifest.jsonl"
    first_temp = _trial(tmp_path / "run" / "trial-first.json.tmp", answer="first")
    second_temp = _trial(tmp_path / "run" / "trial-second.json.tmp", answer="second")

    gcp_resume_state.finalize_trial(
        scenario_file=scenario_file,
        trial_index=1,
        temp_output=first_temp,
        output_path=output,
        latency_file=latency_file,
        manifest_file=manifest_file,
        run_name="resume-smoke",
        batch_id="batch-1",
        start_epoch=10.0,
        end_epoch=12.5,
        return_code=0,
    )

    gcp_resume_state.finalize_trial(
        scenario_file=scenario_file,
        trial_index=1,
        temp_output=second_temp,
        output_path=output,
        latency_file=latency_file,
        manifest_file=manifest_file,
        run_name="resume-smoke",
        batch_id="batch-1",
        start_epoch=20.0,
        end_epoch=23.0,
        return_code=0,
    )

    manifest_rows = [
        json.loads(line)
        for line in manifest_file.read_text(encoding="utf-8").splitlines()
    ]
    rerun = manifest_rows[-1]
    assert rerun["divergent"] is True
    assert rerun["original_output_sha256"] != rerun["final_output_sha256"]


def test_validate_run_artifacts_flags_missing_outputs(tmp_path: Path) -> None:
    scenario_file = _scenario(tmp_path / "multi_01.json")

    result = gcp_resume_state.validate_run_artifacts(
        run_dir=tmp_path / "run",
        scenario_glob=scenario_file.as_posix(),
        trials=1,
        run_name="batch_Y",
        require_latency=True,
    )

    assert result["valid"] is False
    assert result["expected"] == 1
    assert result["complete"] == 0
    assert result["reason"] == "incomplete_trajectory_artifacts"
    assert result["missing"][0]["reason"] == "missing_json"


def test_validate_run_artifacts_counts_terminal_failures_as_complete(
    tmp_path: Path,
) -> None:
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    scenario_file = _scenario(scenarios_dir / "multi_01.json")
    run_dir = tmp_path / "run"
    run_name = "batch_Y"
    success_output = _trial(
        run_dir / f"{run_name}_multi_01_run01.json",
        success=True,
    )
    failure_output = _trial(
        run_dir / f"{run_name}_multi_01_run02.json",
        success=False,
    )
    _latency(run_dir / "latencies.jsonl", scenario_file, 1, success_output)
    _latency(run_dir / "latencies.jsonl", scenario_file, 2, failure_output)

    result = gcp_resume_state.validate_run_artifacts(
        run_dir=run_dir,
        scenario_glob=scenario_file.as_posix(),
        trials=2,
        run_name=run_name,
        require_latency=True,
    )

    assert result["valid"] is True
    assert result["expected"] == 2
    assert result["complete"] == 2
    assert result["success"] == 1
    assert result["failure"] == 1
    assert result["missing"] == []


def test_validate_run_artifacts_accepts_space_separated_scenario_list(
    tmp_path: Path,
) -> None:
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    first_scenario = _scenario(scenarios_dir / "multi_01.json", "SGT-001")
    second_scenario = _scenario(scenarios_dir / "wo_04.json", "SGT-018")
    run_dir = tmp_path / "run"
    run_name = "batch_Y"

    first_output = _trial(
        run_dir / f"{run_name}_multi_01_run01.json",
        success=True,
    )
    second_output = _trial(
        run_dir / f"{run_name}_wo_04_run01.json",
        success=True,
    )
    _latency(run_dir / "latencies.jsonl", first_scenario, 1, first_output)
    _latency(run_dir / "latencies.jsonl", second_scenario, 1, second_output)

    result = gcp_resume_state.validate_run_artifacts(
        run_dir=run_dir,
        scenario_glob=f"{first_scenario.as_posix()} {second_scenario.as_posix()}",
        trials=1,
        run_name=run_name,
        require_latency=True,
    )

    assert result["valid"] is True
    assert result["expected"] == 2
    assert result["complete"] == 2
    assert result["missing"] == []
