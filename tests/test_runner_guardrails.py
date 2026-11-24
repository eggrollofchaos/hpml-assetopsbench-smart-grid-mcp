"""Text guardrails for shell-runner wiring that is awkward to unit-test live."""

from __future__ import annotations

import subprocess
import os
from pathlib import Path


def _function_body(script: str, start_marker: str, end_marker: str) -> str:
    start = script.index(start_marker)
    end = script.index(end_marker, start)
    return script[start:end]


def test_per_trial_aat_exports_mcp_server_environment() -> None:
    script = Path("scripts/run_experiment.sh").read_text(encoding="utf-8")
    body = _function_body(
        script,
        "run_agent_as_tool_trial()",
        "run_agent_as_tool_batch()",
    )

    export_line = (
        "export AAT_MCP_SERVER_PYTHON AAT_MCP_SERVER_LAUNCH_MODE "
        "AAT_MCP_CLIENT_TIMEOUT_SECONDS"
    )
    invocation = '(cd "$REPO_ROOT" && "${cmd[@]}")'
    assert export_line in body
    assert body.index(export_line) < body.index(invocation)


def test_aat_batch_infrastructure_failures_are_not_masked() -> None:
    script = Path("scripts/run_experiment.sh").read_text(encoding="utf-8")
    body = _function_body(
        script,
        'if [ "$ORCHESTRATION" = "agent_as_tool" ]',
        "else\n\nfor SCENARIO_FILE",
    )

    assert 'run_agent_as_tool_batch "$RUN_DIR" || true' not in body
    assert "BATCH_RC=0" in body
    assert "BATCH_RC=$?" in body
    assert "record_batch_infrastructure_failure" in body
    assert "Infrastructure failure detected; exiting nonzero." in script


def _batch_harness(script: str) -> str:
    preservation_start = script.index("prepare_previous_run_artifacts()")
    preservation_end = script.index("\n}\n", preservation_start) + 3
    preservation = script[preservation_start:preservation_end]
    decision = _function_body(
        script,
        "optimized_batch_infrastructure_failure()",
        "run_verified_pe_trial()",
    )
    start = script.index("# Cell C optimized:")
    end = script.index("else\n\nfor SCENARIO_FILE", start)
    batch = script[start:end] + "\nfi\n"
    summary_start = script.index("SUMMARY_OUTPUTS=()")
    summary_end = script.index("record_infrastructure_error()", summary_start)
    summary_outputs = script[summary_start:summary_end]
    return "\n".join(
        [
            'RUN_DIR="$1"',
            'SCENARIO_FILES=("$2")',
            "TRIALS=1",
            "RUN_BASENAME=run",
            'LATENCY_FILE="$RUN_DIR/latencies.jsonl"',
            'RESUME_MANIFEST_FILE="$RUN_DIR/resume_manifest.jsonl"',
            'META_FILE="$RUN_DIR/meta.json"',
            'VLLM_LOG="$RUN_DIR/vllm.log"',
            'CONFIG_FILE="$RUN_DIR/config.json"',
            'SUMMARY_FILE="$RUN_DIR/summary.json"',
            'CELL_DIR="$RUN_DIR"',
            'HARNESS_LOG="$RUN_DIR/harness.log"',
            "ORCHESTRATION=agent_as_tool",
            "MCP_MODE=optimized",
            "SMARTGRID_RESUME=0",
            "SMARTGRID_FORCE_RERUN=0",
            "AAT_RUNNER_TEMPLATE=",
            "INFRA_FAIL=0",
            "INFRASTRUCTURE_ERRORS=0",
            "BATCH_MODE=1",
            "PASS=0; FAIL=0; TOTAL=0; RESUME_SKIPPED=0; RESUME_RERUN=0",
            "date() { echo fixed-token; }",
            "record_infrastructure_error() { INFRASTRUCTURE_ERRORS=$((INFRASTRUCTURE_ERRORS + 1)); INFRA_FAIL=1; }",
            'record_batch_infrastructure_failure() { INFRA_FAIL=1; if [ "$INFRASTRUCTURE_ERRORS" -eq 0 ]; then INFRASTRUCTURE_ERRORS=1; fi; }',
            'trial_succeeded() { echo "$1" >>"$RUN_DIR/output-reads"; [ -f "$1" ]; }',
            'run_agent_as_tool_batch() { : >"$RUN_DIR/runner-called"; return 2; }',
            "mv_count=0",
            'mv() { mv_count=$((mv_count + 1)); if [ "${FAIL_MOVE_AT:-0}" -eq "$mv_count" ]; then return 1; fi; command mv "$@"; }',
            'mkdir "$RUN_DIR/.batch-previous.fixed-token-$$"',
            'printf "existing-archive-bytes\\n" >"$RUN_DIR/.batch-previous.fixed-token-$$/sentinel"',
            preservation,
            "prepare_previous_run_artifacts || true",
            decision,
            batch,
            summary_outputs,
            'printf "PASS=%s FAIL=%s TOTAL=%s INFRA_FAIL=%s INFRASTRUCTURE_ERRORS=%s SUMMARY_COUNT=%s\\n" "$PASS" "$FAIL" "$TOTAL" "$INFRA_FAIL" "$INFRASTRUCTURE_ERRORS" "${#SUMMARY_OUTPUTS[@]}"',
        ]
    )


def test_aat_batch_preserves_old_requested_artifacts_and_counts_fresh_misses(
    tmp_path: Path,
) -> None:
    script = Path("scripts/run_experiment.sh").read_text(encoding="utf-8")
    requested = "run_scenario_run01.json"
    (tmp_path / requested).write_bytes(b"prior-success-bytes\n")
    (tmp_path / "run_unrelated_run01.json").write_bytes(b"unrequested-bytes\n")
    (tmp_path / "_batch_stats.json").write_bytes(b"prior-stats\n")
    (tmp_path / "_batch_latencies.jsonl").write_bytes(b"prior-latencies\n")

    result = subprocess.run(
        [
            "bash",
            "-c",
            _batch_harness(script),
            "bash",
            str(tmp_path),
            "scenario.json",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS=0 FAIL=1 TOTAL=1 INFRA_FAIL=1" in result.stdout
    assert "SUMMARY_COUNT=0" in result.stdout
    assert (tmp_path / "runner-called").exists()
    archives = list(tmp_path.glob(".batch-previous.fixed-token-*/"))
    assert len(archives) == 2  # pre-existing archive plus this invocation's archive
    fresh_archive = next(
        path for path in archives if (path / "batch" / requested).exists()
    )
    assert (
        fresh_archive / "batch" / requested
    ).read_bytes() == b"prior-success-bytes\n"
    assert (
        fresh_archive / "batch" / "_batch_stats.json"
    ).read_bytes() == b"prior-stats\n"
    assert (
        fresh_archive / "batch" / "_batch_latencies.jsonl"
    ).read_bytes() == b"prior-latencies\n"
    existing_archive = next(path for path in archives if (path / "sentinel").exists())
    assert (existing_archive / "sentinel").read_bytes() == b"existing-archive-bytes\n"
    assert (
        tmp_path / "run_unrelated_run01.json"
    ).read_bytes() == b"unrequested-bytes\n"
    assert not (tmp_path / "output-reads").exists()


def test_aat_batch_preservation_failure_blocks_runner_and_all_old_reads(
    tmp_path: Path,
) -> None:
    script = Path("scripts/run_experiment.sh").read_text(encoding="utf-8")
    requested = "run_scenario_run01.json"
    (tmp_path / requested).write_bytes(b"old-output\n")
    (tmp_path / "_batch_stats.json").write_bytes(b"old-stats\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            _batch_harness(script),
            "bash",
            str(tmp_path),
            "scenario.json",
        ],
        env={**os.environ, "FAIL_MOVE_AT": "2"},
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS=0 FAIL=1 TOTAL=1 INFRA_FAIL=1" in result.stdout
    assert "SUMMARY_COUNT=0" in result.stdout
    assert not (tmp_path / "runner-called").exists()
    assert not (tmp_path / "output-reads").exists()
    archives = list(tmp_path.glob(".batch-previous.fixed-token-*"))
    assert len(archives) == 2
    moved_archive = next(
        path for path in archives if (path / "batch" / requested).exists()
    )
    assert (moved_archive / "batch" / requested).read_bytes() == b"old-output\n"
    assert (tmp_path / "_batch_stats.json").read_bytes() == b"old-stats\n"


def test_aat_batch_infrastructure_failure_decision_executes() -> None:
    script = Path("scripts/run_experiment.sh").read_text(encoding="utf-8")
    start = script.index("optimized_batch_infrastructure_failure()")
    end = script.index("\n}\n", start) + 3
    helper = script[start:end]
    cases = [
        (("2", "1", "0"), True),
        (("1", "0", "0"), True),
        (("1", "1", "1"), True),
        (("1", "1", "0"), False),
        (("0", "1", "0"), False),
    ]
    for args, expected_failure in cases:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f'{helper}\nif optimized_batch_infrastructure_failure "$@"; then exit 0; else exit 1; fi',
                "bash",
                *args,
            ],
            check=False,
        )
        assert (result.returncode == 0) is expected_failure, args


def test_effective_interpreter_fallback_precedes_resume_identity() -> None:
    script = Path("scripts/run_experiment.sh").read_text(encoding="utf-8")
    fallback = script.index('AOB_PYTHON="${AOB_PYTHON:-$AOB_PATH/.venv/bin/python}"')
    identity_check = script.index("validate-run-identity")
    identity_args_end = script.index("--vllm-extra-args", identity_check)
    identity_args = script[identity_check:identity_args_end]
    assert fallback < identity_check
    assert (
        'AAT_MCP_SERVER_PYTHON="$REPO_ROOT/.venv/bin/python"' in script[:identity_check]
    )
    assert (
        'AAT_MCP_SERVER_PYTHON_VERSION="$(python_version "$AAT_MCP_SERVER_PYTHON")"'
        in script[:identity_check]
    )
    assert (
        '--aat-mcp-server-python-version "$AAT_MCP_SERVER_PYTHON_VERSION"'
        in identity_args
    )
    assert '--aat-parallel-tool-calls "$AAT_PARALLEL_TOOL_CALLS"' in identity_args
    assert '--aat-tool-choice "${AAT_TOOL_CHOICE:-}"' in identity_args
    assert "export AAT_PARALLEL_TOOL_CALLS" in script[:identity_check]
    assert 'VLLM_ACTIVATE="$REPO_ROOT/.venv-insomnia/bin/activate"' in script
    assert 'VLLM_ACTIVATE="$PROJECT_ROOT/.venv-insomnia/bin/activate"' in script
    assert 'source "$VLLM_ACTIVATE"' in script
    assert "requires a usable .venv-insomnia Python and activate script" in script
    assert 'for path in "${batch_sources[@]}"' in script
    assert 'if [ "$batch_selected" -eq 1 ]; then' in script


def test_gcp_batch_driver_hard_fails_incomplete_artifacts() -> None:
    script = Path("scripts/run_gcp_context_batch.sh").read_text(encoding="utf-8")

    assert "validate-run-shell" in script
    assert "incomplete trajectory artifacts" in script
    assert 'status="artifact_failed"' in script
    assert 'existing_status" = "complete"' in script
    assert "SMARTGRID_SKIP_LOCAL_MODEL_PREFLIGHT" in script


def test_plan_execute_repo_local_keeps_smartgrid_server_overrides() -> None:
    script = Path("scripts/run_experiment.sh").read_text(encoding="utf-8")
    body = _function_body(
        script,
        "run_plan_execute_trial()",
        "run_json_stdout_trial()",
    )

    assert 'PLAN_EXECUTE_REPO_LOCAL="${PLAN_EXECUTE_REPO_LOCAL:-}"' in script
    assert '[ "$ORCHESTRATION" = "plan_execute" ]' in script
    assert '[ "$ENABLE_SMARTGRID_SERVERS" = "1" ]' in script
    assert "PLAN_EXECUTE_REPO_LOCAL=1" in script
    assert '[ "${PLAN_EXECUTE_REPO_LOCAL:-0}" = "1" ]' in body
    assert "wrapper_cmd+=(--disable-self-ask)" in body
    assert 'wrapper_cmd+=("${SERVER_ARGS[@]}")' in body

    upstream_cli = body.split(
        'local -a cmd=(uv run --python "$AOB_UV_PYTHON" plan-execute', 1
    )[1]
    assert 'cmd+=("${SERVER_ARGS[@]}")' not in upstream_cli


def test_repo_local_pe_runner_can_disable_self_ask() -> None:
    runner = Path("scripts/plan_execute_self_ask_runner.py").read_text(encoding="utf-8")

    assert '"--disable-self-ask"' in runner
    assert "if args.disable_self_ask" in runner
    assert "augmented_question=args.question" in runner
