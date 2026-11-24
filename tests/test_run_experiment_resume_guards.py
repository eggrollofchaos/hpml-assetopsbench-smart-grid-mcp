from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
from scripts import gcp_resume_state

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_experiment.sh"
RUN_REPOS: dict[str, Path] = {}


def _isolated_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    from scripts.gcp_resume_state import _RUNNER_CODE_FILES

    for relative in _RUNNER_CODE_FILES:
        source = REPO_ROOT / relative
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    insomnia_env = project / "scripts" / "insomnia_env.sh"
    insomnia_env.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "insomnia_env.sh", insomnia_env)
    (project / "data").symlink_to(REPO_ROOT / "data", target_is_directory=True)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "isolated fixture",
        ],
        cwd=project,
        check=True,
    )
    return project


def _fixture(tmp_path: Path, *, model_id: str = "model-a") -> tuple[Path, Path]:
    _isolated_project(tmp_path)
    aob = tmp_path / "aob"
    aob.mkdir()
    (aob / "pyproject.toml").write_text("[build-system]\nrequires = []\n")
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"id": "SGT-SMOKE", "text": "prompt-a"}) + "\n")
    runner = tmp_path / "runner.sh"
    runner.write_text(
        "#!/bin/bash\n"
        'printf \'%s\\n\' \'{"success":true,"answer":"ok","history":[]}\' > "$OUTPUT_PATH"\n'
    )
    runner.chmod(0o755)
    config = tmp_path / "config.env"
    config.write_text(
        "\n".join(
            [
                "EXPERIMENT_NAME=resume_smoke",
                "EXPERIMENT_CELL=Z",
                "EXPERIMENT_FAMILY=smoke",
                f"SCENARIOS_GLOB={scenario}",
                "SCENARIO_SET_NAME=smoke-set",
                f"MODEL_ID={model_id}",
                "MODEL_PROVIDER=local",
                "ORCHESTRATION=hybrid",
                "MCP_MODE=baseline",
                "TRIALS=1",
                "LAUNCH_VLLM=0",
                "ENABLE_SMARTGRID_SERVERS=0",
                f"AOB_PATH={aob}",
                f"HYBRID_RUNNER_TEMPLATE={runner}",
            ]
        )
        + "\n"
    )
    return config, scenario


def _run_wrapper(
    config: Path,
    run_id: str,
    *,
    resume: bool = False,
    env_overrides: dict[str, str] | None = None,
    env_unset: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SMARTGRID_RUN_ID"] = run_id
    env["SMARTGRID_RESUME"] = "1" if resume else "0"
    project = config.parent / "project"
    RUN_REPOS[run_id] = project
    env["SLURM_SUBMIT_DIR"] = str(project)
    env.setdefault("GPU_TYPE", "test-gpu")
    env.update(env_overrides or {})
    for name in env_unset:
        env.pop(name, None)
    return subprocess.run(
        ["bash", str(RUN_SCRIPT), str(config)],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_dir(run_id: str) -> Path:
    return RUN_REPOS[run_id] / "benchmarks" / "cell_Z_hybrid" / "raw" / run_id


def _create_venv(path: Path) -> Path:
    subprocess.run([sys.executable, "-m", "venv", str(path)], check=True)
    return path / "bin" / "python"


def _offline_vllm_stubs(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    (fake_modules / "torch.py").write_text(
        "__version__ = 'fixture'\n"
        "class _Cuda:\n"
        "    def is_available(self): return True\n"
        "    def device_count(self): return 1\n"
        "    def init(self): return None\n"
        "    def get_device_name(self, index): return 'fixture GPU'\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (fake_bin / "nvidia-smi").write_text(
        '#!/bin/sh\n[ "$1" = "-L" ] && echo \'GPU 0: fixture\'\nexit 0\n',
        encoding="utf-8",
    )
    (fake_bin / "setsid").write_text(
        '#!/bin/sh\nprintf \'%s\n\' "$*" >> "$FAKE_VLLM_LAUNCH_LOG"\nexit 0\n',
        encoding="utf-8",
    )
    (fake_bin / "curl").write_text(
        '#!/bin/sh\ncase "$*" in\n'
        "  */v1/models*) printf '%s\n' "
        '\'{"data":[{"id":"Llama-3.1-8B-Instruct"}]}\' ;;\n'
        "esac\nexit 0\n",
        encoding="utf-8",
    )
    for command in ("nvidia-smi", "setsid", "curl"):
        (fake_bin / command).chmod(0o755)
    uv_path = shutil.which("uv")
    assert uv_path is not None
    (fake_bin / "uv").write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "log = pathlib.Path(os.environ['UV_TEST_LOG'])\n"
        "if args[:2] == ['python', 'find']:\n"
        "    result = subprocess.run([os.environ['REAL_UV'], *args], "
        "text=True, capture_output=True, check=False)\n"
        "    log.write_text(json.dumps({'kind': 'find', "
        "'virtual_env': os.environ.get('VIRTUAL_ENV'), "
        "'path': result.stdout.strip(), 'rc': result.returncode}) + '\\n')\n"
        "    sys.stdout.write(result.stdout); sys.stderr.write(result.stderr)\n"
        "    raise SystemExit(result.returncode)\n"
        "if args and args[0] == 'run':\n"
        "    lines = log.read_text().splitlines() if log.exists() else []\n"
        "    if not any(json.loads(line).get('kind') == 'preflight' "
        "for line in lines):\n"
        "        python = args[args.index('--python') + 1]\n"
        "        result = subprocess.run([python, '-'], input=sys.stdin.read(), "
        "text=True, capture_output=True, check=False)\n"
        "        with log.open('a') as out:\n"
        "            out.write(json.dumps({'kind': 'preflight', "
        "'python': python, 'rc': result.returncode, "
        "'stdout': result.stdout, 'stderr': result.stderr}) + '\\n')\n"
        "        sys.stdout.write(result.stdout); sys.stderr.write(result.stderr)\n"
        "        raise SystemExit(result.returncode)\n"
        "    with log.open('a') as out:\n"
        "        out.write(json.dumps({'kind': 'runner-stopped', "
        "'args': args}) + '\\n')\n"
        "    print('fixture stops before model execution', file=sys.stderr)\n"
        "    raise SystemExit(97)\n"
        "raise SystemExit('unexpected uv fixture command: ' + repr(args))\n",
        encoding="utf-8",
    )
    (fake_bin / "uv").chmod(0o755)
    return fake_bin, fake_modules


def _run_from_repo(
    config: Path,
    run_id: str,
    repo_root: Path,
    *,
    env_overrides: dict[str, str] | None = None,
    env_unset: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "SMARTGRID_RUN_ID": run_id,
            "SMARTGRID_RESUME": "0",
            "SLURM_SUBMIT_DIR": str(repo_root),
            "GPU_TYPE": "fixture-gpu",
        }
    )
    env.update(env_overrides or {})
    for name in env_unset:
        env.pop(name, None)
    RUN_REPOS[run_id] = repo_root
    return subprocess.run(
        ["bash", str(RUN_SCRIPT), str(config)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _optimized_batch_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = _isolated_project(tmp_path)
    aob = tmp_path / "aob"
    aob.mkdir()
    (aob / "pyproject.toml").write_text("[build-system]\nrequires = []\n")
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    scenario = scenario_dir / "scenario.json"
    scenario.write_text(json.dumps({"id": "SGT-BATCH", "text": "prompt"}) + "\n")
    config = tmp_path / "batch.env"
    config.write_text(
        "\n".join(
            [
                "EXPERIMENT_NAME=batch_smoke",
                "EXPERIMENT_CELL=C",
                "EXPERIMENT_FAMILY=smoke",
                f"SCENARIOS_GLOB={scenario_dir}/*.json",
                "SCENARIO_SET_NAME=batch-smoke",
                "MODEL_ID=model-a",
                "MODEL_PROVIDER=local",
                "ORCHESTRATION=agent_as_tool",
                "MCP_MODE=optimized",
                "TRIALS=1",
                "LAUNCH_VLLM=0",
                "AAT_MCP_SERVER_LAUNCH_MODE=uv",
                "ENABLE_SMARTGRID_SERVERS=0",
                f"AOB_PATH={aob}",
                "AAT_RUNNER_TEMPLATE=",
            ]
        )
        + "\n"
    )
    return project, config, scenario


def test_fresh_reuse_snapshots_all_overwritten_artifacts_as_independent_bytes(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    run_id = f"snapshot-all-{tmp_path.name}"
    first = _run_wrapper(config, run_id)
    assert first.returncode == 0, first.stderr
    run_dir = _run_dir(run_id)
    cell_dir = run_dir.parents[1]
    protected = {
        run_dir / "harness.log": b"harness sentinel\x00",
        run_dir / "latencies.jsonl": b"latency sentinel\x00",
        run_dir / "resume_manifest.jsonl": b"manifest sentinel\x00",
        run_dir / "meta.json": b"meta sentinel\x00",
        run_dir / "vllm.log": b"vllm sentinel\x00",
        cell_dir / "config.json": b"cell config sentinel\x00",
        cell_dir / "summary.json": b"cell summary sentinel\x00",
    }
    for path, content in protected.items():
        path.write_bytes(content)

    rerun = _run_wrapper(config, run_id)
    assert rerun.returncode == 0, rerun.stderr
    archives = sorted(run_dir.glob(".batch-previous.*"))
    assert len(archives) == 1
    archive = archives[0]
    mapping = {
        run_dir / "harness.log": archive / "run" / "harness.log",
        run_dir / "latencies.jsonl": archive / "run" / "latencies.jsonl",
        run_dir / "resume_manifest.jsonl": archive / "run" / "resume_manifest.jsonl",
        run_dir / "meta.json": archive / "run" / "meta.json",
        run_dir / "vllm.log": archive / "run" / "vllm.log",
        cell_dir / "config.json": archive / "cell" / "config.json",
        cell_dir / "summary.json": archive / "cell" / "summary.json",
    }
    for source, destination in mapping.items():
        assert destination.read_bytes() == protected[source]
        assert destination.stat().st_ino != source.stat().st_ino


def test_snapshot_copy_failure_keeps_all_sentinel_bytes_and_skips_child(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    run_id = f"snapshot-copy-failure-{tmp_path.name}"
    assert _run_wrapper(config, run_id).returncode == 0
    run_dir = _run_dir(run_id)
    cell_dir = run_dir.parents[1]
    protected = {
        run_dir / "harness.log": b"harness sentinel",
        run_dir / "latencies.jsonl": b"latency sentinel",
        run_dir / "resume_manifest.jsonl": b"manifest sentinel",
        run_dir / "meta.json": b"meta sentinel",
        run_dir / "vllm.log": b"vllm sentinel",
        cell_dir / "config.json": b"cell config sentinel",
        cell_dir / "summary.json": b"cell summary sentinel",
    }
    for path, content in protected.items():
        path.write_bytes(content)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    count_file = tmp_path / "cp-count"
    fake_cp = fake_bin / "cp"
    fake_cp.write_text(
        "#!/bin/bash\n"
        'count=0; [ ! -f "$CP_COUNT_FILE" ] || count=$(cat "$CP_COUNT_FILE")\n'
        'count=$((count + 1)); printf "%s" "$count" > "$CP_COUNT_FILE"\n'
        'if [ "$count" -eq 2 ]; then exit 9; fi\n'
        'exec /bin/cp "$@"\n',
        encoding="utf-8",
    )
    fake_cp.chmod(0o755)
    child_count = _run_dir(run_id).parents[3] / "child-count"
    result = _run_wrapper(
        config,
        run_id,
        env_overrides={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CP_COUNT_FILE": str(count_file),
        },
    )
    assert result.returncode == 2
    assert "preservation failed" in result.stderr
    assert not child_count.exists()
    for path, content in protected.items():
        assert path.read_bytes() == content


def test_snapshot_refuses_symlink_without_touching_target_or_launching(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    run_id = f"snapshot-symlink-{tmp_path.name}"
    project = config.parent / "project"
    cell_config = project / "benchmarks/cell_Z_hybrid/config.json"
    cell_config.parent.mkdir(parents=True)
    target = tmp_path / "protected-target.json"
    target.write_bytes(b"protected config bytes")
    cell_config.symlink_to(target)

    result = _run_wrapper(config, run_id)
    assert result.returncode == 2
    assert "symlinked run artifact" in result.stderr
    assert cell_config.is_symlink()
    assert target.read_bytes() == b"protected config bytes"
    assert not (_run_dir(run_id) / "harness.log").exists()
    assert not (_run_dir(run_id) / "meta.json").exists()


def test_optimized_mid_move_failure_keeps_run_sentinels_and_archives_moved_output(
    tmp_path: Path,
) -> None:
    project, config, _ = _optimized_batch_fixture(tmp_path)
    run_id = f"batch-mid-move-failure-{tmp_path.name}"
    run_dir = project / "benchmarks/cell_C_mcp_optimized/raw" / run_id
    cell_dir = run_dir.parents[1]
    run_dir.mkdir(parents=True)
    run_basename = (
        f"{subprocess.check_output(['date', '+%Y-%m-%d'], text=True).strip()}"
        "_C_model-a_agent_as_tool_optimized"
    )
    old_trial = run_dir / f"{run_basename}_scenario_run01.json"
    old_trial.write_bytes(b"old trial bytes")
    old_stats = run_dir / "_batch_stats.json"
    old_stats.write_bytes(b"old stats bytes")
    old_latencies = run_dir / "_batch_latencies.jsonl"
    old_latencies.write_bytes(b"old batch latency bytes")
    protected = {
        run_dir / "harness.log": b"harness sentinel",
        run_dir / "latencies.jsonl": b"latency sentinel",
        run_dir / "resume_manifest.jsonl": b"manifest sentinel",
        run_dir / "meta.json": b"meta sentinel",
        run_dir / "vllm.log": b"vllm sentinel",
        cell_dir / "config.json": b"cell config sentinel",
        cell_dir / "summary.json": b"cell summary sentinel",
    }
    for path, content in protected.items():
        path.write_bytes(content)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    count_file = tmp_path / "mv-count"
    child_marker = tmp_path / "uv-called"
    fake_mv = fake_bin / "mv"
    fake_mv.write_text(
        "#!/bin/bash\n"
        'count=0; [ ! -f "$MV_COUNT_FILE" ] || count=$(cat "$MV_COUNT_FILE")\n'
        'count=$((count + 1)); printf "%s" "$count" > "$MV_COUNT_FILE"\n'
        'if [ "$count" -eq 2 ]; then exit 9; fi\n'
        'exec /bin/mv "$@"\n',
        encoding="utf-8",
    )
    fake_mv.chmod(0o755)
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = python ] && [ "$2" = find ]; then printf \'%s\\n\' "$TEST_UV_PYTHON"; exit 0; fi\n'
        'touch "$UV_CALLED_MARKER"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), str(config)],
        cwd=project,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SLURM_SUBMIT_DIR": str(project),
            "SMARTGRID_RUN_ID": run_id,
            "MV_COUNT_FILE": str(count_file),
            "UV_CALLED_MARKER": str(child_marker),
            "TEST_UV_PYTHON": sys.executable,
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "preservation failed" in result.stderr
    assert not child_marker.exists()
    assert old_stats.read_bytes() == b"old stats bytes"
    archives = sorted(run_dir.glob(".batch-previous.*"))
    assert len(archives) == 1
    moved_trial = archives[0] / "batch" / old_trial.name
    assert moved_trial.read_bytes() == b"old trial bytes"
    assert not old_trial.exists()
    for path, content in protected.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize(
    (
        "failure_kind",
        "batch_rc",
        "infra_count",
        "stats_kind",
        "expected_exit",
        "expected_infra",
    ),
    [
        ("model", "1", "0", "valid", 0, False),
        ("infra", "1", "1", "valid", 1, True),
        ("model", "0", "0", "missing", 1, True),
        ("model", "0", "0", "malformed", 1, True),
    ],
)
def test_optimized_wrapper_summarizes_fresh_batch_outcomes_in_isolated_repo(
    tmp_path: Path,
    failure_kind: str,
    batch_rc: str,
    infra_count: str,
    stats_kind: str,
    expected_exit: int,
    expected_infra: bool,
) -> None:
    project, config, scenario = _optimized_batch_fixture(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = python ] && [ "$2" = find ]; then printf \'%s\\n\' "$TEST_UV_PYTHON"; exit 0; fi\n'
        'printf \'%s\\n\' "$*" >> "$FAKE_UV_CALLS"\n'
        'if [[ " $* " == *" python - "* ]]; then cat >/dev/null; exit 0; fi\n'
        "out=''\nbase=''\nscenario_glob=''\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        '    --output-dir) out="$2"; shift 2 ;;\n'
        '    --run-basename) base="$2"; shift 2 ;;\n'
        '    --scenarios-glob) scenario_glob="$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        'mkdir -p "$out"\n'
        "for scenario in $scenario_glob; do\n"
        '  name="${scenario##*/}"; name="${name%.json}"\n'
        '  if [ "$FAKE_BATCH_INFRA_ERRORS" = 0 ]; then\n'
        '    printf \'%s\\n\' \'{"success":false,"state":"complete_failure","model_error":true,"error":"MaxTurnsExceeded: budget"}\' > "$out/${base}_${name}_run01.json"\n'
        "  else\n"
        '    printf \'%s\\n\' \'{"success":false,"state":"infrastructure_error","infrastructure_error":true,"error":"RuntimeError: failed"}\' > "$out/${base}_${name}_run01.json"\n'
        "  fi\n"
        '  printf \'{"scenario_file":"%s","trial_index":1,"latency_seconds":1.0}\\n\' "$scenario" >> "$out/_batch_latencies.jsonl"\n'
        "done\n"
        'if [ "$FAKE_BATCH_STATS" = valid ]; then printf \'{"infrastructure_errors":%s,"terminal_records":1}\\n\' "$FAKE_BATCH_INFRA_ERRORS" > "$out/_batch_stats.json"; fi\n'
        "if [ \"$FAKE_BATCH_STATS\" = malformed ]; then printf '%s\\n' 'not-json' > \"$out/_batch_stats.json\"; fi\n"
        'exit "$FAKE_BATCH_RC"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    run_id = f"isolated-batch-{failure_kind}-{tmp_path.name}"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SLURM_SUBMIT_DIR": str(project),
            "SMARTGRID_RUN_ID": run_id,
            "FAKE_BATCH_RC": batch_rc,
            "FAKE_BATCH_INFRA_ERRORS": infra_count,
            "FAKE_BATCH_STATS": stats_kind,
            "TEST_UV_PYTHON": sys.executable,
            "FAKE_UV_CALLS": str(tmp_path / "uv-calls.log"),
        }
    )
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), str(config)],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    run_dir = project / "benchmarks" / "cell_C_mcp_optimized" / "raw" / run_id
    assert result.returncode == expected_exit, result.stderr
    summary = json.loads(
        (project / "benchmarks" / "cell_C_mcp_optimized" / "summary.json").read_text()
    )
    meta = json.loads((run_dir / "meta.json").read_text())
    assert summary["scenarios_attempted"] == 1
    assert summary["scenarios_completed"] == 0
    assert summary["failure_count"] == 1
    assert summary["infrastructure_failure"] is expected_infra
    assert meta["infrastructure_failure"] is expected_infra
    assert summary["run_status"] == "failed"
    assert meta["run_status"] == "failed"
    uv_calls = (tmp_path / "uv-calls.log").read_text().splitlines()
    assert any(call.startswith(f"run --python {sys.executable} ") for call in uv_calls)
    aat_identity = json.loads(meta["run_identity"]["aat_uv_interpreter_identity"])
    expected_uv = gcp_resume_state.interpreter_identity(
        Path(sys.executable),
        subprocess.check_output([sys.executable, "--version"], text=True).strip(),
    )
    assert aat_identity == expected_uv
    assert (
        json.loads(meta["run_identity"]["aat_mcp_interpreter_identity"]) == expected_uv
    )
    assert scenario.is_file()


@pytest.mark.parametrize("value", ["0", "1"])
def test_smartgrid_server_flag_accepts_documented_exact_values(
    tmp_path: Path, value: str
) -> None:
    config, _ = _fixture(tmp_path)
    config.write_text(
        config.read_text().replace(
            "ENABLE_SMARTGRID_SERVERS=0", f"ENABLE_SMARTGRID_SERVERS={value}"
        )
    )
    result = _run_wrapper(
        config,
        f"flag-{value}-{tmp_path.name}",
        env_overrides={"DRY_RUN": "1"},
    )
    assert result.returncode == 0, result.stderr


def test_vanilla_plan_execute_uv_uses_resolved_assetopsbench_interpreter(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    config.write_text(
        config.read_text().replace("ORCHESTRATION=hybrid", "ORCHESTRATION=plan_execute")
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "uv-calls.log"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = python ] && [ "$2" = find ]; then printf \'%s\\n\' "$TEST_UV_PYTHON"; exit 0; fi\n'
        'printf \'%s\\n\' "$*" >> "$UV_CALL_LOG"\n'
        'if [[ " $* " == *" plan-execute "* ]]; then printf \'%s\\n\' \'{"success":true,"answer":"ok","history":[]}\'; exit 0; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    run_id = f"aob-uv-python-pin-{tmp_path.name}"
    result = _run_wrapper(
        config,
        run_id,
        env_overrides={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "TEST_UV_PYTHON": sys.executable,
            "UV_CALL_LOG": str(call_log),
        },
    )
    assert result.returncode == 0, result.stderr
    calls = call_log.read_text().splitlines()
    assert any(
        call.startswith(f"run --python {sys.executable} plan-execute ")
        for call in calls
    )
    meta = json.loads((_run_dir(run_id) / "meta.json").read_text())
    expected = gcp_resume_state.interpreter_identity(
        Path(sys.executable),
        subprocess.check_output([sys.executable, "--version"], text=True).strip(),
    )
    assert json.loads(meta["run_identity"]["aob_uv_interpreter_identity"]) == expected
    assert (
        meta["python_interpreter_provenance"]["assetopsbench_uv_context"]
        == sys.executable
    )


def test_vllm_runner_interpreter_is_selected_before_identity_capture(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    config.write_text(config.read_text().replace("LAUNCH_VLLM=0", "LAUNCH_VLLM=1"))
    project = config.parent / "project"
    insomnia_python = project / ".venv-insomnia" / "bin" / "python"
    insomnia_python.parent.mkdir(parents=True)
    (insomnia_python.parent / "activate").write_text(":\n", encoding="utf-8")
    marker = tmp_path / "selected-python-invocations.log"
    insomnia_python.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$PYTHON_MARKER"\n'
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    insomnia_python.chmod(0o755)

    run_id = f"vllm-python-identity-{tmp_path.name}"
    result = _run_wrapper(
        config,
        run_id,
        env_overrides={
            "DRY_RUN": "1",
            "GPU_TYPE": "fixture-gpu",
            "PYTHON_MARKER": str(marker),
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.find("Dry run enabled") >= 0
    invocations = marker.read_text().splitlines()
    assert invocations.count("--version") >= 2
    assert not (_run_dir(run_id) / "meta.json").exists()


@pytest.mark.parametrize("placement", ["worktree", "shared", "both"])
def test_insomnia_interpreter_is_pinned_before_resume_identity(
    tmp_path: Path, placement: str
) -> None:
    config, _ = _fixture(tmp_path)
    config.write_text(config.read_text().replace("LAUNCH_VLLM=0", "LAUNCH_VLLM=1"))
    shared = config.parent / "project"
    worktree = tmp_path / "linked-worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=shared,
        check=True,
        capture_output=True,
        text=True,
    )
    selected_root = worktree if placement in {"worktree", "both"} else shared
    selected = selected_root / ".venv-insomnia/bin/python"
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$PYTHON_MARKER"\n'
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    selected.chmod(0o755)
    (selected.parent / "activate").write_text(":\n", encoding="utf-8")
    if placement == "both":
        shared_python = shared / ".venv-insomnia/bin/python"
        shared_python.parent.mkdir(parents=True)
        shared_python.write_text("#!/bin/sh\nexit 19\n", encoding="utf-8")
        shared_python.chmod(0o755)
        (shared_python.parent / "activate").write_text(":\n", encoding="utf-8")
    marker = tmp_path / "worktree-python-invocations.log"
    env = os.environ.copy()
    env.update(
        {
            "SLURM_SUBMIT_DIR": str(worktree),
            "SMARTGRID_RUN_ID": f"worktree-python-{tmp_path.name}",
            "GPU_TYPE": "fixture-gpu",
            "DRY_RUN": "1",
            "PYTHON_MARKER": str(marker),
        }
    )
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), str(config)],
        cwd=worktree,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = marker.read_text().splitlines()
    assert any(call.startswith("--version") for call in calls)
    assert any(f"--python {selected}" in call for call in calls)
    assert not (
        worktree
        / "benchmarks/cell_Z_hybrid/raw"
        / env["SMARTGRID_RUN_ID"]
        / "meta.json"
    ).exists()


@pytest.mark.parametrize(
    ("mcp_mode", "placement", "explicit_server_python"),
    [
        ("baseline", "shared", False),
        ("optimized", "shared", False),
        ("baseline", "worktree", False),
        ("baseline", "both", False),
        ("baseline", "shared", True),
    ],
)
def test_vllm_aat_mcp_uses_selected_or_explicit_server_python_through_preflight(
    tmp_path: Path,
    mcp_mode: str,
    placement: str,
    explicit_server_python: bool,
) -> None:
    config, _ = _fixture(tmp_path)
    config.write_text(
        config.read_text()
        .replace(
            "EXPERIMENT_CELL=Z",
            "EXPERIMENT_CELL=C" if mcp_mode == "optimized" else "EXPERIMENT_CELL=B",
        )
        .replace("ORCHESTRATION=hybrid", "ORCHESTRATION=agent_as_tool")
        .replace("MCP_MODE=baseline", f"MCP_MODE={mcp_mode}")
        .replace("LAUNCH_VLLM=0", "LAUNCH_VLLM=1")
        + "\nAAT_MCP_SERVER_LAUNCH_MODE=python\n",
        encoding="utf-8",
    )
    shared = config.parent / "project"
    worktree = tmp_path / "linked-worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=shared,
        check=True,
        capture_output=True,
        text=True,
    )
    selected_root = worktree if placement in {"worktree", "both"} else shared
    selected_python = _create_venv(selected_root / ".venv-insomnia")
    if placement == "both":
        _create_venv(shared / ".venv-insomnia")
    override_python = None
    if explicit_server_python:
        override_python = _create_venv(tmp_path / "server-override")

    fake_bin, fake_modules = _offline_vllm_stubs(tmp_path)
    site_packages = sysconfig.get_paths()["purelib"]
    uv_log = tmp_path / "uv-calls.jsonl"
    launch_log = tmp_path / "vllm-launch.log"
    server_python = override_python or selected_python
    run_id = f"aat-{mcp_mode}-{placement}-{explicit_server_python}-{tmp_path.name}"
    cell_dir = (
        "cell_C_mcp_optimized" if mcp_mode == "optimized" else "cell_B_mcp_baseline"
    )
    env_overrides = {
        "PATH": os.pathsep.join((str(fake_bin), os.environ["PATH"])),
        "PYTHONPATH": os.pathsep.join((str(fake_modules), site_packages)),
        "REAL_UV": shutil.which("uv") or "",
        "UV_TEST_LOG": str(uv_log),
        "FAKE_VLLM_LAUNCH_LOG": str(launch_log),
        "VLLM_STARTUP_TIMEOUT": "1",
        "UV_OFFLINE": "1",
        "DRY_RUN": "0",
        "AAT_MCP_SERVER_LAUNCH_MODE": "python",
    }
    if override_python is not None:
        env_overrides["AAT_MCP_SERVER_PYTHON"] = str(override_python)
    else:
        env_overrides["AAT_MCP_SERVER_PYTHON"] = ""
    result = _run_from_repo(config, run_id, worktree, env_overrides=env_overrides)

    run_dir = worktree / "benchmarks" / cell_dir / "raw" / run_id
    assert (run_dir / "meta.json").is_file(), result.stderr
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["python_interpreter_provenance"]["aat_mcp_server"] == str(server_python)
    server_identity = json.loads(meta["run_identity"]["aat_mcp_interpreter_identity"])
    assert server_identity["version"].startswith("Python ")
    harness = (run_dir / "harness.log").read_text(encoding="utf-8")
    assert "AaT runtime dependency preflight passed." in harness
    assert "AaT MCP server dependency preflight passed." in harness
    assert "torch=fixture" in harness
    assert launch_log.is_file()
    uv_events = [json.loads(line) for line in uv_log.read_text().splitlines()]
    find_event = next(event for event in uv_events if event["kind"] == "find")
    assert find_event["virtual_env"] == str(selected_python.parent.parent)
    assert Path(find_event["path"]) == selected_python
    assert next(event for event in uv_events if event["kind"] == "preflight")["rc"] == 0
    assert any(event["kind"] == "runner-stopped" for event in uv_events)


def test_uv_python_find_uses_activated_insomnia_env_not_inherited_venv(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    config.write_text(
        config.read_text()
        .replace("ORCHESTRATION=hybrid", "ORCHESTRATION=agent_as_tool")
        .replace("LAUNCH_VLLM=0", "LAUNCH_VLLM=1")
        + "\nAAT_MCP_SERVER_LAUNCH_MODE=uv\n",
        encoding="utf-8",
    )
    shared = config.parent / "project"
    worktree = tmp_path / "linked-worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=shared,
        check=True,
        capture_output=True,
        text=True,
    )
    selected_python = _create_venv(shared / ".venv-insomnia")
    inherited_python = _create_venv(tmp_path / "inherited-venv")
    fake_bin, _fake_modules = _offline_vllm_stubs(tmp_path)
    uv_log = tmp_path / "uv-find.jsonl"
    inherited_path = os.pathsep.join(
        (str(inherited_python.parent), str(fake_bin), os.environ["PATH"])
    )
    result = _run_from_repo(
        config,
        f"uv-env-order-{tmp_path.name}",
        worktree,
        env_overrides={
            "PATH": inherited_path,
            "VIRTUAL_ENV": str(inherited_python.parent.parent),
            "UV_TEST_LOG": str(uv_log),
            "REAL_UV": shutil.which("uv") or "",
            "UV_OFFLINE": "1",
            "DRY_RUN": "1",
            "AAT_MCP_SERVER_LAUNCH_MODE": "uv",
            "AAT_MCP_SERVER_PYTHON": "",
        },
        env_unset=("UV_PROJECT_ENVIRONMENT", "UV_PYTHON"),
    )

    assert result.returncode == 0, result.stderr
    assert "Dry run enabled" in result.stdout
    event = json.loads(uv_log.read_text(encoding="utf-8"))
    assert event["kind"] == "find"
    assert event["virtual_env"] == str(selected_python.parent.parent)
    assert Path(event["path"]) == selected_python


def test_vllm_without_selected_insomnia_environment_refuses_before_run_artifacts(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    config.write_text(config.read_text().replace("LAUNCH_VLLM=0", "LAUNCH_VLLM=1"))
    run_id = f"missing-insomnia-python-{tmp_path.name}"
    result = _run_wrapper(config, run_id, env_overrides={"DRY_RUN": "1"})
    assert result.returncode == 2
    assert (
        "requires a usable .venv-insomnia Python and activate script" in result.stderr
    )
    assert not _run_dir(run_id).exists()


@pytest.mark.parametrize(
    ("success", "expected_successes", "expected_failures"),
    [(True, 1, 0), (False, 0, 1)],
)
def test_serial_resume_keeps_terminal_trial_and_batch_sidecars_in_place(
    tmp_path: Path, success: bool, expected_successes: int, expected_failures: int
) -> None:
    config, _ = _fixture(tmp_path)
    runner = config.parent / "runner.sh"
    payload = (
        '{"success":true,"answer":"ok","history":[]}'
        if success
        else '{"success":false,"answer":"failed","error":"MaxTurnsExceeded: model failure","history":[]}'
    )
    runner.write_text(
        "#!/bin/bash\n"
        "printf 'run\\n' >> \"$RUNNER_CALLS\"\n"
        f"printf '%s\\n' '{payload}' > \"$OUTPUT_PATH\"\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    calls = tmp_path / "runner-calls.log"
    run_id = f"serial-terminal-{success}-{tmp_path.name}"
    first = _run_wrapper(config, run_id, env_overrides={"RUNNER_CALLS": str(calls)})
    assert first.returncode == 0, first.stderr
    run_dir = _run_dir(run_id)
    terminal_output = next(run_dir.glob("*_run01.json"))
    terminal_bytes = terminal_output.read_bytes()
    sidecars = {
        run_dir / "_batch_stats.json": b"current batch stats",
        run_dir / "_batch_latencies.jsonl": b"current batch latencies",
    }
    for path, content in sidecars.items():
        path.write_bytes(content)

    resumed = _run_wrapper(
        config,
        run_id,
        resume=True,
        env_overrides={"RUNNER_CALLS": str(calls)},
    )
    assert resumed.returncode == 0, resumed.stderr
    summary = json.loads((run_dir.parent.parent / "summary.json").read_text())
    assert calls.read_text().splitlines() == ["run"]
    assert summary["resume_skipped_count"] == 1
    assert summary["resume_rerun_count"] == 0
    assert summary["scenarios_completed"] == expected_successes
    assert summary["failure_count"] == expected_failures
    assert terminal_output.read_bytes() == terminal_bytes
    assert all(path.read_bytes() == content for path, content in sidecars.items())
    assert not list(run_dir.glob(".batch-previous.*/batch/*"))


@pytest.mark.parametrize(
    ("configured", "inherited", "unset"),
    [("0", None, True), ("0", "1", False), ("1", None, True), ("1", "0", False)],
)
def test_metadata_and_resume_use_configured_server_flag(
    tmp_path: Path, configured: str, inherited: str | None, unset: bool
) -> None:
    config, _ = _fixture(tmp_path)
    config.write_text(
        config.read_text().replace(
            "ENABLE_SMARTGRID_SERVERS=0", f"ENABLE_SMARTGRID_SERVERS={configured}"
        )
    )
    run_id = f"server-flag-parity-{configured}-{inherited}-{tmp_path.name}"
    overrides = {} if inherited is None else {"ENABLE_SMARTGRID_SERVERS": inherited}
    unset_vars = ("ENABLE_SMARTGRID_SERVERS",) if unset else ()

    first = _run_wrapper(config, run_id, env_overrides=overrides, env_unset=unset_vars)
    assert first.returncode == 0, first.stderr
    run_dir = _run_dir(run_id)
    before = json.loads((run_dir / "meta.json").read_text())
    assert before["run_identity"]["enable_smartgrid_servers"] is (configured == "1")

    resumed = _run_wrapper(
        config, run_id, resume=True, env_overrides=overrides, env_unset=unset_vars
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads((run_dir / "meta.json").read_text())["run_identity"][
        "enable_smartgrid_servers"
    ] is (configured == "1")


def test_selected_server_override_bytes_bind_resume_identity(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    selected = tmp_path / "selected_iot.py"
    relocated = tmp_path / "relocated_iot.py"
    selected.write_bytes(b"# selected source v1\n")
    relocated.write_bytes(b"# selected source v1\n")
    run_id = f"server-override-hash-{tmp_path.name}"
    first = _run_wrapper(
        config, run_id, env_overrides={"SERVER_IOT_PATH": str(selected)}
    )
    assert first.returncode == 0, first.stderr
    meta_path = _run_dir(run_id) / "meta.json"
    same_bytes = _run_wrapper(
        config, run_id, resume=True, env_overrides={"SERVER_IOT_PATH": str(relocated)}
    )
    assert same_bytes.returncode == 0, same_bytes.stderr
    before = meta_path.read_bytes()
    relocated.write_bytes(b"# selected source v2\n")
    changed = _run_wrapper(
        config, run_id, resume=True, env_overrides={"SERVER_IOT_PATH": str(relocated)}
    )
    assert changed.returncode == 2
    assert "server_iot_sha256" in changed.stderr
    assert meta_path.read_bytes() == before


def test_smartgrid_server_flag_rejects_trailing_whitespace_before_run_writes(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    config.write_text(
        config.read_text().replace(
            "ENABLE_SMARTGRID_SERVERS=0", "ENABLE_SMARTGRID_SERVERS='1 '"
        )
    )
    run_id = f"flag-invalid-{tmp_path.name}"
    result = _run_wrapper(config, run_id, env_overrides={"DRY_RUN": "1"})

    assert result.returncode == 2
    assert "ENABLE_SMARTGRID_SERVERS must be exactly 0 or 1" in result.stderr
    assert not _run_dir(run_id).exists()


def test_resume_unknown_execution_gpu_refuses_before_meta_overwrite(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    run_id = f"resume-unknown-gpu-{tmp_path.name}"
    try:
        assert _run_wrapper(config, run_id).returncode == 0
        run_dir = _run_dir(run_id)
        before = (run_dir / "meta.json").read_bytes()

        resumed = _run_wrapper(
            config,
            run_id,
            resume=True,
            env_overrides={"GPU_TYPE": "unknown"},
        )

        assert resumed.returncode == 2
        assert "hardware identity unavailable" in resumed.stderr
        assert (run_dir / "meta.json").read_bytes() == before
    finally:
        shutil.rmtree(_run_dir(run_id), ignore_errors=True)


def test_resume_remote_gpu_identity_change_refuses_before_meta_overwrite(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    run_id = f"resume-remote-gpu-change-{tmp_path.name}"
    try:
        assert (
            _run_wrapper(
                config,
                run_id,
                env_overrides={"GPU_TYPE": "remote-server-a100"},
            ).returncode
            == 0
        )
        run_dir = _run_dir(run_id)
        before = (run_dir / "meta.json").read_bytes()

        resumed = _run_wrapper(
            config,
            run_id,
            resume=True,
            env_overrides={"GPU_TYPE": "remote-server-a6000"},
        )

        assert resumed.returncode == 2
        assert "gpu_type" in resumed.stderr
        assert (run_dir / "meta.json").read_bytes() == before
    finally:
        shutil.rmtree(_run_dir(run_id), ignore_errors=True)


def test_resume_changed_model_refuses_before_meta_overwrite(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    run_id = f"resume-model-{tmp_path.name}"
    try:
        assert _run_wrapper(config, run_id).returncode == 0
        run_dir = _run_dir(run_id)
        before = (run_dir / "meta.json").read_bytes()
        config.write_text(
            config.read_text().replace("MODEL_ID=model-a", "MODEL_ID=model-b")
        )

        resumed = _run_wrapper(config, run_id, resume=True)

        assert resumed.returncode == 2
        assert "refusing resume" in resumed.stderr
        assert (run_dir / "meta.json").read_bytes() == before
    finally:
        shutil.rmtree(_run_dir(run_id), ignore_errors=True)


def test_resume_changed_effective_guard_refuses_before_any_artifact_write(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    run_id = f"resume-effective-guard-{tmp_path.name}"
    assert _run_wrapper(config, run_id).returncode == 0
    run_dir = _run_dir(run_id)
    meta_before = (run_dir / "meta.json").read_bytes()
    log_before = (run_dir / "harness.log").read_bytes()
    resumed = _run_wrapper(
        config,
        run_id,
        resume=True,
        env_overrides={"ENABLE_MISSING_EVIDENCE_GUARD": " YeS "},
    )
    assert resumed.returncode == 2
    assert "missing_evidence_guard" in resumed.stderr
    assert (run_dir / "meta.json").read_bytes() == meta_before
    assert (run_dir / "harness.log").read_bytes() == log_before


def test_metadata_records_template_hash_without_template_command_text(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    secret_marker = "TEMPLATE_SECRET_MARKER"
    lines = config.read_text().splitlines()
    config.write_text(
        "\n".join(
            (
                f"HYBRID_RUNNER_TEMPLATE='{line.split('=', 1)[1]} # {secret_marker}'"
                if line.startswith("HYBRID_RUNNER_TEMPLATE=")
                else line
            )
            for line in lines
        )
        + "\n"
    )
    run_id = f"template-hash-{tmp_path.name}"
    result = _run_wrapper(config, run_id)
    assert result.returncode == 0, result.stderr
    metadata = (_run_dir(run_id) / "meta.json").read_text(encoding="utf-8")
    payload = json.loads(metadata)
    assert payload["run_identity"]["runner_template_sha256"]
    assert secret_marker not in metadata


def test_cross_date_resume_summary_uses_actual_skipped_and_rerun_paths(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    config.write_text(config.read_text().replace("TRIALS=1", "TRIALS=2"))
    project = config.parent / "project"
    runner = tmp_path / "metrics-runner.sh"
    runner.write_text(
        "#!/bin/bash\n"
        'case "$OUTPUT_PATH" in\n'
        "  *_run01.json.tmp) input=100; output=10; tools=1 ;;\n"
        '  *) count_file="${OUTPUT_PATH}.count"; count=0; '
        '[ ! -f "$count_file" ] || count=$(cat "$count_file"); '
        'count=$((count + 1)); printf "%s" "$count" > "$count_file"; '
        'if [ "$count" -eq 1 ]; then input=900; output=90; tools=9; '
        "else input=200; output=20; tools=2; fi ;;\n"
        "esac\n"
        'printf \'{"success":true,"answer":"ok","usage":{"input_tokens":%s,"output_tokens":%s,"total_tokens":%s},"tool_call_count":%s,"history":[]}\\n\' \\\n'
        '  "$input" "$output" "$((input + output))" "$tools" > "$OUTPUT_PATH"\n',
        encoding="utf-8",
    )
    runner.chmod(0o755)
    config.write_text(
        config.read_text().replace(
            f"HYBRID_RUNNER_TEMPLATE={config.parent}/runner.sh",
            f"HYBRID_RUNNER_TEMPLATE={runner}",
        )
    )
    run_id = f"cross-date-summary-{tmp_path.name}"
    first = _run_wrapper(config, run_id)
    assert first.returncode == 0, first.stderr
    run_dir = _run_dir(run_id)
    current_trial = next(run_dir.glob("*_scenario_run01.json"))
    previous_trial = run_dir / f"2026-11-13{current_trial.name[10:]}"
    current_trial.rename(previous_trial)
    latency_file = run_dir / "latencies.jsonl"
    rows = [json.loads(line) for line in latency_file.read_text().splitlines() if line]
    previous_path = previous_trial.relative_to(project).as_posix()
    rows = [
        {
            **row,
            "output_path": previous_path,
        }
        for row in rows
        if row.get("trial_index") == 1
    ]
    latency_file.write_text("".join(json.dumps(row) + "\n" for row in rows))

    resumed = _run_wrapper(config, run_id, resume=True)
    assert resumed.returncode == 0, resumed.stderr
    summary = json.loads(
        (run_dir.parents[1] / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["scenarios_attempted"] == 2
    assert summary["scenarios_completed"] == 2
    assert summary["resume_skipped_count"] == 1
    assert summary["resume_rerun_count"] == 1
    assert summary["input_tokens_total"] == 300
    assert summary["output_tokens_total"] == 30
    assert summary["total_tokens_total"] == 330
    assert summary["tool_call_count_total"] == 3
    assert summary["tokens_usage_trial_count"] == 2
    assert previous_trial.is_file()


def test_resume_changed_scenario_text_refuses_before_meta_overwrite(
    tmp_path: Path,
) -> None:
    config, scenario = _fixture(tmp_path)
    run_id = f"resume-scenario-{tmp_path.name}"
    try:
        assert _run_wrapper(config, run_id).returncode == 0
        run_dir = _run_dir(run_id)
        before = (run_dir / "meta.json").read_bytes()
        scenario.write_text(json.dumps({"id": "SGT-SMOKE", "text": "prompt-b"}) + "\n")

        resumed = _run_wrapper(config, run_id, resume=True)

        assert resumed.returncode == 2
        assert "refusing resume" in resumed.stderr
        assert (run_dir / "meta.json").read_bytes() == before
    finally:
        shutil.rmtree(_run_dir(run_id), ignore_errors=True)


@pytest.mark.parametrize(
    ("variable", "initial", "changed", "identity_field"),
    [
        ("VLLM_DTYPE", "float16", "bfloat16", "vllm_dtype"),
        (
            "VLLM_MODEL_PATH",
            "models/model-a",
            "models/model-b",
            "vllm_model_path",
        ),
    ],
)
def test_resume_changed_effective_runtime_setting_refuses_before_meta_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    initial: str,
    changed: str,
    identity_field: str,
) -> None:
    config, _ = _fixture(tmp_path)
    run_id = f"resume-runtime-{variable.lower()}-{tmp_path.name}"
    try:
        monkeypatch.setenv(variable, initial)
        assert _run_wrapper(config, run_id).returncode == 0
        run_dir = _run_dir(run_id)
        before = (run_dir / "meta.json").read_bytes()
        monkeypatch.setenv(variable, changed)

        resumed = _run_wrapper(config, run_id, resume=True)

        assert resumed.returncode == 2
        assert identity_field in resumed.stderr
        assert (run_dir / "meta.json").read_bytes() == before
    finally:
        shutil.rmtree(_run_dir(run_id), ignore_errors=True)


def test_child_failure_with_success_payload_returns_nonzero(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    runner = tmp_path / "runner.sh"
    runner.write_text(
        "#!/bin/bash\n"
        'printf \'%s\\n\' \'{"success":true,"answer":"ok","history":[]}\' > "$OUTPUT_PATH"\n'
        "exit 23\n"
    )
    runner.chmod(0o755)
    run_id = f"child-failure-{tmp_path.name}"
    try:
        result = _run_wrapper(config, run_id)
        assert result.returncode == 1
        assert "Infrastructure failure detected" in result.stdout
        manifest = json.loads(
            (_run_dir(run_id) / "resume_manifest.jsonl").read_text().splitlines()[-1]
        )
        assert manifest["reason"] == "execution_failed:23"
        assert not list(_run_dir(run_id).glob("*_run*.json"))
    finally:
        shutil.rmtree(_run_dir(run_id), ignore_errors=True)


def test_resume_helper_error_does_not_reuse_previous_trial_state(
    tmp_path: Path,
) -> None:
    temp_repo = tmp_path / "repo"
    temp_repo.mkdir()
    (temp_repo / "scripts").mkdir()
    shutil.copy2(RUN_SCRIPT, temp_repo / "scripts" / "run_experiment.sh")
    (temp_repo / "data").symlink_to(REPO_ROOT / "data", target_is_directory=True)
    (temp_repo / "scripts" / "gcp_resume_state.py").write_text("""#!/usr/bin/env python3
import json
import pathlib
import sys

_RUN_IDENTITY_FIELDS = ()
def _shell_flag_enabled(value):
    return str(value or '') == '1'
def _processed_data_sha256(_repo_root):
    return {}
def _runner_code_sha256(_repo_root):
    return 'fixture-code-hash'
def effective_behavior_identity(**_kwargs):
    return {
        'missing_evidence_guard': False,
        'missing_evidence_repair': False,
        'explicit_fault_risk_adjudication': False,
        'missing_evidence_repair_max_attempts': 0,
        'missing_evidence_repair_max_attempts_per_target': 0,
        'enable_self_ask': False,
        'plan_execute_repo_local': False,
        'aat_openai_agents_version': None,
        'aat_mcp_version': None,
        'aat_litellm_version': None,
        'aat_mcp_client_timeout_seconds': None,
        'runner_template_sha256': None,
        'server_iot_sha256': None,
        'server_fmsr_sha256': None,
        'server_tsfm_sha256': None,
        'server_wo_sha256': None,
    }

if __name__ == '__main__':
    args = sys.argv[1:]
    command = args[0]
    def value(name):
        return args[args.index(name) + 1]
    if command == 'interpreter-identity':
        print(json.dumps({'version': value('--version'), 'binary_sha256': 'fixture'}))
        raise SystemExit(0)
    if command == 'validate-run-identity':
        raise SystemExit(0)
    if command == 'trial-status-shell':
        count_path = pathlib.Path(value('--run-dir')) / 'helper-count'
        count = int(count_path.read_text()) if count_path.exists() else 0
        count_path.write_text(str(count + 1))
        if count == 0:
            print('RESUME_STATE=complete_success')
            print('RESUME_COMPLETE=1')
            print('RESUME_SUCCESS=1')
            print('RESUME_OUTPUT_PATH=' + value('--output-path'))
            print('RESUME_REASON=terminal_trial')
            raise SystemExit(0)
        raise SystemExit(7)
    if command == 'preserve-incomplete' or command == 'manifest-event':
        raise SystemExit(0)
    if command == 'finalize-trial-shell':
        print('FINAL_STATE=complete_success')
        print('FINAL_SUCCESS=1')
        print('FINAL_REASON=trial_executed')
        raise SystemExit(0)
    raise SystemExit(2)
""")
    (temp_repo / "scripts" / "gcp_resume_state.py").chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=temp_repo, check=True)
    subprocess.run(["git", "add", "."], cwd=temp_repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=temp_repo,
        check=True,
    )
    aob = tmp_path / "aob"
    aob.mkdir()
    (aob / "pyproject.toml").write_text("[build-system]\nrequires = []\n")
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    for name in ("one", "two"):
        (scenario_dir / f"{name}.json").write_text(
            json.dumps({"id": f"SGT-{name}", "text": name}) + "\n"
        )
    child = tmp_path / "child.sh"
    child.write_text(
        "#!/bin/bash\n"
        'printf \'%s\\n\' \'{"success":true,"answer":"ok","history":[]}\' > "$OUTPUT_PATH"\n'
        'printf x >> "$REPO_ROOT/child-count"\n'
    )
    child.chmod(0o755)
    config = tmp_path / "config.env"
    config.write_text(
        "\n".join(
            [
                "EXPERIMENT_NAME=stale_state_smoke",
                "EXPERIMENT_CELL=Z",
                "EXPERIMENT_FAMILY=smoke",
                f"SCENARIOS_GLOB={scenario_dir}/*.json",
                "SCENARIO_SET_NAME=smoke-set",
                "MODEL_ID=model-a",
                "MODEL_PROVIDER=local",
                "ORCHESTRATION=hybrid",
                "MCP_MODE=baseline",
                "TRIALS=1",
                "LAUNCH_VLLM=0",
                "ENABLE_SMARTGRID_SERVERS=0",
                f"AOB_PATH={aob}",
                f"HYBRID_RUNNER_TEMPLATE={child}",
            ]
        )
        + "\n"
    )
    run_id = f"stale-state-{tmp_path.name}"
    env = os.environ.copy()
    env.update({"SMARTGRID_RUN_ID": run_id, "SMARTGRID_RESUME": "1"})
    result = subprocess.run(
        ["bash", "scripts/run_experiment.sh", str(config)],
        cwd=temp_repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "resume status helper failed" in result.stderr
    assert len((temp_repo / "child-count").read_text()) == 1
