"""Synthetic regression coverage for the legacy failure-evidence renderer."""

from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import audit_failure_evidence as audit
from scripts import build_failure_taxonomy as build


def _score(**overrides: object) -> dict:
    score = {
        "run_name": "run-1",
        "scenario_id": "SGT-001",
        "trial_index": 1,
        "judge_model": "judge-a",
        "judge_prompt_version": "prompt-v1",
        "dim_task_completion": True,
        "dim_data_retrieval_accuracy": True,
        "dim_agent_sequence_correct": True,
        "dim_hallucinations": False,
        "dim_generalized_result_verification": True,
        "dim_clarity_and_justification": True,
        "score_6d": 1.0,
        "suggestions": "",
    }
    score.update(overrides)
    return score


def _evidence_row() -> dict[str, str]:
    return {
        "cell": "Y",
        "run_name": "run-1",
        "scenario_id": "SGT-001",
        "trial_index": "1",
        "taxonomy_label": "low_task_completion",
        "symptom": "synthetic symptom",
        "candidate_mitigation": "synthetic mitigation",
        "evidence_note": "synthetic fixture",
        "audit_status": "confirmed",
    }


def _configure_render(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "_load_evidence_rows", lambda: [_evidence_row()])


def test_failed_dimensions_match_builder_polarity_and_canonical_order() -> None:
    score = _score(
        dim_hallucinations=True,
        dim_task_completion=False,
        dim_agent_sequence_correct=False,
    )

    assert audit._failed_dimensions(score) == [
        "task_completion",
        "agent_sequence_correct",
        "hallucinations",
    ]
    assert audit._failed_dimensions(score) == build.auto_taxonomy(score)[1]


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda score: score.pop("dim_task_completion"), "missing dimensions"),
        (lambda score: score.__setitem__("dim_hallucinations", None), "non-boolean"),
        (lambda score: score.__setitem__("dim_task_completion", 0), "non-boolean"),
        (
            lambda score: score.__setitem__("dim_clarity_and_justification", "false"),
            "non-boolean",
        ),
    ],
)
def test_malformed_rubric_refuses_before_briefs_output_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change,
    message: str,
) -> None:
    score = _score()
    change(score)
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(json.dumps(score) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "SCORES_JSONL", scores_path)
    monkeypatch.setattr(audit, "_load_evidence_rows", lambda: [_evidence_row()])
    destination = tmp_path / "briefs.md"
    destination.write_bytes(b"protected-existing-output\n")

    with pytest.raises(ValueError, match=message):
        audit.cmd_briefs(Namespace(out=str(destination)))

    assert destination.read_bytes() == b"protected-existing-output\n"


def test_briefs_retain_distinct_judge_variants_and_last_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    records = [
        _score(
            judge_model=model,
            judge_prompt_version=prompt,
            score_6d=mark,
            trajectory_file=f"benchmarks/{model}-{prompt}.json",
        )
        for model, prompt, mark in (
            ("judge-a", "prompt-v1", 0.1),
            ("judge-a", "prompt-v2", 0.2),
            ("judge-b", "prompt-v1", 0.3),
            ("judge-b", "prompt-v2", 0.4),
        )
    ]
    records.append(
        _score(
            judge_model="judge-a",
            judge_prompt_version="prompt-v1",
            score_6d=0.9,
            trajectory_file="benchmarks/judge-a-prompt-v1-last.json",
        )
    )
    records.append(
        _score(
            judge_model="judge-c",
            judge_prompt_version="assetopsbench-6d-v1",
            score_6d=0.7,
            trajectory_file="benchmarks/judge-c-explicit-v1.json",
        )
    )
    legacy_default = _score(
        judge_model="judge-c",
        score_6d=0.8,
        trajectory_file="benchmarks/judge-c-legacy-v1.json",
    )
    legacy_default.pop("judge_prompt_version")
    records.append(legacy_default)
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    for record in records:
        trajectory = tmp_path / record["trajectory_file"]
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        trajectory.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit, "SCORES_JSONL", scores_path)
    monkeypatch.setattr(audit, "_load_evidence_rows", lambda: [_evidence_row()])
    destination = tmp_path / "briefs.md"

    assert audit.cmd_briefs(Namespace(out=str(destination))) == 0
    progress = capsys.readouterr().out
    output = destination.read_text(encoding="utf-8")

    assert progress == f"Wrote {destination} (evidence rows: 1)\n"
    assert output.count("JSONL judge identity:") == 5
    assert "judge_model='judge-a'; judge_prompt_version='prompt-v1'" in output
    assert "judge_model='judge-a'; judge_prompt_version='prompt-v2'" in output
    assert "judge_model='judge-b'; judge_prompt_version='prompt-v1'" in output
    assert "judge_model='judge-b'; judge_prompt_version='prompt-v2'" in output
    assert "judge_model='judge-c'; judge_prompt_version='assetopsbench-6d-v1'" in output
    assert "JSONL score_6d=0.9" in output
    assert "JSONL score_6d=0.1" not in output
    assert "JSONL score_6d=0.8" in output
    assert "JSONL score_6d=0.7" not in output
    assert "judge-a-prompt-v1-last.json" in output
    assert "judge-c-legacy-v1.json" in output


def test_legacy_prompt_and_model_identity_match_writer_coercion() -> None:
    legacy = _score(judge_prompt_version=None, judge_model=None)
    assert audit._score_identity(legacy) == (
        "run-1",
        "SGT-001",
        1,
        "None",
        "assetopsbench-6d-v1",
    )
    assert audit._score_identity(_score(judge_prompt_version=""))[4] == (
        "assetopsbench-6d-v1"
    )


def test_no_matching_judge_record_emits_one_unmatched_brief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(audit, "SCORES_JSONL", scores_path)
    monkeypatch.setattr(audit, "_load_evidence_rows", lambda: [_evidence_row()])

    assert audit.cmd_briefs(Namespace(out=None)) == 0
    output = capsys.readouterr().out

    assert output.count("(no matching record for run+scenario+trial)") == 1


def test_complete_crlf_frontmatter_is_preserved_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_render(monkeypatch)
    destination = tmp_path / "report.md"
    crlf = bytes((13, 10))
    header = crlf.join([b"---", b"title: retained", b"owner: test", b"---", b""])
    destination.write_bytes(header + b"old body sentinel")

    assert audit.cmd_render(Namespace(out=str(destination))) == 0

    rendered = destination.read_bytes()
    assert rendered.startswith(header + b"\n# Failure-evidence audit")
    assert b"old body sentinel" not in rendered


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
def test_utf8_bom_frontmatter_is_preserved_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, newline: bytes
) -> None:
    _configure_render(monkeypatch)
    destination = tmp_path / "report.md"
    header = b"\xef\xbb\xbf" + newline.join(
        [b"---", b"title: retained", b"owner: test", b"---", b""]
    )
    destination.write_bytes(header + b"old body sentinel")

    assert audit.cmd_render(Namespace(out=str(destination))) == 0

    rendered = destination.read_bytes()
    assert rendered.startswith(header + b"\n# Failure-evidence audit")
    assert b"owner: test" in rendered
    assert b"old body sentinel" not in rendered


def test_unterminated_bom_frontmatter_refuses_without_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_render(monkeypatch)
    destination = tmp_path / "report.md"
    sentinel = b"\xef\xbb\xbf---\r\ntitle: incomplete\r\nbody sentinel\r\n"
    destination.write_bytes(sentinel)

    with pytest.raises(ValueError, match="unterminated"):
        audit.cmd_render(Namespace(out=str(destination)))

    assert destination.read_bytes() == sentinel


def test_missing_destination_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_render(monkeypatch)
    destination = tmp_path / "new-report.md"

    assert audit.cmd_render(Namespace(out=str(destination))) == 0

    assert destination.read_bytes().startswith(b"# Failure-evidence audit")


def test_destination_io_error_preserves_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_render(monkeypatch)
    destination = tmp_path / "report.md"
    sentinel = b"---\nowner: protected\n---\nbody sentinel\n"
    destination.write_bytes(sentinel)
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path == destination:
            raise PermissionError("denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(PermissionError, match="denied"):
        audit.cmd_render(Namespace(out=str(destination)))

    assert original_read_bytes(destination) == sentinel


def test_destination_decode_error_preserves_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_render(monkeypatch)
    destination = tmp_path / "report.md"
    sentinel = b"\xff---\nbody sentinel\n"
    destination.write_bytes(sentinel)

    with pytest.raises(UnicodeDecodeError):
        audit.cmd_render(Namespace(out=str(destination)))

    assert destination.read_bytes() == sentinel


def test_unterminated_frontmatter_refuses_without_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_render(monkeypatch)
    destination = tmp_path / "report.md"
    sentinel = b"---\ntitle: incomplete\nbody sentinel\n"
    destination.write_bytes(sentinel)

    with pytest.raises(ValueError, match="unterminated"):
        audit.cmd_render(Namespace(out=str(destination)))

    assert destination.read_bytes() == sentinel


def test_symlink_destination_refuses_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_render(monkeypatch)
    target = tmp_path / "target.md"
    target.write_bytes(b"target sentinel")
    destination = tmp_path / "report.md"
    destination.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        audit.cmd_render(Namespace(out=str(destination)))

    assert target.read_bytes() == b"target sentinel"
    assert destination.is_symlink()


def test_direct_script_and_package_import_work() -> None:
    package_import = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts import audit_failure_evidence; "
            "assert audit_failure_evidence.DEFAULT_JUDGE_PROMPT_VERSION == "
            "'assetopsbench-6d-v1'",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    direct_script = subprocess.run(
        [sys.executable, str(Path(audit.__file__).resolve()), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert package_import.returncode == 0, package_import.stderr
    assert direct_script.returncode == 0, direct_script.stderr
    assert "usage:" in direct_script.stdout
