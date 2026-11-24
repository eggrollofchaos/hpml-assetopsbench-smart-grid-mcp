"""Regression coverage for label-bound failure-taxonomy audit decisions."""

from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import audit_failure_taxonomy as audit
from scripts import build_failure_taxonomy as build


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _configure_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    taxonomy = root / "results/metrics/failure_taxonomy_current.csv"
    monkeypatch.setattr(audit, "REPO_ROOT", root)
    monkeypatch.setattr(audit, "TAXONOMY_CSV", taxonomy)
    monkeypatch.setattr(audit, "DECISIONS_TSV", root / "data/audit/decisions.tsv")
    return taxonomy


def _decision(reviewed_label: str = "low_task_completion") -> dict[str, str]:
    return {
        "run_name": "run-1",
        "scenario_id": "SGT-001",
        "trial_index": "1",
        "audit_decision": "confirmed",
        "berkeley_label": "task_verification",
        "failure_stage": "verification",
        "audit_note": "Observed evidence supports the label.",
        "audit_decision_source": "reviewer:2026-11-14",
        "reviewed_auto_taxonomy_label": reviewed_label,
    }


def _taxonomy_row(label: str = "low_task_completion") -> dict[str, str]:
    return {
        "run_name": "run-1",
        "scenario_id": "SGT-001",
        "trial_index": "1",
        "auto_taxonomy_label": label,
        "audit_status": "stratified_sample",
    }


def test_matching_binding_merges_and_replays_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taxonomy = _configure_paths(monkeypatch, tmp_path)
    fields = list(_taxonomy_row())
    _write_csv(taxonomy, fields, [_taxonomy_row()])
    _write_tsv(audit.DECISIONS_TSV, list(_decision()), [_decision()])

    args = Namespace(decisions=None)
    assert audit.cmd_add_columns(args) == 0
    first = taxonomy.read_bytes()
    with taxonomy.open(encoding="utf-8", newline="") as f:
        merged = next(csv.DictReader(f))
    assert merged["audit_status"] == "manual_confirmed"
    assert [merged[c] for c in audit.AUDIT_COLUMNS] == [
        _decision()[c] for c in audit.AUDIT_COLUMNS
    ]

    assert audit.cmd_add_columns(args) == 0
    assert taxonomy.read_bytes() == first


def test_changed_label_refuses_before_target_csv_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taxonomy = _configure_paths(monkeypatch, tmp_path)
    _write_csv(taxonomy, list(_taxonomy_row()), [_taxonomy_row("low_hallucinations")])
    _write_tsv(audit.DECISIONS_TSV, list(_decision()), [_decision()])
    before = taxonomy.read_bytes()

    with pytest.raises(ValueError, match="does not match current label"):
        audit.cmd_add_columns(Namespace(decisions=None))

    assert taxonomy.read_bytes() == before


def test_unbound_sidecar_refuses_before_target_csv_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taxonomy = _configure_paths(monkeypatch, tmp_path)
    _write_csv(taxonomy, list(_taxonomy_row()), [_taxonomy_row()])
    unbound_fields = [c for c in _decision() if c != "reviewed_auto_taxonomy_label"]
    unbound_decision = {c: _decision()[c] for c in unbound_fields}
    _write_tsv(audit.DECISIONS_TSV, unbound_fields, [unbound_decision])
    before = taxonomy.read_bytes()

    with pytest.raises(ValueError, match="not label-bound"):
        audit.cmd_add_columns(Namespace(decisions=None))

    assert taxonomy.read_bytes() == before


def test_blank_binding_refuses_before_target_csv_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taxonomy = _configure_paths(monkeypatch, tmp_path)
    _write_csv(taxonomy, list(_taxonomy_row()), [_taxonomy_row()])
    decision = _decision("")
    _write_tsv(audit.DECISIONS_TSV, list(decision), [decision])
    before = taxonomy.read_bytes()

    with pytest.raises(ValueError, match="missing reviewed_auto_taxonomy_label"):
        audit.cmd_add_columns(Namespace(decisions=None))

    assert taxonomy.read_bytes() == before


def test_duplicate_key_refuses_before_target_csv_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taxonomy = _configure_paths(monkeypatch, tmp_path)
    _write_csv(taxonomy, list(_taxonomy_row()), [_taxonomy_row()])
    duplicate_sidecar = tmp_path / "custom-decisions.tsv"
    stale = _decision("low_hallucinations")
    current = _decision("low_task_completion")
    _write_tsv(duplicate_sidecar, list(stale), [stale, current])
    before = taxonomy.read_bytes()

    with pytest.raises(ValueError, match="duplicate decision key"):
        audit.main(["add-columns", "--decisions", str(duplicate_sidecar)])

    assert taxonomy.read_bytes() == before


def test_duplicate_label_binding_headers_refuse_before_target_csv_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taxonomy = _configure_paths(monkeypatch, tmp_path)
    _write_csv(taxonomy, list(_taxonomy_row()), [_taxonomy_row()])
    duplicate_sidecar = tmp_path / "custom-duplicate-headers.tsv"
    stale = _decision("low_hallucinations")
    current = _decision("low_task_completion")
    headers = list(stale) + ["reviewed_auto_taxonomy_label", "extra_unique_note"]
    values = [stale[column] for column in stale]
    values.extend([current["reviewed_auto_taxonomy_label"], "allowed unique column"])
    with duplicate_sidecar.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(headers)
        writer.writerow(values)
    before = taxonomy.read_bytes()

    with pytest.raises(ValueError, match="duplicate column names"):
        audit.main(["add-columns", "--decisions", str(duplicate_sidecar)])

    assert taxonomy.read_bytes() == before


def test_existing_46_decisions_replay_and_regenerated_labels_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = audit.REPO_ROOT
    old_sidecar_path = repo_root / "data/audit/issue194_decisions.tsv"
    bound_sidecar_path = repo_root / "data/audit/issue194_decisions_labelbound.tsv"
    canonical_taxonomy = repo_root / "results/metrics/failure_taxonomy_current.csv"
    old_bytes = old_sidecar_path.read_bytes()

    with old_sidecar_path.open(encoding="utf-8", newline="") as f:
        old_decisions = list(csv.DictReader(f, delimiter="\t"))
    with bound_sidecar_path.open(encoding="utf-8", newline="") as f:
        bound_decisions = list(csv.DictReader(f, delimiter="\t"))

    def key(row: dict[str, str]) -> tuple[str, str, int]:
        return (row["run_name"], row["scenario_id"], int(row["trial_index"]))

    assert len(old_decisions) == len(bound_decisions) == 46
    assert [
        {column: row[column] for column in old_decisions[0]} for row in bound_decisions
    ] == old_decisions

    # Replaying into a scratch copy preserves the historical manual decisions.
    scratch = tmp_path / "replay"
    taxonomy = _configure_paths(monkeypatch, scratch)
    taxonomy.parent.mkdir(parents=True, exist_ok=True)
    taxonomy.write_bytes(canonical_taxonomy.read_bytes())
    decisions_path = scratch / "data/audit/decisions.tsv"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_bytes(bound_sidecar_path.read_bytes())
    monkeypatch.setattr(audit, "DECISIONS_TSV", decisions_path)

    with taxonomy.open(encoding="utf-8", newline="") as f:
        before_rows = list(csv.DictReader(f))
    before_by_key = {key(row): row for row in before_rows}
    assert len(before_by_key) == len(before_rows)
    assert audit.cmd_add_columns(Namespace(decisions=None)) == 0
    with taxonomy.open(encoding="utf-8", newline="") as f:
        after_rows = list(csv.DictReader(f))
    after_by_key = {key(row): row for row in after_rows}
    assert len(after_rows) == len(before_rows)
    assert sum(row["audit_status"] == "manual_confirmed" for row in after_rows) == 46
    by_decision = {key(row): row for row in bound_decisions}
    for row_key, decision in by_decision.items():
        assert [before_by_key[row_key][c] for c in audit.AUDIT_COLUMNS] == [
            decision[c] for c in audit.AUDIT_COLUMNS
        ]
        assert [after_by_key[row_key][c] for c in audit.AUDIT_COLUMNS] == [
            decision[c] for c in audit.AUDIT_COLUMNS
        ]
        assert [after_by_key[row_key][c] for c in audit.AUDIT_COLUMNS] == [
            before_by_key[row_key][c] for c in audit.AUDIT_COLUMNS
        ]
    assert old_sidecar_path.read_bytes() == old_bytes

    # Generate corrected labels entirely in memory from the source inputs,
    # retain the historical sample keys in scratch, and prove the bound audit
    # refuses before touching this changed-label CSV.
    regenerated = build.build_rows(
        repo_root / "results/metrics/scenario_scores.jsonl",
        repo_root / "results/metrics/evidence_registry.csv",
    )
    regenerated_by_key = {key(row): row for row in regenerated}
    assert set(by_decision) <= set(regenerated_by_key)
    corrected = [dict(row) for row in regenerated]
    for row in corrected:
        row["audit_status"] = (
            "stratified_sample" if key(row) in by_decision else row["audit_status"]
        )
    assert any(
        regenerated_by_key[row_key]["auto_taxonomy_label"]
        != decision["reviewed_auto_taxonomy_label"]
        for row_key, decision in by_decision.items()
    )

    corrected_root = tmp_path / "regenerated"
    corrected_taxonomy = corrected_root / "results/metrics/failure_taxonomy_current.csv"
    _write_csv(corrected_taxonomy, build.OUTPUT_COLUMNS, corrected)
    monkeypatch.setattr(audit, "REPO_ROOT", corrected_root)
    monkeypatch.setattr(audit, "TAXONOMY_CSV", corrected_taxonomy)
    monkeypatch.setattr(audit, "DECISIONS_TSV", bound_sidecar_path)
    unchanged = corrected_taxonomy.read_bytes()

    with pytest.raises(ValueError, match="does not match current label"):
        audit.cmd_add_columns(Namespace(decisions=None))

    assert corrected_taxonomy.read_bytes() == unchanged
