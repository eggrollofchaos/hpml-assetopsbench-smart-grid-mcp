---
title: EVD-009 Resume Provenance And Exit Propagation Closeout
slug: evd009-resume-provenance-and-exit-propagation-closeout
type: retrospective
status: live
created: 2026-09-24
updated: 2026-11-14
owner: Team 13
scope: project
project: hpml-assetopsbench-smart-grid-mcp
tags: [experiment-wrapper, resume, provenance, failure-propagation]
work-items: []
related:
  - scripts/run_experiment.sh
  - scripts/gcp_resume_state.py
  - tests/test_run_experiment_resume_guards.py
  - docs/knowledge-base/learnings/2026-06-05_aat-smoke-runtime-preflight.md
agent: Codex
agent-provider: OpenAI
agent-interface: Codex Desktop
agent-session-id: 01a0d213-f261-71d1-ad9c-39a09bf4b316
session-label: "FND: Guard resume provenance and exits"
invocation-context: "session-closeout: closeable"
session-lifecycle: closeable
session-closeout-note: .agent-sessions/closed/session-closeout-01a0d213-f261-71d1-ad9c-39a09bf4b316.md
---

# EVD-009 Resume Provenance And Exit Propagation Closeout - 2026-09-24

## Metadata

- Unit: EVD-009 C03/H01 experiment-wrapper repair
- Unit type: Correctness fix with offline regression coverage
- Status: Complete on local feature branch; parent integration remains
- Repo: hpml-assetopsbench-smart-grid-mcp
- Branch / commit: `codex-fnd/resume-guards-20260924` / `2e5edb5e777f7f2a05641a325c71f6bd40696b77`
- Work item: EVD-009-C03-H01
- Agent: Codex / OpenAI / Codex Desktop
- Agent session ID: 01a0d213-f261-71d1-ad9c-39a09bf4b316
- Parent context: delegated implementation lane for the full-codebase audit
- Sources inspected: frozen audit finding C03/H01; `scripts/run_experiment.sh`; `scripts/gcp_resume_state.py`; focused and full test suites; parent follow-up reproduction; project coordination overlay

## 1. Work Completed

| What | Why | How | Evidence |
|---|---|---|---|
| Added immutable resume identity validation before metadata writes | A resumed run could be relabeled after model/scenario drift | Bind config, runner code, Git revision, scenario set, model/decode settings, serving/quantization, and effective vLLM settings; refuse mismatch before creating/truncating run artifacts | Commit `2e5edb5`; changed model and scenario-text regressions; env-only dtype/model-path probe returns 2 |
| Propagated child, helper, manifest, and finalizer failures | Shell status masking could report success after failed execution or reuse stale state | Check every helper status, clear per-trial shell variables, reject success-shaped payloads with nonzero child exit, and return nonzero infrastructure status | Commit `2e5edb5`; child exit-23 and helper-exit-7 regressions |
| Sent parent completion and follow-up receipts | Parent owns integration and review/landing | Reported commit, reproduction, tests, and no-push/no-GPU boundary via native task message | Parent task `019e7468-1e59-7ff0-b90f-73a1ab116ca2` |

## 2. Decisions Addressed

| Item | Type | Resolution | Rationale | Evidence |
|---|---|---|---|---|
| What counts as resumable-run identity? | decision | Effective runtime inputs include inherited/default vLLM path, dtype, served name, parser, generation config, extra args, serving stack, quantization, and launch mode, not only config-file bytes | Environment overrides can change evidence while leaving the source config hash unchanged | Parent follow-up reproduction; `gcp_resume_state.py` identity fields |
| How should failed child output be treated? | decision | Nonzero child exit is infrastructure failure even when the output JSON claims success | Exit status is authoritative for execution health; success-shaped bytes must not mask a crash | `finalize_trial`; child-failure regression |

## 3. Issues Encountered And Resolved

| Issue | Impact | Resolution | Verification | Prevention / Learning |
|---|---|---|---|---|
| Initial identity guard omitted effective runtime overrides | Same-config resume with changed `VLLM_DTYPE` or `VLLM_MODEL_PATH` was accepted | Added the effective serving/quantization/vLLM field set to both writer and validator | Before fix: exit 0; after fix: exit 2 naming identity mismatches | Source config hashing is insufficient when defaults or environment overrides participate in execution |
| Ambient Python lacked `mcp` during an early full-suite attempt | Initial collection result looked environment-blocked | Re-ran with the repository `.venv` selected by project instructions | `256 passed, 3 subtests passed in 25.92s` |

## 4. Remaining Ideas, Decisions, Questions

| Item | Type | Priority | Time Horizon | Owner / Next Action | Tracking |
|---|---|---|---|---|---|
| Parent integration and review of `2e5edb5` | follow-up | P1 | immediate | Parent task owner; integrate through normal branch/review workflow | Parent task `019e7468-1e59-7ff0-b90f-73a1ab116ca2` |
| Broader audit lanes | follow-up | P1 | near-term | Sibling lanes retain ownership of their assigned findings | Parent coordination state |

## 5. Remaining Issues

| Issue | Risk | Priority | Time Horizon | Owner / Next Action | Tracking |
|---|---|---|---|---|---|
| No remote publication or PR was performed in this delegated lane | Integration is not yet on the parent branch | P1 | immediate | Parent decides landing path; no push was authorized in this lane | Parent receipt |

## 6. Learnings

### Local

- Resume identity must be checked before `mkdir`, metadata writes, or harness truncation.
- A helper's nonzero status must be observed before evaluating its stdout as shell assignments; otherwise stale variables can turn a failed trial into a skip.

### Project

- Runtime preflight and artifact provenance are coupled: the same effective serving/runtime settings that make a smoke reproducible must be included in the run identity.

### Global Candidates

- No separate global KB write was made. The reusable rule is captured in this project retrospective and the regression suite; promote only if another repository needs the same resume contract.

## 7. Strategic Fit

- Task / sprint: EVD-009 audit repair
- Epic / initiative: SmartGridBench full-codebase correctness audit
- Repo / project: HPML Spring 2026 Team 13 benchmark harness
- Global framework: Fail closed on evidence identity; propagate infrastructure failures to callers
