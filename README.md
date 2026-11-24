---
status: canonical-index
scope: team-repo
owner: Team 13
canonical: true
---

# HPML Final Project: MCP-Based Industrial Agent Benchmarking for Smart Grid Operations

> **Course:** COMS E6998: High Performance Machine Learning
> **Institution:** Columbia University
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui
> **Mentor:** Dr. Dhaval Patel, IBM Research

---

## Team Information

- **Team Name:** Team 13 / District 1101 (SmartGridBench)
- **Members:**
  - Akshat Bhandari (ab6174) — Scenarios and evaluation: 6-dimension LLM-as-judge
    scorer and eval harness, judge audit logs, scenario corpus expansion, generated-
    scenario validation, failure-taxonomy classification and audit, DGA report card
  - Aaron Fan (af3623) — Infrastructure and serving: Insomnia/Slurm and GCP setup,
    vLLM serving (KV-cache and INT8 studies), profiling capture with W&B and torch
    profiler, shared Agent-as-Tool runner, automated scenario generator
  - Tanisha Rathod (tr2828) — Data and MCP servers: processed-data pipeline from
    public datasets, four MCP server domains with hardening and tests, IEC 60599 /
    IEEE C57.104 knowledge plugin, hand-crafted scenarios, W&B evidence dashboard
  - Wei Alexander Xin (wax1) — Team lead / PM: experiment design and runner,
    Self-Ask and Verified Plan-Execute orchestration, Agent-as-Tool parity
    validation, failure-taxonomy pipeline, DGA fidelity validator, analysis
    notebooks, NeurIPS and class final reports, presentation
- **Mentor:** Dr. Dhaval Patel, IBM Research

## Submission

- **GitHub repository:** https://github.com/HPML6998-S26-Team13/hpml-assetopsbench-smart-grid-mcp
- **Final presentation:** [`deliverables/SmartGridBench-Final-Presentation.pptx`](deliverables/SmartGridBench-Final-Presentation.pptx) and [`deliverables/SmartGridBench-Final-Presentation.pdf`](deliverables/SmartGridBench-Final-Presentation.pdf)
- **Final report:** [`deliverables/SmartGridBench_Final_Paper.pdf`](deliverables/SmartGridBench_Final_Paper.pdf)
- **Experiment-tracking dashboard:** https://wandb.ai/assetopsbench-smartgrid/assetopsbench-smartgrid/reports/SmartGridBench-Final-Evidence-Dashboard--VmlldzoxNjgyODI4NA
- **Upstream proposals (open, awaiting review):** [IBM/AssetOpsBench #287](https://github.com/IBM/AssetOpsBench/pull/287) and [#292](https://github.com/IBM/AssetOpsBench/pull/292)

## Upstream Contributions

Two pull requests propose the Smart Grid domain to IBM's AssetOpsBench. As of November 24, 2026, both are open, ready for review and passing the DCO check; neither has been merged.

- [IBM/AssetOpsBench #287](https://github.com/IBM/AssetOpsBench/pull/287) (opened May 7, 2026): Smart Grid transformer MCP servers (IoT, FMSR/DGA, TSFM/RUL, work orders), a direct-call adapter, the 36-scenario corpus plus 5 negative-check fixtures, data-provenance documentation and unit tests.
- [IBM/AssetOpsBench #292](https://github.com/IBM/AssetOpsBench/pull/292) (opened May 11, 2026): the 25 scenarios added after the paper freeze, stacked on #287, bringing the proposed upstream corpus to 61.

## 1. Problem Statement

<p align="center">
  <img src="docs/images/power-transformer-substation.jpg" alt="Power transformer substation" width="600">
</p>

This project extends IBM's [AssetOpsBench](https://github.com/IBM/AssetOpsBench) industrial
AI agent benchmark (467 scenarios across 6 HuggingFace subsets) by adding a 7th domain -- Smart
Grid power transformers:

1. Creating 36 paper-grade validated maintenance scenarios for **Smart Grid transformers** (31 hand-authored plus 5 promoted from generated candidates) using public-safe synthetic telemetry data. The judged quality results use the 31 hand-authored scenarios. The repo now holds 61 scenario files: 25 more were added after the NeurIPS deadline (PR #199) and have not been evaluated
2. Wrapping four AssetOpsBench tool domains (IoT, TSFM, FMSR, WO) as **MCP servers**
3. Profiling and optimizing the **LLM agent inference pipeline** when operating through MCP
4. Comparing two **orchestration paradigms** (Agent-as-Tool vs Plan-Execute) on end-to-end multi-domain scenarios

## 2. Model/Application Description

We serve Llama-3.1-8B-Instruct with vLLM on A100 GPUs, profile with `nvidia-smi` and vLLM's
PyTorch-profiler hook, and compare direct with MCP-mediated tool calls to measure protocol
overhead. The optimization rows are (C) optimized MCP transport and (D) optimized MCP plus an
optimized serving bundle (INT8 weights, fp8 KV cache, prefix caching). D is a bundle, not a
single-knob ablation. Runs are linked to W&B where captured.

| Component | Project implementation |
|---|---|
| Actor model | Llama-3.1-8B-Instruct served through vLLM on A100 GPUs |
| Hosted top-ups | Llama-3.3-70B-Instruct via IBM watsonx (separate sample sizes; not pooled with the 8B matrix) |
| Outcome judge | Llama-4-Maverick-17B-128E-Instruct (fp8) via IBM watsonx, 6-dimension rubric (`scripts/judge_trajectory.py`) |
| In-loop verifier | In Verified Plan-Execute cells, the actor model itself checks intermediate tool results; this is separate from the outcome judge |
| Application | Industrial smart-grid transformer-maintenance extension of AssetOpsBench |
| Tool interface | Four Smart Grid MCP servers (IoT, TSFM, FMSR, work orders), plus a direct-call path |
| Configurations | A direct AaT · B MCP AaT · C optimized MCP · D optimized MCP + optimized serving · Y Plan-Execute · YS PE + Self-Ask · Z Verified PE · ZS Verified PE + Self-Ask |
| Profiling stack | `nvidia-smi` sampling, vLLM PyTorch-profiler traces, W&B run linkage, committed result tables |

### Datasets

| Dataset | Rows | Primary Agent | License | Source |
|---|---|---|---|---|
| [Power Transformers FDD & RUL](https://www.kaggle.com/datasets/yuriykatser/power-transformers-fdd-and-rul) | 3,000 files x 420 | TSFM, IoT | CC0 | Kaggle |
| [DGA Fault Classification](https://www.kaggle.com/datasets/bantipatel20/dissolved-gas-analysis-of-transformer) | 201 | FMSR | CC0 | Kaggle |
| [Transformer Health Index](https://www.kaggle.com/datasets/easonlai/sample-power-transformers-health-condition-dataset) | 470 | FMSR | ODbL | Kaggle |
| [Current & Voltage Monitoring](https://www.kaggle.com/datasets/sreshta140/ai-transformer-monitoring) | 19,352 | IoT, TSFM | © Authors | Kaggle |
| [Smart Grid Fault Records](https://www.kaggle.com/datasets/ziya07/power-system-faults-dataset) | 506 | WO | CC0 | Kaggle |

The repo's tracked `data/processed/` artifacts are kept synthetic/public-safe. Restricted-source Kaggle joins remain a local benchmarking path, not a tracked publication path.

### Why Llama-3.1-8B-Instruct?

| Model | Params | FP16 VRAM | Tool-calling support | Fit for this project |
|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | 8B | ~16GB | Good, well-documented, strong vLLM support | Best -- fits A6000, enables rapid iteration, INT8 optimization is meaningful (16→8GB) |
| Phi-4-14B | 14B | ~28GB | Strong reasoning, less proven for tool calling | Reasonable but less community tooling |
| Mistral-Small-24B | 24B | ~48GB | Good | Needs A100, fewer experiment iterations per dollar |
| Gemma-3-27B | 27B | ~54GB | Good | Needs A100 80GB, overkill for benchmarking |

We select Llama-3.1-8B-Instruct for its favorable profiling characteristics: it fits comfortably on
an A6000 with room for KV-cache experiments, and INT8 quantization produces a meaningful
memory reduction. We optionally compare against Llama-3.3-70B (AssetOpsBench's default model)
via WatsonX API to assess scaling effects.

*Selection rationale from April 2026. Early smoke tests ran on an Insomnia A6000; the reported
experiments ran on A100 GPUs, and the hosted Llama-3.3-70B rows appear as separately sized top-ups.*

## 3. Final Results Summary

Results below are from the frozen class report and NeurIPS paper (experiment code `1913c6e`). They are judge-scored and descriptive; no significance tests are reported.

**31-scenario core matrix** (Llama-3.1-8B-Instruct actor, 5 trials per scenario, 155 judged trajectories per row; source [`results/metrics/gcp_post175_core31_summary.csv`](results/metrics/gcp_post175_core31_summary.csv)):

| Cell | Configuration | Judge mean | Judge pass |
|---|---|---|---|
| A | Agent-as-Tool, direct calls | 0.474 | 67/155 (43.2%) |
| B | Agent-as-Tool, MCP | 0.442 | 57/155 (36.8%) |
| C | Agent-as-Tool, optimized MCP transport | 0.473 | 61/155 (39.4%) |
| Y | Plan-Execute | 0.582 | 76/155 (49.0%) |
| YS | Plan-Execute + Self-Ask | 0.606 | 78/155 (50.3%) |
| Z | Verified Plan-Execute | 0.599 | 86/155 (55.5%) |
| ZS | Verified Plan-Execute + Self-Ask | 0.620 | 80/155 (51.6%) |

**Latency** is measured on a separate six-scenario A100 cohort (30 trials per row): median 8.51 s for direct Agent-as-Tool, 12.22 s over MCP and 37.92 s for Verified Plan-Execute. See the final report, Table V, for p95 and the optimized-transport and optimized-serving rows.

| Other evidence | Location |
|---|---|
| Scenario corpus | 61 files in `data/scenarios/` plus 5 negative fixtures; 36 paper-grade; 31 in the judged core. The 25 post-deadline additions are not evaluated. |
| Evidence floor | 2,420 paper-grade judged trajectories, gated by [`results/metrics/evidence_registry.csv`](results/metrics/evidence_registry.csv) |
| Profiling | [`results/metrics/profiling_inventory.csv`](results/metrics/profiling_inventory.csv) |
| First-capture notebook exports (6 scenarios, preliminary) | [`notebook02_latency_summary.csv`](results/metrics/notebook02_latency_summary.csv), [`notebook03_self_ask_ablation.csv`](results/metrics/notebook03_self_ask_ablation.csv) |

## 4. Repository Structure

```
.
├── README.md                     # This file - project overview, submission links, reproducibility
├── LICENSE                       # MIT license
├── requirements.txt              # Python dependencies (ibm-watsonx-ai, pandas, etc)
├── .github/workflows/            # CI (Black formatting check)
├── deliverables/                 # CourseWorks final report/deck files uploaded with the submission
│
├── data/                         # Data pipeline + processed datasets - see data/README.md
│   ├── build_processed.py        #   Downloads + joins 5 Kaggle datasets via synthesized transformer_id
│   ├── generate_synthetic.py     #   Offline synthetic equivalent (no Kaggle needed)
│   ├── processed/                #   6 public-safe synthetic CSVs tracked in git (asset_metadata, dga_records, …)
│   ├── scenarios/                #   AssetOpsBench-format scenario files - see data/scenarios/README.md
│   ├── knowledge/                #   Structured standards artifacts for PS B generation - see data/knowledge/README.md
│   └── raw/                      #   GITIGNORED raw Kaggle downloads
│
├── mcp_servers/                  # 4 MCP servers on shared base - see mcp_servers/README.md
│   ├── base.py                   #   Shared data loader + utilities
│   ├── iot_server/               #   Sensor reads, asset metadata
│   ├── fmsr_server/              #   Failure search, IEC 60599 Rogers Ratio DGA analysis
│   ├── tsfm_server/              #   RUL forecast, z-score anomaly, OLS trend
│   └── wo_server/                #   Work order CRUD, downtime estimation
│
├── scripts/                      # Utility scripts - see scripts/README.md
│   ├── verify_watsonx.py         #   WatsonX access verification + latency benchmarking
│   ├── run_experiment.sh         #   Canonical SmartGridBench runner (PE baseline + Self-Ask/Verified PE follow-ons)
│   ├── setup_insomnia.sh         #   Shared Insomnia environment setup
│   ├── vllm_serve.sh             #   Self-hosted vLLM serve + smoke path on Insomnia
│   ├── test_inference.sh         #   Sanity checks against a live vLLM endpoint
│   └── benchmark_prompts/        #   Prompt templates for latency tests
│
├── benchmarks/                   # Raw latency/throughput/judge runs - see benchmarks/README.md
│   ├── cell_A_direct/            #   Direct-tool baseline artifacts
│   ├── cell_B_mcp_baseline/      #   MCP baseline artifacts
│   ├── cell_C_mcp_optimized/     #   Optimized MCP path artifacts
│   ├── cell_D/                   #   Optimized MCP + optimized serving bundle
│   ├── cell_Y_plan_execute/      #   Plan-Execute (Y) and PE + Self-Ask (YS) runs
│   ├── cell_Z_hybrid/            #   Verified PE (Z) and Verified PE + Self-Ask (ZS) runs (directory name is historical)
│   └── cell_*70B/, other cell_*  #   Hosted-70B top-ups and follow-on ablations - see benchmarks/README.md
│
├── configs/                      # Per-cell experiment .env files - see configs/README.md
├── tests/                        # pytest suite
├── notebooks/                    # Jupyter notebooks - see notebooks/README.md
├── profiling/                    # PyTorch Profiler + Nsight - see profiling/README.md
├── results/                      # Curated metrics + figures - see results/README.md
│
├── docs/                         # Living authored documentation - see docs/README.md
│   └── knowledge-base/reference/               #   Lower-churn class / mentor / setup reference docs
├── planning/                     # Meeting agendas + working notes
└── reports/                      # Build notes, historical submissions, and report work products - see reports/README.md
    └── archive/                  #   Superseded drafts
```

## 5. Reproducibility Instructions

### A. Environment Setup

#### Prerequisites

- Python 3.11 for the team environment (the upstream AssetOpsBench Plan-Execute client pins 3.12; see `docs/methods_python_version_disclosure.md`)
- Docker (only for the optional upstream CouchDB harness smoke test)
- CUDA-capable GPU (16GB+ VRAM recommended)
- Access to Columbia Insomnia cluster or Google Cloud GPU instance
- WatsonX API key (via [Codabench](https://www.codabench.org/competitions/10206/))

#### Installation

```bash
git clone https://github.com/HPML6998-S26-Team13/hpml-assetopsbench-smart-grid-mcp.git
cd hpml-assetopsbench-smart-grid-mcp
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Optional overlays:

```bash
# interactive notebook authoring
uv pip install -r requirements-notebooks.txt

# Insomnia / cluster serving stack
uv pip install -r requirements-insomnia.txt
```

`requirements.txt` now also includes the portable AssetOpsBench PE-client
dependencies (`litellm` and `mcp[cli]`) needed by the repo-local Self-Ask PE /
Verified PE runners. The cluster overlay in `requirements-insomnia.txt` layers
vLLM and CUDA-specific pins on top of that base.

### B. Experiment Tracking Dashboard

- W&B dashboard (final evidence): https://wandb.ai/assetopsbench-smartgrid/assetopsbench-smartgrid/reports/SmartGridBench-Final-Evidence-Dashboard--VmlldzoxNjgyODI4NA
- W&B project (all runs): https://wandb.ai/assetopsbench-smartgrid/assetopsbench-smartgrid

### C. Dataset

The repo tracks public-safe synthetic processed data under `data/processed/` and
AssetOpsBench-format Smart Grid scenarios under `data/scenarios/`. Raw Kaggle
downloads remain gitignored under `data/raw/` and are not required to inspect
the submitted results.

```bash
python data/generate_synthetic.py
```

### D. Training

This project benchmarks inference and tool orchestration; it does not fine-tune
the base Llama model. The model-serving path is configured through `vLLM`,
WatsonX-compatible LiteLLM endpoints, and the `configs/*.env` experiment files.

### E. Evaluation

Judge outputs and summarized metrics are committed under `results/`. To score a
new completed run directory, use the judge helper with the run directory and
matching output path:

```bash
python scripts/judge_trajectory.py \
  --run-dir benchmarks/cell_B_mcp_baseline/raw/<run-id> \
  --scenario-dir data/scenarios \
  --out results/metrics/scenario_scores.jsonl \
  --log-dir results/judge_logs
```

### F. Profiling

Profiling is enabled through the runner environment and summarized in
`results/metrics/profiling_inventory.csv`.

```bash
ENABLE_WANDB=1 TORCH_PROFILE=1 bash scripts/run_experiment.sh configs/example_baseline.env
python scripts/build_profiling_inventory.py
```

### G. Quickstart: Reproduce the Baseline Runner Path

To reproduce one Agent-as-Tool baseline row (Plan-Execute variants: `configs/example_pe_self_ask.env`, `configs/example_verified_pe.env`; final A100 matrix configs: `configs/final_matrix_5x6/`):

```bash
bash scripts/run_experiment.sh configs/example_baseline.env
```

See:
- [docs/orchestration_wiring.md](docs/orchestration_wiring.md) for what is
  wired today versus only adapter-ready
- [benchmarks/cell_Y_plan_execute/](benchmarks/cell_Y_plan_execute/) for the
  first kept WatsonX smoke proof run and artifact layout
- [docs/insomnia_runbook.md](docs/insomnia_runbook.md) for the shared Insomnia
  self-hosted vLLM path

## 6. Results and Observations

- The first-capture (six-scenario) AaT direct, MCP baseline, and optimized MCP latency comparison is
  exported in `results/metrics/notebook02_latency_summary.csv`.
- The first-capture orchestration and Self-Ask comparison is exported in
  `results/metrics/notebook03_self_ask_ablation.csv`.
- The final submitted claim set is backed by committed result tables, W&B run
  links, benchmark artifacts, and exported report/deck deliverables.

## Project Timeline: What Was Claimed at Each Stage

Each stage reports a fixed evidence cohort, so numbers from different stages should not be pooled.

### (a) NeurIPS 2026 Evaluations & Datasets submission (May 7, 2026)

- **Artifact:** [`reports/SmartOpsBench_NeurIPS_2026.pdf`](reports/SmartOpsBench_NeurIPS_2026.pdf) (anonymous 16-page submission; the paper is titled "SmartGridBench", the filename is historical).
- **Corpus frozen at submission:** 36 validated scenarios (31 hand-authored + 5 promoted from generated candidates) and 5 negative schema fixtures.
- **Claims:** on the 31-scenario × 5-trial core matrix, MCP Agent-as-Tool did not improve judged quality over direct tool calls (57/155 vs 67/155 judge passes), and Plan-Execute-family orchestration raised judged quality at higher latency (Verified Plan-Execute 86/155). A separate six-scenario A100 cohort measured latency. Evidence floor: 2,420 judged trajectories.
- **Outcome:** not accepted (September 2026).

### (b) COMS E6998 final presentation (May 7) and final report (May 10, 2026)

- **Artifacts:** [`deliverables/SmartGridBench_Final_Paper.pdf`](deliverables/SmartGridBench_Final_Paper.pdf) (5-page IEEE report) and the [final deck](deliverables/SmartGridBench-Final-Presentation.pdf).
- **Claims:** the same corpus scope and core matrix as (a), plus the matched six-scenario A100 latency table, profiling coverage, and limitations (the synthetic DGA data does not yet pass a real-distribution realism check; judge scores are not domain-expert review).
- After the deadline the repo corpus grew to 61 scenario files (PR #199). The 25 additions are not part of any evaluated claim.

### (c) Revised manuscript (September 2026)

A revised manuscript was submitted to a peer-reviewed conference in September 2026 and is under double-blind review. It is not linked or described here while review is anonymous.

### (d) Ongoing work

Planned, not yet run or released:

- a matched direct-vs-MCP comparison over all 61 scenarios;
- the same comparison with a second actor model family;
- blinded human ratings of agent outcomes, and independent domain-expert review of scenario realism.

No results from this work are reported in this repository yet.

### Preprint

An arXiv preprint is planned and has not been posted. This README will link it once it is available.

### Errata

- **NeurIPS submission, Table 3 caption:** the caption says the rows were "all judged with Llama-3.1-8B-Instruct". Llama-3.1-8B-Instruct was the actor. The outcome judge for every row was Llama-4-Maverick-17B-128E-Instruct (fp8, hosted on watsonx), as recorded in the `judge_model` field of [`results/metrics/scenario_scores.jsonl`](results/metrics/scenario_scores.jsonl) and in `results/judge_logs/`. The reported counts are unchanged, and the submitted PDF is kept unmodified.

<details>
<summary>April 2026 working snapshot (historical)</summary>

*Last updated: Apr 18, 2026 - Apr 16 post-call audit resolved the WO architecture decision, closed the first-WandB-run milestone, and narrowed the active risk to overdue W2 closeout plus W3 profiling / PS B execution.*

**Week 1 (complete):**
- [x] Problem statement finalized (four contributions)
- [x] Research proposal drafted and shared with mentor via Overleaf; mentor endorsed NeurIPS 2026 Datasets & Benchmarks track
- [x] GitHub repo scaffolded, WandB team created, **repo now public** (Apr 7)
- [x] 5 Kaggle datasets identified, AssetOpsBench forked and reviewed
- [x] Compute confirmed (Insomnia cluster + GCP credits) and compute plan committed (`docs/archive/compute_plan.md`)
- [x] WatsonX API access received from mentor (Apr 5) and verified end-to-end - 6 Llama models available; Llama-4-Maverick-17B and Llama-3.3-70B-instruct benchmarked (`docs/knowledge-base/reference/watsonx_access.md`)
- [x] Data pipeline + tracked public-safe processed datasets landed (`data/processed/` with synthetic asset metadata, DGA records, failure modes, fault records, RUL labels, and sensor readings - development-ready and safe to publish)
- [x] MCP server skeletons landed for all four domains (IoT, FMSR, TSFM, WO) on a shared base class with substantive domain logic (IEC 60599 Rogers Ratio DGA analysis, RUL forecast, anomaly detection, work-order CRUD)
- [x] `docs/data_pipeline.tex` paper section drafted
- [x] **Mid-point report submitted** (`reports/2026-04-06_midpoint_submission.pdf`) to Courseworks on Mon Apr 6

**Week 2 (landed on canonical history):**
- [x] GitHub Projects reset as the canonical planning surface, with weekly iterations, workstream parent issues, and delivery-gate milestones
- [x] Successful Insomnia A6000 vLLM serve smoke test for Llama-3.1-8B-Instruct, with kept smoke-test logs and fixed serve/test scripts
- [x] First real SmartGridBench WandB run is live and back-linked to committed benchmark artifacts
- [x] Plan-Execute is wired to the team’s Smart Grid MCP servers as a real experiment condition, with a successful WatsonX-hosted 70B / Mac 1-scenario smoke proof run under `benchmarks/cell_Y_plan_execute/`
- [x] Scenario realism validation note landed with IEEE / IEC grounded findings for Dhaval-facing review

**Still open from the original W2 critical path / backlog:**
- [ ] `#3` Canonical benchmark scenario proof on the AssetOpsBench stack (Akshat)
- [ ] `#58` Benchmark-Llama-path validation closeout plus `#9-#12` MCP hardening/tests (Tanisha) - PR `#115` now contains a real Insomnia A6000 / self-hosted 8B / all-4-server proof run plus branch-specific long-context serve notes, but the merge/readme cleanup is still open (see `docs/insomnia_runbook.md`)
- [ ] `#7` / `#59` profiling wrappers and the first profiling-linked experiment capture path (Aaron)
- [ ] `#15` / `#17` / `#18` / `#20` scenario-count, judge, and first trajectory artifacts (Akshat)

**Paper-grade scenario corpus:** 36 validated SmartGridBench scenarios — 31 hand-authored + 5 promoted generated. This is the canonical count used for the final NeurIPS and CourseWorks claims (frozen at NeurIPS submission, 2026-05-07).

**Current repo corpus:** 61 scenario files under `data/scenarios/` plus 5 negative validation fixtures. The 25 scenarios beyond the 36 paper-grade canonical were added post-submission via PR #199 (#55 batch A and B) to grow the repo corpus and are deliberately not part of any evaluated/judged claims.

**Resolved during the Apr 16 post-call audit:**
- [x] `#13` WO architecture review closed as a documented keep-vs-pivot decision
- [x] `#28` closed against the first real shared WandB run and its committed benchmark artifacts

**W3 focus (Apr 14-20):**
- Experiment 1 profiling captures plus profiling-to-WandB linkage (`#25`, `#27`)
- Problem Statement B kickoff: Knowledge Plugin, first generation prototype, and evaluation methodology (`#50`, `#2`, `#51`)
- NeurIPS abstract outline and title candidates (`#77`)
- Runbook consolidation for the infra / serve / profiling path (`#37`)

**Committed W3-W5 tracks:**
- Experiment 1: MCP overhead and optimization (Direct vs MCP-baseline vs MCP-optimized)
- Experiment 2: orchestration comparison and failure analysis
- Problem Statement B extension: scenario generation pipeline, Knowledge Plugin, and validation methodology
- NeurIPS 2026 draft first, then back-port to the class IEEE report format

**Current default scope decision:** the working comparison is still **vanilla Agent-as-Tool vs vanilla Plan-Execute**. The repo now also has an active local mitigation stream for PE + Self-Ask and an optional Verified PE third-method prototype, but neither should muddy the honesty of the core AaT vs PE story. The primary local benchmark model remains self-hosted Llama-3.1-8B-Instruct with 70B reserved for selective WatsonX spot-checks.

</details>

## 7. Notes

### Repository Navigation

- [docs/README.md](docs/README.md) - living docs index and runbook navigation
- [scripts/README.md](scripts/README.md) - executable entrypoints and helper scripts
- [configs/README.md](configs/README.md) - benchmark config schema and cell mapping
- [data/README.md](data/README.md) - data pipeline and processed dataset policy
- [data/scenarios/README.md](data/scenarios/README.md) - scenario authoring rules and validation
- [mcp_servers/README.md](mcp_servers/README.md) - Smart Grid MCP server layout
- [benchmarks/README.md](benchmarks/README.md) - raw benchmark artifact layout
- [profiling/README.md](profiling/README.md) - profiling capture workflow

### Key Dates

| Date | Milestone |
|---|---|
| Mon Apr 6 | Mid-point report submitted (CourseWorks) |
| Tue May 5 | NeurIPS 2026 Evaluations & Datasets abstract registered |
| Thu May 7 | NeurIPS 2026 full paper submitted; final class presentation delivered; IBM AssetOpsBench PR #287 opened |
| Fri May 8 | 25 additional scenarios merged to this repo (PR #199); not evaluated |
| Sun May 10 | CourseWorks final report and package submitted |
| Mon May 11 | IBM AssetOpsBench PR #292 opened |
| Sep 2026 | NeurIPS submission not accepted; revised manuscript submitted for double-blind review |
| Pending | arXiv preprint (not yet posted) |

### References and Resources

- [AssetOpsBench](https://github.com/IBM/AssetOpsBench) -- IBM's industrial AI agent benchmark
- [AssetOpsBench on HuggingFace](https://huggingface.co/datasets/ibm-research/AssetOpsBench)
- [AssetOpsBench competition](https://www.codabench.org/competitions/10206/)
- [Model Context Protocol](https://modelcontextprotocol.io/) -- open protocol for LLM tool integration
- [vLLM](https://github.com/vllm-project/vllm) -- High-throughput LLM serving
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Weights & Biases](https://wandb.ai/site)

See also: [docs/knowledge-base/reference/project_reference.md](docs/knowledge-base/reference/project_reference.md) for class requirements, grading, and mentor guidance.

### AI Use Disclosure

Per the HPML AI Use Policy, Team 13 used AI assistance during this project.

**Tools used:** Claude and Codex.

**Specific purpose:** We used AI tools for brainstorming, debugging support, code navigation, runbook/checklist assistance, and prose polishing. Experimental design decisions, reported measurements, profiling interpretations, quantitative analysis, and final claims were checked by team members against committed code, experiment logs, dashboard records, and exported result artifacts.

**Sections affected:** README/runbook wording, report and presentation polishing, checklist maintenance, and selected implementation/debugging workflows.

**How we verified correctness:** We ran the reported experiments ourselves, preserved result artifacts under `benchmarks/`, `results/`, and `reports/`, cross-checked headline claims against the committed evidence tables, and manually reviewed final public-facing wording before submission.

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that AI assistance is disclosed here.

### License

MIT; see [`LICENSE`](LICENSE).

### Citation

If citing the course project, use the final report:

> A. Bhandari, A. Fan, T. Rathod, and W. A. Xin. "SmartGridBench: MCP-Based Industrial Agent Benchmarking for Smart Grid Transformer Operations." COMS E6998 High Performance Machine Learning Final Project, Columbia University, Spring 2026.

A preprint citation will be added once it is posted.

### Contact

For project questions, use the GitHub issue tracker or contact the Team 13 members listed above.
