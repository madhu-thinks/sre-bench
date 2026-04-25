# Hackathon Execution Plan

## 1) SFT vs RL Decision
- Current strategy: start from a capable instruct base model and use RL (GRPO) with objective environment rewards.
- Why: this project has strong programmatic verification via environment outcomes and rubric scoring.
- Optional next step: add a small SFT warm-start set only if rollout success becomes too sparse.

## 2) Team Split (Recommended)
- Person A (Environment): `reset/step/state`, runtime stability, deployment.
- Person B (Rewards/Verifier): rubric quality, anti-hacking checks, failure-case coverage.
- Person C (Training): TRL + Unsloth experiments, metrics tracking, run management.
- Person D (Demo/Product): Space demo, before/after examples, benchmark narrative.

## 3) One-Day Execution Checklist
- Phase 1: lock task scope and verifier contract.
- Phase 2: validate environment locally (`python test_local.py`).
- Phase 3: deploy environment to Space and confirm `/health`.
- Phase 4: run small GRPO job and inspect sampled trajectories.
- Phase 5: run baseline-vs-trained evaluation script and export report artifacts.
- Phase 6: prepare final demo with measurable improvement and safeguards.

## 4) Common Mistakes Guardrail
- Avoid zero-success tasks at start. Use curriculum (`easy -> medium -> hard`).
- Do not rely on one reward only. Keep multiple rubric dimensions active.
- Do not trust reward alone. Inspect trajectories and failure modes.
- Always include timeout/step-budget handling and blast-radius penalties.
- Export model with the intended merged path and verify inference immediately.

## 5) Learning Resources
- RL Mega Lecture: https://www.youtube.com/watch?v=Jew4lhAiqnw
- OpenEnv Workshop: https://www.youtube.com/watch?v=1jU05MlENOI
- OpenEnv docs: https://github.com/meta-pytorch/OpenEnv
- TRL docs: https://huggingface.co/docs/trl

## 6) Evidence Artifacts To Produce
- Training metrics log: `sre_bench_grpo_outputs/rollout_metrics.jsonl`
- Baseline-vs-trained JSON report: `sre_bench_grpo_outputs/baseline_vs_trained_eval.json`
- Baseline-vs-trained Markdown report: `sre_bench_grpo_outputs/baseline_vs_trained_eval.md`
