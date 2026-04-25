# Anti Reward-Hacking Checklist

## Threats We Explicitly Handle
- Random destructive actions on healthy services.
- Step-looping until budget exhaustion.
- Guessing without investigation evidence.
- Invalid tool or malformed action usage.

## Implemented Safeguards
- Multi-component rubric (root cause, time, hypothesis, blast radius, postmortem).
- Blast-radius penalties for harmful fix actions.
- Hard step budget with escalation penalty.
- Unknown tool handling.
- Hypothesis checkpoints at specific steps.
- Optional LLM-judge plus heuristic fallback for postmortem quality.

## Runtime Controls
- Step budget: 20.
- Timeout for LLM-judge HTTP client requests.
- Difficulty control for curriculum and staged rollout complexity.

## Tests To Run Before Training/Submission
- Local smoke and edge cases:
  - `python test_local.py`
- Live environment health:
  - `Invoke-RestMethod https://<your-space>.hf.space/health`
- Baseline-vs-trained report generation:
  - `python training/evaluate_baseline_vs_trained.py --env_url https://<your-space>.hf.space --episodes 25 --model_path ./sre_bench_final_model`

## Operational Monitoring During Training
- Reward trend.
- Success rate and timeout rate.
- Average steps-to-completion.
- Per-rubric component averages.
- Periodic trajectory sample inspection.

## Incident Response If Drift Appears
- Stop long run.
- Sample and inspect latest trajectories.
- Identify reward exploit pattern.
- Patch verifier/reward guardrails.
- Re-run short validation before scaling again.
