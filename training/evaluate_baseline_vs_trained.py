"""
Evaluate baseline-vs-trained performance on SRE-Bench.

Usage examples:
  python training/evaluate_baseline_vs_trained.py --env_url http://localhost:8000 --episodes 25
  python training/evaluate_baseline_vs_trained.py --env_url https://<space>.hf.space --episodes 25 --model_path ./sre_bench_final_model
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    from sre_bench import SreBenchAction, SreBenchEnv
except ImportError:
    from client import SreBenchEnv
    from models import SreBenchAction


SERVICES = [
    "frontend",
    "auth-api",
    "order-service",
    "payment-gateway",
    "database",
    "redis-queue",
    "load-balancer",
]

ROOT_CAUSE_GUESSES = [
    "db_connection_exhaustion",
    "oom_killed",
    "network_partition",
    "retry_storm",
    "config_missing_env",
    "disk_full",
    "cpu_throttle",
    "memory_leak",
]


SYSTEM_PROMPT = """You are a Senior Site Reliability Engineer tackling a live production incident.
Use the tool API and output each action as:
<tool>{"tool_name": "...", "arguments": {...}, "hypothesis": "..."}</tool>
"""


def parse_action_from_text(text: str) -> SreBenchAction:
    """Extract a tool call from model output."""
    match = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    if not match:
        return SreBenchAction(
            tool_name="grep_logs",
            arguments={"service": "frontend", "pattern": "error"},
            hypothesis="FAILED_TO_GENERATE_TOOL_XML",
        )
    try:
        data = json.loads(match.group(1).strip())
        return SreBenchAction(
            tool_name=data.get("tool_name", "grep_logs"),
            arguments=data.get("arguments", {}),
            hypothesis=data.get("hypothesis", ""),
        )
    except json.JSONDecodeError:
        return SreBenchAction(
            tool_name="grep_logs",
            arguments={"service": "frontend", "pattern": "error"},
            hypothesis="JSON_DECODE_ERROR",
        )


def random_baseline_action(step: int) -> SreBenchAction:
    """Simple weak baseline policy."""
    if step >= 10 and random.random() < 0.35:
        guessed_service = random.choice(SERVICES)
        guessed_fault = random.choice(ROOT_CAUSE_GUESSES)
        return SreBenchAction(
            tool_name="resolve_incident",
            arguments={
                "root_cause": guessed_fault,
                "fix_applied": f"Restarted {guessed_service} and rolled back deploy",
            },
            hypothesis="Attempting resolution with current guess",
        )

    roll = random.random()
    if roll < 0.35:
        return SreBenchAction(
            tool_name="get_error_rate",
            arguments={"service": random.choice(SERVICES)},
            hypothesis="Checking error rate",
        )
    if roll < 0.70:
        return SreBenchAction(
            tool_name="get_metrics",
            arguments={"service": random.choice(SERVICES)},
            hypothesis="Checking metrics",
        )
    return SreBenchAction(
        tool_name="grep_logs",
        arguments={"service": random.choice(SERVICES), "pattern": "ERROR"},
        hypothesis="Scanning logs",
    )


def _episode_record(
    reward: float,
    steps: int,
    final_output: str,
    scores: Optional[Dict[str, float]],
) -> Dict:
    final_text = final_output or ""
    resolved = "=== INCIDENT RESOLVED ===" in final_text
    timed_out = "STEP BUDGET EXHAUSTED" in final_text
    return {
        "reward": float(reward),
        "steps": steps,
        "resolved": resolved,
        "timed_out": timed_out,
        "scores": scores or {},
    }


def run_baseline_eval(
    env_url: str,
    episodes: int,
    difficulty: str,
) -> List[Dict]:
    records: List[Dict] = []
    with SreBenchEnv(base_url=env_url).sync() as env:
        for _ in range(episodes):
            result = env.reset(difficulty=difficulty)
            obs = result.observation
            done = result.done
            reward = result.reward
            step = 0

            while not done and step < 20:
                step += 1
                action = random_baseline_action(step)
                result = env.step(action)
                obs = result.observation
                done = result.done
                reward = result.reward

            records.append(_episode_record(reward, step, obs.tool_output, obs.scores))
    return records


def _load_model(model_path: str):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_trained_eval(
    env_url: str,
    model_path: str,
    episodes: int,
    difficulty: str,
) -> List[Dict]:
    import torch

    model, tokenizer = _load_model(model_path)
    records: List[Dict] = []

    with SreBenchEnv(base_url=env_url).sync() as env:
        for _ in range(episodes):
            result = env.reset(difficulty=difficulty)
            obs = result.observation
            done = result.done
            reward = result.reward
            step = 0

            chat_history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs.tool_output},
            ]

            while not done and step < 20:
                step += 1
                inputs = tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")

                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=256,
                        temperature=0.2,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                action = parse_action_from_text(response)
                result = env.step(action)

                obs = result.observation
                done = result.done
                reward = result.reward

                chat_history.append({"role": "assistant", "content": response})
                chat_history.append({"role": "user", "content": obs.tool_output})

            records.append(_episode_record(reward, step, obs.tool_output, obs.scores))
    return records


def summarize(records: List[Dict]) -> Dict:
    metrics = {
        "episodes": len(records),
        "avg_reward": 0.0,
        "success_rate": 0.0,
        "timeout_rate": 0.0,
        "avg_steps": 0.0,
        "scores": {},
    }
    if not records:
        return metrics

    metrics["avg_reward"] = sum(r["reward"] for r in records) / len(records)
    metrics["success_rate"] = sum(1 for r in records if r["resolved"]) / len(records)
    metrics["timeout_rate"] = sum(1 for r in records if r["timed_out"]) / len(records)
    metrics["avg_steps"] = sum(r["steps"] for r in records) / len(records)

    score_sums = defaultdict(float)
    score_counts = defaultdict(int)
    for row in records:
        for key, value in row["scores"].items():
            score_sums[key] += float(value)
            score_counts[key] += 1
    metrics["scores"] = {
        key: score_sums[key] / score_counts[key]
        for key in sorted(score_sums)
        if score_counts[key]
    }
    return metrics


def print_summary(name: str, summary: Dict):
    print(f"\n{name}")
    print(f"  Episodes:      {summary['episodes']}")
    print(f"  Avg Reward:    {summary['avg_reward']:.4f}")
    print(f"  Success Rate:  {summary['success_rate']:.2%}")
    print(f"  Timeout Rate:  {summary['timeout_rate']:.2%}")
    print(f"  Avg Steps:     {summary['avg_steps']:.2f}")
    if summary["scores"]:
        print("  Avg Rubric Scores:")
        for key, value in summary["scores"].items():
            print(f"    - {key}: {value:.4f}")


def write_markdown_report(path: str, output: Dict):
    lines = []
    lines.append("# Baseline vs Trained Evaluation")
    lines.append("")
    lines.append(f"- Generated: {output['generated_at_utc']}")
    lines.append(f"- Env URL: `{output['env_url']}`")
    lines.append(f"- Difficulty: `{output['difficulty']}`")
    lines.append(f"- Episodes per agent: {output['episodes_per_agent']}")
    lines.append("")
    lines.append("| Agent | Avg Reward | Success Rate | Timeout Rate | Avg Steps |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for agent_name in ["baseline", "trained"]:
        summary = output.get(agent_name, {}).get("summary")
        if not summary:
            continue
        lines.append(
            f"| {agent_name} | {summary['avg_reward']:.4f} | "
            f"{summary['success_rate']:.2%} | {summary['timeout_rate']:.2%} | "
            f"{summary['avg_steps']:.2f} |"
        )
    lines.append("")
    lines.append("## Rubric Score Averages")
    lines.append("")
    lines.append("| Agent | Root Cause | Time | Hypothesis | Blast Radius | Postmortem |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for agent_name in ["baseline", "trained"]:
        summary = output.get(agent_name, {}).get("summary")
        if not summary:
            continue
        scores = summary.get("scores", {})
        lines.append(
            f"| {agent_name} | "
            f"{scores.get('root_cause_accuracy', 0.0):.4f} | "
            f"{scores.get('time_to_resolution', 0.0):.4f} | "
            f"{scores.get('hypothesis_quality', 0.0):.4f} | "
            f"{scores.get('blast_radius_control', 0.0):.4f} | "
            f"{scores.get('postmortem_quality', 0.0):.4f} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_url", type=str, default="http://localhost:8000")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model folder.")
    parser.add_argument(
        "--output_json",
        type=str,
        default="sre_bench_grpo_outputs/baseline_vs_trained_eval.json",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="sre_bench_grpo_outputs/baseline_vs_trained_eval.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    baseline_records = run_baseline_eval(args.env_url, args.episodes, args.difficulty)
    baseline_summary = summarize(baseline_records)
    print_summary("Baseline", baseline_summary)

    trained_summary = None
    trained_records = None
    trained_error = None

    if args.model_path:
        try:
            trained_records = run_trained_eval(args.env_url, args.model_path, args.episodes, args.difficulty)
            trained_summary = summarize(trained_records)
            print_summary("Trained", trained_summary)
        except Exception as exc:  # pragma: no cover
            trained_error = str(exc)
            print(f"\n[WARN] Trained evaluation failed: {trained_error}")
    else:
        print("\n[INFO] --model_path not provided. Skipping trained-agent evaluation.")

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "env_url": args.env_url,
        "difficulty": args.difficulty,
        "episodes_per_agent": args.episodes,
        "baseline": {
            "summary": baseline_summary,
            "records": baseline_records,
        },
        "trained": {
            "summary": trained_summary,
            "records": trained_records,
            "error": trained_error,
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    write_markdown_report(args.output_md, output)

    print(f"\nSaved JSON report: {args.output_json}")
    print(f"Saved Markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
