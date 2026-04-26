---
title: SRE-Bench
sdk: docker
app_port: 8000
---

# 🚨 SRE-Bench: Production Incident Training Environment

**A high-fidelity, long-horizon reinforcement learning environment for training Senior SRE Agents.**

[![OpenEnv Support](https://img.shields.io/badge/OpenEnv-Supported-blue?logo=meta)](https://github.com/meta-pytorch/OpenEnv)
[![TRL Support](https://img.shields.io/badge/TRL-Supported-orange?logo=huggingface)](https://huggingface.co/docs/trl)

---

## 📈 SRE-Bench Agent Training Results (Live Environment)

**Can an LLM learn to resolve production incidents?**
Yes. Using GRPO and Curriculum Learning on an Unsloth-optimized Qwen2.5-3B model, our agent successfully learned to identify root causes and resolve incidents autonomously.

*(See docs/sre_bench_reward_curve.png for our full training visualization quadrant).*

**Performance on 32 Real Episodes:**
* **Untrained Baseline:** ~0.05 reward (frequent hallucinations, failed to use tools correctly)
* **AFTER Training (Easy Curriculum):** **0.458 average reward (+8.5% overall improvement)**
* **PEAK Achieved:** **0.94 reward** (Agent correctly identified root cause and resolved incident in minimum steps)
* **Resolution Rate:** **90.6%** on the Easy incident tier. 

**Curriculum Learning Breakdown:**
* **Easy Tier:** Avg = 0.458 
* **Medium / Hard Tiers:** Introduced at rollouts 17+, highlighting the need for extended compute time, but proving the environment scales difficulty correctly.

---

## 💡 The Problem
In modern infrastructure, 3 AM production outages don't look like simple logic puzzles. They look like **cascading failures, noisy logs, partial observability, and high-stakes decision making.** 

Most LLM agents fail spectacularly at this because they lack **long-horizon planning**. They hallucinate root causes immediately, loop on dead ends, and destroy healthy services (blast radius). 

## 🛠️ The Solution (What We Built)
**SRE-Bench** is a simulated 7-service microservice cluster that generates realistic outages on demand. Designed specifically for the Meta PyTorch OpenEnv Hackathon, it forces agents to investigate *before* acting.

**Features:**
*   **8 Real-World Fault Types:** Network partitions, OOM kills, database connection exhaustion, retry storms, memory leaks, CPU throttling, missing configs, and disk full events.
*   **Cascading Dependency Engine:** If the database degrades, the API slows down. Agents must trace the error up the stack.
*   **Noisy Observation Space:** The `grep_logs` tool returns a mix of real errors, upstream cascade warnings, and red-herring background noise.
*   **Multi-Dimensional Composable Rubric:** We don't just score "did it fix it?". We evaluate Root Cause Accuracy (30%), Time-to-Resolution (25%), Hypothesis Quality (20%), Blast Radius Control (15%), and Postmortem Quality (10%).

---

## 🏆 Hackathon Judges: Why This Wins
This environment adheres strictly to Theme 2 (Composable Rubrics) and Theme 3.1 (Multi-Step Tool Use/GRPO). 
1. **Un-gameable Rubric:** The agent *must* declare its hypothesis at steps 3 and 6. It loses points if it restarts healthy services trying to guess the fix.
2. **True RL Post-Training Need:** Standard supervise fine-tuning fails here because there are multiple valid paths to investigate. We use **GRPO** via OpenEnv and TRL to teach the agent the *strategy* of an SRE.

---

## 🚀 Getting Started

### 1. Run the Environment (Local or HF Space)
You can run this locally or deploy it directly to a Hugging Face Space using the `openenv` CLI.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sre-bench.git
cd sre-bench

# Install requirements
pip install -e .

# Test the environment locally (runs an 8-fault smoke test)
python test_local.py

# Start the OpenEnv FastAPI Server natively
uv run --project . server
```

### 2. Training with Unsloth & GRPO (Google Colab)
We provide a plug-and-play Colab training script optimized for T4 GPUs.

1. Open [Google Colab](https://colab.research.google.com)
2. Load the `training/train_grpo.py` script.
3. It uses `Unsloth` to load a 3B model in 4-bit, and uses `TRL GRPOTrainer` to sample trajectories from the running SRE-Bench server, optimizing for our 5-part rubric.

### 3. Baseline vs Trained Evaluation
Generate a judge-friendly before/after report with objective metrics:

```bash
python training/evaluate_baseline_vs_trained.py \
  --env_url http://localhost:8000 \
  --episodes 25 \
  --model_path ./sre_bench_final_model
```

Outputs:
- `sre_bench_grpo_outputs/baseline_vs_trained_eval.json`
- `sre_bench_grpo_outputs/baseline_vs_trained_eval.md`

### 4. Hackathon Ops Docs
- Execution plan and team split: `docs/HACKATHON_EXECUTION_PLAN.md`
- Anti reward-hacking checklist: `docs/ANTI_REWARD_HACKING.md`

---

## 🧩 The 11 SRE Tools
The agent interacts with the environment exclusively through these tools:

*   **Investigation:** `grep_logs`, `get_metrics`, `get_error_rate`, `describe_pod`, `check_db_connections`
*   **Action:** `restart_service`, `scale_replicas`, `fix_disk`, `fix_network`, `rollback_deploy`
*   **Conclusion:** `resolve_incident(root_cause, fix_applied)`

---

## 📊 Evaluation Rubric Breakdown

| Component | Weight | Logic |
| :--- | :--- | :--- |
| **Root Cause** | 30% | Exact match on injected fault + targeting the correct origin service. |
| **Time** | 25% | Max step budget is 20. Bonus applied for resolution under 8 steps. |
| **Hypothesis** | 15% | Agent's `<hypothesis>` tag at steps 3 & 6 is evaluated for trajectory correctness. |
| **Blast Radius** | 15% | Agent loses points if it takes destructive actions on healthy services. |
| **Postmortem** | 15% | **LLM-as-a-Judge**: A Qwen-2.5-7B model evaluates the quality, depth, and accuracy of the report. |

---

*Built for the 2026 Meta PyTorch OpenEnv Hackathon.*
