"""
SRE-Bench GRPO Training Script
Optimised for Google Colab (T4 GPU compatible) using Unsloth & TRL.

This script demonstrates how to train a small language model (e.g., Llama-3-8B or Qwen2.5-1.5B)
to resolve production incidents using SRE-Bench and GRPO (Group Relative Policy Optimization).

Setup in Colab:
1. Requirements:
    !pip install unsloth "trl[peft]" openenv-core
    !pip install -e .  # install sre_bench environment
    
2. Run this script!
"""

import os
import re
import json
from collections import defaultdict
import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

try:
    from sre_bench import SreBenchEnv, SreBenchAction
except ImportError:
    from client import SreBenchEnv
    from models import SreBenchAction

# Patch TRL for Unsloth optimizations
PatchDPOTrainer()

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"  # Fast, highly capable small model
MAX_SEQ_LENGTH = 4096
LORA_RANK = 16
ENV_URL = "http://localhost:8000"  # Requires FastAPI server running
TRAIN_MAX_STEPS = 50
NUM_GENERATIONS = 4
TOTAL_CURRICULUM_ROLLOUTS = TRAIN_MAX_STEPS * NUM_GENERATIONS
# Rollout-based curriculum progression.
# 0-30%: easy, 30-70%: medium, 70-100%: hard.
CURRICULUM_STAGES = (
    ("easy", 0.30),
    ("medium", 0.70),
    ("hard", 1.00),
)

global_rollout_counter = 0
global_last_difficulty = None
global_metric_sums = defaultdict(float)
global_metric_counts = defaultdict(int)
global_outcome_counts = defaultdict(int)
METRICS_LOG_PATH = "sre_bench_grpo_outputs/rollout_metrics.jsonl"

# Prompt Template instructing the model how to output tools
SYSTEM_PROMPT = """You are a Senior Site Reliability Engineer tackling a live production incident.
You must investigate the microservice cluster and fix the root cause.
The cluster has 7 services: frontend, auth-api, order-service, payment-gateway, database, redis-queue, load-balancer.

You have access to predefined tools. Stop generating text after formulating a tool call!
Output your tool calls in exact JSON format wrapped in XML tags, like this:
<tool>
{"tool_name": "grep_logs", "arguments": {"service": "database", "pattern": "ERROR"}, "hypothesis": "Checking DB for errors"}
</tool>

If you believe you have fixed it, call the resolve_incident tool:
<tool>
{"tool_name": "resolve_incident", "arguments": {"root_cause": "memory_leak", "fix_applied": "restarted frontend"}, "hypothesis": "fixed"}
</tool>

Always print your chain-of-thought analysis before executing a tool.
"""

# -----------------------------------------------------------------------
# Environment Integrations
# -----------------------------------------------------------------------

def parse_action_from_text(text: str) -> SreBenchAction:
    """Extracts JSON tool call from LLM generation."""
    match = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    if not match:
        # Fallback penalty action if model fails to output correct XML
        return SreBenchAction(
            tool_name="grep_logs", 
            arguments={"service": "unknown", "pattern": ""},
            hypothesis="FAILED TO PARSE TOOL XML"
        )
    
    try:
        data = json.loads(match.group(1).strip())
        return SreBenchAction(
            tool_name=data.get("tool_name", "grep_logs"),
            arguments=data.get("arguments", {}),
            hypothesis=data.get("hypothesis", "")
        )
    except json.JSONDecodeError:
        return SreBenchAction(
            tool_name="grep_logs", 
            arguments={"service": "unknown", "pattern": ""},
            hypothesis="JSON DECODE ERROR"
        )


def get_curriculum_difficulty(rollout_index: int) -> str:
    """Return curriculum difficulty for a given rollout index."""
    progress = (rollout_index + 1) / max(1, TOTAL_CURRICULUM_ROLLOUTS)
    for difficulty, threshold in CURRICULUM_STAGES:
        if progress <= threshold:
            return difficulty
    return "hard"


def generate_trajectory(model, tokenizer, prompt_text: str, difficulty: str = "hard") -> dict:
    """
    Rolls out a full episode in SRE-Bench using the model.
    Returns the final environment reward.
    """
    with SreBenchEnv(base_url=ENV_URL).sync() as env:
        # reset(**kwargs) is supported by OpenEnv clients and enables curriculum control.
        result = env.reset(difficulty=difficulty)
        obs = result.observation
        
        chat_history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs.tool_output}
        ]
        
        done = False
        step = 0
        total_reward = 0.0
        
        while not done and step < 20:
            # Tokenize & Generate
            inputs = tokenizer.apply_chat_template(
                chat_history, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs, 
                    max_new_tokens=256, 
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            chat_history.append({"role": "assistant", "content": response})
            
            # Execute in env
            action = parse_action_from_text(response)
            result = env.step(action)
            obs = result.observation
            done = result.done
            total_reward = result.reward # in SRE-Bench, reward is only populated at the end
            
            chat_history.append({"role": "user", "content": obs.tool_output})
            step += 1
            
        resolved = "=== INCIDENT RESOLVED ===" in obs.tool_output
        timed_out = "STEP BUDGET EXHAUSTED" in obs.tool_output
        successful = bool(obs.scores and obs.scores.get("root_cause_accuracy", 0.0) >= 1.0 and not timed_out)

        # Return reward and scores for monitoring
        if hasattr(obs, 'scores') and obs.scores:
            return {
                "reward": total_reward,
                "scores": obs.scores,
                "difficulty": difficulty,
                "resolved": resolved,
                "timed_out": timed_out,
                "successful": successful,
                "steps": step,
            }
        else:
            return {
                "reward": total_reward,
                "scores": {},
                "difficulty": difficulty,
                "resolved": resolved,
                "timed_out": timed_out,
                "successful": successful,
                "steps": step,
            }

# -----------------------------------------------------------------------
# GRPO Reward Function Wrapper
# -----------------------------------------------------------------------

def env_reward_func(prompts, completions, **kwargs):
    """
    GRPO calls this function to evaluate model completions.
    Because OpenEnv involves multi-step interaction, we must rollout 
    an episode to get the objective reward.
    (Note: True GRPO typically requires online sampling inside the trainer loop. 
     This acts as a synchronous bridge, instantiating the env per rollout).
    """
    global global_model, global_tokenizer, global_rollout_counter, global_last_difficulty
    global global_metric_sums, global_metric_counts, global_outcome_counts
    
    rewards = []
    for i, (prompt, comp) in enumerate(zip(prompts, completions)):
        rollout_index = global_rollout_counter
        difficulty = get_curriculum_difficulty(rollout_index)
        if difficulty != global_last_difficulty:
            print(
                f"[Curriculum] Switching to {difficulty.upper()} "
                f"at rollout {rollout_index + 1}/{TOTAL_CURRICULUM_ROLLOUTS}"
            )
            global_last_difficulty = difficulty

        global_rollout_counter += 1

        # We start an episode with the prompt as the initial alert (mocking actual rollout)
        result = generate_trajectory(global_model, global_tokenizer, prompt, difficulty=difficulty)
        r = result["reward"]
        scores = result["scores"]
        resolved = result["resolved"]
        timed_out = result["timed_out"]
        successful = result["successful"]
        steps = result["steps"]
        rewards.append(r)
        
        metric_row = {
            "rollout": rollout_index + 1,
            "difficulty": difficulty,
            "reward": r,
            "scores": scores,
            "resolved": resolved,
            "timed_out": timed_out,
            "successful": successful,
            "steps": steps,
        }
        with open(METRICS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_row) + "\n")

        global_metric_sums["reward"] += r
        global_metric_counts["reward"] += 1
        for score_name, score_value in scores.items():
            global_metric_sums[score_name] += float(score_value)
            global_metric_counts[score_name] += 1
        global_outcome_counts["episodes"] += 1
        global_outcome_counts["resolved"] += int(resolved)
        global_outcome_counts["timed_out"] += int(timed_out)
        global_outcome_counts["successful"] += int(successful)
        global_metric_sums["steps"] += float(steps)
        global_metric_counts["steps"] += 1
        
        # Log detailed metrics and sample generations every 10 episodes
        if i % 10 == 0:
            print(
                f"Episode {i} | Rollout {rollout_index + 1} | "
                f"Difficulty={difficulty} | Reward={r:.4f} | "
                f"Resolved={resolved} | Successful={successful} | Timeout={timed_out} | Steps={steps}"
            )
            if scores:
                print(f"  Scores: {scores}")
            # Log a sample completion (first 200 chars)
            comp_text = comp[0] if isinstance(comp, list) else str(comp)
            print(f"  Sample Generation: {comp_text[:200]}...")

        if (rollout_index + 1) % 20 == 0:
            avg_reward = global_metric_sums["reward"] / max(1, global_metric_counts["reward"])
            print(f"[Metrics] Avg Reward after {rollout_index + 1} rollouts: {avg_reward:.4f}")
            avg_steps = global_metric_sums["steps"] / max(1, global_metric_counts["steps"])
            success_rate = global_outcome_counts["resolved"] / max(1, global_outcome_counts["episodes"])
            true_success_rate = global_outcome_counts["successful"] / max(1, global_outcome_counts["episodes"])
            timeout_rate = global_outcome_counts["timed_out"] / max(1, global_outcome_counts["episodes"])
            print(f"  [Metrics] Avg steps: {avg_steps:.2f}")
            print(f"  [Metrics] Resolve-called rate: {success_rate:.2%}")
            print(f"  [Metrics] Correct-resolution rate: {true_success_rate:.2%}")
            print(f"  [Metrics] Timeout rate: {timeout_rate:.2%}")
            for score_name in [
                "root_cause_accuracy",
                "time_to_resolution",
                "hypothesis_quality",
                "blast_radius_control",
                "postmortem_quality",
            ]:
                if global_metric_counts[score_name]:
                    avg_score = global_metric_sums[score_name] / global_metric_counts[score_name]
                    print(f"  [Metrics] Avg {score_name}: {avg_score:.4f}")
    
    return rewards

# -----------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------

def main():
    print("Loading Unsloth Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6, 
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Expose globally for the reward function to access 
    # (Hacky but works well in standard notebook/script workflows)
    global global_model, global_tokenizer, global_rollout_counter, global_last_difficulty
    global global_metric_sums, global_metric_counts, global_outcome_counts
    global_model = model
    global_tokenizer = tokenizer
    global_rollout_counter = 0
    global_last_difficulty = None
    global_metric_sums = defaultdict(float)
    global_metric_counts = defaultdict(int)
    global_outcome_counts = defaultdict(int)
    os.makedirs(os.path.dirname(METRICS_LOG_PATH), exist_ok=True)
    with open(METRICS_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")
    
    # Create Dummy Prompts Dataset. 
    # In SRE-Bench, reset() gives a random fault, so the prompt is just "Start Episode".
    dummy_dataset = Dataset.from_dict({
        "prompt": ["Start Incident Response"] * 128  # 128 episodes
    })
    
    training_args = GRPOConfig(
        output_dir="sre_bench_grpo_outputs",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        logging_steps=1,
        max_steps=TRAIN_MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=NUM_GENERATIONS, # Group size for GRPO
        max_prompt_length=512,
        max_completion_length=1024, # Needs room for CoT + XML
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        report_to="none"
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[env_reward_func],
        args=training_args,
        train_dataset=dummy_dataset,
    )
    
    print("\nStarting GRPO Training...\n")
    trainer.train()
    
    print("\nSaving final model...")
    # For 4-bit LoRA models, merge properly to avoid quality loss
    model.save_pretrained_merged("sre_bench_final_model", tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained("sre_bench_final_model")
    print("Done!")

if __name__ == "__main__":
    main()
