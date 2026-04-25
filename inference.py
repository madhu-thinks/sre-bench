"""
SRE-Bench Inference Script
Runs a full incident response episode using a trained RL model.
Matches the Priority_panic standard for hackathon submissions.

Usage:
    python inference.py --model_path ./sre_bench_final_model --env_url http://localhost:8000
"""

import argparse
import re
import json
import torch
from unsloth import FastLanguageModel
try:
    from sre_bench import SreBenchEnv, SreBenchAction
except ImportError:
    from client import SreBenchEnv
    from models import SreBenchAction

# ---------------------------------------------------------------------------
# Logic to parse the agent's XML tool call
# ---------------------------------------------------------------------------

def parse_action(text: str) -> SreBenchAction:
    """Extracts JSON tool call from <tool> tags."""
    match = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    if not match:
        return SreBenchAction(
            tool_name="grep_logs", 
            arguments={"service": "frontend", "pattern": "error"},
            hypothesis="FAILED_TO_GENERATE_XML"
        )
    try:
        data = json.loads(match.group(1).strip())
        return SreBenchAction(
            tool_name=data.get("tool_name", "grep_logs"),
            arguments=data.get("arguments", {}),
            hypothesis=data.get("hypothesis", "")
        )
    except:
        return SreBenchAction(
            tool_name="grep_logs", 
            arguments={"service": "frontend", "pattern": "error"},
            hypothesis="JSON_DECODE_ERROR"
        )

# ---------------------------------------------------------------------------
# Main Inference Loop
# ---------------------------------------------------------------------------

def run_episode(model_path, env_url):
    print(f"\n[1/3] Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print(f"[2/3] Connecting to Environment at {env_url}...")
    with SreBenchEnv(base_url=env_url).sync() as env:
        result = env.reset()
        obs = result.observation
        
        print("\n" + "="*60)
        print("🚨 INCIDENT STARTED")
        print(f"Alert: {obs.alert}")
        print("="*60 + "\n")

        chat_history = [
            {"role": "system", "content": "You are a Senior SRE. Use <tool>JSON</tool> tags to act."},
            {"role": "user", "content": obs.tool_output}
        ]
        
        done = False
        step = 1
        
        while not done and step <= 20:
            print(f"--- Step {step} ---")
            
            # Tokenize & Generate
            inputs = tokenizer.apply_chat_template(
                chat_history, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to("cuda")
            
            outputs = model.generate(
                inputs, 
                max_new_tokens=512, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            print(f"Agent analysis:\n{response.split('<tool>')[0].strip()}")
            
            # Parse & Execute
            action = parse_action(response)
            print(f"Taking Action: {action.tool_name}({action.arguments})")
            
            result = env.step(action)
            obs = result.observation
            done = result.done
            
            chat_history.append({"role": "assistant", "content": response})
            chat_history.append({"role": "user", "content": obs.tool_output})
            
            print(f"Env Result: {obs.tool_output[:200]}...")
            step += 1
            
        print("\n" + "="*60)
        print("🏁 EPISODE FINISHED")
        print(f"Final Reward: {result.reward}")
        if obs.scores:
            print("\nRubric Breakdown:")
            for k, v in obs.scores.items():
                print(f"  - {k}: {v:.2f}")
        print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_url", type=str, default="http://localhost:8000")
    args = parser.parse_args()
    
    run_episode(args.model_path, args.env_url)
