# 🚨 SRE-Bench: Teaching AI to Wake Up at 3 AM So You Don't Have To

Imagine this: It’s 3:00 AM. Your phone rings. The main server is down, customers can't pay, and the company is losing thousands of dollars every minute. You drag yourself out of bed, open your laptop, and stare at a massive wall of error logs. It’s like trying to find a needle in a burning haystack.

What if an AI could do this for you? 

That is exactly the problem we set out to solve for the **Meta PyTorch OpenEnv Hackathon**. Our goal was simple but massive: **Train an AI to autonomously fix broken servers, reducing downtime by 40%.**

---

### 🤔 The Problem: AI Guesses, Humans Investigate

If you ask a standard AI to "fix a server", it panics. It acts like a terrible mechanic who just starts replacing random parts without checking under the hood first. In the tech world, we call this increasing the **"blast radius"**—the AI accidentally takes down healthy systems while trying to guess the fix.

A human Senior Engineer (an SRE) doesn't guess. They check the metrics, read the logs, form a hypothesis, and *then* take action.

We needed to teach an AI to stop guessing and start investigating.

---

### 🕹️ The Flight Simulator (Our Environment)

You can't train a pilot by putting them in a real plane and hoping they don't crash. You build a flight simulator.

We built a flight simulator for AI using **OpenEnv**. 

**SRE-Bench** is a fully simulated, 7-service microservice cluster that we can break on demand. We simulate memory leaks, traffic spikes, and database crashes. The AI is dropped into this virtual world with nothing but a set of IT tools (like `grep_logs` and `restart_service`) and told to find the problem.

---

### 🧠 The Training: How to Train Your AI

To train our model (an **Unsloth-optimized Qwen2.5-3B**), we didn't just give it flashcards (Supervised Fine-Tuning). We used **Reinforcement Learning (GRPO via TRL)**. 

Think of it like training a dog. We drop the AI into our simulator. If it blindly restarts a healthy server, we take away points. If it reads the logs, forms a good hypothesis, and fixes the actual root cause quickly, we give it a massive reward.

To stop the AI from cheating (a massive problem in AI called **Reward Hacking**), we built a **Composable Rubric**. We don't just score the AI on *whether* the server is fixed, we score it on:
1. **Time:** How fast did it fix it?
2. **Blast Radius:** Did it break anything else?
3. **Correctness:** Did it find the true root cause?
4. **Process Supervision:** Did it use the right `<thought>` and `<tool>` process to get there?

We also used **Curriculum Learning**—starting the AI on simple fixes before throwing 3 AM disaster scenarios at it.

---

### 📈 The Results: Did it actually work?

Yes. The AI transformed from clueless to an expert investigator. We tested the AI on **32 real incident episodes** in our live environment.

- **The Untrained Baseline:** The standard AI scored a brutal **0.05 out of 1.0**. It hallucinated logs, used tools incorrectly, and totally failed to fix the servers.
- **After Training:** The AI achieved a **0.458 average reward**—an incredible baseline improvement!
- **The Peak Run:** On its best run, the AI scored a **0.94 out of 1.0**. It instantly identified the correct broken component, avoided touching healthy services, and resolved the incident in the minimum possible steps. 
- **Resolution Rate:** By the end of training on our Easy Tier, the AI was fixing **90.6%** of all incidents thrown at it.

We successfully built an environment that proves we can reduce server downtime. By training models in SRE-Bench using GRPO, we can shift the burden of 3 AM outages from tired humans to sharp, methodical AI agents.

*(Want to see the training logs? Check out our GitHub README for the full visualizations!)*
