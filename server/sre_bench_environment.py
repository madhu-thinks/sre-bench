"""
SRE-Bench Environment Implementation.

Simulates production microservice incidents for LLM agent training.
The agent must triage, investigate, and resolve the incident using
real SRE tool calls, with partial observability and noisy logs.

Episode lifecycle:
  reset()  → injects a random fault, returns the PagerDuty alert
  step()   → processes a tool call, updates cluster, returns observation
  state    → current episode metadata (step count, episode_id)
  (agent calls resolve_incident() to end episode, or hits step budget)

Reward rubric (5 components):
  1. Root Cause Accuracy  (0.30) - matched injected fault type + service
  2. Time to Resolution   (0.25) - fewer steps = higher score
  3. Hypothesis Quality   (0.20) - correct intermediate hypothesis at steps 3 & 6
  4. Blast Radius Control (0.15) - penalty for harming healthy services
  5. Postmortem Quality   (0.10) - coherence of final explanation
"""

import os
import json
import httpx
import re
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SreBenchAction, SreBenchObservation
    from .sre_engine import SREEngine, FaultType, SERVICES
except ImportError:
    from models import SreBenchAction, SreBenchObservation
    from server.sre_engine import SREEngine, FaultType, SERVICES


# ---------------------------------------------------------------------------
# Rubric constants
# ---------------------------------------------------------------------------

MAX_STEPS = 20          # Hard step budget
STEP_BUDGET_BONUS_THRESHOLD = 8   # Resolve in ≤ 8 steps for full time bonus

# Root cause alias mapping — agent can use natural language that maps to a FaultType
ROOT_CAUSE_ALIASES = {
    FaultType.DB_CONNECTION_EXHAUSTION: [
        "db_connection_exhaustion", "database connection exhaustion",
        "connection pool exhausted", "db connections", "connection exhaustion",
        "pool exhausted", "db pool", "connection pool",
    ],
    FaultType.OOM_KILLED: [
        "oom_killed", "out of memory", "oom", "memory limit",
        "oomkilled", "pod oom", "memory exhaustion", "java heap",
    ],
    FaultType.NETWORK_PARTITION: [
        "network_partition", "network partition", "network issue",
        "connectivity issue", "network split", "partition",
    ],
    FaultType.RETRY_STORM: [
        "retry_storm", "retry storm", "queue depth", "redis queue",
        "message queue", "backlog", "retry cascade",
    ],
    FaultType.CONFIG_MISSING_ENV: [
        "config_missing_env", "missing env var", "missing environment variable",
        "config missing", "env var missing", "configuration error",
        "missing config", "crashloopbackoff",
    ],
    FaultType.DISK_FULL: [
        "disk_full", "disk full", "no space left", "disk space",
        "storage full", "log volume full",
    ],
    FaultType.CPU_THROTTLE: [
        "cpu_throttle", "cpu throttle", "cpu throttling",
        "cpu limit", "cpu starvation", "high cpu", "cpu pressure",
    ],
    FaultType.MEMORY_LEAK: [
        "memory_leak", "memory leak", "gradual memory growth",
        "heap growth", "memory growth",
    ],
}


def _match_root_cause(declared: str, actual: FaultType) -> bool:
    """Check if the agent's declared root cause matches the injected fault."""
    declared_lower = declared.lower().strip()
    aliases = ROOT_CAUSE_ALIASES.get(actual, [])
    # Exact enum match
    if declared_lower == actual.value:
        return True
    # Alias match
    return any(alias in declared_lower for alias in aliases)


def _match_service(declared_fix: str, actual_service: str) -> bool:
    """Check if the fix text references the correct affected service."""
    return actual_service.lower() in declared_fix.lower()


# ---------------------------------------------------------------------------
# Rubric scorer
# ---------------------------------------------------------------------------

class RubricScorer:
    """
    Computes the 5-component reward rubric at episode end.

    All sub-scores are in [0.0, 1.0]. Final reward is a weighted sum.
    """

    WEIGHTS = {
        "root_cause_accuracy":  0.30,
        "time_to_resolution":   0.25,
        "hypothesis_quality":   0.15,
        "blast_radius_control": 0.15,
        "postmortem_quality":   0.15,  # Increased weight for LLM-judged PM
    }

    def __init__(self):
        # Hypothesis checkpoints: {step_number: hypothesis_text}
        self.hypothesis_log: dict = {}
        # Blast radius tracker: list of (service, was_harmful)
        self.action_log: list = []
        self.hf_token = os.getenv("HF_TOKEN")
        self.api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    def _run_llm_grader(self, root_cause: str, fix: str, actual_fault: str, actual_service: str) -> float:
        """Connects to HF Router to grade the postmortem explanation."""
        if not self.hf_token:
            return 0.5  # Neutral fallback if no token

        prompt = f"""You are an expert SRE Auditor. Grade the quality of an incident postmortem.
Actual Incident: The service '{actual_service}' suffered a '{actual_fault}'.
Agent Explanation: 
- Root Cause: {root_cause}
- Fix Applied: {fix}

Score the explanation from 0.0 to 1.0 based on:
1. Accuracy: Did they identify the correct fault and service?
2. Technical Depth: Is the explanation coherent?
3. Resolution: Does the fix actually solve that specific fault?

Output ONLY a JSON object: {{"score": float, "reasoning": "brief string"}}
"""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{self.api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.hf_token}"},
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data['choices'][0]['message']['content']
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        result = json.loads(match.group())
                        return float(result.get("score", 0.5))
        except Exception:
            pass
        return 0.5

    def log_hypothesis(self, step: int, hypothesis: Optional[str]):
        if hypothesis:
            self.hypothesis_log[step] = hypothesis

    def log_action(self, tool_name: str, service: str, was_harmful: bool):
        """Track actions that might constitute blast radius damage."""
        if tool_name in {"rollback_deploy", "restart_service", "scale_replicas",
                         "fix_disk", "fix_network"}:
            self.action_log.append({
                "tool": tool_name, "service": service, "harmful": was_harmful
            })

    def compute(
        self,
        declared_root_cause: str,
        declared_fix: str,
        actual_fault: FaultType,
        actual_service: str,
        step_count: int,
        resolved_by_agent: bool,
    ) -> dict:
        """Compute all rubric components and return scores + final reward."""
        scores = {}

        # ── 1. Root Cause Accuracy (30%) ──────────────────────────────────
        cause_correct = _match_root_cause(declared_root_cause, actual_fault)
        service_correct = _match_service(declared_fix, actual_service)
        if cause_correct and service_correct:
            scores["root_cause_accuracy"] = 1.0
        elif cause_correct or service_correct:
            scores["root_cause_accuracy"] = 0.5
        else:
            scores["root_cause_accuracy"] = 0.0

        # ── 2. Time to Resolution (25%) ────────────────────────────────────
        if not resolved_by_agent:
            scores["time_to_resolution"] = 0.0
        elif step_count <= STEP_BUDGET_BONUS_THRESHOLD:
            scores["time_to_resolution"] = 1.0
        else:
            # Linear decay: full score at 8 steps, 0 at MAX_STEPS
            decay = (MAX_STEPS - step_count) / (MAX_STEPS - STEP_BUDGET_BONUS_THRESHOLD)
            scores["time_to_resolution"] = max(0.0, decay)

        # ── 3. Hypothesis Quality (20%) ────────────────────────────────────
        # Check hypotheses at step 3 and step 6 for correctness
        checkpoint_scores = []
        for checkpoint in [3, 6]:
            hyp = self.hypothesis_log.get(checkpoint)
            if hyp:
                if _match_root_cause(hyp, actual_fault):
                    checkpoint_scores.append(1.0)
                elif actual_service.lower() in hyp.lower():
                    checkpoint_scores.append(0.5)  # right service, wrong fault
                else:
                    checkpoint_scores.append(0.0)
        if checkpoint_scores:
            scores["hypothesis_quality"] = sum(checkpoint_scores) / len(checkpoint_scores)
        else:
            scores["hypothesis_quality"] = 0.0  # Never stated a hypothesis

        # ── 4. Blast Radius Control (15%) ──────────────────────────────────
        harmful_actions = sum(1 for a in self.action_log if a["harmful"])
        total_fix_actions = len(self.action_log)
        if total_fix_actions == 0:
            # No fix actions at all → couldn't have caused blast radius
            scores["blast_radius_control"] = 0.5 if resolved_by_agent else 0.0
        else:
            # Penalise each harmful action
            harm_ratio = harmful_actions / total_fix_actions
            scores["blast_radius_control"] = max(0.0, 1.0 - harm_ratio)

        # ── 5. Postmortem Quality (15%) ────────────────────────────────────
        # Use LLM Judge (matches Priority_panic standard)
        llm_score = self._run_llm_grader(
            declared_root_cause, declared_fix, actual_fault.value, actual_service
        )
        
        # Combine LLM judge with heuristic fallback
        postmortem_text = f"{declared_root_cause} {declared_fix}".lower()
        has_fault_ref = any(
            alias in postmortem_text
            for alias in ROOT_CAUSE_ALIASES.get(actual_fault, [actual_fault.value])
        )
        has_service_ref = actual_service.lower() in postmortem_text
        heuristic_score = (has_fault_ref + has_service_ref) / 2
        
        # LLM score is weighted 80%, heuristic fallback 20%
        scores["postmortem_quality"] = (llm_score * 0.8) + (heuristic_score * 0.2)

        # ── Final weighted reward ──────────────────────────────────────────
        final_reward = sum(
            scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS
        )
        scores["final_reward"] = round(final_reward, 4)
        scores["_weights"] = self.WEIGHTS
        return scores


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SreBenchEnvironment(Environment):
    """
    SRE-Bench: Production Incident Training Environment.

    Simulates production outages in a 7-service microservice cluster.
    The agent must triage, investigate, and resolve the incident using
    tool calls, with partial observability, noisy logs, and real cascades.

    Supports concurrent sessions for parallel training rollouts.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: str = "hard"):
        from .sre_engine import Difficulty
        self._difficulty = Difficulty(difficulty.lower())
        self._engine = SREEngine()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scorer = RubricScorer()
        self._alert: str = ""
        self._episode_active: bool = False

    # ─────────────────────────────────────────────────────────────────────
    # OpenEnv API
    # ─────────────────────────────────────────────────────────────────────

    def reset(self) -> SreBenchObservation:
        """
        Start a fresh episode.

        Injects a random fault into the cluster and returns the firing alert.
        The agent does NOT know which fault was injected — it must investigate.
        """
        cluster = self._engine.new_episode(difficulty=self._difficulty)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scorer = RubricScorer()
        self._alert = cluster.alert_fired
        self._episode_active = True

        return SreBenchObservation(
            tool_output=(
                f"=== INCIDENT RESPONSE INITIATED ===\n"
                f"Incident ID: {cluster.incident_id}\n"
                f"Alert: {cluster.alert_fired}\n\n"
                f"Available services: {', '.join(SERVICES)}\n\n"
                f"Available tools:\n"
                f"  grep_logs(service, pattern)\n"
                f"  get_metrics(service, window='5m')\n"
                f"  get_error_rate(service)\n"
                f"  describe_pod(name)\n"
                f"  check_db_connections()\n"
                f"  rollback_deploy(service)\n"
                f"  restart_service(service)\n"
                f"  scale_replicas(service, n)\n"
                f"  fix_disk(service)\n"
                f"  fix_network(service)\n"
                f"  resolve_incident(root_cause, fix_applied)\n\n"
                f"You have {MAX_STEPS} steps. Triage, investigate, and resolve."
            ),
            alert=cluster.alert_fired,
            step=0,
            steps_remaining=MAX_STEPS,
            episode_ended=False,
            done=False,
            reward=0.0,
        )

    def step(self, action: SreBenchAction) -> SreBenchObservation:  # type: ignore[override]
        """
        Execute one SRE tool call and return the result.

        If the agent calls resolve_incident, the episode ends and
        the full rubric score is computed and returned.
        If the step budget is exhausted, the episode ends with a penalty.
        """
        if not self._episode_active:
            return SreBenchObservation(
                tool_output="Episode is not active. Call reset() to start a new episode.",
                alert=self._alert, step=0, steps_remaining=0,
                episode_ended=True, done=True, reward=0.0,
            )

        self._state.step_count += 1
        step = self._state.step_count
        cluster = self._engine.cluster

        # Log hypothesis if provided
        self._scorer.log_hypothesis(step, action.hypothesis)

        # Update cluster step counter
        if cluster:
            cluster.step_count = step

        tool = action.tool_name
        args = action.arguments
        output = ""
        reward = 0.0
        done = False
        scores = None

        # ── Dispatch tool calls ───────────────────────────────────────────
        if tool == "grep_logs":
            service = args.get("service", "")
            pattern = args.get("pattern", "")
            output = self._engine.grep_logs(service, pattern)

        elif tool == "get_metrics":
            service = args.get("service", "")
            window = args.get("window", "5m")
            output = self._engine.get_metrics(service, window)

        elif tool == "get_error_rate":
            service = args.get("service", "")
            output = self._engine.get_error_rate(service)

        elif tool == "describe_pod":
            name = args.get("name", "")
            output = self._engine.describe_pod(name)

        elif tool == "check_db_connections":
            output = self._engine.check_db_connections()

        elif tool == "rollback_deploy":
            service = args.get("service", "")
            msg, helpful = self._engine.rollback_deploy(service)
            harmful = not helpful and bool(service) and service in SERVICES
            self._scorer.log_action(tool, service, harmful)
            output = msg

        elif tool == "restart_service":
            service = args.get("service", "")
            msg, helpful = self._engine.restart_service(service)
            harmful = not helpful and bool(service) and service in SERVICES
            self._scorer.log_action(tool, service, harmful)
            output = msg

        elif tool == "scale_replicas":
            service = args.get("service", "")
            n = int(args.get("n", 3))
            msg, helpful = self._engine.scale_replicas(service, n)
            harmful = not helpful and bool(service) and service in SERVICES
            self._scorer.log_action(tool, service, harmful)
            output = msg

        elif tool == "fix_disk":
            service = args.get("service", "")
            msg, helpful = self._engine.fix_disk(service)
            harmful = not helpful and bool(service) and service in SERVICES
            self._scorer.log_action(tool, service, harmful)
            output = msg

        elif tool == "fix_network":
            service = args.get("service", "")
            msg, helpful = self._engine.fix_network(service)
            harmful = not helpful and bool(service) and service in SERVICES
            self._scorer.log_action(tool, service, harmful)
            output = msg

        elif tool == "resolve_incident":
            root_cause = args.get("root_cause", "")
            fix_applied = args.get("fix_applied", "")
            result = self._engine.resolve_incident(root_cause, fix_applied)

            scores = self._scorer.compute(
                declared_root_cause=root_cause,
                declared_fix=fix_applied,
                actual_fault=cluster.fault_type if cluster else FaultType.DB_CONNECTION_EXHAUSTION,
                actual_service=cluster.fault_service if cluster else "database",
                step_count=step,
                resolved_by_agent=True,
            )
            reward = scores["final_reward"]
            done = True
            self._episode_active = False

            # Strip internal keys not meant for the Observation model
            display_scores = {k: v for k, v in scores.items()
                              if k not in {"final_reward", "_weights"}}

            output = (
                f"=== INCIDENT RESOLVED ===\n"
                f"Declared root cause: {root_cause}\n"
                f"Fix applied:         {fix_applied}\n"
                f"Actual fault:        {result.get('actual_fault_type', 'unknown')}\n"
                f"Actual service:      {result.get('actual_fault_service', 'unknown')}\n\n"
                f"Rubric Scores:\n"
                + "\n".join(
                    f"  {k}: {v:.3f} (weight {self._scorer.WEIGHTS.get(k, 0):.0%})"
                    for k, v in display_scores.items()
                )
                + f"\n\n  FINAL REWARD: {reward:.4f}"
            )

        else:
            output = (
                f"Unknown tool '{tool}'. Available tools: grep_logs, get_metrics, "
                f"get_error_rate, describe_pod, check_db_connections, rollback_deploy, "
                f"restart_service, scale_replicas, fix_disk, fix_network, resolve_incident"
            )

        # ── Budget exhaustion ─────────────────────────────────────────────
        if not done and step >= MAX_STEPS:
            scores = self._scorer.compute(
                declared_root_cause="",
                declared_fix="",
                actual_fault=cluster.fault_type if cluster else FaultType.DB_CONNECTION_EXHAUSTION,
                actual_service=cluster.fault_service if cluster else "database",
                step_count=step,
                resolved_by_agent=False,
            )
            reward = scores["final_reward"] * 0.3  # Escalation penalty
            done = True
            self._episode_active = False
            output += (
                f"\n\nSTEP BUDGET EXHAUSTED - Incident escalated to senior SRE.\n"
                f"Partial credit reward: {reward:.4f}"
            )

        # Clean scores for serialisation (strip internal keys)
        clean_scores = None
        if scores is not None:
            clean_scores = {k: v for k, v in scores.items()
                            if k not in {"final_reward", "_weights"}}

        return SreBenchObservation(
            tool_output=output,
            alert=self._alert,
            step=step,
            steps_remaining=max(0, MAX_STEPS - step),
            episode_ended=done,
            scores=clean_scores,
            done=done,
            reward=reward,
            metadata={
                "tool_called": tool,
                "step": step,
                "fault_type": cluster.fault_type.value if cluster else None,
            },
        )

    @property
    def state(self) -> State:
        return self._state
