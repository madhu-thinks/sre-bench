"""SRE-Bench Environment Client."""

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SreBenchAction, SreBenchObservation
except ImportError:
    # Fallback for flat-repo execution.
    from models import SreBenchAction, SreBenchObservation


class SreBenchEnv(EnvClient[SreBenchAction, SreBenchObservation, State]):
    """
    Client for the SRE-Bench Environment.

    Connects to the running FastAPI server (local or HF Space) and
    provides a clean interface for training scripts.

    Sync usage (for TRL/Unsloth training loops):
        with SreBenchEnv(base_url="http://localhost:8000").sync() as client:
            result = client.reset()
            obs = result.observation
            print(obs.alert)

            result = client.step(SreBenchAction(
                tool_name="get_metrics",
                arguments={"service": "database"},
                hypothesis="I suspect database connection exhaustion",
            ))
            print(result.observation.tool_output)

    Async usage:
        async with SreBenchEnv(base_url="http://localhost:8000") as client:
            result = await client.reset()
            result = await client.step(SreBenchAction(...))
    """

    def _step_payload(self, action: SreBenchAction) -> Dict:
        return {
            "tool_name":  action.tool_name,
            "arguments":  action.arguments,
            "hypothesis": action.hypothesis,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SreBenchObservation]:
        obs_data = payload.get("observation", {})
        observation = SreBenchObservation(
            tool_output=obs_data.get("tool_output", ""),
            alert=obs_data.get("alert", ""),
            step=obs_data.get("step", 0),
            steps_remaining=obs_data.get("steps_remaining", 20),
            episode_ended=obs_data.get("episode_ended", False),
            scores=obs_data.get("scores"),
            incident_summary=obs_data.get("incident_summary"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
