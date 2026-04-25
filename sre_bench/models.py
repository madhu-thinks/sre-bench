"""
Data models for the SRE-Bench environment.

Defines the Action and Observation types used by the agent and the environment.
The agent issues tool-call actions; the environment returns structured observations.
"""

from typing import Any, Dict, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SreBenchAction(Action):
    """
    A single tool call issued by the SRE agent.

    The agent selects a tool_name and provides arguments as a flat dict.
    Supported tool names:
        grep_logs           - args: service (str), pattern (str)
        get_metrics         - args: service (str), window (str, optional)
        get_error_rate      - args: service (str)
        describe_pod        - args: name (str)
        check_db_connections - args: (none)
        rollback_deploy     - args: service (str)
        restart_service     - args: service (str)
        scale_replicas      - args: service (str), n (int)
        fix_disk            - args: service (str)
        fix_network         - args: service (str)
        resolve_incident    - args: root_cause (str), fix_applied (str)
    """

    tool_name: str = Field(
        ...,
        description="Name of the SRE tool to invoke.",
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value arguments for the tool.",
    )
    # Optional: agent's stated hypothesis at this step (used for rubric scoring)
    hypothesis: Optional[str] = Field(
        default=None,
        description=(
            "Agent's current working hypothesis about the root cause. "
            "Providing this at steps 3 and 6 contributes to the Hypothesis Quality score."
        ),
    )


class SreBenchObservation(Observation):
    """
    Observation returned to the agent after each action.

    Contains the tool output plus episode metadata so the agent
    knows its progress and current incident context.
    """

    # The raw output of the tool call (log lines, metric dump, etc.)
    tool_output: str = Field(
        default="",
        description="Raw output returned by the invoked tool.",
    )
    # The firing alert shown at the start of the episode
    alert: str = Field(
        default="",
        description="The original PagerDuty-style alert that triggered this incident.",
    )
    # Current step (1-indexed after reset)
    step: int = Field(
        default=0,
        description="Current episode step count.",
    )
    # Remaining steps before forced escalation
    steps_remaining: int = Field(
        default=20,
        description="Steps remaining before the episode is automatically terminated.",
    )
    # Whether the episode has ended
    episode_ended: bool = Field(
        default=False,
        description="True if the episode has ended (resolved or budget exhausted).",
    )
    # Sub-scores (populated on episode end)
    scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-rubric-component scores. Only populated when episode_ended=True.",
    )
    # Human-readable incident summary (populated on episode end)
    incident_summary: Optional[str] = Field(
        default=None,
        description="Final incident summary. Only populated when episode_ended=True.",
    )
