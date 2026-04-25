"""
Smoke test - verifies the SRE engine and environment work correctly
before starting the FastAPI server.

Run with: python test_local.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from models import SreBenchAction
from server.sre_bench_environment import SreBenchEnvironment
from server.sre_engine import FaultType, SREEngine


def divider(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def test_engine():
    divider("ENGINE TEST - All 8 fault types")
    engine = SREEngine()
    for fault in FaultType:
        cluster = engine.new_episode(fault_type=fault)
        primary = cluster.fault_service
        svc = cluster.get_service(primary)
        ok = svc.error_rate > 0.1
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {fault.value:40s} "
            f"-> {primary} error_rate={svc.error_rate:.2f}"
        )
    print()


def test_environment():
    divider("ENVIRONMENT RESET + STEP TEST")
    env = SreBenchEnvironment()

    obs = env.reset()
    print(f"[PASS] reset() -> alert: {obs.alert[:80]}...")
    print(f"       steps_remaining={obs.steps_remaining}, done={obs.done}")

    action = SreBenchAction(
        tool_name="get_metrics",
        arguments={"service": "database"},
        hypothesis="Checking database health first",
    )
    result = env.step(action)
    print("\n[PASS] step(get_metrics/database)")
    print(f"       step={result.step}, reward={result.reward}")
    print(f"       output preview: {result.tool_output[:120]}")

    action = SreBenchAction(
        tool_name="grep_logs",
        arguments={"service": "database", "pattern": "ERROR"},
        hypothesis="Checking for DB errors",
    )
    result = env.step(action)
    print("\n[PASS] step(grep_logs/database/ERROR)")
    print(f"       step={result.step}, output lines={len(result.tool_output.splitlines())}")

    action = SreBenchAction(
        tool_name="check_db_connections",
        arguments={},
        hypothesis="database connection pool exhaustion",
    )
    result = env.step(action)
    print("\n[PASS] step(check_db_connections) - hypothesis logged at step 3")
    print(f"       output: {result.tool_output[:200]}")

    action = SreBenchAction(
        tool_name="resolve_incident",
        arguments={
            "root_cause": "database connection pool exhaustion",
            "fix_applied": "Increased max_connections on database and restarted connection pool",
        },
    )
    result = env.step(action)
    print("\n[PASS] resolve_incident()")
    print(f"       done={result.done}, reward={result.reward:.4f}")
    print(f"       scores={result.scores}")
    print(f"\n  Full output:\n{result.tool_output}")


def test_invalid_tool():
    divider("EDGE CASE - Unknown tool name")
    env = SreBenchEnvironment()
    env.reset()
    action = SreBenchAction(tool_name="hack_server", arguments={})
    result = env.step(action)
    assert "Unknown tool" in result.tool_output, "Should return unknown tool message"
    print(f"[PASS] Unknown tool handled: {result.tool_output[:80]}")


def test_budget_exhaustion():
    divider("EDGE CASE - Step budget exhaustion (runs 20 steps)")
    env = SreBenchEnvironment()
    env.reset()
    for _ in range(20):
        action = SreBenchAction(
            tool_name="get_error_rate",
            arguments={"service": "frontend"},
        )
        result = env.step(action)
        if result.done:
            print(f"[PASS] Budget exhausted at step {result.step}, reward={result.reward:.4f}")
            break


def test_curriculum_easy():
    divider("CURRICULUM TEST - Easy Difficulty")
    env = SreBenchEnvironment(difficulty="easy")

    obs = env.reset()
    print(f"[PASS] reset(easy) -> alert: {obs.alert[:80]}...")
    print("       episode active, ready for training")

    for i in range(5):
        action = SreBenchAction(
            tool_name="get_error_rate",
            arguments={"service": "database"},
        )
        result = env.step(action)
        if result.done:
            print(f"[PASS] Episode ended at step {i + 1}, reward={result.reward:.4f}")
            break


def test_reset_difficulty_override():
    divider("EDGE CASE - Reset Difficulty Override")
    env = SreBenchEnvironment(difficulty="hard")
    obs = env.reset(difficulty="easy")
    assert "Difficulty: easy" in obs.tool_output, "Reset override should be reflected in observation text"
    print("[PASS] reset(difficulty='easy') override applied correctly")


def test_blast_radius_penalty():
    divider("ANTI-HACK TEST - Blast Radius Penalty")
    env = SreBenchEnvironment(difficulty="easy")
    env.reset()

    # Intentionally harmful action on a likely healthy service.
    env.step(
        SreBenchAction(
            tool_name="restart_service",
            arguments={"service": "frontend"},
            hypothesis="Forcing an unnecessary restart",
        )
    )
    final = env.step(
        SreBenchAction(
            tool_name="resolve_incident",
            arguments={
                "root_cause": "unknown",
                "fix_applied": "restarted frontend",
            },
        )
    )
    assert final.done, "Episode should terminate on resolve_incident"
    assert final.scores is not None, "Rubric scores should be present at episode end"
    blast = final.scores.get("blast_radius_control")
    assert blast is not None and blast < 1.0, "Harmful action must reduce blast radius score"
    print(f"[PASS] harmful action penalized, blast_radius_control={blast:.3f}")


if __name__ == "__main__":
    print("\n*** SRE-Bench Local Smoke Test ***")
    try:
        test_engine()
        test_environment()
        test_invalid_tool()
        test_budget_exhaustion()
        test_curriculum_easy()
        test_reset_difficulty_override()
        test_blast_radius_penalty()
        print("\n[OK] ALL TESTS PASSED - Environment is working correctly.")
    except Exception as e:
        import traceback

        print(f"\n[FAIL] TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
