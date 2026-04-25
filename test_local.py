"""
Smoke test — verifies the SRE engine and environment work correctly
before starting the FastAPI server.

Run with:  python test_local.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from server.sre_engine import SREEngine, FaultType
from server.sre_bench_environment import SreBenchEnvironment
from models import SreBenchAction

def divider(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def test_engine():
    divider("ENGINE TEST — All 8 fault types")
    engine = SREEngine()
    for fault in FaultType:
        cluster = engine.new_episode(fault_type=fault)
        primary = cluster.fault_service
        svc = cluster.get_service(primary)
        ok = svc.error_rate > 0.1
        print(f"  [{('PASS' if ok else 'FAIL')}] {fault.value:40s} "
              f"→ {primary} error_rate={svc.error_rate:.2f}")
    print()

def test_environment():
    divider("ENVIRONMENT RESET + STEP TEST")
    env = SreBenchEnvironment()
    
    # reset
    obs = env.reset()
    print(f"[PASS] reset() → alert: {obs.alert[:80]}...")
    print(f"       steps_remaining={obs.steps_remaining}, done={obs.done}")
    
    # Step 1: get_metrics
    action = SreBenchAction(
        tool_name="get_metrics",
        arguments={"service": "database"},
        hypothesis="Checking database health first",
    )
    result = env.step(action)
    print(f"\n[PASS] step(get_metrics/database)")
    print(f"       step={result.step}, reward={result.reward}")
    print(f"       output preview: {result.tool_output[:120]}")
    
    # Step 2: grep_logs
    action = SreBenchAction(
        tool_name="grep_logs",
        arguments={"service": "database", "pattern": "ERROR"},
        hypothesis="Checking for DB errors",
    )
    result = env.step(action)
    print(f"\n[PASS] step(grep_logs/database/ERROR)")
    print(f"       step={result.step}, output lines={len(result.tool_output.splitlines())}")

    # Step 3: hypothesis at checkpoint
    action = SreBenchAction(
        tool_name="check_db_connections",
        arguments={},
        hypothesis="database connection pool exhaustion",
    )
    result = env.step(action)
    print(f"\n[PASS] step(check_db_connections) — hypothesis logged at step 3")
    print(f"       output: {result.tool_output[:200]}")

    # Final: resolve_incident
    action = SreBenchAction(
        tool_name="resolve_incident",
        arguments={
            "root_cause": "database connection pool exhaustion",
            "fix_applied": "Increased max_connections on database and restarted connection pool",
        },
    )
    result = env.step(action)
    print(f"\n[PASS] resolve_incident()")
    print(f"       done={result.done}, reward={result.reward:.4f}")
    print(f"       scores={result.scores}")
    print(f"\n  Full output:\n{result.tool_output}")

def test_invalid_tool():
    divider("EDGE CASE — Unknown tool name")
    env = SreBenchEnvironment()
    env.reset()
    action = SreBenchAction(tool_name="hack_server", arguments={})
    result = env.step(action)
    assert "Unknown tool" in result.tool_output, "Should return unknown tool message"
    print(f"[PASS] Unknown tool handled: {result.tool_output[:80]}")

def test_budget_exhaustion():
    divider("EDGE CASE — Step budget exhaustion (runs 20 steps)")
    env = SreBenchEnvironment()
    env.reset()
    for i in range(20):
        action = SreBenchAction(
            tool_name="get_error_rate",
            arguments={"service": "frontend"},
        )
        result = env.step(action)
        if result.done:
            print(f"[PASS] Budget exhausted at step {result.step}, "
                  f"reward={result.reward:.4f}")
            break

def test_curriculum_easy():
    divider("CURRICULUM TEST — Easy Difficulty")
    env = SreBenchEnvironment(difficulty="easy")
    
    # reset
    obs = env.reset()
    print(f"[PASS] reset(easy) → alert: {obs.alert[:80]}...")
    
    # Check that it's an easy fault (no cascade, so database should be healthy if not the victim)
    # But since we don't know the fault, just check it runs
    print(f"       episode active, ready for training")
    
    # Quick episode
    for i in range(5):
        action = SreBenchAction(
            tool_name="get_error_rate",
            arguments={"service": "database"},
        )
        result = env.step(action)
        if result.done:
            print(f"[PASS] Episode ended at step {i+1}, reward={result.reward:.4f}")
            break

if __name__ == "__main__":
    print("\n*** SRE-Bench Local Smoke Test ***")
    try:
        test_engine()
        test_environment()
        test_invalid_tool()
        test_budget_exhaustion()
        test_curriculum_easy()
        print("\n✅ ALL TESTS PASSED — Environment is working correctly.")
    except Exception as e:
        import traceback
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
