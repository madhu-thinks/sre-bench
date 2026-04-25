"""
SRE-Bench Simulation Engine.

Models a realistic microservice cluster, injects faults, propagates cascades,
and generates noisy-but-authentic logs and metrics for agent observation.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVICES = [
    "frontend",
    "auth-api",
    "order-service",
    "payment-gateway",
    "database",
    "redis-queue",
    "load-balancer",
]

# Dependency graph: which services depend on which.
# If a dependency degrades, dependants feel the impact.
DEPENDENCY_GRAPH: Dict[str, List[str]] = {
    "frontend":        ["load-balancer", "auth-api"],
    "auth-api":        ["database", "redis-queue"],
    "order-service":   ["database", "payment-gateway", "redis-queue"],
    "payment-gateway": ["database"],
    "database":        [],
    "redis-queue":     [],
    "load-balancer":   ["frontend"],
}

# How much a degraded dependency raises a service's error rate (0-1 multiplier).
CASCADE_FACTOR = 0.35


class ServiceStatus(str, Enum):
    HEALTHY   = "healthy"
    DEGRADED  = "degraded"
    CRITICAL  = "critical"
    DOWN      = "down"


class FaultType(str, Enum):
    DB_CONNECTION_EXHAUSTION  = "db_connection_exhaustion"
    OOM_KILLED                = "oom_killed"               # Bad deploy → pods OOMKilled
    NETWORK_PARTITION         = "network_partition"
    RETRY_STORM               = "retry_storm"
    CONFIG_MISSING_ENV        = "config_missing_env"       # Env var missing on deploy
    DISK_FULL                 = "disk_full"                # Log volume fills up
    CPU_THROTTLE              = "cpu_throttle"             # Noisy-neighbour on node
    MEMORY_LEAK               = "memory_leak"              # Gradual degradation


# Difficulty levels for curriculum learning
class Difficulty(str, Enum):
    EASY = "easy"      # Obvious faults, no cascading
    MEDIUM = "medium"  # Faults with cascading effects
    HARD = "hard"      # All faults, full complexity

# Faults by difficulty
EASY_FAULTS = {FaultType.DISK_FULL, FaultType.CONFIG_MISSING_ENV}
MEDIUM_FAULTS = {FaultType.OOM_KILLED, FaultType.CPU_THROTTLE, FaultType.MEMORY_LEAK}
HARD_FAULTS = {FaultType.DB_CONNECTION_EXHAUSTION, FaultType.NETWORK_PARTITION, FaultType.RETRY_STORM}


# Which service is the *primary* victim for each fault type.
FAULT_PRIMARY_SERVICE: Dict[FaultType, str] = {
    FaultType.DB_CONNECTION_EXHAUSTION: "database",
    FaultType.OOM_KILLED:               "order-service",
    FaultType.NETWORK_PARTITION:        "payment-gateway",
    FaultType.RETRY_STORM:              "redis-queue",
    FaultType.CONFIG_MISSING_ENV:       "auth-api",
    FaultType.DISK_FULL:                "load-balancer",
    FaultType.CPU_THROTTLE:             "order-service",
    FaultType.MEMORY_LEAK:              "frontend",
}


# ---------------------------------------------------------------------------
# Service State
# ---------------------------------------------------------------------------

@dataclass
class ServiceState:
    name:          str
    status:        ServiceStatus = ServiceStatus.HEALTHY
    error_rate:    float = 0.0          # 0.0 – 1.0
    cpu_pct:       float = random.uniform(10, 35)
    memory_pct:    float = random.uniform(20, 50)
    latency_p99:   float = random.uniform(40, 120)  # ms
    pod_count:     int   = 3
    restart_count: int   = 0
    deploy_version: str  = "v1.4.2"
    last_deploy:   str   = "2026-04-24T08:00:00Z"
    disk_pct:      float = random.uniform(20, 45)


# ---------------------------------------------------------------------------
# Cluster State
# ---------------------------------------------------------------------------

@dataclass
class ClusterState:
    services:      Dict[str, ServiceState] = field(default_factory=dict)
    fault_type:    Optional[FaultType]     = None
    fault_service: Optional[str]           = None
    incident_id:   str                     = ""
    alert_fired:   str                     = ""
    step_count:    int                     = 0
    resolved:      bool                    = False
    resolution:    Optional[str]           = None

    def get_service(self, name: str) -> ServiceState:
        return self.services[name]


# ---------------------------------------------------------------------------
# Log Templates — realistic log lines per fault type
# ---------------------------------------------------------------------------

# Red-herring log lines that appear during any incident to add noise.
RED_HERRING_LOGS = [
    "WARN  [health-check] Slow health check response: 312ms (threshold: 300ms)",
    "INFO  [metrics] Prometheus scrape: 847 samples collected",
    "WARN  [gc] G1GC pause 87ms - within acceptable range",
    "INFO  [audit] User auth token refreshed for session-id=a3f9b",
    "DEBUG [cache] Cache miss ratio: 0.12 (expected ~0.10)",
    "WARN  [net] TCP retransmit rate: 0.8% (monitoring threshold: 1%)",
    "INFO  [deploy] Heartbeat OK for version v1.4.2",
    "INFO  [cron] Daily log rotation triggered for /var/log/app",
    "WARN  [rate-limit] Client 10.0.2.45 near rate limit (490/500 rps)",
]

FAULT_LOG_SIGNATURES: Dict[FaultType, List[str]] = {
    FaultType.DB_CONNECTION_EXHAUSTION: [
        "ERROR [db-pool] Timeout acquiring connection after 30000ms - pool exhausted",
        "ERROR [db-pool] HikariPool-1 - Connection is not available, request timed out after 30000ms",
        "WARN  [db-pool] Pool size: 100/100 active, 0 idle",
        "ERROR [order-service] SQL query failed: PSQLException: FATAL: remaining connection slots are reserved",
        "ERROR [payment-gateway] Database connection refused: max_connections=100 reached",
        "WARN  [db] pg_stat_activity: 98 connections active, 2 reserved for superuser",
    ],
    FaultType.OOM_KILLED: [
        "ERROR [kubelet] Container 'order-service' exceeded memory limit (512Mi), OOMKilled",
        "WARN  [order-service] JVM heap usage: 498MB / 512MB (97%)",
        "ERROR [kubelet] Pod order-service-7d9f8b-xk2p9 restarted 4 times in last 10 minutes",
        "ERROR [order-service] OutOfMemoryError: Java heap space",
        "WARN  [order-service] GC overhead limit exceeded: 95% time in GC",
        "ERROR [k8s-events] Back-off restarting failed container order-service",
    ],
    FaultType.NETWORK_PARTITION: [
        "ERROR [payment-gateway] Connection to 10.0.1.45:5432 timed out after 5000ms",
        "ERROR [payment-gateway] Cannot reach database: dial tcp 10.0.1.45:5432: i/o timeout",
        "WARN  [network] Packet loss detected on eth0: 42% drop rate",
        "ERROR [order-service] payment-gateway health check failed: connection refused",
        "WARN  [envoy] Upstream cluster payment-gateway: 0/3 healthy endpoints",
        "ERROR [circuit-breaker] payment-gateway circuit OPEN after 5 consecutive failures",
    ],
    FaultType.RETRY_STORM: [
        "WARN  [redis-queue] Queue depth: 48293 messages (threshold: 10000)",
        "ERROR [order-service] Redis BLPOP timed out after 5000ms, retrying...",
        "WARN  [redis] Connected clients: 987 (max: 1000)",
        "ERROR [redis-queue] OOM command not allowed when used memory > 'maxmemory'",
        "WARN  [order-service] Retry attempt 7/10 for job processor - backoff 3200ms",
        "ERROR [redis] MISCONF: Redis is configured to save RDB snapshots, but is currently not able to persist on disk",
    ],
    FaultType.CONFIG_MISSING_ENV: [
        "FATAL [auth-api] Missing required environment variable: JWT_SECRET_KEY",
        "ERROR [auth-api] Application startup failed: configuration validation error",
        "ERROR [auth-api] KeyError: 'JWT_SECRET_KEY' not found in environment",
        "WARN  [k8s] Pod auth-api-6b8c9f-mp4k1 in CrashLoopBackOff state",
        "ERROR [frontend] Auth service unavailable: 503 Service Unavailable",
        "ERROR [auth-api] Cannot bind to port 8080: startup sequence incomplete",
    ],
    FaultType.DISK_FULL: [
        "ERROR [nginx] open() '/var/log/nginx/access.log' failed (28: No space left on device)",
        "ERROR [load-balancer] Failed to write to disk: No space left on device",
        "WARN  [disk] Filesystem /dev/sda1: 99% used (threshold: 85%)",
        "ERROR [filebeat] Cannot write to log buffer: disk full",
        "WARN  [logrotate] Failed to rotate logs: insufficient disk space",
        "ERROR [load-balancer] Health check endpoint unresponsive: write error",
    ],
    FaultType.CPU_THROTTLE: [
        "WARN  [order-service] CPU throttle: 78% of time throttled by cgroup",
        "WARN  [k8s] Node gke-prod-pool-1-abc: CPU pressure detected",
        "ERROR [order-service] Request timeout: processing exceeded 30s deadline",
        "WARN  [order-service] Thread pool exhausted: 200/200 threads active",
        "WARN  [jvm] Safepoint spin time > 1000ms: CPU starvation suspected",
        "ERROR [gateway] upstream order-service: 504 Gateway Timeout",
    ],
    FaultType.MEMORY_LEAK: [
        "WARN  [frontend] Heap usage: 78% (was 42% 2 hours ago)",
        "WARN  [node] RSS memory growing: 1.2GB → 1.8GB over 4 hours",
        "WARN  [frontend] Memory growth rate: +150MB/hour — possible leak",
        "ERROR [frontend] ENOMEM: Cannot allocate memory",
        "WARN  [k8s] Pod frontend-5c9d7f-lp8r2 approaching memory limit (80%)",
        "INFO  [frontend] Slow response times correlating with memory growth",
    ],
}

# Alert messages shown at start of episode (what the on-call engineer receives).
FAULT_ALERT_MESSAGES: Dict[FaultType, str] = {
    FaultType.DB_CONNECTION_EXHAUSTION: (
        "[CRITICAL] PagerDuty: order-service error rate > 15% | "
        "Alert: order-service.error_rate | Value: 18.4% | Threshold: 5%"
    ),
    FaultType.OOM_KILLED: (
        "[CRITICAL] PagerDuty: order-service pod restart loop detected | "
        "Alert: order-service.restart_count | Value: 6 restarts/10min"
    ),
    FaultType.NETWORK_PARTITION: (
        "[CRITICAL] PagerDuty: payment-gateway elevated error rate | "
        "Alert: payment-gateway.error_rate | Value: 24.1% | Threshold: 5%"
    ),
    FaultType.RETRY_STORM: (
        "[CRITICAL] PagerDuty: redis-queue depth critical | "
        "Alert: redis-queue.queue_depth | Value: 48293 | Threshold: 10000"
    ),
    FaultType.CONFIG_MISSING_ENV: (
        "[CRITICAL] PagerDuty: auth-api CrashLoopBackOff | "
        "Alert: auth-api.pod_status | Value: CrashLoopBackOff"
    ),
    FaultType.DISK_FULL: (
        "[CRITICAL] PagerDuty: load-balancer disk usage critical | "
        "Alert: load-balancer.disk_pct | Value: 99% | Threshold: 85%"
    ),
    FaultType.CPU_THROTTLE: (
        "[WARNING] PagerDuty: order-service high latency | "
        "Alert: order-service.latency_p99 | Value: 8420ms | Threshold: 2000ms"
    ),
    FaultType.MEMORY_LEAK: (
        "[WARNING] PagerDuty: frontend memory growth anomaly | "
        "Alert: frontend.memory_pct | Value: 78% | Rising for 4h"
    ),
}


# ---------------------------------------------------------------------------
# Cluster Engine
# ---------------------------------------------------------------------------

class SREEngine:
    """
    Simulation engine for a microservice cluster with injected faults.

    Manages:
    - Initialising healthy cluster state
    - Injecting a random (or specified) fault
    - Propagating cascades through dependency graph
    - Generating realistic (noisy) logs and metrics per service
    - Processing fix actions and updating state
    """

    def __init__(self):
        self._cluster: Optional[ClusterState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_episode(self, fault_type: Optional[FaultType] = None, difficulty: Difficulty = Difficulty.HARD) -> ClusterState:
        """Create a new episode with a fresh cluster and injected fault."""
        cluster = self._build_healthy_cluster()
        
        # Select fault based on difficulty if not specified
        if fault_type is None:
            if difficulty == Difficulty.EASY:
                chosen_fault = random.choice(list(EASY_FAULTS))
            elif difficulty == Difficulty.MEDIUM:
                chosen_fault = random.choice(list(MEDIUM_FAULTS))
            else:  # HARD
                chosen_fault = random.choice(list(FaultType))
        else:
            chosen_fault = fault_type
            
        self._inject_fault(cluster, chosen_fault)
        
        # Apply cascading based on difficulty
        if difficulty != Difficulty.EASY:
            self._propagate_cascade(cluster)
            
        cluster.incident_id = f"INC-{int(time.time()) % 100000:05d}"
        cluster.alert_fired = FAULT_ALERT_MESSAGES[chosen_fault]
        self._cluster = cluster
        return cluster

    @property
    def cluster(self) -> Optional[ClusterState]:
        return self._cluster

    # ------------------------------------------------------------------
    # Tool Implementations (called by environment step())
    # ------------------------------------------------------------------

    def grep_logs(self, service: str, pattern: str) -> str:
        """Return log lines for a service, filtered by pattern (case-insensitive)."""
        if service not in SERVICES:
            return f"Error: unknown service '{service}'. Known: {', '.join(SERVICES)}"
        if self._cluster is None:
            return "Error: no active episode"

        all_logs = self._generate_logs_for_service(service)
        matched = [l for l in all_logs if pattern.lower() in l.lower()]

        # Simulate occasional timeouts (5% chance) to add realism
        if random.random() < 0.05:
            return f"[TIMEOUT] grep_logs({service}): connection timed out after 10s. Retry."

        if not matched:
            return f"No log lines matching '{pattern}' found in {service} logs."
        return "\n".join(matched)

    def get_metrics(self, service: str, window: str = "5m") -> str:
        """Return metric snapshot for a service."""
        if service not in SERVICES:
            return f"Error: unknown service '{service}'. Known: {', '.join(SERVICES)}"
        if self._cluster is None:
            return "Error: no active episode"

        svc = self._cluster.get_service(service)
        # Add slight jitter to metric values (real metrics are noisy)
        jitter = lambda v, pct=0.05: round(v * (1 + random.uniform(-pct, pct)), 2)

        return (
            f"Metrics for {service} [{window}]:\n"
            f"  status:       {svc.status}\n"
            f"  error_rate:   {jitter(svc.error_rate)*100:.1f}%\n"
            f"  cpu_pct:      {jitter(svc.cpu_pct):.1f}%\n"
            f"  memory_pct:   {jitter(svc.memory_pct):.1f}%\n"
            f"  latency_p99:  {jitter(svc.latency_p99):.0f}ms\n"
            f"  pod_count:    {svc.pod_count}\n"
            f"  restart_count:{svc.restart_count}\n"
            f"  disk_pct:     {jitter(svc.disk_pct):.1f}%"
        )

    def get_error_rate(self, service: str) -> str:
        """Return current error rate for a service."""
        if service not in SERVICES:
            return f"Error: unknown service '{service}'. Known: {', '.join(SERVICES)}"
        if self._cluster is None:
            return "Error: no active episode"

        svc = self._cluster.get_service(service)
        jitter = lambda v: round(v * (1 + random.uniform(-0.03, 0.03)), 4)
        rate = jitter(svc.error_rate) * 100
        return f"{service} error rate: {rate:.2f}%"

    def describe_pod(self, name: str) -> str:
        """Describe a pod (service name used as pod name prefix)."""
        # Accept either 'database' or 'database-0' style names
        service = name.split("-")[0] if "-" in name else name
        # Special handling for compound names like 'order-service'
        if service not in SERVICES:
            # Try joining first two parts for services like 'order-service'
            parts = name.split("-")
            for i in range(len(parts), 0, -1):
                candidate = "-".join(parts[:i])
                if candidate in SERVICES:
                    service = candidate
                    break

        if service not in SERVICES:
            return f"Error: cannot find pod for '{name}'. Known services: {', '.join(SERVICES)}"
        if self._cluster is None:
            return "Error: no active episode"

        svc = self._cluster.get_service(service)
        pod_suffix = f"{random.randint(10000, 99999):x}"
        pod_name = f"{service}-{svc.deploy_version.replace('.', '')}-{pod_suffix}"
        conditions = "Ready=True" if svc.status == ServiceStatus.HEALTHY else "Ready=False"

        return (
            f"Pod: {pod_name}\n"
            f"  Service:        {service}\n"
            f"  Status:         {svc.status}\n"
            f"  Restarts:       {svc.restart_count}\n"
            f"  CPU usage:      {svc.cpu_pct:.1f}%\n"
            f"  Memory usage:   {svc.memory_pct:.1f}%\n"
            f"  Deploy version: {svc.deploy_version}\n"
            f"  Last deploy:    {svc.last_deploy}\n"
            f"  Conditions:     {conditions}"
        )

    def check_db_connections(self) -> str:
        """Check database connection pool status."""
        if self._cluster is None:
            return "Error: no active episode"

        db = self._cluster.get_service("database")
        if self._cluster.fault_type == FaultType.DB_CONNECTION_EXHAUSTION:
            return (
                "DB Connection Pool Status:\n"
                "  max_connections:    100\n"
                "  active_connections: 100\n"
                "  idle_connections:   0\n"
                "  waiting_clients:    47\n"
                "  pool_status:        EXHAUSTED\n"
                "  avg_wait_time:      28400ms"
            )
        active = int(db.cpu_pct * 0.6)
        idle = max(0, 100 - active - 5)
        return (
            f"DB Connection Pool Status:\n"
            f"  max_connections:    100\n"
            f"  active_connections: {active}\n"
            f"  idle_connections:   {idle}\n"
            f"  waiting_clients:    0\n"
            f"  pool_status:        OK"
        )

    def rollback_deploy(self, service: str) -> Tuple[str, bool]:
        """
        Roll back a service to the previous deploy.
        Returns (message, was_helpful) — helpful only if fault is deploy-related.
        """
        if service not in SERVICES:
            return f"Error: unknown service '{service}'", False
        if self._cluster is None:
            return "Error: no active episode", False

        svc = self._cluster.get_service(service)
        deploy_faults = {FaultType.OOM_KILLED, FaultType.CONFIG_MISSING_ENV}
        primary = self._cluster.fault_service

        was_helpful = (
            self._cluster.fault_type in deploy_faults and service == primary
        )
        if was_helpful:
            svc.status = ServiceStatus.HEALTHY
            svc.error_rate = 0.02
            svc.restart_count = 0
            svc.memory_pct = 35.0
            svc.deploy_version = "v1.4.1"
            self._propagate_cascade(self._cluster)
            return (
                f"Rolled back {service} to v1.4.1. Pods restarting. "
                f"Error rate dropping. Deploy hash: a3f2b1c",
                True,
            )
        else:
            svc.restart_count += 1  # unnecessary action = blast radius
            return (
                f"Rolled back {service} to v1.4.1. "
                f"No improvement observed — issue may not be deploy-related.",
                False,
            )

    def restart_service(self, service: str) -> Tuple[str, bool]:
        """
        Restart a service. Helpful for some faults, harmful blast radius for others.
        """
        if service not in SERVICES:
            return f"Error: unknown service '{service}'", False
        if self._cluster is None:
            return "Error: no active episode", False

        svc = self._cluster.get_service(service)
        primary = self._cluster.fault_service

        restart_faults = {FaultType.MEMORY_LEAK, FaultType.DISK_FULL}
        was_helpful = (
            self._cluster.fault_type in restart_faults and service == primary
        )

        if was_helpful:
            svc.status = ServiceStatus.HEALTHY
            svc.error_rate = 0.01
            svc.memory_pct = 30.0
            svc.restart_count += 1
            self._propagate_cascade(self._cluster)
            return (
                f"Restarted {service}. Service recovered. "
                f"Memory cleared, error rate normalising.",
                True,
            )
        elif service != primary:
            # Restarting a healthy service = unnecessary blast radius penalty
            svc.restart_count += 1
            return (
                f"Restarted {service}. Service came back up (was already healthy). "
                f"No effect on incident.",
                False,
            )
        else:
            # Correct service but wrong fix — temporary relief then degrades again
            svc.restart_count += 1
            return (
                f"Restarted {service}. Briefly recovered but issue re-emerged. "
                f"Root cause not addressed.",
                False,
            )

    def scale_replicas(self, service: str, n: int) -> Tuple[str, bool]:
        """Scale service replicas. Helpful for CPU throttle, harmful otherwise if not the root cause."""
        if service not in SERVICES:
            return f"Error: unknown service '{service}'", False
        if self._cluster is None:
            return "Error: no active episode", False

        svc = self._cluster.get_service(service)
        primary = self._cluster.fault_service
        was_helpful = (
            self._cluster.fault_type == FaultType.CPU_THROTTLE and service == primary
        )

        if was_helpful and n > svc.pod_count:
            svc.pod_count = n
            svc.cpu_pct = max(30, svc.cpu_pct / (n / 3))
            svc.latency_p99 = max(120, svc.latency_p99 * 0.4)
            svc.error_rate = 0.01
            svc.status = ServiceStatus.HEALTHY
            self._propagate_cascade(self._cluster)
            return (
                f"Scaled {service} to {n} replicas. CPU throttle relieved. "
                f"Latency dropping back to normal.",
                True,
            )
        elif n <= svc.pod_count:
            return f"Scale request ignored: {service} already has {svc.pod_count} replicas ≥ {n}.", False
        else:
            svc.pod_count = n
            return (
                f"Scaled {service} to {n} replicas. No significant improvement. "
                f"Scaling may not be related to root cause.",
                False,
            )

    def fix_disk(self, service: str) -> Tuple[str, bool]:
        """Clear disk space on a service. Only useful if disk_full is the fault."""
        if service not in SERVICES:
            return f"Error: unknown service '{service}'", False
        if self._cluster is None:
            return "Error: no active episode", False

        svc = self._cluster.get_service(service)
        was_helpful = (
            self._cluster.fault_type == FaultType.DISK_FULL and service == self._cluster.fault_service
        )
        if was_helpful:
            svc.disk_pct = 35.0
            svc.status = ServiceStatus.HEALTHY
            svc.error_rate = 0.01
            self._propagate_cascade(self._cluster)
            return f"Cleared old log files on {service}. Disk now at 35%. Service recovering.", True
        return f"Cleared logs on {service}. Disk was not full — no impact on incident.", False

    def fix_network(self, service: str) -> Tuple[str, bool]:
        """Fix network partition for a service."""
        if service not in SERVICES:
            return f"Error: unknown service '{service}'", False
        if self._cluster is None:
            return "Error: no active episode", False

        svc = self._cluster.get_service(service)
        was_helpful = (
            self._cluster.fault_type == FaultType.NETWORK_PARTITION
            and service == self._cluster.fault_service
        )
        if was_helpful:
            svc.status = ServiceStatus.HEALTHY
            svc.error_rate = 0.02
            svc.latency_p99 = 80.0
            self._propagate_cascade(self._cluster)
            return f"Network partition resolved for {service}. Connectivity restored.", True
        return f"Checked network for {service}. No partition found.", False

    def resolve_incident(self, root_cause: str, fix_applied: str) -> Dict:
        """
        End the episode. Agent declares root cause and fix.
        Returns structured result for rubric scoring.
        """
        if self._cluster is None:
            return {"error": "no active episode"}

        self._cluster.resolved = True
        self._cluster.resolution = f"root_cause={root_cause}|fix={fix_applied}"
        return {
            "declared_root_cause": root_cause,
            "declared_fix": fix_applied,
            "actual_fault_type": self._cluster.fault_type.value,
            "actual_fault_service": self._cluster.fault_service,
            "step_count": self._cluster.step_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_healthy_cluster(self) -> ClusterState:
        """Build a healthy cluster with realistic baseline metrics."""
        services = {}
        for name in SERVICES:
            services[name] = ServiceState(
                name=name,
                status=ServiceStatus.HEALTHY,
                error_rate=random.uniform(0.001, 0.008),  # baseline 0.1–0.8%
                cpu_pct=random.uniform(15, 40),
                memory_pct=random.uniform(25, 55),
                latency_p99=random.uniform(40, 150),
                pod_count=random.choice([2, 3, 3, 4]),
                restart_count=0,
                disk_pct=random.uniform(20, 50),
            )
        return ClusterState(services=services)

    def _inject_fault(self, cluster: ClusterState, fault: FaultType):
        """Apply the primary fault to its target service."""
        cluster.fault_type = fault
        primary = FAULT_PRIMARY_SERVICE[fault]
        cluster.fault_service = primary
        svc = cluster.get_service(primary)

        if fault == FaultType.DB_CONNECTION_EXHAUSTION:
            svc.status = ServiceStatus.CRITICAL
            svc.error_rate = 0.72
            svc.cpu_pct = 90.0
            svc.latency_p99 = 30000.0

        elif fault == FaultType.OOM_KILLED:
            svc.status = ServiceStatus.CRITICAL
            svc.error_rate = 0.55
            svc.memory_pct = 98.0
            svc.restart_count = 6
            svc.latency_p99 = 15000.0

        elif fault == FaultType.NETWORK_PARTITION:
            svc.status = ServiceStatus.DOWN
            svc.error_rate = 0.95
            svc.latency_p99 = 30000.0

        elif fault == FaultType.RETRY_STORM:
            svc.status = ServiceStatus.CRITICAL
            svc.error_rate = 0.48
            svc.cpu_pct = 88.0
            svc.memory_pct = 91.0

        elif fault == FaultType.CONFIG_MISSING_ENV:
            svc.status = ServiceStatus.DOWN
            svc.error_rate = 1.0
            svc.restart_count = 8
            svc.pod_count = 0

        elif fault == FaultType.DISK_FULL:
            svc.status = ServiceStatus.CRITICAL
            svc.error_rate = 0.65
            svc.disk_pct = 99.5

        elif fault == FaultType.CPU_THROTTLE:
            svc.status = ServiceStatus.DEGRADED
            svc.error_rate = 0.18
            svc.cpu_pct = 95.0
            svc.latency_p99 = 8500.0

        elif fault == FaultType.MEMORY_LEAK:
            svc.status = ServiceStatus.DEGRADED
            svc.error_rate = 0.12
            svc.memory_pct = 82.0
            svc.latency_p99 = 3400.0

    def _propagate_cascade(self, cluster: ClusterState):
        """Propagate fault effects through the dependency graph."""
        for service_name, depends_on in DEPENDENCY_GRAPH.items():
            svc = cluster.get_service(service_name)
            if svc.name == cluster.fault_service:
                continue  # primary already set

            max_dep_error = 0.0
            for dep_name in depends_on:
                dep = cluster.get_service(dep_name)
                if dep.error_rate > max_dep_error:
                    max_dep_error = dep.error_rate

            if max_dep_error > 0.05:
                cascade_error = max_dep_error * CASCADE_FACTOR
                if cascade_error > svc.error_rate:
                    svc.error_rate = min(0.95, cascade_error)
                    if svc.error_rate > 0.4:
                        svc.status = ServiceStatus.CRITICAL
                    elif svc.error_rate > 0.15:
                        svc.status = ServiceStatus.DEGRADED
                    svc.latency_p99 = min(30000, svc.latency_p99 * (1 + cascade_error))

    def _generate_logs_for_service(self, service: str) -> List[str]:
        """Generate a realistic mix of fault-specific and red-herring logs."""
        if self._cluster is None:
            return []

        fault = self._cluster.fault_type
        primary = self._cluster.fault_service

        # Pull fault-specific logs only for the primary service
        fault_lines = []
        if service == primary and fault in FAULT_LOG_SIGNATURES:
            fault_lines = FAULT_LOG_SIGNATURES[fault].copy()

        # Add some cascaded errors for dependant services
        cascade_lines = []
        svc = self._cluster.get_service(service)
        if svc.error_rate > 0.15 and service != primary:
            cascade_lines = [
                f"ERROR [{service}] Upstream {primary} returning 503: connection timeout",
                f"WARN  [{service}] Circuit breaker threshold reached: {primary} unhealthy",
                f"ERROR [{service}] Request failed: dependency {primary} unavailable",
            ]

        # Red herrings always present
        herring_count = random.randint(3, 6)
        herring = random.sample(RED_HERRING_LOGS, min(herring_count, len(RED_HERRING_LOGS)))

        # Shuffle everything together so the agent can't just read line-by-line
        all_lines = fault_lines + cascade_lines + herring
        random.shuffle(all_lines)

        # Prepend realistic timestamps
        base_ts = int(time.time()) - 300
        result = []
        for i, line in enumerate(all_lines):
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(base_ts + i * 7))
            result.append(f"{ts} {line}")
        return result
