"""
Compatibility module for flat-repo usage.

Allows imports like:
    from sre_bench import SreBenchEnv, SreBenchAction, SreBenchObservation
without requiring an installed package directory layout.
"""

from client import SreBenchEnv
from models import SreBenchAction, SreBenchObservation

__all__ = ["SreBenchEnv", "SreBenchAction", "SreBenchObservation"]
