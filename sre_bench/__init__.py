# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SRE-Bench: Production Incident Training Environment.

Public API — import from here in training scripts and client code.

Example:
    from sre_bench import SreBenchEnv, SreBenchAction, SreBenchObservation

    with SreBenchEnv(base_url="http://localhost:8000").sync() as client:
        result = client.reset()
        print(result.observation.alert)
"""

from .client import SreBenchEnv
from .models import SreBenchAction, SreBenchObservation

__all__ = ["SreBenchEnv", "SreBenchAction", "SreBenchObservation"]
