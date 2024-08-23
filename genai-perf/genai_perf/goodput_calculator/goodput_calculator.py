#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from genai_perf.metrics import Metrics


class GoodputCalculator(ABC):
    """A base class to calculate goodput according to goodput constraints."""

    MS_TO_NS_CONVERSION = 1e6
    INVALID_GOODPUT = [-1.0]

    def __init__(
        self,
        goodput_constraints: Dict[str, float],
        metric: Metrics,
        benchmark_duration: float,
    ) -> None:
        self._goodput_constraints = goodput_constraints
        self._benchmark_duration = benchmark_duration
        self._metric = metric
        self._goodput_val: Optional[List[float]] = None
        self._slo_names = {
            "request_latency": "request_latencies",
        }

    def compute(self) -> None:
        """
        Compute the goodput result.

        The compute method sets the valid goodput constraints from user's
        inputs, aggregates request metric values, counts the number of good requests,
        and calculates the final goodput.
        """
        self._set_valid_slos()
        self._combine_requests_metric_values()
        good_count = self._count_good_reqs()
        self._compute_goodput(good_count)

    @abstractmethod
    def _set_valid_slos(self) -> None:
        """Set the valid goodput constraints while logging any invalid ones."""
        pass

    @abstractmethod
    def _combine_requests_metric_values(self) -> None:
        """
        Combine values from the metrics that match with the valid
        goodput constraints at a per request level.
        """
        pass

    @abstractmethod
    def _count_good_reqs(self) -> Optional[int]:
        """Count the number of good requests according to goodput constraints."""
        pass

    @abstractmethod
    def _compute_goodput(self, good_count) -> None:
        """Compute the goodput."""
        pass

    @property
    def goodput(self) -> Optional[List[float]]:
        return self._goodput_val

    def get_slo_name(self, metric_name: str) -> str:
        """Returns the plural name of a given metric."""
        if metric_name in self._slo_names:
            return self._slo_names[metric_name]
        else:
            raise KeyError(f"No metric named '{metric_name}' exists.")
