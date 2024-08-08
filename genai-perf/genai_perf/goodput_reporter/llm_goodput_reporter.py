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

from typing import Dict
from genai_perf.goodput_reporter.goodput_reporter import GoodputReporter
from genai_perf.metrics.llm_metrics import LLMMetrics


class LLMGoodputReporter(GoodputReporter):
    """A subclass to report goodput for language models."""

    def __init__(self,
                goodput_constraints: Dict[str, float],
                metric: LLMMetrics,
                benchmark_duration: float,
    ) -> None:
        super().__init__(goodput_constraints, metric, benchmark_duration)
        

    def set_valid_slos(self) -> None:
        """Check user's Service Level Objectives (SLOs) inputs. 
        Set the valid ones while logging the invalid ones. 
        """
        invalid_slos = []
        self._valid_slos = {}
        valid_names = [metric.name for metric in self._metric.request_metrics]

        for slo_name, slo_value in self._goodput_constraints.items(): 
            if self._metric.get_base_name(slo_name) not in valid_names:
                invalid_slos.append(slo_name)
            else:
                self._valid_slos[slo_name] = slo_value * self.MS_TO_NS_CONVERSION
        if invalid_slos:
            print("These are valid request metrics", self._metric.request_metrics)
            print(self._metric.request_metrics[0].name)
            raise ValueError(f"Invalid SLOs found: {', '.join(invalid_slos)}, "
                            "Make sure these are supported request metrics.")

    def combine_requests_metric_values(self) -> None:
        """Combine metric values at per request level.
        Only the metrics from valid SLOs.
        """
        metric_data = self._metric.data 
        requests_metric_values = [metric_data[key] for key in self._valid_slos]
        self._combined_requests_metric_values = list(zip(*requests_metric_values))

    def count_good_reqs(self) -> None:
        """Count the number of good requests according to SLOs."""
        target_metric_values = list(self._valid_slos.values())
        requests_metric_values = self._combined_requests_metric_values
        good_req_count = 0

        for request_metric_values in requests_metric_values:
            if all(val < slo 
                   for val, slo in zip(request_metric_values, target_metric_values)
            ):
                good_req_count += 1
        self._good_req_count = good_req_count
    
    def compute_goodput(self) -> None:
        """Compute the goodput."""
        self._goodput = [self._good_req_count / self._benchmark_duration]
        