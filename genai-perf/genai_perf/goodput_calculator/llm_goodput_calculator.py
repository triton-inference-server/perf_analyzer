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

from typing import Dict, Optional
import genai_perf.logging as logging
from genai_perf.goodput_calculator.goodput_calculator import GoodputCalculator
from genai_perf.metrics.llm_metrics import LLMMetrics

logger = logging.getLogger(__name__)

class LLMGoodputCalculator(GoodputCalculator):
    """
    A subclass to calculate goodput for LLMs according to LLM-related SLOs.
    """

    def __init__(self,
                goodput_constraints: Dict[str, float],
                metric: LLMMetrics,
                benchmark_duration: float,
    ) -> None:
        super().__init__(goodput_constraints, metric, benchmark_duration)
        # (TMA-1975 related) The order is hardcoded as below due to the hardcoded order
        # in LLMMetirc class. We would eventually want to impose some consistent order 
        # for time-related metrics and throughput related metrics.
        self._valid_time_related_names = [
            item.name for item in metric.request_time_metrics
        ]
        self._valid_throughput_related_names = [
            item.name for item in metric.request_throughput_metrics
        ]
        self._valid_metric_names = (
            self._valid_time_related_names + self._valid_throughput_related_names
        )
        self._has_time_target = False
        self._has_throughput_target = False

    def _set_valid_slos(self) -> None:
        """
        Check users' Service Level Objectives (SLOs) inputs. 
        Set the valid ones while logging the invalid ones. 
        """
        invalid_slos = []
        self._valid_time_related_slos = {}
        self._valid_throughput_related_slos = {}
        for slo_name, slo_value in self._goodput_constraints.items():
            try:
                base_name = self._metric.get_base_name(slo_name)
                if base_name in self._valid_metric_names:
                    if base_name in self._valid_time_related_names:
                        self._valid_time_related_slos[slo_name] = (
                            slo_value * self.MS_TO_NS_CONVERSION
                        )
                    elif base_name in self._valid_throughput_related_names:
                        self._valid_throughput_related_slos[slo_name] = (
                            slo_value 
                        )
            except KeyError:            
                invalid_slos.append(slo_name)
        if self._valid_time_related_slos:
            self._has_time_target = True
        if self._valid_throughput_related_slos:
            self._has_throughput_target = True 
        if invalid_slos:
            valid_slos_list = ', '.join(self._valid_metric_names)
            logger.info(f"Invalid SLOs found: {', '.join(invalid_slos)}. "
                        f"The goodput will be N/A. "
                        f"Valid SLOs are: {valid_slos_list} in plural forms.")
            self._goodput = None

    def _combine_requests_metric_values(self) -> None:
        """
        Combine values from the metrics that match with the valid SLOs at a
        per request level.  
        """
        if self.goodput is None:
            return
        
        if self._has_time_target:
            requests_time_metric_values = [
                self._metric.data[key] for key in self._valid_time_related_slos
            ]
            self._combined_requests_time_metric_values = list(
                zip(*requests_time_metric_values)
            )

        if self._has_throughput_target:
            requests_throughput_metric_values = [
                self._metric.data[key] for key in self._valid_throughput_related_slos
            ] 
            self._combined_requests_throughput_metric_values = list(
                zip(*requests_throughput_metric_values)
            )

    def _count_good_reqs(self) -> Optional[int]:
        """Count the number of good requests according to SLOs."""
        if self.goodput is None:
            return self.goodput        
        target_time_metric_values = []
        target_throughput_metric_values = []
        if self._has_time_target:
            num_of_requests = len(self._combined_requests_time_metric_values)
            target_time_metric_values = list(self._valid_time_related_slos.values())
        if self._has_throughput_target:
            num_of_requests = len(self._combined_requests_throughput_metric_values)
            target_throughput_metric_values = list(
                self._valid_throughput_related_slos.values()
            )                        

        good_req_count = 0
        for idx in range(num_of_requests):
            is_good_request = True
            request_time_metric_values = []
            request_throughput_metric_values = []
            if self._has_time_target:
                request_time_metric_values = (
                    self._combined_requests_time_metric_values[idx]
                )
            if self._has_throughput_target:
                request_throughput_metric_values = (
                    self._combined_requests_throughput_metric_values[idx]
                )
            for val, slo in zip(request_time_metric_values, target_time_metric_values):
                if val > slo:
                    is_good_request = False
                    break
            else:
                for val, slo in zip(
                    request_throughput_metric_values, target_throughput_metric_values
                ):
                    if val < slo:
                        is_good_request = False
                        break
            
            if is_good_request:
                good_req_count += 1

        return good_req_count
    
    def _compute_goodput(self, good_count) -> None:
        """Compute the goodput."""
        if self.goodput is None:
            return
        else:
            self._goodput = [good_count / self._benchmark_duration]
