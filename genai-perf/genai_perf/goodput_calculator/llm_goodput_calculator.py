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

from typing import Dict, List, Optional, Union

import genai_perf.logging as logging
from genai_perf.goodput_calculator.goodput_calculator import GoodputCalculator
from genai_perf.metrics.metrics import Metrics

logger = logging.getLogger(__name__)


class LLMGoodputCalculator(GoodputCalculator):
    """
    A subclass to calculate goodput for LLMs according to
    LLM-related goodput constraints.
    """

    def __init__(
        self,
        goodput_constraints: Dict[str, float],
        metric: Metrics,
        benchmark_duration: float,
    ) -> None:
        super().__init__(goodput_constraints, metric, benchmark_duration)

        self._set_valid_metric_names()

        self._has_time_target = False
        self._has_throughput_target = False

        self._add_slo_mapping()

    def _set_valid_metric_names(self) -> None:
        self._valid_time_related_names = [
            item.name for item in self._metric.request_time_metrics
        ]
        self._valid_throughput_related_names = [
            item.name for item in self._metric.request_throughput_metrics
        ]
        self._valid_metric_names = (
            self._valid_time_related_names + self._valid_throughput_related_names
        )

    def _add_slo_mapping(self) -> None:
        self._slo_names["time_to_first_token"] = "time_to_first_tokens"
        self._slo_names["inter_token_latency"] = "inter_token_latencies"
        self._slo_names["output_token_throughput_per_request"] = (
            "output_token_throughputs_per_request"
        )
        self._slo_names["image_throughput"] = "image_throughputs"
        self._slo_names["image_latency"] = "image_latencies"

    def _set_valid_slos(self) -> None:
        invalid_slos = []
        self._valid_time_related_slos = {}
        self._valid_throughput_related_slos = {}
        for slo_name, slo_value in self._goodput_constraints.items():
            if slo_name in self._valid_time_related_names:
                self._valid_time_related_slos[slo_name] = (
                    slo_value * self.MS_TO_NS_CONVERSION
                )
                self._has_time_target = True
            elif slo_name in self._valid_throughput_related_names:
                self._valid_throughput_related_slos[slo_name] = slo_value
                self._has_throughput_target = True
            else:
                invalid_slos.append(slo_name)

        if invalid_slos:
            valid_slos_list = ", ".join(self._valid_metric_names)
            logger.info(
                f"Invalid Service Level Objectives found: {', '.join(invalid_slos)}. "
                f"Valid Service Level Objectives are: {valid_slos_list}."
            )
            self._goodput_val = self.INVALID_GOODPUT

    def _combine_requests_metric_values(self) -> None:
        if self.goodput == self.INVALID_GOODPUT:
            return

        if self._has_time_target:
            time_names = [
                self.get_slo_name(key) for key in self._valid_time_related_slos
            ]
            requests_time_metric_values = [
                self._metric.data[name] for name in time_names
            ]

            self._combined_requests_time_metric_values = list(
                zip(*requests_time_metric_values)
            )

        if self._has_throughput_target:
            throughput_names = [
                self.get_slo_name(key) for key in self._valid_throughput_related_slos
            ]
            requests_throughput_metric_values = [
                self._metric.data[name] for name in throughput_names
            ]

            self._combined_requests_throughput_metric_values = list(
                zip(*requests_throughput_metric_values)
            )

    def _count_good_reqs(self) -> Optional[int]:
        if self.goodput == self.INVALID_GOODPUT:
            return None
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
            request_time_metric_values: List[float] = []
            request_throughput_metric_values: List[float] = []
            if self._has_time_target:
                request_time_metric_values = list(
                    self._combined_requests_time_metric_values[idx]
                )
            if self._has_throughput_target:
                request_throughput_metric_values = list(
                    self._combined_requests_throughput_metric_values[idx]
                )

            for val, slo in zip(request_time_metric_values, target_time_metric_values):
                if val > slo:
                    is_good_request = False
                    break
            if is_good_request:
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
        if self.goodput == self.INVALID_GOODPUT:
            return
        else:
            self._goodput_val = [good_count / self._benchmark_duration]
