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

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from genai_perf.metrics.statistics import Statistics
from genai_perf.metrics.telemetry_metrics import TelemetryMetrics


class TelemetryStatistics:
    """A class that aggregates various statistics from telemetry metrics class."""

    def __init__(self, metrics: TelemetryMetrics):
        self._metrics = metrics
        self._stats_dict: DefaultDict[str, Any] = defaultdict(lambda: defaultdict(dict))
        self._statistics = Statistics(metrics)

        self._add_units()
        for attr, data in self._metrics.data.items():
            if self._should_skip(data):
                continue
            for gpu_index, gpu_data in data.items():
                self._stats_dict[attr][gpu_index]["avg"] = (
                    self._statistics._calculate_mean(gpu_data)
                )
            if not self._is_constant_metric(attr):
                percentile_results = self._statistics._calculate_percentiles(gpu_data)
                for percentile_label, percentile_value in percentile_results.items():
                    self._stats_dict[attr][gpu_index][
                        percentile_label
                    ] = percentile_value
                min, max = self._statistics._calculate_minmax(gpu_data)
                self._stats_dict[attr][gpu_index]["min"] = min
                self._stats_dict[attr][gpu_index]["max"] = max
                self._stats_dict[attr][gpu_index]["std"] = (
                    self._statistics._calculate_std(gpu_data)
                )

    def scale_data(self) -> None:
        SCALING_FACTORS = {
            "energy_consumption": 1e-6,  # joules to megajoules (MJ)
            "gpu_memory_used": 1e-9,  # bytes to gigabytes (GB)
            "total_gpu_memory": 1e-9,  # bytes to gigabytes (GB)
            "gpu_utilization": 100,  # ratio to percentage (%)
        }
        for metric, data in self._stats_dict.items():
            if metric in SCALING_FACTORS:
                factor = SCALING_FACTORS[metric]
                for key, gpu_data in data.items():
                    if key != "unit":
                        for stat, value in gpu_data.items():
                            self._stats_dict[metric][key][stat] = value * factor

    def _should_skip(self, data: Dict[str, List[float]]) -> bool:
        if len(data) == 0:
            return True
        return False

    def _add_units(self) -> None:
        for metric in self._metrics.telemetry_metrics:
            self._stats_dict[metric.name]["unit"] = metric.unit

    def _is_constant_metric(self, attr: str) -> bool:
        return attr in ["gpu_power_limit", "total_gpu_memory"]

    @property
    def stats_dict(self) -> Dict:
        return self._stats_dict
