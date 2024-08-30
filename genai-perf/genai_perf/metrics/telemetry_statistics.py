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

from typing import DefaultDict, Dict, List, Any
from collections import defaultdict

from genai_perf.metrics.telemetry_metrics import TelemetryMetrics
from genai_perf.metrics.statistics_util import StatisticsUtil

class TelemetryStatistics:
    """A class that aggregates various statistics from telemetry metrics class.
    """
    
    def __init__(self, metrics: TelemetryMetrics):
        self._metrics = metrics
        self._stats_dict: DefaultDict[str, Any] = defaultdict(
            lambda: defaultdict(lambda: defaultdict())
        )
        
        self._add_units()
        for attr, data in self._metrics.data.items():
            if self._should_skip(data):
                continue
            self._calculate_statistics(data, attr)

    def _should_skip(self, data: Dict[str, List[float]]) -> bool:
        """Checks if some metrics should be skipped."""
        if len(data) == 0:
            return True
        return False
    
    def _calculate_statistics(self, data: Dict[str, List[float]], attr: str) -> None:
        for gpu_index, gpu_data in data.items():
            self._calculate_mean(gpu_data, attr, gpu_index)
            if not self._is_constant_metric(attr):
                self._calculate_percentiles(gpu_data, attr, gpu_index)
                self._calculate_minmax(gpu_data, attr, gpu_index)
                self._calculate_std(gpu_data, attr, gpu_index)
    
    def _calculate_mean(self, data: List[float], attr: str, gpu_index: str) -> None:
        avg = StatisticsUtil.calculate_mean(data)
        setattr(self, f"avg_{attr}_{gpu_index}", avg)
        self._stats_dict[attr][gpu_index]["avg"] = avg
        
    def _calculate_percentiles(self, data: List[float], attr: str, gpu_index: str) -> None:
        percentile_results = StatisticsUtil.calculate_percentiles(data)
        p25 = percentile_results[0]
        p50 = percentile_results[1]
        p75 = percentile_results[2]
        p90 = percentile_results[3]
        p95 = percentile_results[4]
        p99 = percentile_results[5]
        setattr(self, f"p25_{attr}_{gpu_index}", p25)
        setattr(self, f"p50_{attr}_{gpu_index}", p50)
        setattr(self, f"p75_{attr}_{gpu_index}", p75)
        setattr(self, f"p90_{attr}_{gpu_index}", p90)
        setattr(self, f"p95_{attr}_{gpu_index}", p95)
        setattr(self, f"p99_{attr}_{gpu_index}", p99)
        self._stats_dict[attr][gpu_index]["p99"] = p99
        self._stats_dict[attr][gpu_index]["p95"] = p95
        self._stats_dict[attr][gpu_index]["p90"] = p90
        self._stats_dict[attr][gpu_index]["p75"] = p75
        self._stats_dict[attr][gpu_index]["p50"] = p50
        self._stats_dict[attr][gpu_index]["p25"] = p25
        
    def _calculate_minmax(self, data: List[float], attr: str, gpu_index: str) -> None:
        min, max = StatisticsUtil.calculate_minmax(data)
        setattr(self, f"min_{attr}_{gpu_index}", min)
        setattr(self, f"max_{attr}_{gpu_index}", max)
        self._stats_dict[attr][gpu_index]["max"] = max
        self._stats_dict[attr][gpu_index]["min"] = min
        
    def _calculate_std(self, data: List[float], attr: str, gpu_index: str) -> None:
        std = StatisticsUtil.calculate_std(data)
        setattr(self, f"std_{attr}_{gpu_index}", std)
        self._stats_dict[attr][gpu_index]["std"] = std
        
    def _add_units(self) -> None:
        for metric in self._metrics.telemetry_metrics:
            self._stats_dict[metric.name]['unit'] = metric.unit
             
    def _is_constant_metric(self, attr: str) -> bool:
        return attr in ["gpu_power_limit", "total_gpu_memory"]
    
    @property
    def stats_dict(self) -> Dict:
        return self._stats_dict
    
        
        