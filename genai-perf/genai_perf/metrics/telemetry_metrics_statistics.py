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
from typing import DefaultDict, Dict, List

import numpy as np
from genai_perf.metrics import MetricMetadata, TelemetryMetrics


class TelemetryMetricsStatistics:
    def __init__(self, telemetry_metrics: TelemetryMetrics):
        self._metrics = telemetry_metrics
        self._stats_dict: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        self._aggregate_metrics()

    def _aggregate_metrics(self):
        for attr, data in self._metrics.data.items():
            if self._should_skip(data, attr):
                if len(data) > 0:
                    unit = self._get_metric_unit(attr)
                    # Take the first measurement value for gpu_power_limit and total_gpu_memory since they are constant
                    self._stats_dict[attr] = {
                        "unit": unit,
                        **{f"gpu{i}": value for i, value in enumerate(data[0])},
                    }
                continue
            unit = self._get_metric_unit(attr)
            # Aggregating non-constant metrics
            self._stats_dict[attr]["unit"] = unit
            self._calculate_statistics(data, attr)

    def _should_skip(self, data, attr):
        # Skip constant metrics or empty data
        return len(data) == 0 or attr in ["gpu_power_limit", "total_gpu_memory"]

    def _calculate_statistics(self, data, attr):
        self._calculate_mean(data, attr)
        self._calculate_percentiles(data, attr)
        self._calculate_minmax(data, attr)

    def _get_metric_unit(self, attr: str) -> str:
        for metric in self._metrics.telemetry_metrics:
            if metric.name == attr:
                return metric.unit
        return ""

    def _calculate_mean(self, data, attr):
        mean_values = [round(np.mean(gpu_data), 2) for gpu_data in zip(*data)]
        self._stats_dict[attr]["avg"] = {
            f"gpu{i}": value for i, value in enumerate(mean_values)
        }

    def _calculate_percentiles(self, data, attr):
        percentiles = [
            self._calculate_percentiles_for_gpu(gpu_data) for gpu_data in zip(*data)
        ]
        self._stats_dict[attr]["p25"] = {
            f"gpu{i}": gpu_percentiles[0]
            for i, gpu_percentiles in enumerate(percentiles)
        }
        self._stats_dict[attr]["p50"] = {
            f"gpu{i}": gpu_percentiles[1]
            for i, gpu_percentiles in enumerate(percentiles)
        }
        self._stats_dict[attr]["p75"] = {
            f"gpu{i}": gpu_percentiles[2]
            for i, gpu_percentiles in enumerate(percentiles)
        }
        self._stats_dict[attr]["p90"] = {
            f"gpu{i}": gpu_percentiles[3]
            for i, gpu_percentiles in enumerate(percentiles)
        }
        self._stats_dict[attr]["p95"] = {
            f"gpu{i}": gpu_percentiles[4]
            for i, gpu_percentiles in enumerate(percentiles)
        }
        self._stats_dict[attr]["p99"] = {
            f"gpu{i}": gpu_percentiles[5]
            for i, gpu_percentiles in enumerate(percentiles)
        }

    def _calculate_percentiles_for_gpu(self, gpu_data):
        p25, p50, p75 = np.percentile(gpu_data, [25, 50, 75])
        p90, p95, p99 = np.percentile(gpu_data, [90, 95, 99])
        return [
            round(p25, 2),
            round(p50, 2),
            round(p75, 2),
            round(p90, 2),
            round(p95, 2),
            round(p99, 2),
        ]

    def _calculate_minmax(self, data, attr):
        min_values = [round(np.min(gpu_data), 2) for gpu_data in zip(*data)]
        max_values = [round(np.max(gpu_data), 2) for gpu_data in zip(*data)]
        self._stats_dict[attr]["min"] = {
            f"gpu{i}": value for i, value in enumerate(min_values)
        }
        self._stats_dict[attr]["max"] = {
            f"gpu{i}": value for i, value in enumerate(max_values)
        }

    @property
    def metrics(self) -> TelemetryMetrics:
        """Return the underlying metrics used to calculate the statistics."""
        return self._metrics

    @property
    def stats_dict(self) -> Dict:
        return dict(self._stats_dict)  # Convert defaultdict to regular dict
