# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from statistics import mean
from typing import Any, DefaultDict, Dict, List

from genai_perf.metrics import TelemetryMetrics, TelemetryStatistics


class TelemetryStatsAggregator:
    """
    A class to aggregate telemetry statistics from multiple profile files
    and create an aggregated TelemetryStatistics object.
    """

    def __init__(self, telemetry_dicts: List[Dict[str, Any]]) -> None:
        self._telemetry_stats = TelemetryStatistics(TelemetryMetrics())
        self._telemetry_dicts = telemetry_dicts
        self._aggregate()

    def _get_gpu_ids(self, metric_name: str) -> set:
        """
        Acquires the list of unique GPU IDs for a given metric from the telemetry dictionaries.

        Args:
            metric_name (str): The name of the metric (e.g., "gpu_power_usage").

        Returns:
            set: A set of unique GPU IDs.
        """
        gpu_ids = set()
        for telemetry_dict in self._telemetry_dicts:
            gpu_ids.update(telemetry_dict.get(metric_name, {}).keys())
        gpu_ids.discard("unit")
        return gpu_ids

    def _get_values_for_gpu(self, metric_name: str, gpu_id: str) -> List[float]:
        """
        Acquires the values for a specific GPU and metric from all telemetry dictionaries.

        Args:
            metric_name (str): The name of the metric (e.g., "gpu_power_usage").
            gpu_id (str): The GPU ID (e.g., "gpu0").

        Returns:
            List[float]: A list of values corresponding to the metric and GPU.
        """
        return [
            telemetry_dict[metric_name][gpu_id]["avg"]
            for telemetry_dict in self._telemetry_dicts
            if metric_name in telemetry_dict
            and gpu_id in telemetry_dict[metric_name]
            and "avg" in telemetry_dict[metric_name][gpu_id]
        ]

    def _get_aggregated_value(self, values: List[float], metric_name: str) -> float:
        """
        Aggregates the values based on the metric type.

        Args:
            values (List[float]): The list of values to be aggregated.
            metric_name (str): The metric name to determine the aggregation method.

        Returns:
            float: The aggregated value for the metric.
        """
        if metric_name in [
            "gpu_power_usage",
            "gpu_utilization",
            "energy_consumption",
        ]:
            return mean(values)
        elif metric_name == "gpu_memory_usage":
            return sum(values)
        else:
            return max(values)

    def _aggregate(self) -> None:
        """
        Aggregates telemetry stats from multiple files to create aggregate telemetry statistics.
        """
        if not self._telemetry_dicts:
            return

        aggregated_telemetry_stats_dict: DefaultDict[str, Any] = defaultdict(dict)
        for metric_name in self._telemetry_dicts[0]:
            aggregated_telemetry_stats_dict[metric_name] = {}
            unit = self._telemetry_dicts[0][metric_name].get("unit", "")
            aggregated_telemetry_stats_dict[metric_name]["unit"] = unit

            gpu_ids = self._get_gpu_ids(metric_name)

            for gpu_id in gpu_ids:
                values = self._get_values_for_gpu(metric_name, gpu_id)

                if values:
                    aggregated_value = self._get_aggregated_value(values, metric_name)
                    aggregated_telemetry_stats_dict[metric_name][gpu_id] = {
                        "avg": aggregated_value
                    }

        self._telemetry_stats.set_stats_dict(aggregated_telemetry_stats_dict)

    def get_telemetry_stats(self) -> TelemetryStatistics:
        """
        Returns the aggregated telemetry statistics.
        """
        return self._telemetry_stats
