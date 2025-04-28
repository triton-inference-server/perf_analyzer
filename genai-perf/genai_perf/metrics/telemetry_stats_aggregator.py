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
    and creates an aggregated TelemetryStatistics object.
    """

    def __init__(self, telemetry_dicts: List[Dict[str, Any]]) -> None:
        self._telemetry_stats = TelemetryStatistics(TelemetryMetrics(None))
        self._telemetry_dicts = telemetry_dicts
        self._aggregate()

    def _aggregate(self) -> None:
        """
        Aggregates telemetry stats from multiplt files to create aggregate Telmetry statistcs.
        """
        if not self._telemetry_dicts:
            return

        aggregated_telemetry_stats_dict: DefaultDict[str, Any] = defaultdict(dict)
        for metric_name in self._telemetry_dicts[0]:
            aggregated_telemetry_stats_dict[metric_name] = {}
            unit = self._telemetry_dicts[0][metric_name].get("unit", "")
            aggregated_telemetry_stats_dict[metric_name]["unit"] = unit

            gpu_ids = set()
            for telemetry_dict in self._telemetry_dicts:
                gpu_ids.update(telemetry_dict.get(metric_name, {}).keys())
            gpu_ids.discard("unit")

            for gpu_id in gpu_ids:
                values = [
                    telemetry_dict[metric_name][gpu_id]["avg"]
                    for telemetry_dict in self._telemetry_dicts
                    if metric_name in telemetry_dict
                    and gpu_id in telemetry_dict[metric_name]
                    and "avg" in telemetry_dict[metric_name][gpu_id]
                ]
                if values:
                    if metric_name in [
                        "gpu_power_usage",
                        "gpu_utilization",
                        "energy_consumption",
                    ]:
                        aggregated_telemetry_stats_dict[metric_name][gpu_id] = {
                            "avg": mean(values)
                        }
                    elif metric_name == "gpu_memory_usage":
                        aggregated_telemetry_stats_dict[metric_name][gpu_id] = {
                            "avg": sum(values)
                        }
                    else:
                        aggregated_telemetry_stats_dict[metric_name][gpu_id] = {
                            "avg": max(values)
                        }

        self._telemetry_stats.set_stats_dict(aggregated_telemetry_stats_dict)

    def get_telemetry_stats(self) -> TelemetryStatistics:
        """
        Returns the aggregated telemetry statistics.
        """
        return self._telemetry_stats
