#!/usr/bin/env python3

# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from genai_perf.exceptions import GenAIPerfException
from genai_perf.metrics.statistics import Statistics
from genai_perf.metrics.telemetry_metrics import TelemetryMetrics
from genai_perf.record.record import RecordType
from genai_perf.types import GpuRecords


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

            gpu_index = None
            gpu_data = None

            for gpu_index, gpu_data in data.items():
                self._stats_dict[attr][gpu_index]["avg"] = (
                    self._statistics._calculate_mean(gpu_data)
                )
                if not self._is_constant_metric(attr):
                    if gpu_data is None or gpu_index is None:
                        continue

                    percentile_results = self._statistics._calculate_percentiles(
                        gpu_data
                    )
                    for (
                        percentile_label,
                        percentile_value,
                    ) in percentile_results.items():
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
            "energy_consumption": 1e-9,  # mJ to megajoules (MJ)
            "gpu_memory_used": 1.048576 * 1e-3,  # MiB to gigabytes (GB)
            "total_gpu_memory": 1.048576 * 1e-3,  # MiB to gigabytes (GB)
            "gpu_memory_free": 1.048576e-3,  # MiB to gigabytes (GB)
            "sm_utilization": 100,  # ratio to %
            "pcie_transmit_throughput": 1 / 1024,  # bytes/sec to KB/sec
            "pcie_receive_throughput": 1 / 1024,  # bytes/sec to KB/sec
        }
        for metric, data in self._stats_dict.items():
            if metric in SCALING_FACTORS:
                factor = SCALING_FACTORS[metric]
                for key, gpu_data in data.items():
                    if key != "unit":
                        for stat, value in gpu_data.items():
                            self._stats_dict[metric][key][stat] = value * factor

    def set_stats_dict(self, stats_dict: DefaultDict[str, Any]) -> None:
        self._stats_dict = stats_dict

    def create_records(self) -> GpuRecords:
        """
        Populates and returns a list of Records
        """
        telemetry_records: GpuRecords = {}
        for metric_base_name, metric_info in self.stats_dict.items():
            for gpu_id, gpu_info in metric_info.items():
                if gpu_id == "unit":
                    continue
                elif gpu_id not in telemetry_records:
                    telemetry_records[gpu_id] = {}

                for metric_post_name, metric_value in gpu_info.items():
                    metric_name = metric_base_name + "_" + metric_post_name

                    try:
                        new_record = RecordType.get_all_record_types()[metric_name](
                            metric_value, gpu_id
                        )
                    except KeyError:
                        raise GenAIPerfException(
                            f"{metric_name} is not a valid Record tag."
                        )

                    telemetry_records[gpu_id][metric_name] = new_record

        return telemetry_records

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
    def stats_dict(self) -> Dict[str, Any]:
        return self._stats_dict
