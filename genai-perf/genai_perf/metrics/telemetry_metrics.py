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

from typing import List

from genai_perf.metrics.metrics import MetricMetadata


class TelemetryMetrics:
    """
    A class that contains common telemetry metrics.
    Metrics are stored as lists where each inner list corresponds to multiple measurements per GPU.
    Each measurement is recorded every second.
    """

    TELEMETRY_METRICS = [
        MetricMetadata("gpu_power_usage", "W"),  # Watts (W)
        MetricMetadata("gpu_power_limit", "W"),  # Watts (W)
        MetricMetadata("energy_consumption", "MJ"),  # Megajoules (MJ)
        MetricMetadata("gpu_utilization", "%"),  # Percentage (%)
        MetricMetadata("total_gpu_memory", "GB"),  # Gigabytes (GB)
        MetricMetadata("gpu_memory_used", "GB"),  # Gigabytes (GB)
    ]

    def __init__(
        self,
        gpu_power_usage: List[List[float]] = [],  # Multiple measurements per GPU
        gpu_power_limit: List[List[float]] = [],
        energy_consumption: List[List[float]] = [],
        gpu_utilization: List[List[float]] = [],
        total_gpu_memory: List[List[float]] = [],
        gpu_memory_used: List[List[float]] = [],
    ) -> None:
        self.gpu_power_usage = gpu_power_usage
        self.gpu_power_limit = gpu_power_limit
        self.energy_consumption = energy_consumption
        self.gpu_utilization = gpu_utilization
        self.total_gpu_memory = total_gpu_memory
        self.gpu_memory_used = gpu_memory_used

    def update_metrics(self, measurement_data: dict) -> None:
        """Update the metrics with new measurement data"""
        for metric in self.TELEMETRY_METRICS:
            metric_key = metric.name
            if metric_key in measurement_data:
                getattr(self, metric_key).append(measurement_data[metric_key])

    def __repr__(self):
        attr_strs = []
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                attr_strs.append(f"{k}={v}")
        return f"TelemetryMetrics({','.join(attr_strs)})"

    @property
    def telemetry_metrics(self) -> List[MetricMetadata]:
        return self.TELEMETRY_METRICS

    @property
    def data(self) -> dict:
        """Returns all the metrics."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
