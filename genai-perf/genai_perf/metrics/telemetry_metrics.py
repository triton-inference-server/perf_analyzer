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
from typing import Dict, List, Optional

from genai_perf.metrics.metrics import MetricMetadata


class TelemetryMetrics:
    """
    A class that contains common telemetry metrics.
    Metrics are stored as
        'gpu_power_usage': {
            'gpu0': [27.01]
        },
        'gpu_utilization': {
            'gpu0': [75.5]
        },
        'energy_consumption': {
            'gpu0': [123.56]
        }
    """

    TELEMETRY_METRICS = [
        MetricMetadata("gpu_power_usage", "W"),
        MetricMetadata("gpu_power_limit", "W"),
        MetricMetadata("energy_consumption", "MJ"),
        MetricMetadata("gpu_utilization", "%"),
        MetricMetadata("total_gpu_memory", "GB"),
        MetricMetadata("gpu_memory_used", "GB"),
    ]

    def __init__(
        self,
        gpu_power_usage: Optional[Dict[str, List[float]]] = None,
        gpu_power_limit: Optional[Dict[str, List[float]]] = None,
        energy_consumption: Optional[Dict[str, List[float]]] = None,
        gpu_utilization: Optional[Dict[str, List[float]]] = None,
        total_gpu_memory: Optional[Dict[str, List[float]]] = None,
        gpu_memory_used: Optional[Dict[str, List[float]]] = None,
    ):
        self.gpu_power_usage = defaultdict(list, gpu_power_usage or {})
        self.gpu_power_limit = defaultdict(list, gpu_power_limit or {})
        self.energy_consumption = defaultdict(list, energy_consumption or {})
        self.gpu_utilization = defaultdict(list, gpu_utilization or {})
        self.total_gpu_memory = defaultdict(list, total_gpu_memory or {})
        self.gpu_memory_used = defaultdict(list, gpu_memory_used or {})

    def update_metrics(self, measurement_data: dict) -> None:
        for metric in self.TELEMETRY_METRICS:
            metric_key = metric.name
            if metric_key in measurement_data:
                metric_data = measurement_data[metric_key]
                for gpu_name, values in metric_data.items():
                    getattr(self, metric_key)[gpu_name].extend(values)

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
