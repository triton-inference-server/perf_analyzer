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

import re
from typing import Dict, List

from genai_perf.telemetry_data.telemetry_data_collector import TelemetryDataCollector


class TritonTelemetryDataCollector(TelemetryDataCollector):
    """Class to collect telemetry metrics from Triton server"""

    def _parse_metrics(self, data: str) -> None:
        # Parsing logic for Prometheus metrics
        metrics = {
            "gpu_power_usage": [],
            "gpu_power_limit": [],
            "energy_consumption": [],
            "gpu_utilization": [],
            "total_gpu_memory": [],
            "gpu_memory_used": [],
        }

        for line in data.splitlines():
            if line.startswith("nv_gpu_power_usage"):
                self._extract_metric(line, metrics["gpu_power_usage"])
            elif line.startswith("nv_gpu_power_limit"):
                self._extract_metric(line, metrics["gpu_power_limit"])
            elif line.startswith("nv_energy_consumption"):
                self._extract_metric(line, metrics["energy_consumption"])
            elif line.startswith("nv_gpu_utilization"):
                self._extract_metric(line, metrics["gpu_utilization"])
            elif line.startswith("nv_gpu_memory_total_bytes"):
                self._extract_metric(line, metrics["total_gpu_memory"])
            elif line.startswith("nv_gpu_memory_used_bytes"):
                self._extract_metric(line, metrics["gpu_memory_used"])
        return metrics

    def _extract_metric(
        self, metric_line: str, metric_list: List[List[float]]
    ) -> Dict[str, List[List[float]]]:
        metric_components = metric_line.split()
        metric_value = float(metric_components[1])
        metric_list.append(metric_value)
