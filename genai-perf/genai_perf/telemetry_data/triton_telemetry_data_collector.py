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

from typing import Dict, List

import genai_perf.logging as logging
from genai_perf.telemetry_data.telemetry_data_collector import TelemetryDataCollector

logger = logging.getLogger(__name__)


class TritonTelemetryDataCollector(TelemetryDataCollector):
    """Class to collect telemetry metrics from Triton server"""

    METRIC_NAME_MAPPING = {
        "nv_gpu_power_usage": "gpu_power_usage",
        "nv_gpu_power_limit": "gpu_power_limit",
        "nv_energy_consumption": "energy_consumption",
        "nv_gpu_utilization": "gpu_utilization",
        "nv_gpu_memory_total_bytes": "total_gpu_memory",
        "nv_gpu_memory_used_bytes": "gpu_memory_used",
    }

    def _process_and_update_metrics(self, metrics_data: str) -> None:
        """Process the response from Triton metrics endpoint and update metrics.

        This method extracts metric names and values from the raw data. Metric names
        are extracted from the start of each line up to the '{' character, as all metrics
        follow the format 'metric_name{labels} value'. Only metrics defined in
        METRIC_NAME_MAPPING are processed.

        Args:
            data (str): Raw metrics data from the Triton endpoint.

        Example:
            Given the metric data:
            ```
            nv_gpu_power_usage{gpu_uuid="GPU-abschdinjacgdo65gdj7"} 27.01
            nv_gpu_utilization{gpu_uuid="GPU-abcdef123456"} 75.5
            nv_energy_consumption{gpu_uuid="GPU-xyz789"} 1234.56
            ```

            The metrics are stored as:
            'gpu_power_usage': {
                'gpu0': [27.01]
            },
            'gpu_utilization': {
                'gpu0': [75.5]
            },
            'energy_consumption': {
                'gpu0': [1234.56]
            }
        """

        if not metrics_data.strip():
            logger.info("Response from Triton metrics endpoint is empty")
            return

        gpu_mapping = {}
        gpu_index = 0

        current_measurement_interval = {
            metric.name: {} for metric in self.metrics.TELEMETRY_METRICS
        }  # type: Dict[str, Dict[str, List[float]]]

        for line in metrics_data.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            triton_metric_key = parts[0].split("{")[0]
            metric_value = float(parts[1])

            metric_key = self.METRIC_NAME_MAPPING.get(triton_metric_key, None)
            if not metric_key:
                continue

            gpu_uuid = parts[0].split('gpu_uuid="')[1].split('"')[0]

            if gpu_uuid not in gpu_mapping:
                gpu_mapping[gpu_uuid] = f"gpu{gpu_index}"
                gpu_index += 1

            gpu_label = gpu_mapping[gpu_uuid]

            # Store the metric under the corresponding gpu label
            if gpu_label not in current_measurement_interval[metric_key]:
                current_measurement_interval[metric_key][gpu_label] = []

            current_measurement_interval[metric_key][gpu_label].append(metric_value)

        self.metrics.update_metrics(current_measurement_interval)
