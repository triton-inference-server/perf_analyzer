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

from typing import Dict, List, Optional, Tuple

import genai_perf.logging as logging
from genai_perf.telemetry_data.telemetry_data_collector import TelemetryDataCollector

logger = logging.getLogger(__name__)


class DCGMTelemetryDataCollector(TelemetryDataCollector):
    """Collects telemetry metrics from DCGM metrics endpoint."""

    METRIC_NAME_MAPPING = {
        "DCGM_FI_DEV_POWER_USAGE": "gpu_power_usage",
        "DCGM_FI_DEV_POWER_MGMT_LIMIT": "gpu_power_limit",
        "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": "energy_consumption",
        "DCGM_FI_DEV_GPU_UTIL": "gpu_utilization",
        "DCGM_FI_DEV_FB_USED": "gpu_memory_used",
        "DCGM_FI_DEV_FB_TOTAL": "total_gpu_memory",
    }

    def _process_and_update_metrics(self, metrics_data: str) -> None:
        """Process GPU metrics from the DCGM endpoint and update internal metrics data.

        Each metric line is expected to follow the format:
        metric_name{label1="value1", ...} value

        Only metrics listed in METRIC_NAME_MAPPING are processed. The 'gpu' label is
        extracted from each line and used to group values by GPU (e.g., '0', '1', etc.).
        Parsed values are grouped by metric and GPU and used to update metrics data.
        """

        if not metrics_data.strip():
            logger.info("Response from DCGM metrics endpoint is empty")
            return

        current_measurement_interval: Dict[str, Dict[str, List[float]]] = {
            metric.name: {} for metric in self.metrics.TELEMETRY_METRICS
        }

        for line in metrics_data.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parsed = self._parse_metric_line(line)
            if not parsed:
                continue

            metric_full_name, metric_value = parsed
            metric_name = metric_full_name.split("{")[0]
            mapped_key = self.METRIC_NAME_MAPPING.get(metric_name)
            if not mapped_key:
                continue

            gpu_label = self._extract_gpu_label(metric_full_name)
            if not gpu_label:
                continue

            self._append_metric_value(
                current_measurement_interval, mapped_key, gpu_label, metric_value
            )

        self.metrics.update_metrics(current_measurement_interval)

    def _parse_metric_line(self, line: str) -> Optional[Tuple[str, float]]:
        try:
            metric_full_name, value_str = line.rsplit(" ", 1)
            metric_value = float(value_str.strip())
            return metric_full_name, metric_value
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse metric value from line: {line}")
        return None

    def _extract_gpu_label(self, metric_full_name: str) -> Optional[str]:
        try:
            labels_str = metric_full_name.partition("{")[2].partition("}")[0]
            labels = dict(kv.split("=", 1) for kv in labels_str.split(",") if "=" in kv)
            gpu_value = labels.get("gpu")
            return gpu_value.strip('"') if gpu_value else None
        except Exception as e:
            logger.warning(
                f"Failed to extract GPU label from: {metric_full_name} — {e}"
            )
            return None

    def _append_metric_value(
        self,
        current_measurement_interval: Dict[str, Dict[str, List[float]]],
        mapped_key: str,
        gpu_label: str,
        metric_value: float,
    ) -> None:
        gpu_metrics = current_measurement_interval.setdefault(mapped_key, {})
        gpu_metrics.setdefault(gpu_label, []).append(metric_value)
