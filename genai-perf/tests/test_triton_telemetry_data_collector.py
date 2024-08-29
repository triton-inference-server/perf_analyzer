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

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from genai_perf.metrics.telemetry_metrics import MetricMetadata, TelemetryMetrics
from genai_perf.telemetry_data.triton_telemetry_data_collector import (
    TritonTelemetryDataCollector,
)


class TestTritonTelemetryDataCollector:

    TEST_SERVER_URL: str = "http://tritonserver:8002/metrics"

    @pytest.fixture
    def triton_collector(self) -> TritonTelemetryDataCollector:
        return TritonTelemetryDataCollector(self.TEST_SERVER_URL)

    @pytest.fixture
    def mock_telemetry_metrics(self) -> MagicMock:
        mock_telemetry_metrics = MagicMock(spec=TelemetryMetrics)
        mock_telemetry_metrics.TELEMETRY_METRICS = [
            MetricMetadata("gpu_power_usage", "W"),
            MetricMetadata("gpu_power_limit", "W"),
            MetricMetadata("energy_consumption", "MJ"),
            MetricMetadata("gpu_utilization", "%"),
            MetricMetadata("total_gpu_memory", "GB"),
            MetricMetadata("gpu_memory_used", "GB"),
        ]
        return mock_telemetry_metrics

    @patch.object(TritonTelemetryDataCollector, "metrics", new_callable=PropertyMock)
    def test_process_and_update_metrics_single_gpu(
        self,
        mock_metrics: PropertyMock,
        triton_collector: TritonTelemetryDataCollector,
        mock_telemetry_metrics: MagicMock,
    ) -> None:

        mock_metrics.return_value = mock_telemetry_metrics

        triton_metrics_data = """nv_gpu_power_usage{gpu_uuid="GPU-1234"} 35.0
            nv_gpu_power_limit{gpu_uuid="GPU-1234"} 250.0
            nv_gpu_utilization{gpu_uuid="GPU-1234"} 0.85
            nv_energy_consumption{gpu_uuid="GPU-1234"} 1500000.0
            nv_gpu_memory_total_bytes{gpu_uuid="GPU-1234"} 8000000000.0
            nv_gpu_memory_used_bytes{gpu_uuid="GPU-1234"} 4000000000.0"""

        triton_collector._process_and_update_metrics(triton_metrics_data)

        expected_data = {
            "gpu_power_usage": [35.0],
            "gpu_power_limit": [250.0],
            "gpu_utilization": [85.0],
            "energy_consumption": [1.5],
            "total_gpu_memory": [8.0],
            "gpu_memory_used": [4.0],
        }
        mock_metrics.return_value.update_metrics.assert_called_once_with(expected_data)

    @patch.object(TritonTelemetryDataCollector, "metrics", new_callable=PropertyMock)
    def test_process_and_update_metrics_multiple_gpus(
        self,
        mock_metrics: PropertyMock,
        triton_collector: TritonTelemetryDataCollector,
        mock_telemetry_metrics: MagicMock,
    ) -> None:

        mock_metrics.return_value = mock_telemetry_metrics

        triton_metrics_data = """nv_gpu_power_usage{gpu_uuid="GPU-1234"} 35.0
            nv_gpu_power_usage{gpu_uuid="GPU-5678"} 40.0
            nv_gpu_power_limit{gpu_uuid="GPU-1234"} 250.0
            nv_gpu_power_limit{gpu_uuid="GPU-1234"} 300.0
            nv_gpu_utilization{gpu_uuid="GPU-1234"} 0.85
            nv_gpu_utilization{gpu_uuid="GPU-5678"} 0.90
            nv_energy_consumption{gpu_uuid="GPU-1234"} 1500000.0
            nv_energy_consumption{gpu_uuid="GPU-1234"} 1500000.0
            nv_gpu_memory_total_bytes{gpu_uuid="GPU-1234"} 8000000000.0
            nv_gpu_memory_total_bytes{gpu_uuid="GPU-1234"} 9000000000.0
            nv_gpu_memory_used_bytes{gpu_uuid="GPU-1234"} 4000000000.0
            nv_gpu_memory_used_bytes{gpu_uuid="GPU-1234"} 4500000000.0"""

        triton_collector._process_and_update_metrics(triton_metrics_data)

        expected_data = {
            "gpu_power_usage": [35.0, 40.0],
            "gpu_power_limit": [250.0, 300.0],
            "gpu_utilization": [85.0, 90.0],
            "energy_consumption": [1.5, 1.5],
            "total_gpu_memory": [8.0, 9.0],
            "gpu_memory_used": [4.0, 4.5],
        }

        mock_metrics.return_value.update_metrics.assert_called_once_with(expected_data)

    @patch.object(TritonTelemetryDataCollector, "metrics", new_callable=PropertyMock)
    def test_process_and_update_metrics_empty_data(
        self,
        mock_metrics: PropertyMock,
        triton_collector: TritonTelemetryDataCollector,
        mock_telemetry_metrics: MagicMock,
    ) -> None:

        mock_metrics.return_value = mock_telemetry_metrics

        trtion_metrics_data = ""

        triton_collector._process_and_update_metrics(trtion_metrics_data)

        mock_telemetry_metrics.update_metrics.assert_not_called()
