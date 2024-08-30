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

from genai_perf.metrics.telemetry_metrics import MetricMetadata, TelemetryMetrics


class TestTelemetryMetrics:

    def test_update_metrics(self) -> None:
        telemetry = TelemetryMetrics()
        measurement_data: Dict[str, Dict[str, List[float]]] = {
            "gpu_power_usage": {"gpu0": [11.1], "gpu1": [11.2]},
            "gpu_power_limit": {"gpu0": [101.2], "gpu1": [101.2]},
            "energy_consumption": {"gpu0": [1004.0], "gpu1": [1005.0]},
            "gpu_utilization": {"gpu0": [85.0], "gpu1": [90.0]},
            "total_gpu_memory": {"gpu0": [9000.0], "gpu1": [9000.0]},
            "gpu_memory_used": {"gpu0": [4500.0], "gpu1": [4500.0]},
        }
        telemetry.update_metrics(measurement_data)

        assert telemetry.gpu_power_usage == {"gpu0": [11.1], "gpu1": [11.2]}
        assert telemetry.gpu_power_limit == {"gpu0": [101.2], "gpu1": [101.2]}
        assert telemetry.energy_consumption == {"gpu0": [1004.0], "gpu1": [1005.0]}
        assert telemetry.gpu_utilization == {"gpu0": [85.0], "gpu1": [90.0]}
        assert telemetry.total_gpu_memory == {"gpu0": [9000.0], "gpu1": [9000.0]}
        assert telemetry.gpu_memory_used == {"gpu0": [4500.0], "gpu1": [4500.0]}

    def test_telemetry_metrics_property(self) -> None:
        telemetry = TelemetryMetrics()
        telemetry_metrics: List[MetricMetadata] = telemetry.telemetry_metrics

        assert len(telemetry_metrics) == 6
        assert telemetry_metrics[0].name == "gpu_power_usage"
        assert telemetry_metrics[0].unit == "W"
        assert telemetry_metrics[1].name == "gpu_power_limit"
        assert telemetry_metrics[1].unit == "W"
        assert telemetry_metrics[2].name == "energy_consumption"
        assert telemetry_metrics[2].unit == "MJ"
        assert telemetry_metrics[3].name == "gpu_utilization"
        assert telemetry_metrics[3].unit == "%"
        assert telemetry_metrics[4].name == "total_gpu_memory"
        assert telemetry_metrics[4].unit == "GB"
        assert telemetry_metrics[5].name == "gpu_memory_used"
        assert telemetry_metrics[5].unit == "GB"
