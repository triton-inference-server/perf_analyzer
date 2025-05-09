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
from typing import Dict, List

from genai_perf.metrics.telemetry_metrics import (
    MetricMetadata,
    TelemetryMetricName,
    TelemetryMetrics,
)


class TestTelemetryMetrics:

    def test_initialization(self):
        telemetry = TelemetryMetrics()
        for metric in TelemetryMetricName.values():
            attr = getattr(telemetry, metric.value)
            assert isinstance(attr, defaultdict)
            assert len(attr) == 0

    def test_initialization_with_params(self):
        telemetry = TelemetryMetrics(
            gpu_power_usage={"gpu0": [0.0]},
            gpu_power_limit={"gpu0": [1.0]},
            energy_consumption={"gpu0": [2.0]},
            gpu_utilization={"gpu0": [3.0]},
            total_gpu_memory={"gpu0": [4.0]},
            gpu_memory_used={"gpu0": [5.0]},
            gpu_memory_free={"gpu0": [6.0]},
            gpu_memory_temperature={"gpu0": [7.0]},
            gpu_temperature={"gpu0": [8.0]},
            sm_utilization={"gpu0": [9.0]},
            memory_copy_utilization={"gpu0": [10.0]},
            video_encoder_utilization={"gpu0": [11.0]},
            video_decoder_utilization={"gpu0": [12.0]},
            gpu_clock_sm={"gpu0": [13.0]},
            gpu_clock_memory={"gpu0": [14.0]},
        )

        assert telemetry.gpu_power_usage == {"gpu0": [0.0]}
        assert telemetry.gpu_power_limit == {"gpu0": [1.0]}
        assert telemetry.energy_consumption == {"gpu0": [2.0]}
        assert telemetry.gpu_utilization == {"gpu0": [3.0]}
        assert telemetry.total_gpu_memory == {"gpu0": [4.0]}
        assert telemetry.gpu_memory_used == {"gpu0": [5.0]}
        assert telemetry.gpu_memory_free == {"gpu0": [6.0]}
        assert telemetry.gpu_memory_temperature == {"gpu0": [7.0]}
        assert telemetry.gpu_temperature == {"gpu0": [8.0]}
        assert telemetry.sm_utilization == {"gpu0": [9.0]}
        assert telemetry.memory_copy_utilization == {"gpu0": [10.0]}
        assert telemetry.video_encoder_utilization == {"gpu0": [11.0]}
        assert telemetry.video_decoder_utilization == {"gpu0": [12.0]}
        assert telemetry.gpu_clock_sm == {"gpu0": [13.0]}
        assert telemetry.gpu_clock_memory == {"gpu0": [14.0]}

    def test_update_metrics(self) -> None:
        telemetry = TelemetryMetrics()
        measurement_data: Dict[str, Dict[str, List[float]]] = {
            "gpu_power_usage": {"gpu0": [11.1], "gpu1": [11.2]},
            "gpu_power_limit": {"gpu0": [101.2], "gpu1": [101.2]},
            "energy_consumption": {"gpu0": [1004.0], "gpu1": [1005.0]},
            "gpu_utilization": {"gpu0": [85.0], "gpu1": [90.0]},
            "sm_utilization": {"gpu0": [65.0], "gpu1": [68.0]},
            "memory_copy_utilization": {"gpu0": [55.0], "gpu1": [58.0]},
            "video_encoder_utilization": {"gpu0": [20.0], "gpu1": [25.0]},
            "video_decoder_utilization": {"gpu0": [15.0], "gpu1": [18.0]},
            "gpu_clock_sm": {"gpu0": [1520.0], "gpu1": [1510.0]},
            "gpu_clock_memory": {"gpu0": [5050.0], "gpu1": [5000.0]},
            "total_gpu_memory": {"gpu0": [9000.0], "gpu1": [9000.0]},
            "gpu_memory_used": {"gpu0": [4500.0], "gpu1": [4500.0]},
            "gpu_memory_free": {"gpu0": [4500.0], "gpu1": [4500.0]},
            "gpu_memory_temperature": {"gpu0": [62.0], "gpu1": [63.0]},
            "gpu_temperature": {"gpu0": [72.0], "gpu1": [73.0]},
        }

        telemetry.update_metrics(measurement_data)

        for metric_name, expected in measurement_data.items():
            assert getattr(telemetry, metric_name) == expected

    def test_update_metrics_with_empty_data(self):
        telemetry = TelemetryMetrics()
        telemetry.update_metrics({})
        for metric in TelemetryMetricName.values():
            assert len(getattr(telemetry, metric.value)) == 0

    def test_update_metrics_multiple_times(self):
        telemetry = TelemetryMetrics()
        telemetry.update_metrics({"gpu_power_usage": {"gpu0": [10.5]}})
        telemetry.update_metrics({"gpu_power_usage": {"gpu0": [15.0]}})

        assert telemetry.gpu_power_usage == {"gpu0": [10.5, 15.0]}

    def test_telemetry_metrics_property(self) -> None:
        telemetry = TelemetryMetrics()
        telemetry_metrics: List[MetricMetadata] = telemetry.telemetry_metrics

        assert len(telemetry_metrics) == len(TelemetryMetricName)
        defined_names = {m.name for m in telemetry_metrics}
        enum_names = {m.value for m in TelemetryMetricName.values()}
        assert defined_names == enum_names
