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

import pytest
from genai_perf.exceptions import GenAIPerfException
from genai_perf.metrics.telemetry_metrics import TelemetryMetrics
from genai_perf.metrics.telemetry_statistics import TelemetryStatistics
from genai_perf.record.types.gpu_power_usage_avg import GPUPowerUsageAvg


class TestTelemetryStatistics:

    @pytest.fixture
    def mock_metrics(self) -> TelemetryMetrics:
        telemetry = TelemetryMetrics()
        telemetry.update_metrics(
            {
                "gpu_power_usage": {"gpu0": [10.0, 20.0, 30.0], "gpu1": [40.0, 50.0]},
                "gpu_utilization": {"gpu0": [60.0, 70.0, 80.0], "gpu1": [90.0]},
                "energy_consumption": {"gpu0": [1000000000.0], "gpu1": [2000000000.0]},
                "total_gpu_memory": {"gpu0": [8000000000.0], "gpu1": [16000000000.0]},
                "gpu_memory_used": {"gpu0": [4000], "gpu1": [5000]},
            }
        )
        return telemetry

    @pytest.fixture
    def telemetry_statistics(self, mock_metrics) -> TelemetryStatistics:
        return TelemetryStatistics(mock_metrics)

    def test_initialization(self, telemetry_statistics):
        stats_dict = telemetry_statistics.stats_dict

        assert "gpu_power_usage" in stats_dict
        assert "gpu0" in stats_dict["gpu_power_usage"]
        assert "avg" in stats_dict["gpu_power_usage"]["gpu0"]
        assert isinstance(stats_dict["gpu_power_usage"]["gpu0"]["avg"], float)

    def test_average_calculation(self, telemetry_statistics):
        stats_dict = telemetry_statistics.stats_dict
        assert stats_dict["gpu_power_usage"]["gpu0"]["avg"] == 20.0  # (10+20+30)/3
        assert stats_dict["gpu_power_usage"]["gpu1"]["avg"] == 45.0  # (40+50)/2

    def test_min_max_calculation(self, telemetry_statistics):
        stats_dict = telemetry_statistics.stats_dict
        assert stats_dict["gpu_power_usage"]["gpu0"]["min"] == 10.0
        assert stats_dict["gpu_power_usage"]["gpu0"]["max"] == 30.0
        assert stats_dict["gpu_power_usage"]["gpu1"]["min"] == 40.0
        assert stats_dict["gpu_power_usage"]["gpu1"]["max"] == 50.0

    def test_percentile_calculations(self, telemetry_statistics):
        stats_dict = telemetry_statistics.stats_dict
        assert "p99" in stats_dict["gpu_power_usage"]["gpu0"]
        assert "p95" in stats_dict["gpu_power_usage"]["gpu0"]
        assert "p90" in stats_dict["gpu_power_usage"]["gpu0"]
        assert (
            stats_dict["gpu_power_usage"]["gpu0"]["p99"] == 29.8
        )  # np.percentile([10, 20, 30], 99)

    def test_scaling_data(self, telemetry_statistics):
        telemetry_statistics.scale_data()
        stats_dict = telemetry_statistics.stats_dict

        assert stats_dict["energy_consumption"]["gpu0"]["avg"] == 1.0  # 1 MJ
        assert stats_dict["energy_consumption"]["gpu1"]["avg"] == 2.0  # 2 MJ

        assert (
            round(stats_dict["gpu_memory_used"]["gpu0"]["avg"], 2) == 4.19
        )  # 4.194304 GB
        assert (
            round(stats_dict["gpu_memory_used"]["gpu1"]["avg"], 2) == 5.24
        )  # 5.24288 GB

    def test_create_records(self, telemetry_statistics):
        telemetry_statistics._stats_dict = telemetry_statistics.stats_dict
        telemetry_records = telemetry_statistics.create_records()

        assert "gpu0" in telemetry_records
        assert "gpu_power_usage_avg" in telemetry_records["gpu0"]
        assert isinstance(
            telemetry_records["gpu0"]["gpu_power_usage_avg"], GPUPowerUsageAvg
        )

    def test_create_records_invalid_key(self, telemetry_statistics):
        telemetry_statistics._stats_dict["invalid_metric"] = {"gpu0": {"avg": 100}}
        with pytest.raises(GenAIPerfException):
            telemetry_statistics.create_records()

    def test_empty_data_handling(self):
        empty_metrics = TelemetryMetrics()
        telemetry_statistics = TelemetryStatistics(empty_metrics)
        assert len(telemetry_statistics.stats_dict) > 0
        for metric_key, metric_data in telemetry_statistics.stats_dict.items():
            assert list(metric_data.keys()) == [
                "unit"
            ], f"Expected only 'unit' in {metric_key}, but got {metric_data.keys()}"

        assert all(
            "gpu0" not in metric_data and "gpu1" not in metric_data
            for metric_data in telemetry_statistics.stats_dict.values()
        ), "Expected no GPU-specific stats in empty case"

    def test_constant_metrics(self, telemetry_statistics):
        """Test that constant metrics do not compute unnecessary statistics."""
        stats_dict = telemetry_statistics.stats_dict
        assert "min" not in stats_dict["total_gpu_memory"]["gpu0"]
        assert "max" not in stats_dict["total_gpu_memory"]["gpu0"]
        assert "std" not in stats_dict["total_gpu_memory"]["gpu0"]

    def test_should_skip(self, telemetry_statistics):
        assert telemetry_statistics._should_skip({}) is True
        assert telemetry_statistics._should_skip({"gpu0": [10.0]}) is False
