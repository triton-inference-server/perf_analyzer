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

import pytest
from genai_perf.metrics.telemetry_metrics import TelemetryMetrics
from genai_perf.subcommand.common import merge_telemetry_metrics


class TestMergeTelemetryMetrics:

    @pytest.fixture
    def telemetry_1(self) -> TelemetryMetrics:
        telemetry = TelemetryMetrics()
        telemetry.update_metrics(
            {
                "gpu_power_usage": {"gpu0": [10.0, 20.0], "gpu1": [30.0]},
                "gpu_utilization": {"gpu0": [50.0], "gpu1": [75.0]},
            }
        )
        return telemetry

    @pytest.fixture
    def telemetry_2(self) -> TelemetryMetrics:
        telemetry = TelemetryMetrics()
        telemetry.update_metrics(
            {
                "gpu_power_usage": {"gpu0": [40.0], "gpu1": [60.0]},
                "gpu_utilization": {"gpu0": [80.0], "gpu2": [90.0]},
            }
        )
        return telemetry

    def test_merge_identical_metrics(self, telemetry_1):
        merged = merge_telemetry_metrics([telemetry_1, telemetry_1])

        assert merged.gpu_power_usage["gpu0"] == [10.0, 20.0, 10.0, 20.0]
        assert merged.gpu_power_usage["gpu1"] == [30.0, 30.0]
        assert merged.gpu_utilization["gpu0"] == [50.0, 50.0]
        assert merged.gpu_utilization["gpu1"] == [75.0, 75.0]

        assert set(merged.gpu_power_usage.keys()) == {"gpu0", "gpu1"}
        assert set(merged.gpu_utilization.keys()) == {"gpu0", "gpu1"}

    def test_merge_different_gpus(self, telemetry_1, telemetry_2):
        merged = merge_telemetry_metrics([telemetry_1, telemetry_2])

        assert merged.gpu_power_usage["gpu0"] == [10.0, 20.0, 40.0]
        assert merged.gpu_utilization["gpu0"] == [50.0, 80.0]
        assert merged.gpu_power_usage["gpu1"] == [30.0, 60.0]
        assert merged.gpu_utilization["gpu1"] == [75.0]
        assert merged.gpu_utilization["gpu2"] == [90.0]

        assert set(merged.gpu_power_usage.keys()) == {"gpu0", "gpu1"}
        assert set(merged.gpu_utilization.keys()) == {"gpu0", "gpu1", "gpu2"}

    def test_merge_with_empty_telemetry(self, telemetry_1):
        empty_telemetry = TelemetryMetrics()
        merged = merge_telemetry_metrics([telemetry_1, empty_telemetry])

        assert merged.gpu_power_usage["gpu0"] == [10.0, 20.0]
        assert merged.gpu_utilization["gpu1"] == [75.0]

        assert set(merged.gpu_power_usage.keys()) == {"gpu0", "gpu1"}
        assert set(merged.gpu_utilization.keys()) == {"gpu0", "gpu1"}

    def test_merge_no_metrics(self):
        merged = merge_telemetry_metrics([])
        assert isinstance(merged, TelemetryMetrics)
        assert len(merged.gpu_power_usage) == 0
        assert len(merged.gpu_utilization) == 0
