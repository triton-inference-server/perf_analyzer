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

import csv
from io import StringIO

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.export_data.telemetry_data_exporter_util import (
    export_telemetry_stats_console,
    export_telemetry_stats_csv,
    merge_telemetry_stats_json,
)
from genai_perf.metrics import TelemetryMetrics
from genai_perf.subcommand.subcommand import Subcommand
from rich.console import Console


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

    @pytest.fixture
    def telemetry_data(self):
        return {
            "gpu_power_usage": {
                "gpu0": {"avg": 50.0, "min": 30.0, "max": 60.0, "p99": 59.0},
                "unit": "W",
            },
            "gpu_utilization": {"gpu0": {"avg": 75.0, "p99": 90.0}, "unit": "%"},
            "gpu_memory_used": {"gpu0": {"avg": 4096.0}, "unit": "MB"},
            "total_gpu_memory": {"gpu0": {"avg": 8192.0}, "unit": "MB"},
        }

    def test_merge_identical_metrics(self, telemetry_1):
        config = ConfigCommand(user_config={"model_name": "test_model"})
        merged = Subcommand(config)._merge_telemetry_metrics([telemetry_1, telemetry_1])

        assert merged.gpu_power_usage["gpu0"] == [10.0, 20.0, 10.0, 20.0]
        assert merged.gpu_power_usage["gpu1"] == [30.0, 30.0]
        assert merged.gpu_utilization["gpu0"] == [50.0, 50.0]
        assert merged.gpu_utilization["gpu1"] == [75.0, 75.0]

        assert set(merged.gpu_power_usage.keys()) == {"gpu0", "gpu1"}
        assert set(merged.gpu_utilization.keys()) == {"gpu0", "gpu1"}

    def test_merge_different_gpus(self, telemetry_1, telemetry_2):
        config = ConfigCommand(user_config={"model_name": "test_model"})
        merged = Subcommand(config)._merge_telemetry_metrics([telemetry_1, telemetry_2])

        assert merged.gpu_power_usage["gpu0"] == [10.0, 20.0, 40.0]
        assert merged.gpu_utilization["gpu0"] == [50.0, 80.0]
        assert merged.gpu_power_usage["gpu1"] == [30.0, 60.0]
        assert merged.gpu_utilization["gpu1"] == [75.0]
        assert merged.gpu_utilization["gpu2"] == [90.0]

        assert set(merged.gpu_power_usage.keys()) == {"gpu0", "gpu1"}
        assert set(merged.gpu_utilization.keys()) == {"gpu0", "gpu1", "gpu2"}

    def test_merge_with_empty_telemetry(self, telemetry_1):
        empty_telemetry = TelemetryMetrics()
        config = ConfigCommand(user_config={"model_name": "test_model"})
        merged = Subcommand(config)._merge_telemetry_metrics(
            [telemetry_1, empty_telemetry]
        )

        assert merged.gpu_power_usage["gpu0"] == [10.0, 20.0]
        assert merged.gpu_utilization["gpu1"] == [75.0]

        assert set(merged.gpu_power_usage.keys()) == {"gpu0", "gpu1"}
        assert set(merged.gpu_utilization.keys()) == {"gpu0", "gpu1"}

    def test_merge_no_metrics(self):
        config = ConfigCommand(user_config={"model_name": "test_model"})
        merged = Subcommand(config)._merge_telemetry_metrics([])
        assert isinstance(merged, TelemetryMetrics)
        assert len(merged.gpu_power_usage) == 0
        assert len(merged.gpu_utilization) == 0

    def test_export_telemetry_stats_console(self, telemetry_data):
        console = Console(record=True)
        export_telemetry_stats_console(telemetry_data, ["avg", "min", "max"], console)
        output = console.export_text()
        assert "Power Metrics" in output
        assert "GPU Power Usage" in output
        assert "75" in output

    def test_export_telemetry_stats_csv(self, telemetry_data):
        csv_output = StringIO()
        csv_writer = csv.writer(csv_output)
        export_telemetry_stats_csv(telemetry_data, csv_writer)
        csv_output.seek(0)
        csv_content = csv_output.read()
        assert "GPU Power Usage" in csv_content
        assert "GPU Utilization" in csv_content
        assert "Total GPU Memory" in csv_content

    def test_merge_telemetry_stats_json(self, telemetry_data):
        stats_and_args = {}
        merge_telemetry_stats_json(telemetry_data, stats_and_args)
        assert "telemetry_stats" in stats_and_args
        assert "gpu_power_usage" in stats_and_args["telemetry_stats"]
        assert "gpu_utilization" in stats_and_args["telemetry_stats"]
        assert (
            stats_and_args["telemetry_stats"]["gpu_power_usage"]["gpu0"]["avg"] == 50.0
        )

    def test_export_empty_telemetry_stats_console(self):
        console = Console(record=True)
        export_telemetry_stats_console({}, ["avg", "min", "max"], console)
        output = console.export_text()
        assert output.strip() == ""

    def test_export_empty_telemetry_stats_csv(self):
        csv_output = StringIO()
        csv_writer = csv.writer(csv_output)
        export_telemetry_stats_csv({}, csv_writer)
        csv_output.seek(0)
        csv_content = csv_output.read()
        assert csv_content.strip() == ""

    def test_export_empty_json_stats_csv(self):
        stats_and_args = {}
        merge_telemetry_stats_json({}, stats_and_args)
        assert "telemetry_stats" not in stats_and_args
