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
import json
from io import StringIO
from pathlib import Path
from unittest.mock import create_autospec, mock_open, patch

import genai_perf.parser as parser
import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.metrics import Metrics, TelemetryMetrics, TelemetryStatistics
from genai_perf.metrics.statistics import Statistics
from genai_perf.metrics.telemetry_metrics import TelemetryMetrics
from genai_perf.profile_data_parser import ProfileDataParser
from genai_perf.subcommand.profile import _report_output
from genai_perf.telemetry_data.triton_telemetry_data_collector import (
    TelemetryDataCollector,
)
from rich.console import Console


class TestIntegrationTelemetry:

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

    def test_reporting_for_multiple_metric_urls(
        self, capsys, monkeypatch, telemetry_1, telemetry_2
    ):
        """
        An integration test for reporting Telemetry statistics with multiple metric URLs
        """
        test_args = [
            "genai-perf",
            "profile",
            "--model",
            "test_model",
            "-v",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        args, _, _ = parser.parse_args()

        mock_metrics = create_autospec(Metrics, instance=True)
        mock_statistics = create_autospec(Statistics, instance=True)
        mock_statistics.metrics = mock_metrics

        mock_parser = create_autospec(ProfileDataParser, instance=True)
        mock_parser.get_statistics.return_value = mock_statistics
        mock_parser.metrics = mock_metrics

        telemetry_collectors = [
            create_autospec(TelemetryDataCollector, instance=True),
            create_autospec(TelemetryDataCollector, instance=True),
        ]
        telemetry_collectors[0].get_metrics.return_value = telemetry_1
        telemetry_collectors[1].get_metrics.return_value = telemetry_2

        telemetry_collectors[0].get_statistics.return_value = TelemetryStatistics(
            telemetry_1
        )
        telemetry_collectors[1].get_statistics.return_value = TelemetryStatistics(
            telemetry_2
        )

        mock_file_open = mock_open()
        csv_output = StringIO()
        csv_writer = csv.writer(csv_output)
        console = Console(record=True)

        with patch("builtins.open", mock_file_open) as mocked_file, patch.object(
            Path, "exists", return_value=True
        ), patch("csv.writer", return_value=csv_writer), patch(
            "rich.console.Console.print", side_effect=console.print
        ):
            config = ConfigCommand({"model_name": args.model[0]})
            config = parser.add_cli_options_to_config(config, args)

            _report_output(mock_parser, telemetry_collectors, config)

            mock_file_open.assert_called()

            assert mock_parser.get_statistics.called
            assert telemetry_collectors[0].get_metrics.called
            assert telemetry_collectors[1].get_metrics.called

            written_json = None
            mock_file_handle = mocked_file.return_value
            for call in mock_file_handle.write.mock_calls:
                json_candidate = json.loads(call.args[0])
                if "telemetry_stats" in json_candidate:
                    written_json = json_candidate
                    break

            assert written_json is not None, "JSON output was not generated"
            assert (
                "telemetry_stats" in written_json
            ), "Telemetry statistics missing from JSON"
            assert (
                "gpu_power_usage" in written_json["telemetry_stats"]
            ), "GPU power usage missing from JSON"
            assert (
                written_json["telemetry_stats"]["gpu_power_usage"]["gpu1"]["avg"]
                == 45.0
            ), "Expected average GPU power missing from JSON"

            csv_output.seek(0)
            csv_content = csv_output.read()
            assert "GPU Power Usage" in csv_content, "Missing GPU power usage in CSV"
            assert "GPU Utilization" in csv_content, "Missing GPU utilization in CSV"
            assert "75" in csv_content, "Expected GPU utilization missing from CSV"

            captured = capsys.readouterr()
            assert "Power" in captured.out, "Missing Power section in console output"
            assert (
                "GPU Power Usage" in captured.out
            ), "Missing GPU power usage in console output"
            assert "75" in captured.out, "Expected GPU utilization missing from console"
