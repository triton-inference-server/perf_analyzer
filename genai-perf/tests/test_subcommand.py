# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.input.config_field import ConfigField
from genai_perf.subcommand.subcommand import Subcommand


class TestCommon:
    @patch("genai_perf.subcommand.subcommand.subprocess.run")
    def test_stdout_verbose(self, mock_subprocess_run):
        config = ConfigCommand(user_config={"model_name": "test_model"})
        config.verbose = ConfigField(default=False, value=True)
        perf_analyzer_config = PerfAnalyzerConfig(config)
        subcommand = Subcommand(config)
        subcommand._run_perf_analyzer(
            perf_analyzer_config=perf_analyzer_config,
        )

        # Check that standard output was not redirected.
        for call_args in mock_subprocess_run.call_args_list:
            _, kwargs = call_args
            assert (
                "stdout" not in kwargs or kwargs["stdout"] is None
            ), "With the verbose flag, stdout should not be redirected."

    @patch("genai_perf.subcommand.subcommand.subprocess.run")
    def test_stdout_not_verbose(self, mock_subprocess_run):
        config = ConfigCommand(user_config={"model_name": "test_model"})
        config.verbose = ConfigField(default=False)
        perf_analyzer_config = PerfAnalyzerConfig(config)
        subcommand = Subcommand(config)
        subcommand._run_perf_analyzer(
            perf_analyzer_config=perf_analyzer_config,
        )

        # Check that standard output was redirected.
        for call_args in mock_subprocess_run.call_args_list:
            _, kwargs = call_args
            assert (
                kwargs["stdout"] is subprocess.DEVNULL
            ), "When the verbose flag is not passed, stdout should be redirected to /dev/null."

            class TestSubcommand:
                @patch("genai_perf.subcommand.subcommand.subprocess.run")
                def test_run_perf_analyzer_verbose(self, mock_subprocess_run):
                    config = ConfigCommand(user_config={"model_name": "test_model"})
                    config.verbose = True
                    perf_analyzer_config = MagicMock(spec=PerfAnalyzerConfig)
                    perf_analyzer_config.create_command.return_value = ["mock_command"]
                    perf_analyzer_config.get_profile_export_file.return_value = (
                        "mock_file"
                    )

                    subcommand = Subcommand(config)
                    with patch(
                        "genai_perf.subcommand.subcommand.remove_file"
                    ) as mock_remove_file:
                        subcommand._run_perf_analyzer(perf_analyzer_config)

                        mock_remove_file.assert_called_once_with("mock_file")
                        mock_subprocess_run.assert_called_once_with(
                            ["mock_command"], check=True, stdout=None
                        )

                @patch("genai_perf.subcommand.subcommand.subprocess.run")
                def test_run_perf_analyzer_not_verbose(self, mock_subprocess_run):
                    config = ConfigCommand(user_config={"model_name": "test_model"})
                    config.verbose = False
                    perf_analyzer_config = MagicMock(spec=PerfAnalyzerConfig)
                    perf_analyzer_config.create_command.return_value = ["mock_command"]
                    perf_analyzer_config.get_profile_export_file.return_value = (
                        "mock_file"
                    )

                    subcommand = Subcommand(config)
                    with patch(
                        "genai_perf.subcommand.subcommand.remove_file"
                    ) as mock_remove_file:
                        subcommand._run_perf_analyzer(perf_analyzer_config)

                        mock_remove_file.assert_called_once_with("mock_file")
                        mock_subprocess_run.assert_called_once_with(
                            ["mock_command"], check=True, stdout=subprocess.DEVNULL
                        )

                @patch("genai_perf.subcommand.subcommand.os.makedirs")
                def test_create_artifact_directory(self, mock_makedirs):
                    config = ConfigCommand(user_config={"model_name": "test_model"})
                    perf_analyzer_config = MagicMock(spec=PerfAnalyzerConfig)
                    perf_analyzer_config.get_artifact_directory.return_value = (
                        "mock_directory"
                    )

                    subcommand = Subcommand(config)
                    subcommand._create_artifact_directory(perf_analyzer_config)

                    mock_makedirs.assert_called_once_with(
                        "mock_directory", exist_ok=True
                    )

                @patch("genai_perf.subcommand.subcommand.os.makedirs")
                def test_create_plot_directory(self, mock_makedirs):
                    config = ConfigCommand(user_config={"model_name": "test_model"})
                    config.output = MagicMock()
                    config.output.generate_plots = True
                    perf_analyzer_config = MagicMock(spec=PerfAnalyzerConfig)
                    perf_analyzer_config.get_artifact_directory.return_value = (
                        "mock_directory"
                    )

                    subcommand = Subcommand(config)
                    subcommand._create_plot_directory(perf_analyzer_config)

                    mock_makedirs.assert_called_once_with(
                        os.path.join("mock_directory", "plots"), exist_ok=True
                    )

                @patch("genai_perf.subcommand.subcommand.TritonTelemetryDataCollector")
                def test_create_telemetry_data_collectors(self, mock_collector):
                    config = ConfigCommand(user_config={"model_name": "test_model"})
                    config.endpoint = MagicMock()
                    config.endpoint.service_kind = "triton"
                    config.endpoint.server_metrics_urls = ["http://mock_url"]
                    mock_collector_instance = mock_collector.return_value
                    mock_collector_instance.is_url_reachable.return_value = True

                    subcommand = Subcommand(config)
                    collectors = subcommand._create_telemetry_data_collectors()

                    assert len(collectors) == 1
                    mock_collector.assert_called_once_with("http://mock_url")
                    mock_collector_instance.is_url_reachable.assert_called_once()

                def test_is_config_present_in_results(self):
                    config = ConfigCommand(user_config={"model_name": "test_model"})
                    subcommand = Subcommand(config)
                    subcommand._results = MagicMock()
                    subcommand._create_representation = MagicMock(
                        return_value="mock_representation"
                    )

                    genai_perf_config = MagicMock()
                    perf_analyzer_config = MagicMock()
                    subcommand._results.found_representation.return_value = True

                    result = subcommand._is_config_present_in_results(
                        genai_perf_config, perf_analyzer_config
                    )

                    assert result is True
                    subcommand._create_representation.assert_called_once_with(
                        genai_perf_config, perf_analyzer_config
                    )
                    subcommand._results.found_representation.assert_called_once_with(
                        "mock_representation"
                    )
