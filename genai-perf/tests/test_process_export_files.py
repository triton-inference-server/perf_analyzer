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

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.metrics.statistics import Statistics
from genai_perf.metrics.telemetry_statistics import TelemetryStatistics
from genai_perf.subcommand.process_export_files import ProcessExportFiles


class TestProcessExportFilesHandler:

    @pytest.fixture
    def process_export_files(self, mock_config):
        return ProcessExportFiles(mock_config, None)

    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=ConfigCommand)
        config.process = MagicMock()
        config.process.input_path = Path("test_dir")
        config.model_names = []
        config.input = MagicMock()
        config.endpoint = MagicMock()
        config.perf_analyzer = MagicMock()
        config.tokenizer = MagicMock()
        config.output = MagicMock()
        config.subcommand = "process"
        return config

    @pytest.fixture
    def mock_objectives(self):
        return MagicMock()

    @pytest.fixture
    def mock_perf_stats(self):
        return MagicMock(spec=Statistics)

    @pytest.fixture
    def mock_telemetry_stats(self):
        return MagicMock(spec=TelemetryStatistics)

    @pytest.fixture
    def mock_session_stats(self):
        return {"session1": MagicMock(spec=Statistics)}

    @pytest.fixture
    def mock_pa_profile_data(self):
        return {
            "version": "1.0",
            "service_kind": "triton",
            "endpoint": "localhost:8001",
            "experiments": [
                {
                    "experiment": {"mode": "concurrency", "value": 1},
                    "requests": [
                        {
                            "timestamp": 1745526100322752068,
                            "request_inputs": {
                                "text_input": "Hello world!",
                                "stream": True,
                            },
                            "response_timestamps": [1745526100365982228],
                            "response_outputs": [{"text_output": "Hi!"}],
                        }
                    ],
                }
            ],
        }

    @pytest.fixture
    def mock_gap_profile_data(self):
        return {
            "request_throughput": {"unit": "requests/sec", "avg": 15.0},
            "output_token_throughput": {"unit": "tokens/sec", "avg": 200.0},
            "input_sequence_length": {"unit": "tokens", "avg": 550},
            "output_sequence_length": {"unit": "tokens", "avg": 16},
            "telemetry_stats": {
                "gpu_power_usage": {"unit": "W", "gpu0": {"avg": 22.5}}
            },
            "input_config": {
                "model_names": ["small_model"],
                "input": {
                    "batch_size": 1,
                    "num_dataset_entries": 10,
                    "synthetic_tokens": {"mean": 550, "stddev": 0},
                    "prompt_source": "synthetic",
                },
                "endpoint": {
                    "backend": "vllm",
                    "url": "localhost:8001",
                    "service_kind": "triton",
                    "output_format": "vllm",
                },
                "perf_analyzer": {
                    "path": "perf_analyzer",
                    "stimulus": {"concurrency": 1},
                },
                "tokenizer": {
                    "name": "test-tokenizer",
                    "revision": "main",
                    "trust_remote_code": False,
                },
            },
        }

    def test_process_pa_profile_file_success(
        self, process_export_files, mock_pa_profile_data
    ):
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_pa_profile_data))
        ):
            process_export_files._process_pa_profile_file(Path("mock_pa.json"))
        assert process_export_files._pa_profile_data["version"] == "1.0"

    def test_process_pa_profile_file_file_not_found(self, process_export_files):
        with patch("builtins.open", side_effect=FileNotFoundError()):
            with pytest.raises(GenAIPerfException, match="Error reading file"):
                process_export_files._process_pa_profile_file(Path("missing.json"))

    def test_process_gap_profile_file_success(
        self, process_export_files, mock_gap_profile_data
    ):
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_gap_profile_data))
        ):
            process_export_files._process_gap_profile_file(Path("mock_gap.json"))
        assert process_export_files._input_config["model_names"] == ["small_model"]

    def test_process_gap_profile_file_invalid_json(self, process_export_files):
        with patch("builtins.open", mock_open(read_data="bad_json")):
            with pytest.raises(GenAIPerfException, match="Error reading file"):
                process_export_files._process_gap_profile_file(Path("bad_gap.json"))

    def test_parse_input_directory_success(self, process_export_files):
        mock_subdir = MagicMock()
        mock_subdir.is_dir.return_value = True

        mock_pa_file = MagicMock()
        mock_pa_file.name = "pa_profile.json"
        mock_gap_file = MagicMock()
        mock_gap_file.name = "gap_profile_genai_perf.json"

        mock_subdir.glob.side_effect = lambda pattern: (
            [mock_pa_file, mock_gap_file] if pattern == "*.json" else []
        )

        with patch.object(Path, "iterdir", return_value=[mock_subdir]):
            with patch.object(
                process_export_files, "_process_pa_profile_file"
            ) as mock_pa, patch.object(
                process_export_files, "_process_gap_profile_file"
            ) as mock_gap:
                process_export_files._parse_input_directory(Path("test_dir"))

                mock_pa.assert_called_once_with(mock_pa_file)
                mock_gap.assert_called_once_with(mock_gap_file)

    def test_parse_input_directory_missing_files(self, process_export_files):
        mock_subdir = MagicMock()
        mock_subdir.is_dir.return_value = True

        mock_subdir.glob.return_value = []

        with patch.object(Path, "iterdir", return_value=[mock_subdir]):
            with patch.object(
                process_export_files, "_process_pa_profile_file"
            ) as mock_pa, patch.object(
                process_export_files, "_process_gap_profile_file"
            ) as mock_gap:
                process_export_files._parse_input_directory(Path("test_dir"))

                mock_pa.assert_not_called()
                mock_gap.assert_not_called()

    def test_parse_input_directory_no_subdirs(self, process_export_files):
        with patch.object(Path, "iterdir", return_value=[]):
            with patch.object(
                process_export_files, "_process_pa_profile_file"
            ) as mock_pa, patch.object(
                process_export_files, "_process_gap_profile_file"
            ) as mock_gap:
                process_export_files._parse_input_directory(Path("test_dir"))

                mock_pa.assert_not_called()
                mock_gap.assert_not_called()

    def test_parse_input_directory_invalid_files(self, process_export_files):
        mock_subdir = MagicMock()
        mock_subdir.is_dir.return_value = True

        mock_subdir.glob.return_value = [MagicMock(name="invalid_file.txt")]

        with patch.object(Path, "iterdir", return_value=[mock_subdir]):
            with patch.object(
                process_export_files, "_process_pa_profile_file"
            ) as mock_pa, patch.object(
                process_export_files, "_process_gap_profile_file"
            ) as mock_gap:
                process_export_files._parse_input_directory(Path("test_dir"))

                mock_pa.assert_not_called()
                mock_gap.assert_not_called()

    def test_parse_input_directory_only_pa_file(self, process_export_files):
        mock_subdir = MagicMock()
        mock_subdir.is_dir.return_value = True

        mock_pa_file = MagicMock()
        mock_pa_file.name = "pa_profile.json"

        mock_subdir.glob.side_effect = lambda pattern: (
            [mock_pa_file] if pattern == "*.json" else []
        )

        with patch.object(Path, "iterdir", return_value=[mock_subdir]):
            with patch.object(
                process_export_files, "_process_pa_profile_file"
            ) as mock_pa, patch.object(
                process_export_files, "_process_gap_profile_file"
            ) as mock_gap:
                process_export_files._parse_input_directory(Path("test_dir"))

                mock_pa.assert_not_called()
                mock_gap.assert_not_called()

    def test_parse_input_directory_only_gap_file(self, process_export_files):
        mock_subdir = MagicMock()
        mock_subdir.is_dir.return_value = True

        mock_gap_file = MagicMock()
        mock_gap_file.name = "gap_profile_genai_perf.json"

        mock_subdir.glob.side_effect = lambda pattern: (
            [mock_gap_file] if pattern == "*.json" else []
        )

        with patch.object(Path, "iterdir", return_value=[mock_subdir]):
            with patch.object(
                process_export_files, "_process_pa_profile_file"
            ) as mock_pa, patch.object(
                process_export_files, "_process_gap_profile_file"
            ) as mock_gap:
                process_export_files._parse_input_directory(Path("test_dir"))

                mock_pa.assert_not_called()
                mock_gap.assert_not_called()

    def test_set_model_names(self, process_export_files):
        process_export_files._input_config = {"model_names": ["test_model"]}
        process_export_files._set_model_names()
        assert process_export_files._config.model_names == ["test_model"]

    def test_set_input_fields(self, process_export_files):
        process_export_files._input_config = {"input": {"batch_size": 2}}
        process_export_files._set_input_fields()
        process_export_files._config.input.parse.assert_called_once()

    def test_set_perf_analyzer_fields(self, process_export_files):
        process_export_files._input_config = {"perf_analyzer": {"path": "mock"}}
        process_export_files._set_perf_analyzer_fields()
        process_export_files._config.perf_analyzer.parse.assert_called_once()

    def test_set_endpoint_fields(self, process_export_files):
        process_export_files._input_config = {"endpoint": {"backend": "mock"}}
        process_export_files._config.model_names = ["mock_model"]
        process_export_files._set_endpoint_fields()
        process_export_files._config.endpoint.parse.assert_called_once()

    def test_set_tokenizer_fields(self, process_export_files):
        process_export_files._input_config = {"tokenizer": {"name": "mock_tokenizer"}}
        process_export_files._config.model_names = ["mock_model"]
        process_export_files._set_tokenizer_fields()
        process_export_files._config.tokenizer.parse.assert_called_once()

    @patch.object(ProcessExportFiles, "_set_tokenizer_fields")
    @patch.object(ProcessExportFiles, "_set_endpoint_fields")
    @patch.object(ProcessExportFiles, "_set_perf_analyzer_fields")
    @patch.object(ProcessExportFiles, "_set_input_fields")
    @patch.object(ProcessExportFiles, "_set_model_names")
    def test_update_config_from_profile_data(
        self,
        mock_model,
        mock_input,
        mock_perf,
        mock_endpoint,
        mock_tokenizer,
        process_export_files,
    ):
        process_export_files._update_config_from_profile_data()
        mock_model.assert_called_once()
        mock_input.assert_called_once()
        mock_perf.assert_called_once()
        mock_endpoint.assert_called_once()
        mock_tokenizer.assert_called_once()

    def test_create_merged_profile_export_file(self, process_export_files):
        perf_analyzer_config = MagicMock()
        perf_analyzer_config.get_profile_export_file.return_value = Path("merged.json")
        with patch("builtins.open", mock_open()) as mocked_file:
            process_export_files._create_merged_profile_export_file(
                perf_analyzer_config
            )
        mocked_file.assert_called_once()

    @patch("genai_perf.subcommand.process_export_files.MergedProfileParser")
    def test_calculate_metrics(self, mock_profile_parser, process_export_files):
        perf_analyzer_config = MagicMock()
        perf_analyzer_config.get_profile_export_file.return_value = Path("profile.json")
        process_export_files._tokenizer = MagicMock()

        result = process_export_files._calculate_metrics(perf_analyzer_config)
        mock_profile_parser.assert_called_once()
        assert result == mock_profile_parser.return_value

    def test_set_telemetry_aggregator(self, process_export_files):
        process_export_files._telemetry_dicts = [
            {"gpu_power_usage": {"gpu0": {"avg": 10.0}, "unit": "W"}}
        ]
        process_export_files._set_telemetry_aggregator()
        assert process_export_files._telemetry_aggregator is not None

    def test_create_telemetry_stats(self, process_export_files):
        mock_aggregator = MagicMock()
        mock_aggregator.get_telemetry_stats.return_value = "telemetry_stats"
        process_export_files._telemetry_aggregator = mock_aggregator
        result = process_export_files._create_telemetry_stats()
        assert result == "telemetry_stats"

    @patch("genai_perf.subcommand.process_export_files.OutputReporter")
    def test_add_output_to_artifact_directory(
        self,
        mock_output_reporter,
        process_export_files,
        mock_perf_stats,
        mock_telemetry_stats,
        mock_session_stats,
        mock_objectives,
    ):

        process_export_files._create_perf_stats = MagicMock(
            return_value=mock_perf_stats
        )
        process_export_files._create_telemetry_stats = MagicMock(
            return_value=mock_telemetry_stats
        )
        process_export_files._create_session_stats = MagicMock(
            return_value=mock_session_stats
        )

        perf_analyzer_config = MagicMock()
        objectives = MagicMock()

        process_export_files._add_output_to_artifact_directory(
            perf_analyzer_config, objectives
        )

        process_export_files._create_perf_stats.assert_called_once_with(
            perf_analyzer_config, objectives
        )
        process_export_files._create_telemetry_stats.assert_called_once()
        process_export_files._create_session_stats.assert_called_once_with(
            perf_analyzer_config, objectives
        )

        mock_output_reporter.assert_called_once_with(
            mock_perf_stats,
            mock_telemetry_stats,
            process_export_files._config,
            perf_analyzer_config,
            mock_session_stats,
        )
        mock_output_reporter.return_value.report_output.assert_called_once()

    @patch.object(ProcessExportFiles, "_parse_input_directory")
    @patch.object(ProcessExportFiles, "_update_config_from_profile_data")
    @patch.object(ProcessExportFiles, "_create_objectives_based_on_stimulus")
    @patch.object(ProcessExportFiles, "_create_perf_analyzer_config")
    @patch.object(ProcessExportFiles, "_create_tokenizer")
    @patch.object(ProcessExportFiles, "_create_artifact_directory")
    @patch.object(ProcessExportFiles, "_create_merged_profile_export_file")
    @patch.object(ProcessExportFiles, "_set_data_parser")
    @patch.object(ProcessExportFiles, "_set_telemetry_aggregator")
    @patch.object(ProcessExportFiles, "_add_output_to_artifact_directory")
    def test_process_export_files_end_to_end(
        self,
        mock_add_output,
        mock_set_telemetry,
        mock_set_data_parser,
        mock_create_merged_export,
        mock_create_artifact_dir,
        mock_create_tokenizer,
        mock_create_perf_config,
        mock_create_objectives,
        mock_update_config,
        mock_parse_input,
        process_export_files,
    ):
        mock_create_objectives.return_value = MagicMock()
        mock_create_perf_config.return_value = MagicMock()

        process_export_files.process_export_files()

        mock_parse_input.assert_called_once()
        mock_update_config.assert_called_once()
        mock_create_objectives.assert_called_once()
        mock_create_perf_config.assert_called_once()
        mock_create_tokenizer.assert_called_once()
        mock_create_artifact_dir.assert_called_once()
        mock_create_merged_export.assert_called_once()
        mock_set_data_parser.assert_called_once()
        mock_set_telemetry.assert_called_once()
        mock_add_output.assert_called_once()
