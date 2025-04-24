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

import json
from io import StringIO
from typing import Any, Dict, List, Tuple

import genai_perf.parser as parser
import pytest
from genai_perf.config.input.create_config import CreateConfig
from genai_perf.export_data.json_exporter import JsonExporter
from tests.test_utils import create_default_exporter_config


class TestJsonExporter:

    def create_json_exporter(
        self, monkeypatch, cli_cmd: List[str], stats: Dict[str, Any], **kwargs
    ) -> JsonExporter:
        monkeypatch.setattr("sys.argv", cli_cmd)
        args, _ = parser.parse_args()
        config = CreateConfig.create(args)
        exporter_config = create_default_exporter_config(
            stats=stats,
            config=config,
            **kwargs,
        )
        return JsonExporter(exporter_config)

    @pytest.fixture
    def mock_read_write(self, monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str]]:
        """
        This function will mock the open function for specific files.
        """
        written_data = []

        def custom_open(filename, *args, **kwargs):
            def write(self: Any, content: str) -> int:
                print(f"Writing to {filename}")  # To help with debugging failures
                written_data.append((str(filename), content))
                return len(content)

            tmp_file = StringIO()
            tmp_file.write = write.__get__(tmp_file)
            return tmp_file

        monkeypatch.setattr("builtins.open", custom_open)
        return written_data

    def test_generate_json_export_file(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        """
        Tests that the resulting json file is generated/exported correctly.
        """
        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--backend",
            "tensorrtllm",
            "--artifact-dir",
            "/tmp/test_artifact",
        ]
        json_exporter = self.create_json_exporter(monkeypatch, cli_cmd, stats={})
        json_exporter.export()

        expected_filename = "/tmp/test_artifact/test_model-triton-tensorrtllm-concurrency1/profile_export_genai_perf.json"
        actual_filename, _ = next(iter(mock_read_write))
        assert actual_filename == expected_filename

    def test_generate_json_custom_export_file(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--backend",
            "tensorrtllm",
            "--artifact-dir",
            "/tmp/test_artifact",
            "--profile-export-file",
            "custom_export.json",
        ]
        json_exporter = self.create_json_exporter(monkeypatch, cli_cmd, stats={})
        json_exporter.export()

        expected_json_filename = "/tmp/test_artifact/test_model-triton-tensorrtllm-concurrency1/custom_export_genai_perf.json"
        expected_profile_filename = "custom_export.json"

        actual_json_filename, data = next(iter(mock_read_write))
        assert actual_json_filename == expected_json_filename

        json_output = json.loads(data)
        actual_profile_filename = json_output["input_config"]["output"][
            "profile_export_file"
        ]
        assert actual_profile_filename == expected_profile_filename

    def test_generate_json_stats(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        """
        Tests that the stats are exported correctly.
        """
        stats = {
            "some_metric_1": {
                "unit": "requests/sec",
                "avg": 123,
                "std": 456,
            },
            "some_metric_2": {
                "unit": "ms",
                "p99": 789,
            },
        }

        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--endpoint-type",
            "chat",
        ]
        json_exporter = self.create_json_exporter(monkeypatch, cli_cmd, stats)
        json_exporter.export()

        _, data = next(iter(mock_read_write))
        actual_json_output = json.loads(data)
        del actual_json_output["input_config"]  # only test stats

        assert actual_json_output == stats

    def test_generate_json_input_config(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--endpoint-type",
            "chat",
        ]
        json_exporter = self.create_json_exporter(monkeypatch, cli_cmd, stats={})
        json_exporter.export()

        expected_input_config_sections = [
            "model_names",
            "analyze",
            "endpoint",
            "perf_analyzer",
            "input",
            "output",
            "tokenizer",
            "subcommand",
        ]

        _, data = next(iter(mock_read_write))
        actual_json_output = json.loads(data)

        for expected_input_config_section in expected_input_config_sections:
            assert expected_input_config_section in actual_json_output["input_config"]

    def test_generate_json_goodput_stats(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        """
        Tests that the goodput stats are exported correctly.
        """
        goodput_stats = {
            "some_goodput_metric_1": {
                "unit": "requests/sec",
                "avg": 123,
            },
            "some_goodput_metric_2": {
                "unit": "requests/sec",
                "avg": 456,
            },
        }

        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--endpoint-type",
            "chat",
            "--goodput",
            "some_metric_1:8.0",
            "some_metric_2:2.0",
            "some_metric_3:650.0",
        ]
        json_exporter = self.create_json_exporter(
            monkeypatch, cli_cmd, stats=goodput_stats
        )
        json_exporter.export()

        _, data = next(iter(mock_read_write))
        json_output = json.loads(data)

        input_config = json_output.pop("input_config")
        assert input_config["input"]["goodput"] == {
            "some_metric_1": 8.0,
            "some_metric_2": 2.0,
            "some_metric_3": 650.0,
        }
        assert json_output == goodput_stats

    def test_generate_json_telemetry_stats(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        """
        Tests that the telemetry stats are exported correctly.
        """
        telemetry_stats = {
            "some_telemetry_metric_1": {
                "unit": "W",
                "gpu0": {
                    "avg": 123,
                    "p25": 456,
                    "p50": 789,
                },
            },
            "some_telemetry_metric_2": {
                "unit": "W",
                "gpu0": {
                    "avg": 123,
                    "p25": 456,
                    "p50": 789,
                },
            },
        }

        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--backend",
            "tensorrtllm",
            "--server-metrics-url",
            "http://tritonmetrics:8002/metrics",
        ]

        json_exporter = self.create_json_exporter(
            monkeypatch, cli_cmd, stats={}, telemetry_stats=telemetry_stats
        )
        json_exporter.export()

        _, data = next(iter(mock_read_write))
        json_output = json.loads(data)
        assert json_output["telemetry_stats"] == telemetry_stats

    def test_generate_json_session_stats(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        session_stats = {
            "session-id-123": {
                "some_metric_1": {
                    "unit": "count",
                    "avg": 123,
                },
                "some_metric_2": {
                    "unit": "ms",
                    "avg": 456,
                },
            },
            "session-id-456": {
                "some_metric_1": {
                    "unit": "requests/sec",
                    "avg": 789,
                },
                "some_metric_2": {
                    "unit": "ms",
                    "avg": 1011,
                },
            },
        }

        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--endpoint-type",
            "chat",
        ]
        json_exporter = self.create_json_exporter(
            monkeypatch, cli_cmd, stats={}, session_stats=session_stats
        )
        json_exporter.export()

        _, data = next(iter(mock_read_write))
        json_output = json.loads(data)

        assert "sessions" in json_output
        assert json_output["sessions"] == session_stats
