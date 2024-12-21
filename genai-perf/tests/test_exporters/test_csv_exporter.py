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

import os
from io import StringIO
from pathlib import Path
from typing import Any, List, Tuple

import pytest
from genai_perf import parser
from genai_perf.export_data.csv_exporter import CsvExporter
from genai_perf.export_data.exporter_config import ExporterConfig
from genai_perf.metrics import (
    LLMMetrics,
    Metrics,
    Statistics,
    TelemetryMetrics,
    TelemetryStatistics,
)


class TestCsvExporter:
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

    @pytest.fixture
    def llm_metrics(self) -> LLMMetrics:
        """
        Provides an LLMMetrics object with predefined metrics.
        """
        return LLMMetrics(
            request_throughputs=[123],
            request_latencies=[4, 5, 6],
            time_to_first_tokens=[7, 8, 9],
            time_to_second_tokens=[1, 2, 3],
            inter_token_latencies=[10, 11, 12],
            output_token_throughputs=[456],
            output_sequence_lengths=[1, 2, 3],
            input_sequence_lengths=[5, 6, 7],
        )

    def test_streaming_llm_csv_output(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch, llm_metrics: LLMMetrics
    ) -> None:
        """
        Collect LLM metrics from profile export data and confirm correct values are
        printed in csv.
        """
        argv = [
            "genai-perf",
            "profile",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "chat",
            "--streaming",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        stats = Statistics(metrics=llm_metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.artifact_dir = Path(".")
        config.args = args

        exporter = CsvExporter(config)
        exporter.export()

        expected_content = [
            "Metric,avg,min,max,p99,p95,p90,p75,p50,p25\r\n",
            "Time To First Token (ms),8.00,7.00,9.00,8.98,8.90,8.80,8.50,8.00,7.50\r\n",
            "Time To Second Token (ms),2.00,1.00,3.00,2.98,2.90,2.80,2.50,2.00,1.50\r\n",
            "Request Latency (ms),5.00,4.00,6.00,5.98,5.90,5.80,5.50,5.00,4.50\r\n",
            "Inter Token Latency (ms),11.00,10.00,12.00,11.98,11.90,11.80,11.50,11.00,10.50\r\n",
            "Output Sequence Length,2.00,1.00,3.00,2.98,2.90,2.80,2.50,2.00,1.50\r\n",
            "Input Sequence Length,6.00,5.00,7.00,6.98,6.90,6.80,6.50,6.00,5.50\r\n",
            "\r\n",
            "Metric,Value\r\n",
            "Output Token Throughput (per sec),456.00\r\n",
            "Request Throughput (per sec),123.00\r\n",
            "Request Count (count),3.00\r\n",
        ]
        expected_filename = "profile_export_genai_perf.csv"
        returned_data = [
            data
            for filename, data in mock_read_write
            if os.path.basename(filename) == expected_filename
        ]

        assert returned_data == expected_content

    def test_nonstreaming_llm_csv_output(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch, llm_metrics: LLMMetrics
    ) -> None:
        """
        Collect LLM metrics from profile export data and confirm correct values are
        printed in csv.
        """
        argv = [
            "genai-perf",
            "profile",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "chat",
            "--profile-export-file",
            "custom_export.json",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        stats = Statistics(metrics=llm_metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.artifact_dir = Path(".")
        config.args = args

        exporter = CsvExporter(config)
        exporter.export()

        expected_filename = f"custom_export_genai_perf.csv"
        expected_content = [
            "Metric,avg,min,max,p99,p95,p90,p75,p50,p25\r\n",
            "Request Latency (ms),5.00,4.00,6.00,5.98,5.90,5.80,5.50,5.00,4.50\r\n",
            "Output Sequence Length,2.00,1.00,3.00,2.98,2.90,2.80,2.50,2.00,1.50\r\n",
            "Input Sequence Length,6.00,5.00,7.00,6.98,6.90,6.80,6.50,6.00,5.50\r\n",
            "\r\n",
            "Metric,Value\r\n",
            "Output Token Throughput (per sec),456.00\r\n",
            "Request Throughput (per sec),123.00\r\n",
            "Request Count (count),3.00\r\n",
        ]
        returned_data = [
            data for filename, data in mock_read_write if filename == expected_filename
        ]

        assert returned_data == expected_content

    def test_embedding_csv_output(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        argv = [
            "genai-perf",
            "profile",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "embeddings",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        metrics = Metrics(
            request_throughputs=[123],
            request_latencies=[4, 5, 6],
        )
        stats = Statistics(metrics=metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.artifact_dir = Path(".")
        config.args = args

        exporter = CsvExporter(config)
        exporter.export()

        expected_content = [
            "Metric,avg,min,max,p99,p95,p90,p75,p50,p25\r\n",
            "Request Latency (ms),5.00,4.00,6.00,5.98,5.90,5.80,5.50,5.00,4.50\r\n",
            "\r\n",
            "Metric,Value\r\n",
            "Request Throughput (per sec),123.00\r\n",
            "Request Count (count),3.00\r\n",
        ]
        returned_data = [data for _, data in mock_read_write]
        assert returned_data == expected_content

    def test_valid_goodput_csv_output(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        argv = [
            "genai-perf",
            "profile",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "chat",
            "--streaming",
            "--goodput",
            "request_latency:100",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        metrics = LLMMetrics(
            request_throughputs=[123],
            request_latencies=[4, 5, 6],
            time_to_first_tokens=[7, 8, 9],
            time_to_second_tokens=[1, 2, 3],
            inter_token_latencies=[10, 11, 12],
            output_token_throughputs=[456],
            output_sequence_lengths=[1, 2, 3],
            input_sequence_lengths=[5, 6, 7],
            request_goodputs=[100],
        )
        stats = Statistics(metrics=metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.artifact_dir = Path(".")
        config.args = args

        exporter = CsvExporter(config)
        exporter.export()

        expected_content = "Request Goodput (per sec),100.00\r\n"
        expected_filename = "profile_export_genai_perf.csv"
        returned_data = [
            data
            for filename, data in mock_read_write
            if os.path.basename(filename) == expected_filename
        ]

        assert returned_data[-2] == expected_content

    def test_invalid_goodput_csv_output(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        argv = [
            "genai-perf",
            "profile",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "chat",
            "--streaming",
            "--goodput",
            "request_latenC:100",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        metrics = LLMMetrics(
            request_throughputs=[123],
            request_latencies=[4, 5, 6],
            time_to_first_tokens=[7, 8, 9],
            time_to_second_tokens=[1, 2, 3],
            inter_token_latencies=[10, 11, 12],
            output_token_throughputs=[456],
            output_sequence_lengths=[1, 2, 3],
            input_sequence_lengths=[5, 6, 7],
            request_goodputs=[-1],
        )
        stats = Statistics(metrics=metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.artifact_dir = Path(".")
        config.args = args

        exporter = CsvExporter(config)
        exporter.export()

        expected_content = "Request Goodput (per sec),-1.00\r\n"

        expected_filename = "profile_export_genai_perf.csv"
        returned_data = [
            data
            for filename, data in mock_read_write
            if os.path.basename(filename) == expected_filename
        ]

        assert returned_data[-2] == expected_content

    def test_triton_telemetry_output(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch, llm_metrics: LLMMetrics
    ) -> None:
        argv = [
            "genai-perf",
            "profile",
            "-m",
            "model_name",
            "--service-kind",
            "triton",
            "--streaming",
            "--server-metrics-url",
            "http://tritonserver:8002/metrics",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        telemetry_metrics = TelemetryMetrics(
            gpu_power_usage={"gpu0": [45.2, 46.5]},
            gpu_power_limit={"gpu0": [250.0, 250.0]},
            energy_consumption={"gpu0": [0.5, 0.6]},
            gpu_utilization={"gpu0": [75.0, 80.0]},
            total_gpu_memory={"gpu0": [8.0, 8.0]},
            gpu_memory_used={"gpu0": [4.0, 4.0]},
        )

        stats = Statistics(metrics=llm_metrics)
        telemetry_stats = TelemetryStatistics(telemetry_metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.telemetry_stats = telemetry_stats.stats_dict
        config.metrics = stats.metrics
        config.artifact_dir = Path(".")
        config.args = args

        exporter = CsvExporter(config)
        exporter.export()

        expected_content = [
            "Metric,avg,min,max,p99,p95,p90,p75,p50,p25\r\n",
            "Time To First Token (ms),8.00,7.00,9.00,8.98,8.90,8.80,8.50,8.00,7.50\r\n",
            "Time To Second Token (ms),2.00,1.00,3.00,2.98,2.90,2.80,2.50,2.00,1.50\r\n",
            "Request Latency (ms),5.00,4.00,6.00,5.98,5.90,5.80,5.50,5.00,4.50\r\n",
            "Inter Token Latency (ms),11.00,10.00,12.00,11.98,11.90,11.80,11.50,11.00,10.50\r\n",
            "Output Sequence Length,2.00,1.00,3.00,2.98,2.90,2.80,2.50,2.00,1.50\r\n",
            "Input Sequence Length,6.00,5.00,7.00,6.98,6.90,6.80,6.50,6.00,5.50\r\n",
            "\r\n",
            "Metric,Value\r\n",
            "Output Token Throughput (per sec),456.00\r\n",
            "Request Throughput (per sec),123.00\r\n",
            "Request Count (count),3.00\r\n",
            "\r\n",
            "Metric,GPU,avg,min,max,p99,p95,p90,p75,p50,p25\r\n",
            "GPU Power Usage (W),gpu0,45.85,45.20,46.50,46.49,46.44,46.37,46.17,45.85,45.53\r\n",
            "Energy Consumption (MJ),gpu0,0.55,0.50,0.60,0.60,0.59,0.59,0.57,0.55,0.53\r\n",
            "GPU Utilization (%),gpu0,77.50,75.00,80.00,79.95,79.75,79.50,78.75,77.50,76.25\r\n",
            "GPU Memory Used (GB),gpu0,4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00\r\n",
            "\r\n",
            "Metric,GPU,Value\r\n",
            "GPU Power Limit (W),gpu0,250.00\r\n",
            "Total GPU Memory (GB),gpu0,8.00\r\n",
        ]

        expected_filename = "profile_export_genai_perf.csv"
        returned_data = [
            data
            for filename, data in mock_read_write
            if os.path.basename(filename) == expected_filename
        ]

        assert returned_data == expected_content

    def test_missing_data(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch, llm_metrics: LLMMetrics
    ) -> None:
        """
        Test if missing data does not throw an error and are marked as "N/A".
        """
        argv = [
            "genai-perf",
            "profile",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "chat",
            "--profile-export-file",
            "custom_export.json",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        stats = Statistics(metrics=llm_metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.artifact_dir = Path(".")
        config.args = args

        # Missing data
        del config.stats["request_latency"]["avg"]
        del config.stats["output_sequence_length"]["max"]
        del config.stats["input_sequence_length"]

        exporter = CsvExporter(config)
        exporter.export()

        expected_filename = f"custom_export_genai_perf.csv"
        expected_content = [
            "Metric,avg,min,max,p99,p95,p90,p75,p50,p25\r\n",
            "Request Latency (ms),N/A,4.00,6.00,5.98,5.90,5.80,5.50,5.00,4.50\r\n",
            "Output Sequence Length,2.00,1.00,N/A,2.98,2.90,2.80,2.50,2.00,1.50\r\n",
            "Input Sequence Length,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A\r\n",
            "\r\n",
            "Metric,Value\r\n",
            "Output Token Throughput (per sec),456.00\r\n",
            "Request Throughput (per sec),123.00\r\n",
            "Request Count (count),3.00\r\n",
        ]
        returned_data = [
            data for filename, data in mock_read_write if filename == expected_filename
        ]

        assert returned_data == expected_content
