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


from unittest.mock import patch

import pytest
from genai_perf import parser
from genai_perf.export_data.console_exporter import ConsoleExporter
from genai_perf.export_data.exporter_config import ExporterConfig
from genai_perf.metrics import (
    ImageRetrievalMetrics,
    LLMMetrics,
    Metrics,
    Statistics,
    TelemetryMetrics,
    TelemetryStatistics,
)


class TestConsoleExporter:

    def test_streaming_llm_output(self, monkeypatch, capsys) -> None:
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

        metrics = LLMMetrics(
            request_throughputs=[123],
            request_latencies=[4, 5, 6],
            time_to_first_tokens=[7, 8, 9],
            inter_token_latencies=[10, 11, 12],
            output_token_throughputs=[456],
            output_sequence_lengths=[1, 2, 3],
            input_sequence_lengths=[5, 6, 7],
        )
        stats = Statistics(metrics=metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.args = args

        exporter = ConsoleExporter(config)
        exporter.export()

        expected_content = (
            "                        NVIDIA GenAI-Perf | LLM Metrics                         \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┓\n"
            "┃                         Statistic ┃  avg ┃  min ┃  max ┃  p99 ┃   p90 ┃  p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━┩\n"
            "│          Time to first token (ms) │ 8.00 │ 7.00 │ 9.00 │ 8.98 │  8.80 │ 8.50 │\n"
            "│          Inter token latency (ms) │ 11.… │ 10.… │ 12.… │ 11.… │ 11.80 │ 11.… │\n"
            "│              Request latency (ms) │ 5.00 │ 4.00 │ 6.00 │ 5.98 │  5.80 │ 5.50 │\n"
            "│            Output sequence length │ 2.00 │ 1.00 │ 3.00 │ 2.98 │  2.80 │ 2.50 │\n"
            "│             Input sequence length │ 6.00 │ 5.00 │ 7.00 │ 6.98 │  6.80 │ 6.50 │\n"
            "│ Output token throughput (per sec) │ 456… │  N/A │  N/A │  N/A │   N/A │  N/A │\n"
            "│      Request throughput (per sec) │ 123… │  N/A │  N/A │  N/A │   N/A │  N/A │\n"
            "└───────────────────────────────────┴──────┴──────┴──────┴──────┴───────┴──────┘\n"
        )

        returned_data = capsys.readouterr().out
        assert returned_data == expected_content

    def test_nonstreaming_llm_output(self, monkeypatch, capsys) -> None:
        argv = [
            "genai-perf",
            "profile",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "chat",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        metrics = LLMMetrics(
            request_throughputs=[123],
            request_latencies=[4, 5, 6],
            time_to_first_tokens=[4, 5, 6],  # same as request_latency
            inter_token_latencies=[],  # no ITL
            output_token_throughputs=[456],
            output_sequence_lengths=[1, 2, 3],
            input_sequence_lengths=[5, 6, 7],
        )
        stats = Statistics(metrics=metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.args = args

        exporter = ConsoleExporter(config)
        exporter.export()

        # No TTFT and ITL in the output
        expected_content = (
            "                        NVIDIA GenAI-Perf | LLM Metrics                         \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓\n"
            "┃                         Statistic ┃   avg ┃  min ┃  max ┃  p99 ┃  p90 ┃  p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩\n"
            "│              Request latency (ms) │  5.00 │ 4.00 │ 6.00 │ 5.98 │ 5.80 │ 5.50 │\n"
            "│            Output sequence length │  2.00 │ 1.00 │ 3.00 │ 2.98 │ 2.80 │ 2.50 │\n"
            "│             Input sequence length │  6.00 │ 5.00 │ 7.00 │ 6.98 │ 6.80 │ 6.50 │\n"
            "│ Output token throughput (per sec) │ 456.… │  N/A │  N/A │  N/A │  N/A │  N/A │\n"
            "│      Request throughput (per sec) │ 123.… │  N/A │  N/A │  N/A │  N/A │  N/A │\n"
            "└───────────────────────────────────┴───────┴──────┴──────┴──────┴──────┴──────┘\n"
        )

        returned_data = capsys.readouterr().out
        assert returned_data == expected_content

    def test_embedding_output(self, monkeypatch, capsys) -> None:
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
        config.args = args

        exporter = ConsoleExporter(config)
        exporter.export()

        expected_content = (
            "                   NVIDIA GenAI-Perf | Embeddings Metrics                   \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓\n"
            "┃                    Statistic ┃    avg ┃  min ┃  max ┃  p99 ┃  p90 ┃  p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩\n"
            "│         Request latency (ms) │   5.00 │ 4.00 │ 6.00 │ 5.98 │ 5.80 │ 5.50 │\n"
            "│ Request throughput (per sec) │ 123.00 │  N/A │  N/A │  N/A │  N/A │  N/A │\n"
            "└──────────────────────────────┴────────┴──────┴──────┴──────┴──────┴──────┘\n"
        )

        returned_data = capsys.readouterr().out
        assert returned_data == expected_content

    def test_valid_goodput(self, monkeypatch, capsys) -> None:
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
        config.args = args

        exporter = ConsoleExporter(config)
        exporter.export()

        expected_content = (
            "                        NVIDIA GenAI-Perf | LLM Metrics                         \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┓\n"
            "┃                         Statistic ┃  avg ┃  min ┃  max ┃  p99 ┃   p90 ┃  p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━┩\n"
            "│          Time to first token (ms) │ 8.00 │ 7.00 │ 9.00 │ 8.98 │  8.80 │ 8.50 │\n"
            "│          Inter token latency (ms) │ 11.… │ 10.… │ 12.… │ 11.… │ 11.80 │ 11.… │\n"
            "│              Request latency (ms) │ 5.00 │ 4.00 │ 6.00 │ 5.98 │  5.80 │ 5.50 │\n"
            "│            Output sequence length │ 2.00 │ 1.00 │ 3.00 │ 2.98 │  2.80 │ 2.50 │\n"
            "│             Input sequence length │ 6.00 │ 5.00 │ 7.00 │ 6.98 │  6.80 │ 6.50 │\n"
            "│ Output token throughput (per sec) │ 456… │  N/A │  N/A │  N/A │   N/A │  N/A │\n"
            "│      Request throughput (per sec) │ 123… │  N/A │  N/A │  N/A │   N/A │  N/A │\n"
            "│         Request goodput (per sec) │ 100… │  N/A │  N/A │  N/A │   N/A │  N/A │\n"
            "└───────────────────────────────────┴──────┴──────┴──────┴──────┴───────┴──────┘\n"
        )
        returned_data = capsys.readouterr().out
        assert returned_data == expected_content

    def test_invalid_goodput_output(self, monkeypatch, capsys) -> None:
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
        config.args = args

        exporter = ConsoleExporter(config)
        exporter.export()

        expected_content = (
            "                        NVIDIA GenAI-Perf | LLM Metrics                         \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┓\n"
            "┃                         Statistic ┃  avg ┃  min ┃  max ┃  p99 ┃   p90 ┃  p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━┩\n"
            "│          Time to first token (ms) │ 8.00 │ 7.00 │ 9.00 │ 8.98 │  8.80 │ 8.50 │\n"
            "│          Inter token latency (ms) │ 11.… │ 10.… │ 12.… │ 11.… │ 11.80 │ 11.… │\n"
            "│              Request latency (ms) │ 5.00 │ 4.00 │ 6.00 │ 5.98 │  5.80 │ 5.50 │\n"
            "│            Output sequence length │ 2.00 │ 1.00 │ 3.00 │ 2.98 │  2.80 │ 2.50 │\n"
            "│             Input sequence length │ 6.00 │ 5.00 │ 7.00 │ 6.98 │  6.80 │ 6.50 │\n"
            "│ Output token throughput (per sec) │ 456… │  N/A │  N/A │  N/A │   N/A │  N/A │\n"
            "│      Request throughput (per sec) │ 123… │  N/A │  N/A │  N/A │   N/A │  N/A │\n"
            "│         Request goodput (per sec) │ -1.… │  N/A │  N/A │  N/A │   N/A │  N/A │\n"
            "└───────────────────────────────────┴──────┴──────┴──────┴──────┴───────┴──────┘\n"
        )
        returned_data = capsys.readouterr().out
        assert returned_data == expected_content

    @patch(
        "genai_perf.export_data.console_exporter.ConsoleExporter._construct_table",
        return_value=None,
    )
    @pytest.mark.parametrize(
        "endpoint_type, metrics, expected_title",
        [
            ("chat", LLMMetrics(), "NVIDIA GenAI-Perf | LLM Metrics"),
            ("embeddings", Metrics(), "NVIDIA GenAI-Perf | Embeddings Metrics"),
            ("rankings", Metrics(), "NVIDIA GenAI-Perf | Rankings Metrics"),
            ("vision", LLMMetrics(), "NVIDIA GenAI-Perf | VLM Metrics"),
            (
                "image_retrieval",
                ImageRetrievalMetrics(),
                "NVIDIA GenAI-Perf | Image Retrieval Metrics",
            ),
        ],
    )
    def test_console_title(
        self, mock_table, endpoint_type, metrics, expected_title, monkeypatch, capsys
    ) -> None:
        argv = [
            "genai-perf",
            "profile",
            "-m",
            "model_name",
            "--service-kind",
            "openai",
            "--endpoint-type",
            endpoint_type,
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        stats = Statistics(metrics=metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.metrics = stats.metrics
        config.args = args

        exporter = ConsoleExporter(config)
        exporter.export()

        returned_data = capsys.readouterr().out
        assert expected_title in returned_data

    def test_valid_telemetry_verbose(self, monkeypatch, capsys) -> None:
        argv = [
            "genai-perf",
            "profile",
            "-m",
            "model_name",
            "--service-kind",
            "triton",
            "--streaming",
            "--server-metrics-url",
            "http://tritonmetrics:8002/metrics",
            "--verbose",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args, _ = parser.parse_args()

        metrics = LLMMetrics(
            request_throughputs=[123],
            request_latencies=[4, 5, 6],
            time_to_first_tokens=[7, 8, 9],
            inter_token_latencies=[10, 11, 12],
            output_token_throughputs=[456],
            output_sequence_lengths=[1, 2, 3],
            input_sequence_lengths=[5, 6, 7],
            request_goodputs=[100],
        )

        telemetry_metrics = TelemetryMetrics(
            gpu_power_usage={"gpu0": [45.2, 46.5]},
            gpu_power_limit={"gpu0": [250.0, 250.0]},
            energy_consumption={"gpu0": [0.5, 0.6]},
            gpu_utilization={"gpu0": [75.0, 80.0]},
            total_gpu_memory={"gpu0": [8.0, 8.0]},
            gpu_memory_used={"gpu0": [4.0, 4.0]},
        )
        stats = Statistics(metrics=metrics)
        telemetry_stats = TelemetryStatistics(telemetry_metrics)

        config = ExporterConfig()
        config.stats = stats.stats_dict
        config.telemetry_stats = telemetry_stats.stats_dict
        config.metrics = stats.metrics
        config.args = args

        exporter = ConsoleExporter(config)
        exporter.export()

        expected_content = (
            "                        NVIDIA GenAI-Perf | LLM Metrics                         \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┓\n"
            "┃                         Statistic ┃  avg ┃  min ┃  max ┃  p99 ┃   p90 ┃  p75 ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━┩\n"
            "│          Time to first token (ms) │ 8.00 │ 7.00 │ 9.00 │ 8.98 │  8.80 │ 8.50 │\n"
            "│          Inter token latency (ms) │ 11.… │ 10.… │ 12.… │ 11.… │ 11.80 │ 11.… │\n"
            "│              Request latency (ms) │ 5.00 │ 4.00 │ 6.00 │ 5.98 │  5.80 │ 5.50 │\n"
            "│            Output sequence length │ 2.00 │ 1.00 │ 3.00 │ 2.98 │  2.80 │ 2.50 │\n"
            "│             Input sequence length │ 6.00 │ 5.00 │ 7.00 │ 6.98 │  6.80 │ 6.50 │\n"
            "│ Output token throughput (per sec) │ 456… │  N/A │  N/A │  N/A │   N/A │  N/A │\n"
            "│      Request throughput (per sec) │ 123… │  N/A │  N/A │  N/A │   N/A │  N/A │\n"
            "└───────────────────────────────────┴──────┴──────┴──────┴──────┴───────┴──────┘\n"
            "                NVIDIA GenAI-Perf | Power Metrics                \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃                                                               ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n"
            "│                      GPU Power Usage (W)                      │\n"
            "│ ┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓ │\n"
            "│ ┃ GPU Index ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃ │\n"
            "│ ┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩ │\n"
            "│ │      gpu0 │ 45.85 │ 45.20 │ 46.50 │ 46.49 │ 46.37 │ 46.17 │ │\n"
            "│ └───────────┴───────┴───────┴───────┴───────┴───────┴───────┘ │\n"
            "│                 GPU Power Limit (W)                           │\n"
            "│ ┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓          │\n"
            "│ ┃ GPU Index ┃    avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p75 ┃          │\n"
            "│ ┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩          │\n"
            "│ │      gpu0 │ 250.00 │ N/A │ N/A │ N/A │ N/A │ N/A │          │\n"
            "│ └───────────┴────────┴─────┴─────┴─────┴─────┴─────┘          │\n"
            "│                 Energy Consumption (MJ)                       │\n"
            "│ ┏━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓       │\n"
            "│ ┃ GPU Index ┃  avg ┃  min ┃  max ┃  p99 ┃  p90 ┃  p75 ┃       │\n"
            "│ ┡━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩       │\n"
            "│ │      gpu0 │ 0.55 │ 0.50 │ 0.60 │ 0.60 │ 0.59 │ 0.57 │       │\n"
            "│ └───────────┴──────┴──────┴──────┴──────┴──────┴──────┘       │\n"
            "└───────────────────────────────────────────────────────────────┘\n"
            "            NVIDIA GenAI-Perf | Memory Metrics             \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃                                                         ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n"
            "│                  GPU Memory Used (GB)                   │\n"
            "│ ┏━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓ │\n"
            "│ ┃ GPU Index ┃  avg ┃  min ┃  max ┃  p99 ┃  p90 ┃  p75 ┃ │\n"
            "│ ┡━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩ │\n"
            "│ │      gpu0 │ 4.00 │ 4.00 │ 4.00 │ 4.00 │ 4.00 │ 4.00 │ │\n"
            "│ └───────────┴──────┴──────┴──────┴──────┴──────┴──────┘ │\n"
            "│               Total GPU Memory (GB)                     │\n"
            "│ ┏━━━━━━━━━━━┳━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓      │\n"
            "│ ┃ GPU Index ┃  avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p75 ┃      │\n"
            "│ ┡━━━━━━━━━━━╇━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩      │\n"
            "│ │      gpu0 │ 8.00 │ N/A │ N/A │ N/A │ N/A │ N/A │      │\n"
            "│ └───────────┴──────┴─────┴─────┴─────┴─────┴─────┘      │\n"
            "└─────────────────────────────────────────────────────────┘\n"
            "       NVIDIA GenAI-Perf | Utilization Metrics       \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃                                                   ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n"
            "│                GPU Utilization (%)                │\n"
            "│ ┏━━━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓ │\n"
            "│ ┃ GPU Index ┃ avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p75 ┃ │\n"
            "│ ┡━━━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩ │\n"
            "│ │      gpu0 │  78 │  75 │  80 │  80 │  80 │  79 │ │\n"
            "│ └───────────┴─────┴─────┴─────┴─────┴─────┴─────┘ │\n"
            "└───────────────────────────────────────────────────┘\n"
        )

        returned_data = capsys.readouterr().out
        assert returned_data == expected_content
