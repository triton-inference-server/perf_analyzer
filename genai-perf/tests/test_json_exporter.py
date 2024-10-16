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

import json
import os
from io import StringIO
from typing import Any, List, Tuple

import genai_perf.parser as parser
import pytest
from genai_perf.export_data.exporter_config import ExporterConfig
from genai_perf.export_data.json_exporter import JsonExporter


class TestJsonExporter:
    stats = {
        "request_throughput": {"unit": "requests/sec", "avg": "7"},
        "request_latency": {
            "unit": "ms",
            "avg": 1,
            "p99": 2,
            "p95": 3,
            "p90": 4,
            "p75": 5,
            "p50": 6,
            "p25": 7,
            "max": 8,
            "min": 9,
            "std": 0,
        },
        "time_to_first_token": {
            "unit": "ms",
            "avg": 11,
            "p99": 12,
            "p95": 13,
            "p90": 14,
            "p75": 15,
            "p50": 16,
            "p25": 17,
            "max": 18,
            "min": 19,
            "std": 10,
        },
        "inter_token_latency": {
            "unit": "ms",
            "avg": 21,
            "p99": 22,
            "p95": 23,
            "p90": 24,
            "p75": 25,
            "p50": 26,
            "p25": 27,
            "max": 28,
            "min": 29,
            "std": 20,
        },
        "output_token_throughput": {
            "unit": "tokens/sec",
            "avg": 31,
        },
        "output_token_throughput_per_request": {
            "unit": "tokens/sec",
            "avg": 41,
            "p99": 42,
            "p95": 43,
            "p90": 44,
            "p75": 45,
            "p50": 46,
            "p25": 47,
            "max": 48,
            "min": 49,
            "std": 40,
        },
        "output_sequence_length": {
            "unit": "tokens",
            "avg": 51,
            "p99": 52,
            "p95": 53,
            "p90": 54,
            "p75": 55,
            "p50": 56,
            "p25": 57,
            "max": 58,
            "min": 59,
            "std": 50,
        },
        "input_sequence_length": {
            "unit": "tokens",
            "avg": 61,
            "p99": 62,
            "p95": 63,
            "p90": 64,
            "p75": 65,
            "p50": 66,
            "p25": 67,
            "max": 68,
            "min": 69,
            "std": 60,
        },
    }

    expected_json_output = """
      {
        "request_throughput": {
          "unit": "requests/sec",
          "avg": "7"
          },
          "request_latency": {
              "unit": "ms",
              "avg": 1,
              "p99": 2,
              "p95": 3,
              "p90": 4,
              "p75": 5,
              "p50": 6,
              "p25": 7,
              "max": 8,
              "min": 9,
              "std": 0
          },
          "time_to_first_token": {
              "unit": "ms",
              "avg": 11,
              "p99": 12,
              "p95": 13,
              "p90": 14,
              "p75": 15,
              "p50": 16,
              "p25": 17,
              "max": 18,
              "min": 19,
              "std": 10
          },
          "inter_token_latency": {
              "unit": "ms",
              "avg": 21,
              "p99": 22,
              "p95": 23,
              "p90": 24,
              "p75": 25,
              "p50": 26,
              "p25": 27,
              "max": 28,
              "min": 29,
              "std": 20
          },
          "output_token_throughput": {
              "unit": "tokens/sec",
              "avg": 31
          },
          "output_token_throughput_per_request": {
              "unit": "tokens/sec",
              "avg": 41,
              "p99": 42,
              "p95": 43,
              "p90": 44,
              "p75": 45,
              "p50": 46,
              "p25": 47,
              "max": 48,
              "min": 49,
              "std": 40
          },
          "output_sequence_length": {
              "unit": "tokens",
              "avg": 51,
              "p99": 52,
              "p95": 53,
              "p90": 54,
              "p75": 55,
              "p50": 56,
              "p25": 57,
              "max": 58,
              "min": 59,
              "std": 50
          },
          "input_sequence_length": {
              "unit": "tokens",
              "avg": 61,
              "p99": 62,
              "p95": 63,
              "p90": 64,
              "p75": 65,
              "p50": 66,
              "p25": 67,
              "max": 68,
              "min": 69,
              "std": 60
          },
        "input_config": {
          "model": ["gpt2_vllm"],
          "formatted_model_name": "gpt2_vllm",
          "model_selection_strategy": "round_robin",
          "backend": "vllm",
          "batch_size_image": 1,
          "batch_size_text": 1,
          "endpoint": null,
          "endpoint_type": null,
          "service_kind": "triton",
          "server_metrics_url": null,
          "streaming": true,
          "u": null,
          "num_prompts": 100,
          "output_tokens_mean": -1,
          "output_tokens_mean_deterministic": false,
          "output_tokens_stddev": 0,
          "random_seed": 0,
          "synthetic_input_tokens_mean": 550,
          "synthetic_input_tokens_stddev": 0,
          "image_width_mean": 100,
          "image_width_stddev": 0,
          "image_height_mean": 100,
          "image_height_stddev": 0,
          "image_format": null,
          "concurrency": 1,
          "measurement_interval": 10000,
          "request_rate": null,
          "stability_percentage": 999,
          "generate_plots": false,
          "profile_export_file": "artifacts/gpt2_vllm-triton-vllm-concurrency1/profile_export.json",
          "artifact_dir": "artifacts/gpt2_vllm-triton-vllm-concurrency1",
          "tokenizer": "hf-internal-testing/llama-tokenizer",
          "tokenizer_revision": "main",
          "tokenizer_trust_remote_code": false,
          "verbose": false,
          "goodput": null,
          "subcommand": "profile",
          "prompt_source": "synthetic",
          "extra_inputs": {
            "max_tokens": 256,
            "ignore_eos": true
          }
        }
      }
    """

    @pytest.fixture
    def mock_read_write(self, monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str]]:
        """
        This function will mock the open function for specific files.
        """

        written_data = []

        def custom_open(filename, *args, **kwargs):
            def write(self: Any, content: str) -> int:
                print(f"Writing to {filename}")
                written_data.append((str(filename), content))
                return len(content)

            tmp_file = StringIO()
            tmp_file.write = write.__get__(tmp_file)
            return tmp_file

        monkeypatch.setattr("builtins.open", custom_open)

        return written_data

    def test_generate_json(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "gpt2_vllm",
            "--backend",
            "vllm",
            "--streaming",
            "--extra-inputs",
            "max_tokens:256",
            "--extra-inputs",
            "ignore_eos:true",
        ]
        monkeypatch.setattr("sys.argv", cli_cmd)
        args, _ = parser.parse_args()
        config = ExporterConfig()
        config.stats = self.stats
        config.args = args
        config.extra_inputs = parser.get_extra_inputs_as_dict(args)
        config.artifact_dir = args.artifact_dir
        json_exporter = JsonExporter(config)
        assert json_exporter._stats_and_args == json.loads(self.expected_json_output)
        json_exporter.export()
        expected_filename = "profile_export_genai_perf.json"
        written_data = [
            data
            for filename, data in mock_read_write
            if os.path.basename(filename) == expected_filename
        ]

        assert len(written_data) == 1
        assert json.loads(written_data[0]) == json.loads(self.expected_json_output)

    def test_generate_json_custom_export(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        artifacts_dir = "artifacts/gpt2_vllm-triton-vllm-concurrency1"
        custom_filename = "custom_export.json"
        expected_filename = f"{artifacts_dir}/custom_export_genai_perf.json"
        expected_profile_filename = f"{artifacts_dir}/custom_export.json"
        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "gpt2_vllm",
            "--backend",
            "vllm",
            "--streaming",
            "--extra-inputs",
            "max_tokens:256",
            "--extra-inputs",
            "ignore_eos:true",
            "--profile-export-file",
            custom_filename,
        ]
        monkeypatch.setattr("sys.argv", cli_cmd)
        args, _ = parser.parse_args()
        config = ExporterConfig()
        config.stats = self.stats
        config.args = args
        config.extra_inputs = parser.get_extra_inputs_as_dict(args)
        config.artifact_dir = args.artifact_dir
        json_exporter = JsonExporter(config)
        json_exporter.export()
        written_data = [
            data for filename, data in mock_read_write if filename == expected_filename
        ]

        assert len(written_data) == 1
        expected_json_output = json.loads(self.expected_json_output)
        expected_json_output["input_config"][
            "profile_export_file"
        ] = expected_profile_filename
        assert json.loads(written_data[0]) == expected_json_output

    def test_valid_goodput_json_output(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        valid_goodput_stats = {
            "request_throughput": {"unit": "requests/sec", "avg": "7"},
            "request_latency": {
                "unit": "ms",
                "avg": 1,
                "p99": 2,
                "p95": 3,
                "p90": 4,
                "p75": 5,
                "p50": 6,
                "p25": 7,
                "max": 8,
                "min": 9,
                "std": 0,
            },
            "request_goodput": {
                "unit": "requests/sec",
                "avg": "5",
            },
            "time_to_first_token": {
                "unit": "ms",
                "avg": 11,
                "p99": 12,
                "p95": 13,
                "p90": 14,
                "p75": 15,
                "p50": 16,
                "p25": 17,
                "max": 18,
                "min": 19,
                "std": 10,
            },
            "inter_token_latency": {
                "unit": "ms",
                "avg": 21,
                "p99": 22,
                "p95": 23,
                "p90": 24,
                "p75": 25,
                "p50": 26,
                "p25": 27,
                "max": 28,
                "min": 29,
                "std": 20,
            },
            "output_token_throughput": {
                "unit": "tokens/sec",
                "avg": 31,
            },
            "output_token_throughput_per_request": {
                "unit": "tokens/sec",
                "avg": 41,
                "p99": 42,
                "p95": 43,
                "p90": 44,
                "p75": 45,
                "p50": 46,
                "p25": 47,
                "max": 48,
                "min": 49,
                "std": 40,
            },
            "output_sequence_length": {
                "unit": "tokens",
                "avg": 51,
                "p99": 52,
                "p95": 53,
                "p90": 54,
                "p75": 55,
                "p50": 56,
                "p25": 57,
                "max": 58,
                "min": 59,
                "std": 50,
            },
            "input_sequence_length": {
                "unit": "tokens",
                "avg": 61,
                "p99": 62,
                "p95": 63,
                "p90": 64,
                "p75": 65,
                "p50": 66,
                "p25": 67,
                "max": 68,
                "min": 69,
                "std": 60,
            },
        }

        expected_valid_goodput_json_output = """
            {
                "unit": "requests/sec",
                "avg": "5"
            }
        """

        expected_valid_goodput_json_config = """
            {
                "time_to_first_token": 8.0,
                "inter_token_latency": 2.0,
                "output_token_throughput_per_request": 650.0
            }
        """

        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "gpt2_vllm",
            "--backend",
            "vllm",
            "--streaming",
            "--extra-inputs",
            "max_tokens:256",
            "--extra-inputs",
            "ignore_eos:true",
            "--goodput",
            "time_to_first_token:8.0",
            "inter_token_latency:2.0",
            "output_token_throughput_per_request:650.0",
        ]
        monkeypatch.setattr("sys.argv", cli_cmd)
        args, _ = parser.parse_args()
        config = ExporterConfig()
        config.stats = valid_goodput_stats
        config.args = args
        config.extra_inputs = parser.get_extra_inputs_as_dict(args)
        config.artifact_dir = args.artifact_dir
        json_exporter = JsonExporter(config)
        assert json_exporter._stats_and_args["request_goodput"] == json.loads(
            expected_valid_goodput_json_output
        )
        assert json_exporter._stats_and_args["input_config"]["goodput"] == json.loads(
            expected_valid_goodput_json_config
        )

        json_exporter.export()
        expected_filename = "profile_export_genai_perf.json"
        written_data = [
            data
            for filename, data in mock_read_write
            if os.path.basename(filename) == expected_filename
        ]

        assert len(written_data) == 1
        output_data_dict = json.loads(written_data[0])

        assert output_data_dict["request_goodput"] == json.loads(
            expected_valid_goodput_json_output
        )

        assert output_data_dict["input_config"]["goodput"] == json.loads(
            expected_valid_goodput_json_config
        )

    def test_invalid_goodput_json_output(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:
        invalid_goodput_stats = {
            "request_throughput": {"unit": "requests/sec", "avg": "7"},
            "request_latency": {
                "unit": "ms",
                "avg": 1,
                "p99": 2,
                "p95": 3,
                "p90": 4,
                "p75": 5,
                "p50": 6,
                "p25": 7,
                "max": 8,
                "min": 9,
                "std": 0,
            },
            "request_goodput": {
                "unit": "requests/sec",
                "avg": "-1.0",
            },
            "time_to_first_token": {
                "unit": "ms",
                "avg": 11,
                "p99": 12,
                "p95": 13,
                "p90": 14,
                "p75": 15,
                "p50": 16,
                "p25": 17,
                "max": 18,
                "min": 19,
                "std": 10,
            },
            "inter_token_latency": {
                "unit": "ms",
                "avg": 21,
                "p99": 22,
                "p95": 23,
                "p90": 24,
                "p75": 25,
                "p50": 26,
                "p25": 27,
                "max": 28,
                "min": 29,
                "std": 20,
            },
            "output_token_throughput": {
                "unit": "tokens/sec",
                "avg": 31,
            },
            "output_token_throughput_per_request": {
                "unit": "tokens/sec",
                "avg": 41,
                "p99": 42,
                "p95": 43,
                "p90": 44,
                "p75": 45,
                "p50": 46,
                "p25": 47,
                "max": 48,
                "min": 49,
                "std": 40,
            },
            "output_sequence_length": {
                "unit": "tokens",
                "avg": 51,
                "p99": 52,
                "p95": 53,
                "p90": 54,
                "p75": 55,
                "p50": 56,
                "p25": 57,
                "max": 58,
                "min": 59,
                "std": 50,
            },
            "input_sequence_length": {
                "unit": "tokens",
                "avg": 61,
                "p99": 62,
                "p95": 63,
                "p90": 64,
                "p75": 65,
                "p50": 66,
                "p25": 67,
                "max": 68,
                "min": 69,
                "std": 60,
            },
        }

        expected_invalid_goodput_json_output = """
            {
                "unit": "requests/sec",
                "avg": "-1.0"
            }
        """
        expected_invalid_goodput_json_config = """
            {
                "time_to_first_tokens": 8.0,
                "inter_token_latencies": 2.0,
                "output_token_throughputs_per_requesdt": 650.0
            }
        """
        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "gpt2_vllm",
            "--backend",
            "vllm",
            "--streaming",
            "--extra-inputs",
            "max_tokens:256",
            "--extra-inputs",
            "ignore_eos:true",
            "--goodput",
            "time_to_first_tokens:8.0",
            "inter_token_latencies:2.0",
            "output_token_throughputs_per_requesdt:650.0",
        ]
        monkeypatch.setattr("sys.argv", cli_cmd)
        args, _ = parser.parse_args()
        config = ExporterConfig()
        config.stats = invalid_goodput_stats
        config.args = args
        config.extra_inputs = parser.get_extra_inputs_as_dict(args)
        config.artifact_dir = args.artifact_dir
        json_exporter = JsonExporter(config)
        assert json_exporter._stats_and_args["request_goodput"] == json.loads(
            expected_invalid_goodput_json_output
        )
        print(json_exporter._stats_and_args["input_config"]["goodput"])
        assert json_exporter._stats_and_args["input_config"]["goodput"] == json.loads(
            expected_invalid_goodput_json_config
        )
        json_exporter.export()
        expected_filename = "profile_export_genai_perf.json"

        written_data = [
            data
            for filename, data in mock_read_write
            if os.path.basename(filename) == expected_filename
        ]

        assert len(written_data) == 1
        output_data_dict = json.loads(written_data[0])

        assert output_data_dict["request_goodput"] == json.loads(
            expected_invalid_goodput_json_output
        )

        assert output_data_dict["input_config"]["goodput"] == json.loads(
            expected_invalid_goodput_json_config
        )

    def test_triton_telemetry_output(
        self, monkeypatch, mock_read_write: pytest.MonkeyPatch
    ) -> None:

        telemetry_stats = {
            "gpu_power_usage": {
                "unit": "W",
                "gpu0": {
                    "avg": 80.30575675675676,
                    "p25": 82.569,
                    "p50": 83.597,
                    "p75": 84.485,
                    "p90": 84.589,
                    "p95": 84.7184,
                    "p99": 84.772,
                    "min": 23.858,
                    "max": 84.772,
                },
            },
            "gpu_power_limit": {"unit": "W", "gpu0": {"avg": 300.0}},
            "energy_consumption": {
                "unit": "MJ",
                "gpu0": {
                    "avg": 2.154032905081084,
                    "p25": 2.1533188160000027,
                    "p50": 2.154057423000003,
                    "p75": 2.154666464000003,
                    "p90": 2.155118764000003,
                    "p95": 2.155270593000003,
                    "p99": 2.1553677661200026,
                    "min": 2.1527738520000033,
                    "max": 2.1554224260000026,
                },
            },
            "gpu_utilization": {
                "unit": "%",
                "gpu0": {
                    "avg": 8.72972972972973,
                    "p25": 9.0,
                    "p50": 9.0,
                    "p75": 9.0,
                    "p90": 10.0,
                    "p95": 10.0,
                    "p99": 10.0,
                    "min": 0.0,
                    "max": 10.0,
                },
            },
            "total_gpu_memory": {"unit": "GB", "gpu0": {"avg": 51.52702464}},
            "gpu_memory_used": {
                "unit": "GB",
                "gpu0": {
                    "avg": 26.052919296000006,
                    "p25": 26.052919296000002,
                    "p50": 26.052919296000002,
                    "p75": 26.052919296000002,
                    "p90": 26.052919296000002,
                    "p95": 26.052919296000002,
                    "p99": 26.052919296000002,
                    "min": 26.052919296000002,
                    "max": 26.052919296000002,
                },
            },
        }

        expected_telemetry_json_output = """
            {
                "gpu_power_usage": {
                    "unit": "W",
                    "gpu0": {
                        "avg": 80.30575675675676,
                        "p25": 82.569,
                        "p50": 83.597,
                        "p75": 84.485,
                        "p90": 84.589,
                        "p95": 84.7184,
                        "p99": 84.772,
                        "min": 23.858,
                        "max": 84.772
                    }
                },
                "gpu_power_limit": {
                    "unit": "W",
                    "gpu0": {
                        "avg": 300.0
                    }
                },
                "energy_consumption": {
                    "unit": "MJ",
                    "gpu0": {
                        "avg": 2.154032905081084,
                        "p25": 2.1533188160000027,
                        "p50": 2.154057423000003,
                        "p75": 2.154666464000003,
                        "p90": 2.155118764000003,
                        "p95": 2.155270593000003,
                        "p99": 2.1553677661200026,
                        "min": 2.1527738520000033,
                        "max": 2.1554224260000026
                    }
                },
                "gpu_utilization": {
                    "unit": "%",
                    "gpu0": {
                        "avg": 8.72972972972973,
                        "p25": 9.0,
                        "p50": 9.0,
                        "p75": 9.0,
                        "p90": 10.0,
                        "p95": 10.0,
                        "p99": 10.0,
                        "min": 0.0,
                        "max": 10.0
                    }
                },
                "total_gpu_memory": {
                    "unit": "GB",
                    "gpu0": {
                        "avg": 51.52702464
                    }
                },
                "gpu_memory_used": {
                    "unit": "GB",
                    "gpu0": {
                        "avg": 26.052919296000006,
                        "p25": 26.052919296000002,
                        "p50": 26.052919296000002,
                        "p75": 26.052919296000002,
                        "p90": 26.052919296000002,
                        "p95": 26.052919296000002,
                        "p99": 26.052919296000002,
                        "min": 26.052919296000002,
                        "max": 26.052919296000002
                    }
                }
            }
        """

        cli_cmd = [
            "genai-perf",
            "profile",
            "-m",
            "gpt2_vllm",
            "--service-kind",
            "triton",
            "--streaming",
            "--server-metrics-url",
            "http://tritonmetrics:8002/metrics",
        ]

        monkeypatch.setattr("sys.argv", cli_cmd)
        args, _ = parser.parse_args()
        config = ExporterConfig()
        config.stats = self.stats
        config.telemetry_stats = telemetry_stats
        config.args = args
        config.extra_inputs = parser.get_extra_inputs_as_dict(args)
        config.artifact_dir = args.artifact_dir
        json_exporter = JsonExporter(config)
        assert json_exporter._stats_and_args["telemetry_stats"] == json.loads(
            expected_telemetry_json_output
        )

        json_exporter.export()
        expected_filename = "profile_export_genai_perf.json"
        written_data = [
            data
            for filename, data in mock_read_write
            if os.path.basename(filename) == expected_filename
        ]

        assert len(written_data) == 1
        output_data_dict = json.loads(written_data[0])

        assert output_data_dict["telemetry_stats"] == json.loads(
            expected_telemetry_json_output
        )
