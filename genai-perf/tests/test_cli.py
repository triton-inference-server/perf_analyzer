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

import argparse
from pathlib import Path
from unittest.mock import patch

import genai_perf.logging as logging
import pytest
from genai_perf import __version__, parser
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.inputs.input_constants import (
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
)
from genai_perf.inputs.retrievers.synthetic_image_generator import ImageFormat

# TODO (TPA-1002): move AudioFormat to synthetic audio generator
# from genai_perf.inputs.retrievers.synthetic_audio_generator import AudioFormat
from genai_perf.parser import AudioFormat
from genai_perf.subcommand.common import get_extra_inputs_as_dict


class TestCLIArguments:
    # ================================================
    # PROFILE COMMAND
    # ================================================
    expected_help_output = (
        "CLI to profile LLMs and Generative AI models with Perf Analyzer"
    )
    expected_version_output = f"genai-perf {__version__}"

    @pytest.mark.parametrize(
        "args, expected_output",
        [
            (["-h"], expected_help_output),
            (["--help"], expected_help_output),
            (["--version"], expected_version_output),
        ],
    )
    def test_help_version_arguments_output_and_exit(
        self, monkeypatch, args, expected_output, capsys
    ):
        monkeypatch.setattr("sys.argv", ["genai-perf"] + args)

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        # Check that the exit was successful
        assert excinfo.value.code == 0

        # Capture that the correct message was displayed
        captured = capsys.readouterr()
        assert expected_output in captured.out

    @pytest.mark.parametrize(
        "arg, expected_attributes",
        [
            (
                ["--artifact-dir", "test_artifact_dir"],
                {"artifact_dir": Path("test_artifact_dir")},
            ),
            (
                [
                    "--batch-size-text",
                    "5",
                    "--endpoint-type",
                    "embeddings",
                    "--service-kind",
                    "openai",
                ],
                {"batch_size_text": 5},
            ),
            (
                [
                    "--batch-size-image",
                    "5",
                    "--endpoint-type",
                    "image_retrieval",
                    "--service-kind",
                    "openai",
                ],
                {"batch_size_image": 5},
            ),
            (
                [
                    "-b",
                    "5",
                    "--endpoint-type",
                    "embeddings",
                    "--service-kind",
                    "openai",
                ],
                {"batch_size_text": 5},
            ),
            (["--concurrency", "3"], {"concurrency": 3}),
            (
                ["--endpoint-type", "completions", "--service-kind", "openai"],
                {"endpoint": "v1/completions"},
            ),
            (
                ["--endpoint-type", "chat", "--service-kind", "openai"],
                {"endpoint": "v1/chat/completions"},
            ),
            (
                ["--endpoint-type", "multimodal", "--service-kind", "openai"],
                {"endpoint": "v1/chat/completions"},
            ),
            (
                ["--endpoint-type", "rankings", "--service-kind", "openai"],
                {"endpoint": "v1/ranking"},
            ),
            (
                ["--endpoint-type", "image_retrieval", "--service-kind", "openai"],
                {"endpoint": "v1/infer"},
            ),
            (
                [
                    "--endpoint-type",
                    "chat",
                    "--service-kind",
                    "openai",
                    "--endpoint",
                    "custom/address",
                ],
                {"endpoint": "custom/address"},
            ),
            (
                [
                    "--endpoint-type",
                    "chat",
                    "--service-kind",
                    "openai",
                    "--endpoint",
                    "   /custom/address",
                ],
                {"endpoint": "custom/address"},
            ),
            (
                [
                    "--endpoint-type",
                    "completions",
                    "--service-kind",
                    "openai",
                    "--endpoint",
                    "custom/address",
                ],
                {"endpoint": "custom/address"},
            ),
            (
                ["--extra-inputs", "test_key:test_value"],
                {"extra_inputs": ["test_key:test_value"]},
            ),
            (
                [
                    "--extra-inputs",
                    "test_key:5",
                    "--extra-inputs",
                    "another_test_key:6",
                ],
                {"extra_inputs": ["test_key:5", "another_test_key:6"]},
            ),
            (
                [
                    "--extra-inputs",
                    '{"name": "Wolverine","hobbies": ["hacking", "slashing"],"address": {"street": "1407 Graymalkin Lane, Salem Center","city": "NY"}}',
                ],
                {
                    "extra_inputs": [
                        '{"name": "Wolverine","hobbies": ["hacking", "slashing"],"address": {"street": "1407 Graymalkin Lane, Salem Center","city": "NY"}}'
                    ]
                },
            ),
            (["-H", "header_name:value"], {"header": ["header_name:value"]}),
            (["--header", "header_name:value"], {"header": ["header_name:value"]}),
            (
                ["--header", "header_name:value", "--header", "header_name_2:value_2"],
                {"header": ["header_name:value", "header_name_2:value_2"]},
            ),
            (["--measurement-interval", "100"], {"measurement_interval": 100}),
            (
                ["--model-selection-strategy", "random"],
                {"model_selection_strategy": ModelSelectionStrategy.RANDOM},
            ),
            (["--num-dataset-entries", "101"], {"num_dataset_entries": 101}),
            (["--num-prompts", "101"], {"num_dataset_entries": 101}),
            (["--num-prefix-prompts", "101"], {"num_prefix_prompts": 101}),
            (
                ["--output-tokens-mean", "6"],
                {"output_tokens_mean": 6},
            ),
            (
                ["--osl", "6"],
                {"output_tokens_mean": 6},
            ),
            (
                ["--output-tokens-mean", "6", "--output-tokens-stddev", "7"],
                {"output_tokens_stddev": 7},
            ),
            (
                ["--output-tokens-mean", "6", "--output-tokens-mean-deterministic"],
                {"output_tokens_mean_deterministic": True},
            ),
            (["-p", "100"], {"measurement_interval": 100}),
            (
                ["--profile-export-file", "test.json"],
                {"profile_export_file": Path("test.json")},
            ),
            (["--random-seed", "8"], {"random_seed": 8}),
            (["--request-count", "100"], {"request_count": 100}),
            (
                ["--grpc-method", "package.name.v1.ServiceName/MethodName"],
                {"grpc_method": "package.name.v1.ServiceName/MethodName"},
            ),
            (["--num-requests", "100"], {"request_count": 100}),
            (["--warmup-request-count", "100"], {"warmup_request_count": 100}),
            (["--num-warmup-requests", "100"], {"warmup_request_count": 100}),
            (["--request-rate", "9.0"], {"request_rate": 9.0}),
            (["-s", "99.5"], {"stability_percentage": 99.5}),
            (
                [
                    "--service-kind",
                    "dynamic_grpc",
                    "--grpc-method",
                    "package.name.v1.ServiceName/MethodName",
                ],
                {
                    "service_kind": "dynamic_grpc",
                    "endpoint_type": "dynamic_grpc",
                    "grpc_method": "package.name.v1.ServiceName/MethodName",
                },
            ),
            (["--service-kind", "triton"], {"service_kind": "triton"}),
            (
                ["--service-kind", "tensorrtllm_engine"],
                {"service_kind": "tensorrtllm_engine"},
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "chat"],
                {"service_kind": "openai", "endpoint": "v1/chat/completions"},
            ),
            (["--session-concurrency", "3"], {"session_concurrency": 3}),
            (["--session-turn-delay-mean", "100"], {"session_turn_delay_mean": 100}),
            (
                ["--session-turn-delay-stddev", "100"],
                {"session_turn_delay_stddev": 100},
            ),
            (["--session-turns-mean", "6"], {"session_turns_mean": 6}),
            (["--session-turns-stddev", "7"], {"session_turns_stddev": 7}),
            (["--stability-percentage", "99.5"], {"stability_percentage": 99.5}),
            (["--streaming"], {"streaming": True}),
            (
                ["--synthetic-input-tokens-mean", "6"],
                {"synthetic_input_tokens_mean": 6},
            ),
            (
                ["--isl", "6"],
                {"synthetic_input_tokens_mean": 6},
            ),
            (
                ["--synthetic-input-tokens-stddev", "7"],
                {"synthetic_input_tokens_stddev": 7},
            ),
            (
                ["--prefix-prompt-length", "6"],
                {"prefix_prompt_length": 6},
            ),
            (
                ["--image-width-mean", "123"],
                {"image_width_mean": 123},
            ),
            (
                ["--image-width-stddev", "123"],
                {"image_width_stddev": 123},
            ),
            (
                ["--image-height-mean", "456"],
                {"image_height_mean": 456},
            ),
            (
                ["--image-height-stddev", "456"],
                {"image_height_stddev": 456},
            ),
            (["--image-format", "png"], {"image_format": ImageFormat.PNG}),
            (
                ["--audio-length-mean", "456"],
                {"audio_length_mean": 456},
            ),
            (
                ["--audio-length-stddev", "456"],
                {"audio_length_stddev": 456},
            ),
            (["--audio-format", "wav"], {"audio_format": AudioFormat.WAV}),
            (
                ["--audio-sample-rates", "16", "44.1", "48"],
                {"audio_sample_rates": [16, 44.1, 48]},
            ),
            (["--audio-depths", "16", "32"], {"audio_depths": [16, 32]}),
            (["--tokenizer-trust-remote-code"], {"tokenizer_trust_remote_code": True}),
            (["--tokenizer-revision", "not_main"], {"tokenizer_revision": "not_main"}),
            (["-v"], {"verbose": True}),
            (["--verbose"], {"verbose": True}),
            (["-u", "test_url"], {"u": "test_url"}),
            (["--url", "test_url"], {"u": "test_url"}),
            (
                [
                    "--goodput",
                    "time_to_first_token:5",
                    "output_token_throughput_per_request:6",
                ],
                {
                    "goodput": {
                        "time_to_first_token": 5,
                        "output_token_throughput_per_request": 6,
                    }
                },
            ),
        ],
    )
    def test_non_file_flags_parsed(self, monkeypatch, arg, expected_attributes, capsys):
        logging.init_logging()
        combined_args = ["genai-perf", "profile", "--model", "test_model"] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _, _ = parser.parse_args()

        # Check that the attributes are set correctly
        for key, value in expected_attributes.items():
            assert getattr(args, key) == value

    @pytest.mark.parametrize(
        "models, expected_model_list, formatted_name",
        [
            (
                ["--model", "test_model_A"],
                {"model": ["test_model_A"]},
                {"formatted_model_name": "test_model_A"},
            ),
            (
                ["--model", "test_model_A", "test_model_B"],
                {"model": ["test_model_A", "test_model_B"]},
                {"formatted_model_name": "test_model_A_multi"},
            ),
            (
                ["--model", "test_model_A", "test_model_B", "test_model_C"],
                {"model": ["test_model_A", "test_model_B", "test_model_C"]},
                {"formatted_model_name": "test_model_A_multi"},
            ),
            (
                ["--model", "test_model_A:math", "test_model_B:embedding"],
                {"model": ["test_model_A:math", "test_model_B:embedding"]},
                {"formatted_model_name": "test_model_A:math_multi"},
            ),
        ],
    )
    def test_multiple_model_args(
        self, monkeypatch, models, expected_model_list, formatted_name, capsys
    ):
        logging.init_logging()
        combined_args = ["genai-perf", "profile"] + models
        monkeypatch.setattr("sys.argv", combined_args)
        args, _, _ = parser.parse_args()

        # Check that models are handled correctly
        for key, value in expected_model_list.items():
            assert getattr(args, key) == value

        # Check that the formatted_model_name is correctly generated
        for key, value in formatted_name.items():
            assert getattr(args, key) == value

    def test_file_flags_parsed(self, monkeypatch, mocker):
        mocker.patch.object(Path, "is_file", return_value=True)
        combined_args = [
            "genai-perf",
            "profile",
            "--model",
            "test_model",
            "--input-file",
            "fakefile.txt",
        ]
        monkeypatch.setattr("sys.argv", combined_args)
        args, _, _ = parser.parse_args()
        assert args.input_file == Path(
            "fakefile.txt"
        ), "The file argument should be the path to the file"

    @pytest.mark.parametrize(
        "arg, expected_path",
        [
            (
                ["--service-kind", "openai", "--endpoint-type", "chat"],
                "artifacts/test_model-openai-chat-concurrency1",
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "completions"],
                "artifacts/test_model-openai-completions-concurrency1",
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "rankings"],
                "artifacts/test_model-openai-rankings-concurrency1",
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "image_retrieval"],
                "artifacts/test_model-openai-image_retrieval-concurrency1",
            ),
            (
                ["--service-kind", "triton", "--backend", "tensorrtllm"],
                "artifacts/test_model-triton-tensorrtllm-concurrency1",
            ),
            (
                ["--service-kind", "triton", "--backend", "vllm"],
                "artifacts/test_model-triton-vllm-concurrency1",
            ),
            (
                [
                    "--service-kind",
                    "triton",
                    "--backend",
                    "vllm",
                    "--concurrency",
                    "32",
                ],
                "artifacts/test_model-triton-vllm-concurrency32",
            ),
        ],
    )
    def test_default_profile_export_filepath(
        self, monkeypatch, arg, expected_path, capsys
    ):
        logging.init_logging()
        combined_args = ["genai-perf", "profile", "--model", "test_model"] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _, _ = parser.parse_args()
        config = ConfigCommand({"model_name": args.formatted_model_name})
        config = parser.add_cli_options_to_config(config, args)
        perf_analyzer_config = PerfAnalyzerConfig(config)

        assert perf_analyzer_config.get_artifact_directory() == Path(expected_path)

    @pytest.mark.parametrize(
        "arg, expected_path, expected_output",
        [
            (
                ["--model", "strange/test_model"],
                "artifacts/strange_test_model-triton-tensorrtllm-concurrency1",
                (
                    "Model name 'strange/test_model' cannot be used to create "
                    "artifact directory. Instead, 'strange_test_model' will be used"
                ),
            ),
            (
                [
                    "--model",
                    "hello/world/test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "chat",
                ],
                "artifacts/hello_world_test_model-openai-chat-concurrency1",
                (
                    "Model name 'hello/world/test_model' cannot be used to create "
                    "artifact directory. Instead, 'hello_world_test_model' will be used"
                ),
            ),
        ],
    )
    def test_model_name_artifact_path(
        self, monkeypatch, arg, expected_path, expected_output, capsys
    ):
        logging.init_logging()
        combined_args = ["genai-perf", "profile"] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _, _ = parser.parse_args()

        config = ConfigCommand({"model_name": args.formatted_model_name})
        config = parser.add_cli_options_to_config(config, args)
        perf_analyzer_config = PerfAnalyzerConfig(config)

        assert perf_analyzer_config.get_artifact_directory() == Path(expected_path)

    def test_default_load_level(self, monkeypatch, capsys):
        logging.init_logging()
        monkeypatch.setattr(
            "sys.argv", ["genai-perf", "profile", "--model", "test_model"]
        )
        args, _, _ = parser.parse_args()
        assert args.concurrency == 1

    def test_load_manager_args_with_payload(self, monkeypatch, mocker):
        monkeypatch.setattr(
            "sys.argv",
            [
                "genai-perf",
                "profile",
                "--model",
                "test_model",
                "--input-file",
                "payload:test",
            ],
        )
        mocker.patch.object(Path, "is_file", return_value=True)
        args, _, _ = parser.parse_args()
        assert args.concurrency is None

    def test_load_level_mutually_exclusive(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["genai-perf", "profile", "--concurrency", "3", "--request-rate", "9.0"],
        )
        expected_output = (
            "argument --request-rate: not allowed with argument --concurrency"
        )

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    def test_model_not_provided(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["genai-perf", "profile"])
        expected_output = "the following arguments are required: -m/--model"

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    def test_pass_through_args(self, monkeypatch):
        args = ["genai-perf", "profile", "-m", "test_model"]
        other_args = ["--", "With", "great", "power"]
        monkeypatch.setattr("sys.argv", args + other_args)
        _, _, pass_through_args = parser.parse_args()

        assert pass_through_args == other_args[1:]

    def test_unrecognized_arg(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            [
                "genai-perf",
                "profile",
                "-m",
                "nonexistent_model",
                "--wrong-arg",
            ],
        )
        expected_output = "unrecognized arguments: --wrong-arg"

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    @pytest.mark.parametrize(
        "args, expected_output",
        [
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                ],
                "The --endpoint-type option is required when using the 'openai' service-kind.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint",
                    "custom/address",
                ],
                "The --endpoint-type option is required when using the 'openai' service-kind.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "dynamic_grpc",
                ],
                "The --grpc-method option is required when using the 'dynamic_grpc' service-kind.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--output-tokens-stddev",
                    "5",
                ],
                "The --output-tokens-mean option is required when using --output-tokens-stddev.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--output-tokens-mean-deterministic",
                ],
                "The --output-tokens-mean option is required when using --output-tokens-mean-deterministic.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--output-tokens-mean-deterministic",
                ],
                "The --output-tokens-mean option is required when using --output-tokens-mean-deterministic.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "chat",
                    "--output-tokens-mean",
                    "100",
                    "--output-tokens-mean-deterministic",
                ],
                "The --output-tokens-mean-deterministic option is only supported with the Triton and TensorRT-LLM Engine service-kind",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "embeddings",
                    "--generate-plots",
                ],
                "The --generate-plots option is not currently supported with the embeddings endpoint type",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "rankings",
                    "--generate-plots",
                ],
                "The --generate-plots option is not currently supported with the rankings endpoint type",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "image_retrieval",
                    "--generate-plots",
                ],
                "The --generate-plots option is not currently supported with the image_retrieval endpoint type",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "triton",
                    "--server-metrics-url",
                    "ftp://invalid.com:8002/metrics",
                ],
                "Invalid scheme 'ftp' in URL: ftp://invalid.com:8002/metrics. Use 'http' or 'https'.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "triton",
                    "--server-metrics-url",
                    "http:///metrics",
                ],
                "Invalid domain in URL: http:///metrics. Use a valid hostname or 'localhost'.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "triton",
                    "--server-metrics-url",
                    "http://valid.com:8002/invalidpath",
                ],
                "Invalid URL path '/invalidpath' in http://valid.com:8002/invalidpath. The path must include '/metrics'.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "triton",
                    "--server-metrics-url",
                    "http://valid.com/metrics",
                ],
                "Port missing in URL: http://valid.com/metrics. A port number is required (e.g., ':8002').",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "triton",
                    "--server-metrics-url",
                    "invalid_url",
                ],
                "Invalid scheme '' in URL: invalid_url. Use 'http' or 'https'.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--num-dataset-entries",
                    "0",
                ],
                "The value must be greater than zero.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--num-dataset-entries",
                    "not_number",
                ],
                "The value must be an integer.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "rankings",
                    "--backend",
                    "vllm",
                ],
                "The --backend option should only be used when using the 'triton' service-kind and 'kserve' endpoint-type.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "triton",
                    "--endpoint-type",
                    "rankings",
                    "--backend",
                    "vllm",
                ],
                "Invalid endpoint-type 'rankings' for service-kind 'triton'.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "tensorrtllm_engine",
                    "--endpoint-type",
                    "rankings",
                    "--backend",
                    "vllm",
                ],
                "Invalid endpoint-type 'rankings' for service-kind 'tensorrtllm_engine'.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "kserve",
                    "--backend",
                    "vllm",
                ],
                "Invalid endpoint-type 'kserve' for service-kind 'openai'.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "unknown_service",
                ],
                "--service-kind: invalid choice: 'unknown_service'",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--audio-format",
                    "unknown_format",
                ],
                "--audio-format: invalid choice: 'unknown_format'",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--audio-num-channels",
                    "3",
                ],
                "--audio-num-channels: invalid choice: 3",
            ),
        ],
    )
    def test_conditional_errors(self, args, expected_output, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", args)

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    @pytest.mark.parametrize(
        "args, expected_format",
        [
            (
                ["--service-kind", "openai", "--endpoint-type", "chat"],
                OutputFormat.OPENAI_CHAT_COMPLETIONS,
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "completions"],
                OutputFormat.OPENAI_COMPLETIONS,
            ),
            (
                [
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "completions",
                    "--endpoint",
                    "custom/address",
                ],
                OutputFormat.OPENAI_COMPLETIONS,
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "rankings"],
                OutputFormat.RANKINGS,
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "image_retrieval"],
                OutputFormat.IMAGE_RETRIEVAL,
            ),
            (
                ["--service-kind", "triton", "--backend", "tensorrtllm"],
                OutputFormat.TENSORRTLLM,
            ),
            (["--service-kind", "triton", "--backend", "vllm"], OutputFormat.VLLM),
            (["--service-kind", "tensorrtllm_engine"], OutputFormat.TENSORRTLLM_ENGINE),
        ],
    )
    def test_inferred_output_format(self, monkeypatch, args, expected_format):
        monkeypatch.setattr(
            "sys.argv", ["genai-perf", "profile", "-m", "test_model"] + args
        )

        parsed_args, _, _ = parser.parse_args()
        assert parsed_args.output_format == expected_format

    @pytest.mark.parametrize(
        "args, expected_error",
        [
            (
                ["--extra-inputs", "hi:"],
                "Input name or value is empty in --extra-inputs: hi:\n"
                "Expected input format: 'input_name' or 'input_name:value'",
            ),
            (
                ["--extra-inputs", ":a"],
                "Input name or value is empty in --extra-inputs: :a\n"
                "Expected input format: 'input_name' or 'input_name:value'",
            ),
            (
                ["--extra-inputs", ":a:"],
                "Invalid input format for --extra-inputs: :a:\n"
                "Expected input format: 'input_name' or 'input_name:value'",
            ),
            (
                ["--extra-inputs", "test_key:5", "--extra-inputs", "test_key:6"],
                "Input name already exists in request_inputs dictionary: test_key",
            ),
        ],
    )
    def test_get_extra_inputs_as_dict_warning(self, monkeypatch, args, expected_error):
        combined_args = ["genai-perf", "profile", "-m", "test_model"] + args
        monkeypatch.setattr("sys.argv", combined_args)

        with pytest.raises(ValueError) as exc_info:
            parsed_args, _, _ = parser.parse_args()

        assert str(exc_info.value) == expected_error

    @pytest.mark.parametrize(
        "args, expected_error",
        [
            (
                ["--goodput", "time_to_first_token:-1"],
                "Invalid value found, time_to_first_token: -1.0. The goodput constraint value should be non-negative. ",
            ),
        ],
    )
    def test_goodput_args_warning(self, monkeypatch, args, expected_error):
        combined_args = ["genai-perf", "profile", "-m", "test_model"] + args
        monkeypatch.setattr("sys.argv", combined_args)

        with pytest.raises(ValueError) as exc_info:
            parser.parse_args()

        assert str(exc_info.value) == expected_error

    @pytest.mark.parametrize(
        "args, expected_prompt_source, expected_input_file",
        [
            ([], PromptSource.SYNTHETIC, None),
            (["--input-file", "prompt.txt"], PromptSource.FILE, None),
            (
                ["--input-file", "prompt.txt", "--synthetic-input-tokens-mean", "10"],
                PromptSource.FILE,
                None,
            ),
            (
                ["--input-file", "payload:test.jsonl"],
                PromptSource.PAYLOAD,
                Path("test.jsonl"),
            ),
            (
                ["--input-file", "synthetic:test.jsonl"],
                PromptSource.SYNTHETIC,
                None,
            ),
        ],
    )
    def test_inferred_prompt_source_valid(
        self,
        monkeypatch,
        mocker,
        args,
        expected_prompt_source,
        expected_input_file,
    ):
        mocker.patch.object(Path, "is_file", return_value=True)
        combined_args = ["genai-perf", "profile", "--model", "test_model"] + args
        monkeypatch.setattr("sys.argv", combined_args)
        parsed_args, _, _ = parser.parse_args()
        assert parsed_args.prompt_source == expected_prompt_source
        assert parsed_args.payload_input_file == expected_input_file

    @pytest.mark.parametrize(
        "args",
        [
            (["--input-file", "payload:"]),
            (["--input-file", "payload:input"]),
        ],
    )
    def test_inferred_prompt_source_invalid_payload_input(
        self,
        monkeypatch,
        mocker,
        args,
    ):
        mocker.patch.object(Path, "is_file", return_value=False)
        combined_args = ["genai-perf", "profile", "--model", "test_model"] + args
        monkeypatch.setattr("sys.argv", combined_args)
        with pytest.raises(ValueError):
            parser.parse_args()

    def test_inferred_prompt_source_invalid_input(self, monkeypatch, mocker):
        file_arg = ["--input-file", "invalid_input"]
        mocker.patch.object(Path, "is_file", return_value=False)
        mocker.patch.object(Path, "is_dir", return_value=False)
        combined_args = ["genai-perf", "profile", "--model", "test_model"] + file_arg
        monkeypatch.setattr("sys.argv", combined_args)
        with pytest.raises(SystemExit):
            parser.parse_args()

    @pytest.mark.parametrize(
        "args",
        [
            # negative numbers
            ["--image-width-mean", "-123"],
            ["--image-width-stddev", "-34"],
            ["--image-height-mean", "-123"],
            ["--image-height-stddev", "-34"],
        ],
    )
    def test_positive_image_input_args(self, monkeypatch, args):
        combined_args = ["genai-perf", "profile", "-m", "test_model"] + args
        monkeypatch.setattr("sys.argv", combined_args)

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

    @pytest.mark.parametrize(
        "args",
        [
            # negative numbers
            ["--audio-length-mean", "-123"],
            ["--audio-length-stddev", "-34"],
            ["--audio-sample-rates", "-16"],
            ["--audio-sample-rates", "16", "-44.1"],  # mix
            ["--audio-depths", "-16"],
            ["--audio-depths", "16", "-32"],  # mix
            # zeros
            ["--audio-sample-rates", "0"],
            ["--audio-depths", "0"],
        ],
    )
    def test_positive_audio_input_args(self, monkeypatch, args):
        combined_args = ["genai-perf", "profile", "-m", "test_model"] + args
        monkeypatch.setattr("sys.argv", combined_args)

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

    @pytest.mark.parametrize(
        "args , expected_error_message",
        [
            (
                ["--concurrency", "10"],
                "--concurrency cannot be used with payload input.",
            ),
            (
                ["--request-rate", "5"],
                "--request-rate cannot be used with payload input.",
            ),
            (
                ["--request-count", "3"],
                "--request-count cannot be used with payload input.",
            ),
            (
                ["--warmup-request-count", "7"],
                "--warmup-request-count cannot be used with payload input.",
            ),
        ],
    )
    def test_check_payload_input_args_invalid_args(
        self, monkeypatch, mocker, capsys, args, expected_error_message
    ):
        combined_args = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--input-file",
            "payload:test.jsonl",
        ] + args

        mocker.patch.object(Path, "is_file", return_value=True)
        monkeypatch.setattr("sys.argv", combined_args)
        with pytest.raises(SystemExit):
            parser.parse_args()
        captured = capsys.readouterr()
        assert expected_error_message in captured.err

    def test_check_payload_input_args_valid(self, monkeypatch, mocker):
        valid_args = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--input-file",
            "payload:test.jsonl",
        ]
        mocker.patch.object(Path, "is_file", return_value=True)
        monkeypatch.setattr("sys.argv", valid_args)
        try:
            parser.parse_args()
        except SystemExit:
            pytest.fail("Unexpected error in test")

    def test_print_warnings_payload(self, monkeypatch, mocker):
        expected_warning_message = (
            "--output-tokens-mean is incompatible with output_length"
            " in the payload input file. output-tokens-mean"
            " will be ignored in favour of per payload settings."
        )

        args = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--input-file",
            "payload:test.jsonl",
            "--output-tokens-mean",
            "50",
        ]
        logging.init_logging()
        logger = logging.getLogger("genai_perf.parser")
        mocker.patch.object(Path, "is_file", return_value=True)
        monkeypatch.setattr("sys.argv", args)
        with patch.object(logger, "warning") as mock_logger:
            parser.parse_args()
        mock_logger.assert_any_call(expected_warning_message)

    # ================================================
    # COMPARE SUBCOMMAND
    # ================================================
    expected_compare_help_output = (
        "Subcommand to generate plots that compare multiple profile runs."
    )

    @pytest.mark.parametrize(
        "args, expected_output",
        [
            (["-h"], expected_compare_help_output),
            (["--help"], expected_compare_help_output),
        ],
    )
    def test_compare_help_arguments_output_and_exit(
        self, monkeypatch, args, expected_output, capsys
    ):
        logging.init_logging()
        monkeypatch.setattr("sys.argv", ["genai-perf", "compare"] + args)

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        # Check that the exit was successful
        assert excinfo.value.code == 0

        # Capture that the correct message was displayed
        captured = capsys.readouterr()
        assert expected_output in captured.out

    def test_compare_mutually_exclusive(self, monkeypatch, capsys):
        args = ["genai-perf", "compare", "--config", "hello", "--files", "a", "b", "c"]
        monkeypatch.setattr("sys.argv", args)
        expected_output = "argument -f/--files: not allowed with argument --config"

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    def test_compare_not_provided(self, monkeypatch, capsys):
        args = ["genai-perf", "compare"]
        monkeypatch.setattr("sys.argv", args)
        expected_output = "Either the --config or --files option must be specified."

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    @pytest.mark.parametrize(
        "extra_inputs_list, expected_dict",
        [
            (["test_key:test_value"], {"test_key": "test_value"}),
            (["test_key"], {"test_key": None}),
            (
                ["test_key:1", "another_test_key:2"],
                {"test_key": 1, "another_test_key": 2},
            ),
            (
                [
                    '{"name": "Wolverine","hobbies": ["hacking", "slashing"],"address": {"street": "1407 Graymalkin Lane, Salem Center","city": "NY"}}'
                ],
                {
                    "name": "Wolverine",
                    "hobbies": ["hacking", "slashing"],
                    "address": {
                        "street": "1407 Graymalkin Lane, Salem Center",
                        "city": "NY",
                    },
                },
            ),
        ],
    )
    def test_get_extra_inputs_as_dict(self, extra_inputs_list, expected_dict):
        namespace = argparse.Namespace()
        namespace.extra_inputs = extra_inputs_list
        actual_dict = get_extra_inputs_as_dict(namespace)
        assert actual_dict == expected_dict

    test_triton_metrics_url = "http://tritonmetrics.com:8002/metrics"

    @pytest.mark.parametrize(
        "args_list, expected_url",
        [
            # server-metrics-url is specified
            (
                [
                    "genai-perf",
                    "profile",
                    "--model",
                    "test_model",
                    "--service-kind",
                    "triton",
                    "--server-metrics-url",
                    test_triton_metrics_url,
                ],
                [test_triton_metrics_url],
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "--model",
                    "test_model",
                    "--service-kind",
                    "triton",
                    "--server-metrics-urls",
                    test_triton_metrics_url,
                ],
                [test_triton_metrics_url],
            ),
            # server-metrics-url is not specified
            (
                [
                    "genai-perf",
                    "profile",
                    "--model",
                    "test_model",
                    "--service-kind",
                    "triton",
                ],
                [],
            ),
        ],
    )
    def test_server_metrics_url_arg_valid(self, args_list, expected_url, monkeypatch):
        monkeypatch.setattr("sys.argv", args_list)
        args, _, _ = parser.parse_args()
        assert args.server_metrics_url == expected_url

    def test_tokenizer_args(self, monkeypatch):
        args = [
            "genai-perf",
            "profile",
            "--model",
            "test_model",
            "--tokenizer",
            "test_tokenizer",
            "--tokenizer-trust-remote-code",
            "--tokenizer-revision",
            "test_revision",
        ]
        monkeypatch.setattr("sys.argv", args)
        parsed_args, _, _ = parser.parse_args()

        assert parsed_args.tokenizer == "test_tokenizer"
        assert parsed_args.tokenizer_trust_remote_code
        assert parsed_args.tokenizer_revision == "test_revision"
