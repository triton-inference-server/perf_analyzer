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
from genai_perf.config.input.config_defaults import EndPointDefaults
from genai_perf.config.input.create_config import CreateConfig
from genai_perf.inputs.input_constants import (
    AudioFormat,
    ImageFormat,
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
)
from genai_perf.subcommand.common import get_extra_inputs_as_dict


class TestCLIArguments:
    # ================================================
    # PROFILE COMMAND
    # ================================================
    expected_help_output = (
        "CLI to profile LLMs and Generative AI models with Perf Analyzer"
    )
    expected_version_output = f"genai-perf {__version__}"
    base_args = [
        "genai-perf",
        "profile",
        "--model",
        "test_model",
    ]
    base_config_args = [
        "genai-perf",
        "config",
        "--file",
        "test_config.yaml",
    ]

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
            args, _ = parser.parse_args()
            CreateConfig.create(args)

        # Check that the exit was successful
        assert excinfo.value.code == 0

        # Capture that the correct message was displayed
        captured = capsys.readouterr()
        assert expected_output in captured.out

    @pytest.mark.parametrize(
        "arg, expected_cli_attributes, expected_config_attributes",
        [
            (
                ["--artifact-dir", "test_artifact_dir"],
                {"artifact_dir": Path("test_artifact_dir")},
                {"output.artifact_directory": Path("test_artifact_dir")},
            ),
            (
                [
                    "--batch-size-text",
                    "5",
                    "--endpoint-type",
                    "embeddings",
                ],
                {"batch_size_text": 5},
                {"input.batch_size": 5},
            ),
            (
                [
                    "--batch-size-image",
                    "5",
                    "--endpoint-type",
                    "image_retrieval",
                ],
                {"batch_size_image": 5},
                {"input.image.batch_size": 5},
            ),
            (
                [
                    "-b",
                    "5",
                    "--endpoint-type",
                    "embeddings",
                ],
                {"batch_size_text": 5},
                {"input.batch_size": 5},
            ),
            (
                ["--concurrency", "3"],
                {"concurrency": 3},
                {"perf_analyzer.stimulus": {"concurrency": 3}},
            ),
            (
                ["--endpoint-type", "completions"],
                {"endpoint": None},
                {"endpoint.custom": "v1/completions"},
            ),
            (
                ["--endpoint-type", "chat"],
                {"endpoint": None},
                {"endpoint.custom": "v1/chat/completions"},
            ),
            (
                ["--endpoint-type", "multimodal"],
                {"endpoint": None},
                {"endpoint.custom": "v1/chat/completions"},
            ),
            (
                ["--endpoint-type", "rankings"],
                {"endpoint": None},
                {"endpoint.custom": "v1/ranking"},
            ),
            (
                ["--endpoint-type", "image_retrieval"],
                {"endpoint": None},
                {"endpoint.custom": "v1/infer"},
            ),
            (
                [
                    "--endpoint-type",
                    "chat",
                    "--endpoint",
                    "custom/address",
                ],
                {"endpoint": "custom/address"},
                {"endpoint.custom": "custom/address"},
            ),
            (
                [
                    "--endpoint-type",
                    "chat",
                    "--endpoint",
                    "/custom/address",
                ],
                {"endpoint": "/custom/address"},
                {"endpoint.custom": "custom/address"},
            ),
            (
                [
                    "--endpoint-type",
                    "completions",
                    "--endpoint",
                    "custom/address",
                ],
                {"endpoint": "custom/address"},
                {"endpoint.custom": "custom/address"},
            ),
            (
                ["--extra-inputs", "test_key:test_value"],
                {"extra_inputs": ["test_key:test_value"]},
                {"input.extra": {"test_key": "test_value"}},
            ),
            (
                [
                    "--extra-inputs",
                    "test_key:5",
                    "--extra-inputs",
                    "another_test_key:6",
                ],
                {"extra_inputs": ["test_key:5", "another_test_key:6"]},
                {"input.extra": {"test_key": 5, "another_test_key": 6}},
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
                {
                    "input.extra": {
                        "name": "Wolverine",
                        "hobbies": ["hacking", "slashing"],
                        "address": {
                            "street": "1407 Graymalkin Lane, Salem Center",
                            "city": "NY",
                        },
                    },
                },
            ),
            (
                ["-H", "header_name:value"],
                {"header": ["header_name:value"]},
                {"input.header": ["header_name:value"]},
            ),
            (
                ["--header", "header_name:value"],
                {"header": ["header_name:value"]},
                {"input.header": ["header_name:value"]},
            ),
            (
                ["--header", "header_name:value", "--header", "header_name_2:value_2"],
                {"header": ["header_name:value", "header_name_2:value_2"]},
                {"input.header": ["header_name:value", "header_name_2:value_2"]},
            ),
            (
                ["--measurement-interval", "100"],
                {"measurement_interval": 100},
                {"perf_analyzer.measurement.num": 100},
            ),
            (
                ["--model-selection-strategy", "random"],
                {"model_selection_strategy": "random"},
                {"endpoint.model_selection_strategy": ModelSelectionStrategy.RANDOM},
            ),
            (
                ["--num-dataset-entries", "101"],
                {"num_dataset_entries": 101},
                {"input.num_dataset_entries": 101},
            ),
            (
                ["--num-prompts", "101"],
                {"num_dataset_entries": 101},
                {"input.num_dataset_entries": 101},
            ),
            (
                ["--num-prefix-prompts", "101"],
                {"num_prefix_prompts": 101},
                {"input.prefix_prompt.num": 101},
            ),
            (
                ["--output-tokens-mean", "6"],
                {"output_tokens_mean": 6},
                {"input.output_tokens.mean": 6},
            ),
            (
                ["--osl", "6"],
                {"output_tokens_mean": 6},
                {"input.output_tokens.mean": 6},
            ),
            (
                ["--output-tokens-mean", "6", "--output-tokens-stddev", "7"],
                {"output_tokens_stddev": 7},
                {"input.output_tokens.stddev": 7},
            ),
            (
                ["--output-tokens-mean", "6", "--output-tokens-mean-deterministic"],
                {"output_tokens_mean_deterministic": True},
                {"input.output_tokens.deterministic": True},
            ),
            (
                ["-p", "100"],
                {"measurement_interval": 100},
                {"perf_analyzer.measurement.num": 100},
            ),
            (
                ["--profile-export-file", "test.json"],
                {"profile_export_file": Path("test.json")},
                {"output.profile_export_file": Path("test.json")},
            ),
            (["--random-seed", "8"], {"random_seed": 8}, {"input.random_seed": 8}),
            (
                ["--request-count", "100"],
                {"request_count": 100},
                {"perf_analyzer.measurement.num": 100},
            ),
            (
                ["--num-requests", "100"],
                {"request_count": 100},
                {"perf_analyzer.measurement.num": 100},
            ),
            (
                ["--warmup-request-count", "100"],
                {"warmup_request_count": 100},
                {"perf_analyzer.warmup_request_count": 100},
            ),
            (
                ["--num-warmup-requests", "100"],
                {"warmup_request_count": 100},
                {"perf_analyzer.warmup_request_count": 100},
            ),
            (
                ["--request-rate", "9.0"],
                {"request_rate": 9.0},
                {"perf_analyzer.stimulus": {"request_rate": 9.0}},
            ),
            (
                ["-s", "99.5"],
                {"stability_percentage": 99.5},
                {"perf_analyzer.stability_percentage": 99.5},
            ),
            (
                [
                    "--endpoint-type",
                    "dynamic_grpc",
                    "--grpc-method",
                    "package.name.v1.ServiceName/MethodName",
                ],
                {
                    "grpc_method": "package.name.v1.ServiceName/MethodName",
                },
                {
                    "endpoint.service_kind": "dynamic_grpc",
                    "endpoint.type": "dynamic_grpc",
                    "endpoint.grpc_method": "package.name.v1.ServiceName/MethodName",
                },
            ),
            (
                ["--session-concurrency", "3"],
                {"session_concurrency": 3},
                {"perf_analyzer.stimulus": {"session_concurrency": 3}},
            ),
            (
                ["--session-delay-ratio", "0.5"],
                {"session_delay_ratio": 0.5},
                {"input.sessions.turn_delay.ratio": 0.5},
            ),
            (
                ["--session-turn-delay-mean", "100"],
                {"session_turn_delay_mean": 100},
                {"input.sessions.turn_delay.mean": 100},
            ),
            (
                ["--session-turn-delay-stddev", "100"],
                {"session_turn_delay_stddev": 100},
                {"input.sessions.turn_delay.stddev": 100},
            ),
            (
                ["--session-turns-mean", "6"],
                {"session_turns_mean": 6},
                {"input.sessions.turns.mean": 6},
            ),
            (
                ["--session-turns-stddev", "7"],
                {"session_turns_stddev": 7},
                {"input.sessions.turns.stddev": 7},
            ),
            (
                ["--stability-percentage", "99.5"],
                {"stability_percentage": 99.5},
                {"perf_analyzer.stability_percentage": 99.5},
            ),
            (
                ["--streaming"],
                {"streaming": True},
                {"endpoint.streaming": True},
            ),
            (
                ["--synthetic-input-tokens-mean", "6"],
                {"synthetic_input_tokens_mean": 6},
                {"input.synthetic_tokens.mean": 6},
            ),
            (
                ["--isl", "6"],
                {"synthetic_input_tokens_mean": 6},
                {"input.synthetic_tokens.mean": 6},
            ),
            (
                ["--synthetic-input-tokens-stddev", "7"],
                {"synthetic_input_tokens_stddev": 7},
                {"input.synthetic_tokens.stddev": 7},
            ),
            (
                ["--prefix-prompt-length", "6"],
                {"prefix_prompt_length": 6},
                {"input.prefix_prompt.length": 6},
            ),
            (
                ["--image-width-mean", "123"],
                {"image_width_mean": 123},
                {"input.image.width.mean": 123},
            ),
            (
                ["--image-width-stddev", "123"],
                {"image_width_stddev": 123},
                {"input.image.width.stddev": 123},
            ),
            (
                ["--image-height-mean", "456"],
                {"image_height_mean": 456},
                {"input.image.height.mean": 456},
            ),
            (
                ["--image-height-stddev", "789"],
                {"image_height_stddev": 789},
                {"input.image.height.stddev": 789},
            ),
            (
                ["--image-format", "png"],
                {"image_format": "png"},
                {"input.image.format": ImageFormat.PNG},
            ),
            (
                ["--audio-length-mean", "234"],
                {"audio_length_mean": 234},
                {"input.audio.length.mean": 234},
            ),
            (
                ["--audio-length-stddev", "345"],
                {"audio_length_stddev": 345},
                {"input.audio.length.stddev": 345},
            ),
            (
                ["--audio-format", "wav"],
                {"audio_format": "wav"},
                {"input.audio.format": AudioFormat.WAV},
            ),
            (
                ["--audio-sample-rates", "16", "44.1", "48"],
                {"audio_sample_rates": [16, 44.1, 48]},
                {"input.audio.sample_rates": [16, 44.1, 48]},
            ),
            (
                ["--audio-depths", "16", "32"],
                {"audio_depths": [16, 32]},
                {"input.audio.depths": [16, 32]},
            ),
            (
                ["--tokenizer-trust-remote-code"],
                {"tokenizer_trust_remote_code": True},
                {"tokenizer.trust_remote_code": True},
            ),
            (
                ["--tokenizer-revision", "not_main"],
                {"tokenizer_revision": "not_main"},
                {"tokenizer.revision": "not_main"},
            ),
            (["-v"], {"verbose": True}, {"verbose": True}),
            (["--verbose"], {"verbose": True}, {"verbose": True}),
            (["-u", "test_url"], {"u": "test_url"}, {"endpoint.url": "test_url"}),
            (["--url", "test_url"], {"u": "test_url"}, {"endpoint.url": "test_url"}),
            (
                [
                    "--goodput",
                    "time_to_first_token:5",
                    "output_token_throughput_per_user:6",
                ],
                {
                    "goodput": {
                        "time_to_first_token": 5,
                        "output_token_throughput_per_user": 6,
                    }
                },
                {
                    "input.goodput": {
                        "time_to_first_token": 5,
                        "output_token_throughput_per_user": 6,
                    }
                },
            ),
        ],
    )
    def test_non_file_flags_parsed(
        self,
        monkeypatch,
        arg,
        expected_cli_attributes,
        expected_config_attributes,
        capsys,
    ):
        logging.init_logging()
        combined_args = self.base_args + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()
        config = CreateConfig.create(args)

        # Check that the attributes are set correctly
        for key, value in expected_cli_attributes.items():
            assert getattr(args, key) == value

        for key_str, expected_value in expected_config_attributes.items():
            keys = key_str.split(".")
            value = config

            for key in keys:
                value = getattr(value, key)

            assert value == expected_value

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
        args, _ = parser.parse_args()
        config = CreateConfig.create(args)
        assert args.input_file == Path(
            "fakefile.txt"
        ), "The file argument should be the path to the file"
        assert config.input.file == args.input_file

    @pytest.mark.parametrize(
        "arg, expected_path",
        [
            (
                ["--endpoint-type", "chat"],
                "artifacts/test_model-openai-chat-concurrency1",
            ),
            (
                ["--endpoint-type", "completions"],
                "artifacts/test_model-openai-completions-concurrency1",
            ),
            (
                ["--endpoint-type", "rankings"],
                "artifacts/test_model-openai-rankings-concurrency1",
            ),
            (
                ["--endpoint-type", "image_retrieval"],
                "artifacts/test_model-openai-image_retrieval-concurrency1",
            ),
            (
                ["--backend", "tensorrtllm"],
                "artifacts/test_model-triton-tensorrtllm-concurrency1",
            ),
            (
                ["--backend", "vllm"],
                "artifacts/test_model-triton-vllm-concurrency1",
            ),
            (
                [
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
        combined_args = self.base_args + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()
        config = ConfigCommand({"model_name": args.model})
        config = CreateConfig._add_cli_options_to_config(config, args)
        config.infer_and_check_options()
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
        combined_args = self.base_args + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()

        config = ConfigCommand({"model_names": args.model})
        config = CreateConfig._add_cli_options_to_config(config, args)
        config.infer_and_check_options()
        perf_analyzer_config = PerfAnalyzerConfig(config)

        assert perf_analyzer_config.get_artifact_directory() == Path(expected_path)

    def test_default_load_level(self, monkeypatch, capsys):
        logging.init_logging()
        monkeypatch.setattr("sys.argv", self.base_args)
        args, _ = parser.parse_args()
        config = CreateConfig.create(args)
        assert config.perf_analyzer.stimulus["concurrency"] == 1

    def test_load_manager_args_with_payload(self, monkeypatch, mocker):
        monkeypatch.setattr(
            "sys.argv",
            self.base_args
            + [
                "--input-file",
                "payload:test",
                "--measurement-interval",
                "100",
            ],
        )
        mocker.patch.object(Path, "is_file", return_value=True)
        args, _ = parser.parse_args()
        config = CreateConfig.create(args)
        assert args.concurrency is None
        assert config.perf_analyzer.get_field("stimulus").is_set_by_user is False

    def test_load_level_mutually_exclusive(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            self.base_args + ["--concurrency", "3", "--request-rate", "9.0"],
        )
        expected_output = (
            "argument --request-rate: not allowed with argument --concurrency"
        )

        with pytest.raises(SystemExit) as excinfo:
            args, _ = parser.parse_args()
            CreateConfig.create(args)

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    def test_model_not_provided(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["genai-perf", "profile"])
        expected_error_message = "Required field model_names is not set"

        with pytest.raises(ValueError) as execinfo:
            args, _ = parser.parse_args()
            CreateConfig.create(args)

        assert expected_error_message == execinfo.value.args[0]

    def test_pass_through_args(self, monkeypatch):
        other_args = ["--", "With", "great", "power"]
        monkeypatch.setattr("sys.argv", self.base_args + other_args)
        _, pass_through_args = parser.parse_args()

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

        with pytest.raises(SystemExit):
            args, _ = parser.parse_args()
            CreateConfig.create(args)

        expected_error_message = "error: unrecognized arguments: --wrong-arg"

        # Capture that the correct message was displayed
        captured = capsys.readouterr()
        assert expected_error_message in captured.err

    def test_non_default_create_template_filename(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["genai-perf", "create-template", "--file", "custom_template.yaml"],
        )

        args, _ = parser.parse_args()
        config = CreateConfig.create(args)
        assert config.template_filename == Path("custom_template.yaml")

    @pytest.mark.parametrize(
        "args, expected_error_message",
        [
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--output-tokens-stddev",
                    "5",
                ],
                "User Config: If output tokens stddev is set, mean must also be set",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--output-tokens-mean-deterministic",
                ],
                "User Config: If output tokens deterministic is set, mean must also be set",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--output-tokens-mean-deterministic",
                ],
                "User Config: If output tokens deterministic is set, mean must also be set",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--endpoint-type",
                    "chat",
                    "--output-tokens-mean",
                    "100",
                    "--output-tokens-mean-deterministic",
                ],
                "User Config: input.output_tokens.deterministic is only supported with Triton or TensorRT-LLM Engine service kinds",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--endpoint-type",
                    "embeddings",
                    "--generate-plots",
                ],
                "User Config: generate_plots is not supported with the OutputFormat.OPENAI_EMBEDDINGS output format",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--endpoint-type",
                    "rankings",
                    "--generate-plots",
                ],
                "User Config: generate_plots is not supported with the OutputFormat.RANKINGS output format",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--endpoint-type",
                    "image_retrieval",
                    "--generate-plots",
                ],
                "User Config: generate_plots is not supported with the OutputFormat.IMAGE_RETRIEVAL output format",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--backend",
                    "vllm",
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
                    "--backend",
                    "vllm",
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
                    "--backend",
                    "vllm",
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
                    "--backend",
                    "vllm",
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
                    "--backend",
                    "vllm",
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
                    "--endpoint-type",
                    "rankings",
                    "--backend",
                    "vllm",
                ],
                "The backend should only be used with the following combination: 'service_kind: triton' & 'type: kserve'",
            ),
        ],
    )
    def test_conditional_errors(
        self, args, expected_error_message, monkeypatch, capsys
    ):
        logging.init_logging()
        monkeypatch.setattr("sys.argv", args)

        with pytest.raises(ValueError) as execinfo:
            args, _ = parser.parse_args()
            CreateConfig.create(args)

        assert expected_error_message == execinfo.value.args[0]

    @pytest.mark.parametrize(
        "args, expected_format",
        [
            (
                ["--endpoint-type", "chat"],
                OutputFormat.OPENAI_CHAT_COMPLETIONS,
            ),
            (
                ["--endpoint-type", "completions"],
                OutputFormat.OPENAI_COMPLETIONS,
            ),
            (
                [
                    "--endpoint-type",
                    "completions",
                    "--endpoint",
                    "custom/address",
                ],
                OutputFormat.OPENAI_COMPLETIONS,
            ),
            (
                ["--endpoint-type", "rankings"],
                OutputFormat.RANKINGS,
            ),
            (
                ["--endpoint-type", "image_retrieval"],
                OutputFormat.IMAGE_RETRIEVAL,
            ),
            (
                ["--backend", "tensorrtllm"],
                OutputFormat.TENSORRTLLM,
            ),
            (["--backend", "vllm"], OutputFormat.VLLM),
            (
                ["--endpoint-type", "tensorrtllm_engine"],
                OutputFormat.TENSORRTLLM_ENGINE,
            ),
        ],
    )
    def test_inferred_output_format(self, monkeypatch, args, expected_format):
        monkeypatch.setattr("sys.argv", self.base_args + args)

        args, _ = parser.parse_args()
        config = CreateConfig.create(args)
        assert config.endpoint.output_format == expected_format

    @pytest.mark.parametrize(
        "args, expected_error_message",
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
    def test_get_extra_inputs_as_dict_warning(
        self, monkeypatch, args, expected_error_message
    ):
        combined_args = self.base_args + args
        monkeypatch.setattr("sys.argv", combined_args)

        with pytest.raises(ValueError) as execinfo:
            args, _ = parser.parse_args()
            CreateConfig.create(args)

        assert expected_error_message == execinfo.value.args[0]

    @pytest.mark.parametrize(
        "args, expected_error_message",
        [
            (
                ["--goodput", "time_to_first_token:-1"],
                "Invalid value found, time_to_first_token: -1.0. The goodput constraint value should be non-negative. ",
            ),
        ],
    )
    def test_goodput_args_warning(self, monkeypatch, args, expected_error_message):
        combined_args = self.base_args + args
        monkeypatch.setattr("sys.argv", combined_args)

        with pytest.raises(ValueError) as execinfo:
            args, _ = parser.parse_args()
            CreateConfig.create(args)

        assert expected_error_message == execinfo.value.args[0]

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
        combined_args = self.base_args + args
        monkeypatch.setattr("sys.argv", combined_args)
        parsed_args, _ = parser.parse_args()
        config = CreateConfig.create(parsed_args)
        assert config.input.prompt_source == expected_prompt_source
        assert config.input.payload_file == expected_input_file

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
        combined_args = self.base_args + args
        monkeypatch.setattr("sys.argv", combined_args)
        with pytest.raises(ValueError):
            args, _ = parser.parse_args()
            CreateConfig.create(args)

    def test_inferred_prompt_source_invalid_input(self, monkeypatch, mocker):
        arg = ["--input-file", "invalid_input"]
        mocker.patch.object(Path, "is_file", return_value=False)
        mocker.patch.object(Path, "is_dir", return_value=False)
        combined_args = self.base_args + arg
        monkeypatch.setattr("sys.argv", combined_args)
        with pytest.raises(SystemExit):
            args, _ = parser.parse_args()
            CreateConfig.create(args)

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
        combined_args = self.base_args + args
        monkeypatch.setattr("sys.argv", combined_args)

        with pytest.raises(ValueError) as excinfo:
            args, _ = parser.parse_args()
            CreateConfig.create(args)

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
        combined_args = self.base_args + args
        monkeypatch.setattr("sys.argv", combined_args)

        with pytest.raises(ValueError) as excinfo:
            args, _ = parser.parse_args()
            CreateConfig.create(args)

    @pytest.mark.parametrize(
        "args , expected_error_message",
        [
            (
                ["--concurrency", "10"],
                "User Config: perf_analyzer.stimulus: concurrency is not supported with the payload input source.",
            ),
            (
                ["--request-rate", "5"],
                "User Config: perf_analyzer.stimulus: request_rate is not supported with the payload input source.",
            ),
            (
                ["--request-count", "3"],
                "User Config: perf_analyzer.measurement.mode of request_count is not supported with the payload input source.",
            ),
        ],
    )
    def test_check_payload_input_args_invalid_args(
        self, monkeypatch, mocker, capsys, args, expected_error_message
    ):
        combined_args = (
            self.base_args
            + [
                "--input-file",
                "payload:test.jsonl",
            ]
            + args
        )

        mocker.patch.object(Path, "is_file", return_value=True)
        monkeypatch.setattr("sys.argv", combined_args)
        with pytest.raises(ValueError) as execinfo:
            args, _ = parser.parse_args()
            CreateConfig.create(args)

        assert expected_error_message == execinfo.value.args[0]

    def test_check_payload_input_args_valid(self, monkeypatch, mocker):
        valid_args = self.base_args + [
            "--input-file",
            "payload:test.jsonl",
        ]
        mocker.patch.object(Path, "is_file", return_value=True)
        monkeypatch.setattr("sys.argv", valid_args)
        try:
            args, _ = parser.parse_args()
            CreateConfig.create(args)
        except SystemExit:
            pytest.fail("Unexpected error in test")

    def test_print_warnings_payload(self, monkeypatch, mocker):
        expected_warning_message = (
            "--output-tokens-mean is incompatible with output_length"
            " in the payload input file. output-tokens-mean"
            " will be ignored in favour of per payload settings."
        )

        args = self.base_args + [
            "--input-file",
            "payload:test.jsonl",
            "--output-tokens-mean",
            "50",
            "--measurement-interval",
            "100",
        ]
        logging.init_logging()
        logger = logging.getLogger("genai_perf.config.input.create_config")
        mocker.patch.object(Path, "is_file", return_value=True)
        monkeypatch.setattr("sys.argv", args)
        with patch.object(logger, "warning") as mock_logger:
            args, _ = parser.parse_args()
            CreateConfig.create(args)
        mock_logger.assert_any_call(expected_warning_message)

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
                    "--backend",
                    "tensorrtllm",
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
                    "--backend",
                    "tensorrtllm",
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
                    "--backend",
                    "tensorrtllm",
                ],
                [],
            ),
        ],
    )
    def test_server_metrics_url_arg_valid(self, args_list, expected_url, monkeypatch):
        monkeypatch.setattr("sys.argv", args_list)
        args, _ = parser.parse_args()
        config = CreateConfig.create(args)

        if expected_url:
            assert config.endpoint.server_metrics_urls == expected_url
        else:
            assert (
                config.endpoint.server_metrics_urls
                == EndPointDefaults.SERVER_METRICS_URLS
            )

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
        parsed_args, _ = parser.parse_args()
        config = CreateConfig.create(parsed_args)

        assert parsed_args.tokenizer == "test_tokenizer"
        assert parsed_args.tokenizer_trust_remote_code
        assert parsed_args.tokenizer_revision == "test_revision"

        assert config.tokenizer.name == "test_tokenizer"
        assert config.tokenizer.trust_remote_code
        assert config.tokenizer.revision == "test_revision"

    def test_measurement_group_mutually_exclusive(self, monkeypatch, capsys):
        combined_args = self.base_args + [
            "--request-count",
            "100",
            "--measurement-interval",
            "5000",
        ]
        monkeypatch.setattr("sys.argv", combined_args)

        with pytest.raises(SystemExit) as excinfo:
            args, _ = parser.parse_args()
            CreateConfig.create(args)

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        expected_error = "argument --measurement-interval/-p: not allowed with argument --request-count/--num-requests"
        assert expected_error in captured.err

    @patch("genai_perf.parser.utils.load_yaml", return_value={})
    @patch("pathlib.Path.exists", return_value=True)
    def test_config_file_plus_illegal_cli_options(
        self, mock_yaml, mock_path, monkeypatch
    ):
        combined_args = self.base_config_args + [
            "--model",
            "test_model_name",
            "--request-rate",
            "100",
        ]

        monkeypatch.setattr("sys.argv", combined_args)
        with pytest.raises(ValueError) as execinfo:
            args, _ = parser.parse_args()
            CreateConfig.create(args)

        expected_error_message = "In order to use the CLI to override the config, the --override-config flag must be set."
        assert expected_error_message == execinfo.value.args[0]

    @patch("genai_perf.parser.utils.load_yaml", return_value={})
    @patch("pathlib.Path.exists", return_value=True)
    def test_config_file_plus_override_config_options(
        self, mock_yaml, mock_path, monkeypatch
    ):
        combined_args = self.base_config_args + [
            "--override-config",
            "--model",
            "test_model_name",
            "--request-rate",
            "100",
        ]

        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()
        config = CreateConfig.create(args)

        assert config.model_names == ["test_model_name"]
        assert config.perf_analyzer.stimulus == {"request_rate": 100}

    @patch("genai_perf.parser.utils.load_yaml", return_value={})
    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "genai_perf.config.input.base_config.BaseConfig.check_required_fields_are_set",
        return_value=None,
    )
    def test_config_file_plus_verbose(
        self, mock_yaml, mock_path, mock_check_fields, monkeypatch
    ):
        combined_args = self.base_config_args + [
            "--verbose",
        ]

        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()
        config = CreateConfig.create(args)

        assert config.verbose is True

    # ================================================
    # PROCESS-EXPORT-FILES SUBCOMMAND
    # ================================================
    expected_help_output = (
        "Subcommand to process export files and aggregate the results."
    )

    @pytest.mark.parametrize(
        "args, expected_output",
        [
            (["-h"], expected_help_output),
            (["--help"], expected_help_output),
        ],
    )
    def test_process_export_files_help_arguments_output_and_exit(
        self, monkeypatch, args, expected_output, capsys
    ):
        monkeypatch.setattr("sys.argv", ["genai-perf", "process-export-files"] + args)

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code == 0

        captured = capsys.readouterr()
        assert expected_output in captured.out

    def test_process_export_files_missing_input_path(self, monkeypatch, capsys):
        args = ["genai-perf", "process-export-files"]
        monkeypatch.setattr("sys.argv", args)

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert "input-directory" in captured.err

    def test_process_export_files_input_path(self, monkeypatch, capsys):
        args = ["genai-perf", "process-export-files", "--input-directory", "test_dir"]
        monkeypatch.setattr("sys.argv", args)
        args, _ = parser.parse_args()
        with patch.object(Path, "is_dir", return_value=True):
            config = CreateConfig.create(args)

        assert args.input_path[0] == "test_dir"
        assert config.process.input_path == Path("test_dir")

    def test_process_export_files_input_path(self, monkeypatch, capsys):
        args = ["genai-perf", "process-export-files", "test_dir"]
        monkeypatch.setattr("genai_perf.parser.directory", Path)
        monkeypatch.setattr("sys.argv", args)
        parsed_args, config, _ = parser.parse_args()

        assert parsed_args.input_path[0] == Path("test_dir")
        assert config.subcommand == Subcommand.PROCESS
        assert config.process.input_path == Path("test_dir")

    @pytest.mark.parametrize(
        "arg, expected_artifact_dir, config_artifact_dir",
        [
            (["--artifact-dir", "test_dir"], "test_dir", "test_dir"),
            ([], None, "artifacts"),
        ],
    )
    def test_process_export_files_artifact_dir(
        self, monkeypatch, arg, expected_artifact_dir, config_artifact_dir
    ):
        combined_args = [
            "genai-perf",
            "process-export-files",
            "--input-directory",
            "test_dir",
        ] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()
        with patch.object(Path, "is_dir", return_value=True):
            config = CreateConfig.create(args)

        if expected_artifact_dir is None:
            assert args.artifact_dir is None
        else:
            assert args.artifact_dir == Path(expected_artifact_dir)
        assert config.output.artifact_directory == Path(config_artifact_dir)

    @pytest.mark.parametrize(
        "arg, expected_profile_json_path, config_profile_json_path",
        [
            (["--profile-export-file", "test.json"], "test.json", "test.json"),
            ([], None, "profile_export.json"),
        ],
    )
    def test_process_export_files_profile_export_filepath(
        self, monkeypatch, arg, expected_profile_json_path, config_profile_json_path
    ):
        combined_args = [
            "genai-perf",
            "process-export-files",
            "--input-directory",
            "test_dir",
        ] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()
        with patch.object(Path, "is_dir", return_value=True):
            config = CreateConfig.create(args)

        if expected_profile_json_path is None:
            assert args.profile_export_file is None
        else:
            assert args.profile_export_file == Path(expected_profile_json_path)
        assert config.output.profile_export_file == Path(config_profile_json_path)

    def test_process_export_files_unrecognized_arg(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            [
                "genai-perf",
                "process-export-files",
                "--input-directory",
                "test_dir",
                "--wrong-arg",
            ],
        )

        with pytest.raises(ValueError) as excinfo:
            parser.parse_args()

        assert excinfo.value.code == 2

        captured = capsys.readouterr()
        assert "unrecognized arguments: --wrong-arg" in captured.err

    def test_process_export_files_short_input_directory_option(self, monkeypatch):
        args = ["genai-perf", "process-export-files", "-d", "test_dir"]
        monkeypatch.setattr("sys.argv", args)

        args, _ = parser.parse_args()
        with patch.object(Path, "is_dir", return_value=True):
            config = CreateConfig.create(args)

        assert args.input_path[0] == "test_dir"
        assert config.process.input_path == Path("test_dir")
