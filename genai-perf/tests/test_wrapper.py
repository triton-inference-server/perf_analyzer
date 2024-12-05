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

from argparse import Namespace
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from genai_perf import parser
from genai_perf.constants import DEFAULT_GRPC_URL
from genai_perf.inputs.input_constants import OutputFormat, PromptSource
from genai_perf.wrapper import Profiler


class TestWrapper:
    @pytest.fixture
    def args(self):
        yield Namespace(
            model="test_model",
            verbose=False,
            service_kind="triton",
            formatted_model_name="test_model",
            artifact_dir=Path("test_dir"),
            concurrency=None,
            request_rate=None,
            prompt_source=PromptSource.SYNTHETIC,
            payload_input_file=None,
            u=None,
            profile_export_file=Path("test_profile_export.json"),
            backend=None,
            output_format=None,
        )

    @pytest.mark.parametrize(
        "service_kind, u, output_format, expected",
        [
            (
                "triton",
                None,
                None,
                ["-i", "grpc", "--streaming", "-u", DEFAULT_GRPC_URL],
            ),
            ("triton", "custom:8080", None, ["-i", "grpc", "--streaming"]),
            (
                "triton",
                None,
                OutputFormat.TENSORRTLLM,
                [
                    "-i",
                    "grpc",
                    "--streaming",
                    "-u",
                    DEFAULT_GRPC_URL,
                    "--shape",
                    "max_tokens:1",
                    "--shape",
                    "text_input:1",
                ],
            ),
            ("openai", None, None, ["-i", "http"]),
        ],
    )
    def test_add_protocol_args(self, args, service_kind, u, output_format, expected):
        args.service_kind = service_kind
        args.u = u
        args.output_format = output_format
        result = Profiler.add_protocol_args(args)
        assert result == expected

    @pytest.mark.parametrize(
        "attr, value, expected",
        [
            ("concurrency", 5, ["--concurrency-range", "5"]),
            ("request_rate", 10, ["--request-rate-range", "10"]),
        ],
    )
    def test_add_inference_load_args(self, args, attr, value, expected):
        setattr(args, attr, value)
        result = Profiler.add_inference_load_args(args)
        assert result == expected

    def test_add_payload_args(self, args):
        args.prompt_source = PromptSource.PAYLOAD
        args.payload_input_file = "test_file.json"
        mock_file_content = (
            '{"timestamp": 0, "input_length": 6755, "output_length": 500}\n'
            '{"timestamp": 0, "input_length": 7319, "output_length": 490}\n'
        )

        with patch(
            "genai_perf.wrapper.open", mock_open(read_data=mock_file_content)
        ) as mock_file:
            mock_file.return_value.__enter__.return_value.readlines.return_value = (
                mock_file_content.split("\n")
            )
            result = Profiler.add_payload_args(args)
            assert result == ["--schedule", "0.0,0.0"]

    @pytest.mark.parametrize(
        "arg",
        [
            ([]),
            (["-u", "testurl:1000"]),
            (["--url", "testurl:1000"]),
        ],
    )
    def test_url_exactly_once_triton(self, args, arg, monkeypatch):
        base_args = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--service-kind",
            "triton",
        ] + arg
        monkeypatch.setattr("sys.argv", base_args)
        parsed_args, extra_args = parser.parse_args()

        for key, value in vars(parsed_args).items():
            setattr(args, key, value)

        cmd = Profiler.build_cmd(args, extra_args)
        cmd_string = " ".join(cmd)

        number_of_url_args = cmd_string.count(" -u ") + cmd_string.count(" --url ")
        assert number_of_url_args == 1

    @pytest.mark.parametrize(
        "arg, expected_filepath",
        [
            (
                [],
                "artifacts/test_model-triton-tensorrtllm-concurrency1/profile_export.json",
            ),
            (
                ["--artifact-dir", "test_dir"],
                "test_dir/profile_export.json",
            ),
            (
                ["--artifact-dir", "test_dir", "--profile-export-file", "test.json"],
                "test_dir/test.json",
            ),
        ],
    )
    def test_profile_export_filepath(self, args, monkeypatch, arg, expected_filepath):
        base_args = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--service-kind",
            "triton",
        ] + arg
        monkeypatch.setattr("sys.argv", base_args)
        parsed_args, extra_args = parser.parse_args()

        for key, value in vars(parsed_args).items():
            setattr(args, key, value)

        cmd = Profiler.build_cmd(args, extra_args)
        cmd_string = " ".join(cmd)

        expected_pattern = f"--profile-export-file {expected_filepath}"
        assert expected_pattern in cmd_string

    @pytest.mark.parametrize(
        "arg",
        [
            (["--backend", "tensorrtllm"]),
            (["--backend", "vllm"]),
        ],
    )
    def test_service_triton(self, args, monkeypatch, arg):
        base_args = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--service-kind",
            "triton",
        ] + arg
        monkeypatch.setattr("sys.argv", base_args)
        parsed_args, extra_args = parser.parse_args()
        for key, value in vars(parsed_args).items():
            setattr(args, key, value)
        cmd = Profiler.build_cmd(args, extra_args)
        cmd_string = " ".join(cmd)

        # Ensure the correct arguments are appended.
        assert cmd_string.count(" -i grpc") == 1
        assert cmd_string.count(" --streaming") == 1
        assert cmd_string.count(f"-u {DEFAULT_GRPC_URL}") == 1
        if arg[1] == "tensorrtllm":
            assert cmd_string.count("--shape max_tokens:1") == 1
            assert cmd_string.count("--shape text_input:1") == 1

    @pytest.mark.parametrize(
        "arg",
        [
            (["--endpoint-type", "completions"]),
            (["--endpoint-type", "chat"]),
        ],
    )
    def test_service_openai(self, args, monkeypatch, arg):
        base_args = [
            "genai-perf",
            "profile",
            "-m",
            "test_model",
            "--service-kind",
            "openai",
        ] + arg
        monkeypatch.setattr("sys.argv", base_args)
        parsed_args, extra_args = parser.parse_args()
        for key, value in vars(parsed_args).items():
            setattr(args, key, value)
        cmd = Profiler.build_cmd(args, extra_args)
        cmd_string = " ".join(cmd)

        # Ensure the correct arguments are appended.
        assert cmd_string.count(" -i http") == 1

    def test_build_cmd_for_payload(self, args, monkeypatch):
        mock_file_content = (
            '{"timestamp": 0, "input_length": 6755, "output_length": 500}\n'
            '{"timestamp": 1, "input_length": 7319, "output_length": 490}\n'
        )

        with patch("genai_perf.wrapper.open", mock_open(read_data=mock_file_content)):
            base_args = [
                "genai-perf",
                "profile",
                "-m",
                "test_model",
                "--service-kind",
                "openai",
                "--endpoint-type",
                "chat",
                "--input-file",
                "payload:test_file",
            ]
            monkeypatch.setattr("sys.argv", base_args)
            parsed_args, extra_args = parser.parse_args()

            for key, value in vars(parsed_args).items():
                setattr(args, key, value)

            cmd = Profiler.build_cmd(args, extra_args)
            cmd_string = " ".join(cmd)

            args_to_be_excluded = [
                "--concurrency",
                "--request-rate-range",
                "--request-count",
                "--warmup-request-count",
                "measurement-interval",
                "--stability-percentage",
            ]
            for arg in args_to_be_excluded:
                assert arg not in cmd_string

            assert "--schedule 0.0,1.0" in cmd_string
