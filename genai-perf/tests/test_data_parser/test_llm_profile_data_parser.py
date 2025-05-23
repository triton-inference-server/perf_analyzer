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

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.metrics import LLMMetrics
from genai_perf.metrics.statistics import Statistics
from genai_perf.profile_data_parser import LLMProfileDataParser
from genai_perf.profile_data_parser.profile_data_parser import ResponseFormat
from genai_perf.tokenizer import get_tokenizer
from tests.test_utils import check_statistics, ns_to_sec


def check_llm_metrics(m1: LLMMetrics, m2: LLMMetrics) -> None:
    assert m1.request_latencies == m2.request_latencies
    assert m1.request_throughputs == pytest.approx(m2.request_throughputs)
    assert m1.time_to_first_tokens == m2.time_to_first_tokens
    assert m1.time_to_second_tokens == m2.time_to_second_tokens
    assert m1.inter_token_latencies == m2.inter_token_latencies
    assert m1.output_token_throughputs_per_user == pytest.approx(
        m2.output_token_throughputs_per_user
    )
    assert m1.output_token_throughputs == pytest.approx(m2.output_token_throughputs)
    assert m1.output_sequence_lengths == m2.output_sequence_lengths
    assert m1.input_sequence_lengths == m2.input_sequence_lengths


class TestLLMProfileDataParser:

    ###############################
    # TRITON
    ###############################
    triton_profile_data = {
        "service_kind": "triton",
        "endpoint": "",
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {"text_input": "This is test"},
                        "response_timestamps": [3, 5, 8],
                        "response_outputs": [
                            {"text_output": "I"},
                            {"text_output": " like"},
                            {"text_output": " dogs"},
                        ],
                    },
                    {
                        "timestamp": 2,
                        "request_inputs": {"text_input": "This is test too"},
                        "response_timestamps": [4, 7, 11],
                        "response_outputs": [
                            {"text_output": "I"},
                            {"text_output": " don't"},
                            {"text_output": " cook food"},
                        ],
                    },
                ],
            },
            {
                "experiment": {
                    "mode": "request_rate",
                    "value": 2.0,
                },
                "requests": [
                    {
                        "timestamp": 5,
                        "request_inputs": {"text_input": "This is test"},
                        "response_timestamps": [7, 8, 13, 18],
                        "response_outputs": [
                            {"text_output": "cat"},
                            {"text_output": " is"},
                            {"text_output": " cool"},
                            {"text_output": " too"},
                        ],
                    },
                    {
                        "timestamp": 3,
                        "request_inputs": {"text_input": "This is test too"},
                        "response_timestamps": [6, 8, 11],
                        "response_outputs": [
                            {"text_output": "it's"},
                            {"text_output": " very"},
                            {"text_output": " simple work"},
                        ],
                    },
                ],
            },
        ],
    }

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=triton_profile_data,
    )
    @pytest.mark.parametrize(
        "infer_mode, load_level, expected_metrics",
        [
            (
                "concurrency",
                "10",
                {
                    "request_latencies": [7, 9],
                    "request_throughputs": [1 / ns_to_sec(5)],
                    "time_to_first_tokens": [2, 2],
                    "time_to_second_tokens": [2, 3],
                    "inter_token_latencies": [2, 1],
                    "output_token_throughputs_per_user": [
                        1 / ns_to_sec(2),
                        1 / ns_to_sec(1),
                    ],
                    "output_token_throughputs": [9 / ns_to_sec(10)],
                    "output_sequence_lengths": [3, 6],
                    "input_sequence_lengths": [3, 4],
                },
            ),
            (
                "request_rate",
                "2.0",
                {
                    "request_latencies": [13, 8],
                    "request_throughputs": [2 / ns_to_sec(15)],
                    "time_to_first_tokens": [2, 3],
                    "time_to_second_tokens": [1, 2],
                    "inter_token_latencies": [4, 1],
                    "output_token_throughputs_per_user": [
                        1 / ns_to_sec(4),
                        1 / ns_to_sec(1),
                    ],
                    "output_token_throughputs": [2 / ns_to_sec(3)],
                    "output_sequence_lengths": [4, 6],
                    "input_sequence_lengths": [3, 4],
                },
            ),
        ],
    )
    def test_triton_llm_profile_data(
        self,
        mock_json,
        infer_mode,
        load_level,
        expected_metrics,
    ) -> None:
        """Collect LLM metrics from profile export data and check values.

        Metrics
        * request_latencies
            - experiment 1: [8 - 1, 11 - 2] = [7, 9]
            - experiment 2: [18 - 5, 11 -3] = [13, 8]
        * request_throughputs
            - experiment 1: [2/(11 - 1)] = [1/5]
            - experiment 2: [2/(18 - 3)] = [2/15]
        * time to first tokens
            - experiment 1: [3 - 1, 4 - 2] = [2, 2]
            - experiment 2: [7 - 5, 6 - 3] = [2, 3]
        * time to second tokens
            - experiment 1: [5-3, 7-4] = [2, 3]
            - experiment 2: [8-7, 6-2] = [1, 2]
        * inter token latencies
            - experiment 1: [((8 - 1) - 2)/(3 - 1), ((11 - 2) - 2)/(6 - 1)]
                          : [2.5, 1.4]
                          : [2, 1]  # rounded
            - experiment 2: [((18 - 5) - 2)/(4 - 1), ((11 - 3) - 3)/(6 - 1)]
                          : [11/3, 1]
                          : [4, 1]  # rounded
        * output token throughputs per user
            - experiment 1: [1 / ns_to_sec(2), 1 / ns_to_sec(1)]
            - experiment 2: [1 / ns_to_sec(4), 1 / ns_to_sec(1)]
        * output token throughputs
            - experiment 1: [(3 + 6)/(11 - 1)] = [9/10]
            - experiment 2: [(4 + 6)/(18 - 3)] = [2/3]
        * output sequence lengths
            - experiment 1: [3, 6]
            - experiment 2: [4, 6]
        * input sequence lengths
            - experiment 1: [3, 4]
            - experiment 2: [3, 4]
        """
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("triton_profile_export.json"),
            tokenizer=tokenizer,
        )

        statistics = pd.get_statistics(infer_mode=infer_mode, load_level=load_level)
        metrics = cast(LLMMetrics, statistics.metrics)

        expected_metrics = LLMMetrics(**expected_metrics)
        expected_statistics = Statistics(expected_metrics)

        check_llm_metrics(metrics, expected_metrics)
        check_statistics(statistics, expected_statistics)

        # Check that Records can be created
        records = statistics.create_records()
        assert records is not None
        assert records["request_throughput_avg"] is not None

        # check non-existing profile data
        with pytest.raises(KeyError):
            pd.get_statistics(infer_mode="concurrency", load_level="30")

    ###############################
    # OPENAI COMPLETIONS
    ###############################
    openai_completions_profile_data = {
        "service_kind": "openai",
        "endpoint": "v1/completions",
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {
                            "payload": '{"model":"test_model","prompt":["This is test"],"stream":true}',
                        },
                        "response_timestamps": [3, 5, 8, 12],
                        "response_outputs": [
                            {
                                "response": 'data: {"object":"text_completion","model":"test_model","choices":[{"text":" I"}]}\n\n'
                            },
                            {
                                "response": 'data: {"object":"text_completion","model":"test_model","choices":[{"text":" like"}]}\n\n'
                            },
                            {
                                "response": 'data: {"object":"text_completion","model":"test_model","choices":[{"text":" dogs"}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                    {
                        "timestamp": 2,
                        "request_inputs": {
                            "payload": '{"model":"test_model","prompt":["This is test too"],"stream":true}',
                        },
                        "response_timestamps": [4, 7, 11, 15, 18],
                        "response_outputs": [
                            {
                                "response": 'data: {"object":"text_completion","model":"test_model","choices":[{"text":" I"}]}\n\n'
                            },
                            {
                                "response": 'data: {"object":"text_completion","model":"test_model","choices":[{"text":" like"}]}\n\n'
                            },
                            {
                                "response": 'data: {"object":"text_completion","model":"test_model","choices":[{"text":" cats"}]}\n\n'
                            },
                            {
                                "response": 'data: {"object":"text_completion","model":"test_model","choices":[{"text":" too"}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                ],
            },
        ],
    }

    ###############################
    # OPENAI CHAT COMPLETIONS
    ###############################
    openai_profile_data = {
        "service_kind": "openai",
        "endpoint": "v1/chat/completions",
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {
                            "payload": '{"messages":[{"role":"user","content":"This is test"}],"model":"llama-2-7b","stream":true}',
                        },
                        # the first, and the last two responses will be ignored because they have no "content"
                        "response_timestamps": [3, 5, 8, 12, 13, 14],
                        "response_outputs": [
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"I"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":" like"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":" dogs"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{},"finish_reason":null}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                    {
                        "timestamp": 2,
                        "request_inputs": {
                            "payload": '{"messages":[{"role":"user","content":"This is test too"}],"model":"llama-2-7b","stream":true}',
                        },
                        # the first, and the last two responses will be ignored because they have no "content"
                        "response_timestamps": [4, 7, 11, 15, 18, 19],
                        "response_outputs": [
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"I"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"don\'t"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"cook food"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{},"finish_reason":null}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                ],
            },
        ],
    }

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=openai_profile_data,
    )
    @pytest.mark.parametrize(
        "infer_mode, load_level, expected_metrics",
        [
            (
                "concurrency",
                "10",
                {
                    "request_latencies": [11, 13],
                    "request_throughputs": [1 / ns_to_sec(7)],
                    "time_to_first_tokens": [4, 5],
                    "time_to_second_tokens": [3, 4],
                    "inter_token_latencies": [4, 2],
                    "output_token_throughputs_per_user": [
                        1 / ns_to_sec(4),
                        1 / ns_to_sec(2),
                    ],
                    "output_token_throughputs": [9 / ns_to_sec(14)],
                    "output_sequence_lengths": [3, 6],
                    "input_sequence_lengths": [3, 4],
                },
            ),
        ],
    )
    def test_openai_llm_profile_data(
        self,
        mock_json,
        infer_mode,
        load_level,
        expected_metrics,
    ) -> None:
        """Collect LLM metrics from profile export data and check values.

        Metrics
        * request_latencies
            - experiment 1: [12 - 1, 15 - 2] = [11, 13]
        * request_throughputs
            - experiment 1: [2/(15 - 1)] = [1/7]
        * time to first tokens
            - experiment 1: [5 - 1, 7 - 2] = [4, 5]
        * time to second tokens
            - experiment 1: [8 - 5, 11 - 7] = [3, 4]
        * inter token latencies
            - experiment 1: [((12 - 1) - 4)/(3 - 1), ((15 - 2) - 5)/(6 - 1)]
                          : [3.5, 1.6]
                          : [4, 2]  # rounded
        * output token throughputs per user
            - experiment 1: [1 / ns_to_sec(4), 1 / ns_to_sec(2)]
        * output token throughputs
            - experiment 1: [(3 + 6)/(15 - 1)] = [9/14]
        * output sequence lengths
            - experiment 1: [3, 6]
        * input sequence lengths
            - experiment 1: [3, 4]
        """
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("openai_profile_export.json"),
            tokenizer=tokenizer,
        )

        statistics = pd.get_statistics(infer_mode=infer_mode, load_level=load_level)
        metrics = cast(LLMMetrics, statistics.metrics)

        expected_metrics = LLMMetrics(**expected_metrics)
        expected_statistics = Statistics(expected_metrics)

        check_llm_metrics(metrics, expected_metrics)
        check_statistics(statistics, expected_statistics)

        # check non-existing profile data
        with pytest.raises(KeyError):
            pd.get_statistics(infer_mode="concurrency", load_level="40")

    ###############################
    # OPENAI VISION
    ###############################
    openai_vlm_profile_data = {
        "service_kind": "openai",
        "endpoint": "v1/chat/completions",
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {
                            "payload": '{"messages":[{"role":"user","content":[{"type":"text","text":"This is test"},{"type":"image_url","image_url":{"url":"data:image/png;base64,abcdef"}}]}],"model":"llava-1.6","stream":true}',
                        },
                        # the first, and the last two responses will be ignored because they have no "content"
                        "response_timestamps": [3, 5, 8, 12, 13, 14],
                        "response_outputs": [
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"I"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" like"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" dogs"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":null}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                    {
                        "timestamp": 2,
                        "request_inputs": {
                            "payload": '{"messages":[{"role":"user","content":[{"type":"text","text":"This is test too"},{"type":"image_url","image_url":{"url":"data:image/png;base64,abcdef"}}]}],"model":"llava-1.6","stream":true}',
                        },
                        # the first, and the last two responses will be ignored because they have no "content"
                        "response_timestamps": [4, 7, 11, 15, 18, 19],
                        "response_outputs": [
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"I"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"don\'t"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"cook food"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":null}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                ],
            },
        ],
    }

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=openai_vlm_profile_data,
    )
    @pytest.mark.parametrize(
        "infer_mode, load_level, expected_metrics",
        [
            (
                "concurrency",
                "10",
                {
                    "request_latencies": [11, 13],
                    "request_throughputs": [1 / ns_to_sec(7)],
                    "time_to_first_tokens": [4, 5],
                    "time_to_second_tokens": [3, 4],
                    "inter_token_latencies": [4, 2],
                    "output_token_throughputs_per_user": [
                        1 / ns_to_sec(4),
                        1 / ns_to_sec(2),
                    ],
                    "output_token_throughputs": [9 / ns_to_sec(14)],
                    "output_sequence_lengths": [3, 6],
                    "input_sequence_lengths": [3, 4],
                },
            ),
        ],
    )
    def test_openai_vlm_profile_data(
        self,
        mock_json,
        infer_mode,
        load_level,
        expected_metrics,
    ) -> None:
        """Collect LLM metrics from profile export data and check values.

        Metrics
        * request_latencies
            - experiment 1: [12 - 1, 15 - 2] = [11, 13]
        * request_throughputs
            - experiment 1: [2/(15 - 1)] = [1/7]
        * time to first tokens
            - experiment 1: [5 - 1, 7 - 2] = [4, 5]
        * time to second tokens
            - experiment 1: [8 - 5, 11 - 7] = [3, 4]
        * inter token latencies
            - experiment 1: [((12 - 1) - 4)/(3 - 1), ((15 - 2) - 5)/(6 - 1)]
                          : [3.5, 1.6]
                          : [4, 2]  # rounded
        * output token throughputs per user
            - experiment 1: [1 / ns_to_sec(4), 1 / ns_to_sec(2)]
        * output token throughputs
            - experiment 1: [(3 + 6)/(15 - 1)] = [9/14]
        * output sequence lengths
            - experiment 1: [3, 6]
        * input sequence lengths
            - experiment 1: [3, 4]
        """
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("openai_vlm_profile_export.json"),
            tokenizer=tokenizer,
        )

        statistics = pd.get_statistics(infer_mode=infer_mode, load_level=load_level)
        metrics = cast(LLMMetrics, statistics.metrics)

        expected_metrics = LLMMetrics(**expected_metrics)
        expected_statistics = Statistics(expected_metrics)

        check_llm_metrics(metrics, expected_metrics)
        check_statistics(statistics, expected_statistics)

        # check non-existing profile data
        with pytest.raises(KeyError):
            pd.get_statistics(infer_mode="concurrency", load_level="40")

    ###############################
    # TENSORRTLLM ENGINE
    ###############################
    tensorrtllm_engine_profile_data = {
        "service_kind": "triton_c_api",
        "endpoint": "",
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {
                            "streaming": True,
                            "request_output_len": 3,
                            "min_length": 3,
                            "input_lengths": 3,
                            "input_ids": [
                                111,
                                222,
                                333,
                            ],
                        },
                        "response_timestamps": [3, 5, 8],
                        "response_outputs": [
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 123,
                            },
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 456,
                            },
                            {
                                "output_ids": 789,
                            },
                        ],
                    },
                    {
                        "timestamp": 2,
                        "request_inputs": {
                            "streaming": True,
                            "request_output_len": 3,
                            "min_length": 3,
                            "input_lengths": 4,
                            "input_ids": [
                                111,
                                222,
                                333,
                                444,
                            ],
                        },
                        "response_timestamps": [4, 7, 11],
                        "response_outputs": [
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 123,
                            },
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 456,
                            },
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 789,
                            },
                        ],
                    },
                ],
            },
            {
                "experiment": {
                    "mode": "request_rate",
                    "value": 2.0,
                },
                "requests": [
                    {
                        "timestamp": 5,
                        "request_inputs": {
                            "streaming": True,
                            "request_output_len": 4,
                            "min_length": 4,
                            "input_lengths": 3,
                            "input_ids": [
                                111,
                                222,
                                333,
                            ],
                        },
                        "response_timestamps": [7, 8, 13, 18],
                        "response_outputs": [
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 123,
                            },
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 456,
                            },
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 789,
                            },
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 1011,
                            },
                        ],
                    },
                    {
                        "timestamp": 3,
                        "request_inputs": {
                            "streaming": True,
                            "request_output_len": 3,
                            "min_length": 3,
                            "input_lengths": 4,
                            "input_ids": [
                                111,
                                222,
                                333,
                                444,
                            ],
                        },
                        "response_timestamps": [6, 8, 11],
                        "response_outputs": [
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 123,
                            },
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 456,
                            },
                            {
                                "output_log_probs": [0, 0],
                                "output_ids": 789,
                            },
                        ],
                    },
                ],
            },
        ],
    }

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=tensorrtllm_engine_profile_data,
    )
    @pytest.mark.parametrize(
        "infer_mode, load_level, expected_metrics",
        [
            (
                "concurrency",
                "10",
                {
                    "request_latencies": [7, 9],
                    "request_throughputs": [1 / ns_to_sec(5)],
                    "time_to_first_tokens": [2, 2],
                    "time_to_second_tokens": [2, 3],
                    "inter_token_latencies": [2, 4],
                    "output_token_throughputs_per_user": [
                        1 / ns_to_sec(2),
                        1 / ns_to_sec(4),
                    ],
                    "output_token_throughputs": [3 / ns_to_sec(5)],
                    "output_sequence_lengths": [3, 3],
                    "input_sequence_lengths": [3, 4],
                },
            ),
            (
                "request_rate",
                "2.0",
                {
                    "request_latencies": [13, 8],
                    "request_throughputs": [2 / ns_to_sec(15)],
                    "time_to_first_tokens": [2, 3],
                    "time_to_second_tokens": [1, 2],
                    "inter_token_latencies": [4, 2],
                    "output_token_throughputs_per_user": [
                        1 / ns_to_sec(4),
                        1 / ns_to_sec(2),
                    ],
                    "output_token_throughputs": [7 / ns_to_sec(15)],
                    "output_sequence_lengths": [4, 3],
                    "input_sequence_lengths": [3, 4],
                },
            ),
        ],
    )
    def test_tensorrtllm_engine_llm_profile_data(
        self,
        mock_json,
        infer_mode,
        load_level,
        expected_metrics,
    ) -> None:
        """Collect LLM metrics from profile export data and check values.

        Metrics
        * request_latencies
            - experiment 1: [8 - 1, 11 - 2] = [7, 9]
            - experiment 2: [18 - 5, 11 -3] = [13, 8]
        * request_throughputs
            - experiment 1: [2/(11 - 1)] = [1/5]
            - experiment 2: [2/(18 - 3)] = [2/15]
        * time to first tokens
            - experiment 1: [3 - 1, 4 - 2] = [2, 2]
            - experiment 2: [7 - 5, 6 - 3] = [2, 3]
        * time to second tokens
            - experiment 1: [5 - 3, 7 - 4] = [2, 3]
            - experiment 2: [8 - 7, 8 - 6] = [1, 2]
        * inter token latencies
            - experiment 1: [((8 - 1) - 2)/(3 - 1), ((11 - 2) - 2)/(3 - 1)]
                          : [2.5, 3.5]
                          : [2, 4]  # rounded
            - experiment 2: [((18 - 5) - 2)/(4 - 1), ((11 - 3) - 3)/(3 - 1)]
                          : [11/3, 2.5]
                          : [4, 2]  # rounded
        * output token throughputs per user
            - experiment 1: [1 / ns_to_sec(2), 1 / ns_to_sec(4)]
            - experiment 2: [1 / ns_to_sec(4), 1 / ns_to_sec(2)]
        * output token throughputs
            - experiment 1: [(3 + 3)/(11 - 1)] = [3/5]
            - experiment 2: [(4 + 3)/(18 - 3)] = [7/15]
        * output sequence lengths
            - experiment 1: [3, 3]
            - experiment 2: [4, 3]
        * input sequence lengths
            - experiment 1: [3, 4]
            - experiment 2: [3, 4]
        """
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("tensorrtllm_engine_profile_export.json"),
            tokenizer=tokenizer,
        )

        statistics = pd.get_statistics(infer_mode=infer_mode, load_level=load_level)
        metrics = cast(LLMMetrics, statistics.metrics)

        expected_metrics = LLMMetrics(**expected_metrics)
        expected_statistics = Statistics(expected_metrics)

        check_llm_metrics(metrics, expected_metrics)
        check_statistics(statistics, expected_statistics)

        # check non-existing profile data
        with pytest.raises(KeyError):
            pd.get_statistics(infer_mode="concurrency", load_level="30")

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=openai_profile_data,
    )
    def test_openai_output_token_counts(self, mock_json) -> None:
        output_texts = [
            "Ad",
            "idas",
            " Orig",
            "inals",
            " are",
            " now",
            " available",
            " in",
            " more",
            " than",
        ]
        res_outputs = []
        for text in output_texts:
            response = f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}],"object":"chat.completion.chunk"}}\n\n'
            res_outputs.append({"response": response})

        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("openai_profile_export.json"),
            tokenizer=tokenizer,
        )

        output_token_counts, total_output_token = pd._get_output_token_counts(
            res_outputs
        )
        assert output_token_counts == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # total 10
        assert total_output_token == 9
        assert total_output_token != sum(output_token_counts)

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=triton_profile_data,
    )
    def test_triton_output_token_counts(self, mock_json) -> None:
        output_texts = [
            "Ad",
            "idas",
            " Orig",
            "inals",
            " are",
            " now",
            " available",
            " in",
            " more",
            " than",
        ]
        res_outputs = []
        for text in output_texts:
            res_outputs.append({"text_output": text})

        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("triton_profile_export.json"),
            tokenizer=tokenizer,
        )

        output_token_counts, total_output_token = pd._get_output_token_counts(
            res_outputs
        )
        assert output_token_counts == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # total 10
        assert total_output_token == 9
        assert total_output_token != sum(output_token_counts)

    empty_profile_data = {
        "service_kind": "openai",
        "endpoint": "v1/chat/completions",
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {
                            "payload": '{"messages":[{"role":"user","content":[{"type":"text","text":"This is test"}]}],"model":"llama-2-7b","stream":true}',
                        },
                        "response_timestamps": [3, 5, 8],
                        "response_outputs": [
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                ],
            },
        ],
    }

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=empty_profile_data,
    )
    def test_empty_response(self, mock_json) -> None:
        """Check if it handles all empty responses."""
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)

        # Should not throw error
        _ = LLMProfileDataParser(
            filename=Path("empty_profile_export.json"),
            tokenizer=tokenizer,
        )

    @pytest.mark.parametrize(
        "profile_data, res_outputs, expected_response",
        [
            # OpenAI Completions
            (
                openai_completions_profile_data,
                [
                    {
                        "response": (
                            'data: {"object":"text_completion","model":"test_model","choices":[{"text":"abc"}]}\n\n'
                            'data: {"object":"text_completion","model":"test_model","choices":[{"text":"1234"}]}\n\n'
                            'data: {"object":"text_completion","model":"test_model","choices":[{"text":"helloworld"}]}\n\n'
                        )
                    },
                    {"response": "data: [DONE]\n\n"},
                ],
                '{"object":"text_completion","model":"test_model","choices":[{"text":"abc1234helloworld"}]}',
            ),
            # OpenAI Chat Completions
            (
                openai_profile_data,
                [
                    {
                        "response": (
                            'data: {"choices":[{"delta":{"content":"abc"}}],"object":"chat.completion.chunk"}\n\n'
                            'data: {"choices":[{"delta":{"content":"1234"}}],"object":"chat.completion.chunk"}\n\n'
                            'data: {"choices":[{"delta":{"content":"helloworld"}}],"object":"chat.completion.chunk"}\n\n'
                        )
                    },
                    {"response": "data: [DONE]\n\n"},
                ],
                '{"choices":[{"delta":{"content":"abc1234helloworld"}}],"object":"chat.completion.chunk"}',
            ),
        ],
    )
    def test_merged_sse_responses(
        self, profile_data, res_outputs, expected_response
    ) -> None:
        """Test merging the multiple sse responses."""
        with patch(
            target="genai_perf.profile_data_parser.profile_data_parser.load_json",
            return_value=profile_data,
        ):
            config = ConfigCommand({"model_name": "test_model"})
            config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
            tokenizer = get_tokenizer(config)
            pd = LLMProfileDataParser(
                filename=Path("profile_export.json"),
                tokenizer=tokenizer,
            )

        res_timestamps = [i for i in range(len(res_outputs))]
        pd._preprocess_response(res_timestamps, res_outputs)
        assert res_outputs[0]["response"] == expected_response

    @pytest.mark.parametrize(
        "profile_data, res_outputs, expected_responses",
        [
            # OpenAI Completions
            (
                openai_completions_profile_data,
                [
                    # response 0 and 1 are single SSE response split into two.
                    {
                        "response": 'data: {"object":"text_com',
                    },
                    {
                        "response": 'pletion","model":"test_model","choices":[{"text":"abc"}]}\n\n',
                    },
                    # response 2 and 3 are two separate SSE responses overlapping some parts.
                    {
                        "response": 'data: {"object":"text_completion","model":"test',
                    },
                    {
                        "response": '_model","choices":[{"text":"1234"}]}\n\ndata: {"object":"text_completion","model":"test_model","choices":[{"text":"helloworld"}]}\n\n',
                    },
                    {"response": "data: [DONE]\n\n"},
                ],
                [
                    'data: {"object":"text_completion","model":"test_model","choices":[{"text":"abc"}]}\n\n',
                    'data: {"object":"text_completion","model":"test_model","choices":[{"text":"1234"}]}\n\n',
                    'data: {"object":"text_completion","model":"test_model","choices":[{"text":"helloworld"}]}\n\n',
                ],
            ),
            # OpenAI Chat Completions
            (
                openai_profile_data,
                [
                    # response 0 and 1 are single SSE response split into two.
                    {
                        "response": 'data: {"object":"chat.completion.chunk","choices":[{"de',
                    },
                    {
                        "response": 'lta":{"content":"abc"}}],"model":"test_model"}\n\n',
                    },
                    # response 2 and 3 are two separate SSE responses overlapping some parts.
                    {
                        "response": 'data: {"object":"chat.completion.chunk","choices":[{"de',
                    },
                    {
                        "response": 'lta":{"content":"123"}}],"model":"test_model"}\n\ndata: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"hello"}}],"model":"test_model"}\n\n',
                    },
                    {"response": "data: [DONE]\n\n"},
                ],
                [
                    'data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"abc"}}],"model":"test_model"}\n\n',
                    'data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"123"}}],"model":"test_model"}\n\n',
                    'data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"hello"}}],"model":"test_model"}\n\n',
                ],
            ),
            # Edge case: cut-off exactly after 'data: ' prefix.
            (
                openai_profile_data,
                [
                    # response 0 and 1 are single SSE response split into two.
                    {
                        "response": "data:",
                    },
                    {
                        "response": '{"object":"chat.completion.chunk","choices":[{"delta":{"content":"abc"}}],"model":"test_model"}\n\n',
                    },
                    {"response": "data: [DONE]\n\n"},
                ],
                [
                    'data:{"object":"chat.completion.chunk","choices":[{"delta":{"content":"abc"}}],"model":"test_model"}\n\n',
                ],
            ),
        ],
    )
    def test_splintered_sse_responses(
        self, profile_data, res_outputs, expected_responses
    ) -> None:
        """Check if the parser handles splintered SSE responses."""
        with patch(
            target="genai_perf.profile_data_parser.profile_data_parser.load_json",
            return_value=profile_data,
        ):
            config = ConfigCommand({"model_name": "test_model"})
            config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
            tokenizer = get_tokenizer(config)
            pd = LLMProfileDataParser(
                filename=Path("profile_export.json"),
                tokenizer=tokenizer,
            )

        res_timestamps = [i for i in range(len(res_outputs))]
        pd._preprocess_response(res_timestamps, res_outputs)

        assert len(res_outputs) == len(expected_responses)
        for out, expected_response in zip(res_outputs, expected_responses):
            assert out["response"] == expected_response

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=openai_profile_data,
    )
    def test_handle_non_data_sse_fields(self, mock_json) -> None:
        """Check if the parser can handle SSE comments or event field."""
        res_outputs = [
            {
                "response": ":\n\n",
            },
            {
                "response": ":\n\n",
            },
            {
                "response": 'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"token_id":4477,"role":"assistant","content":"Hello "}}],"model":"meta-llama"}\n\n'
            },
            {
                "response": ":\n\n",
            },
            {
                "response": ":\n\n",
            },
            {
                "response": 'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"token_id":4477,"role":"assistant","content":"world!"}}],"model":"meta-llama"}\n\n'
            },
            {
                "response": "event: some description\n\n",
            },
            {"response": "data: [DONE]\n\n"},
        ]
        res_timestamps = [i for i in range(len(res_outputs))]

        expected_responses = [
            'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"token_id":4477,"role":"assistant","content":"Hello "}}],"model":"meta-llama"}\n\n',
            'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"token_id":4477,"role":"assistant","content":"world!"}}],"model":"meta-llama"}\n\n',
        ]

        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("openai_profile_export.json"),
            tokenizer=tokenizer,
        )

        pd._preprocess_response(res_timestamps, res_outputs)

        assert len(res_outputs) == 2 and len(res_timestamps) == 2
        assert res_outputs[0]["response"] == expected_responses[0]
        assert res_outputs[1]["response"] == expected_responses[1]

    @pytest.mark.parametrize(
        "res_outputs",
        [
            [
                {
                    "response": 'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"token_id":4477,"role":"assistant","content":"Hello "}}],"model":"meta-llama"}\n\n'
                },
                {"response": "event: error: some error occurred.\n\n"},
                {
                    "response": 'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"token_id":4477,"role":"assistant","content":"world!"}}],"model":"meta-llama"}\n\n'
                },
                {"response": "data: [DONE]\n\n"},
            ],
            [
                {
                    "response": 'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"token_id":4477,"role":"assistant","content":"Hello "}}],"model":"meta-llama"}\n\nevent: error: some error occurred.\n\n'
                },
                {
                    "response": 'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"token_id":4477,"role":"assistant","content":"world!"}}],"model":"meta-llama"}\n\n'
                },
                {"response": "data: [DONE]\n\n"},
            ],
        ],
    )
    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=openai_profile_data,
    )
    def test_handle_sse_error(self, mock_json, res_outputs) -> None:
        """Check if the parser can handle SSE error field."""
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("openai_profile_export.json"),
            tokenizer=tokenizer,
        )

        with pytest.raises(GenAIPerfException) as excinfo:
            res_timestamps = [i for i in range(len(res_outputs))]
            pd._preprocess_response(res_timestamps, res_outputs)

        expected_error_msg = "Detected an error event in the SSE response: event: error: some error occurred."
        assert str(excinfo.value) == expected_error_msg

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=openai_profile_data,
    )
    def test_non_sse_response(self, mock_json) -> None:
        """Check if it handles single responses."""
        res_timestamps = [
            0,
        ]
        res_outputs = [
            {
                "response": '{"id":"1","object":"chat.completion","created":2,"model":"hf-internal-testing/llama-tokenizer","choices":[{"index":0,"message":{"role":"assistant","content":"A friend of mine, who is also a cook, writes a blog.","tool_calls":[]},"logprobs":null,"finish_reason":"length","stop_reason":null}],"usage":{"prompt_tokens":47,"total_tokens":1024,"completion_tokens":977}}'
            },
        ]
        expected_response = '{"id":"1","object":"chat.completion","created":2,"model":"hf-internal-testing/llama-tokenizer","choices":[{"index":0,"message":{"role":"assistant","content":"A friend of mine, who is also a cook, writes a blog.","tool_calls":[]},"logprobs":null,"finish_reason":"length","stop_reason":null}],"usage":{"prompt_tokens":47,"total_tokens":1024,"completion_tokens":977}}'

        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("openai_profile_export.json"),
            tokenizer=tokenizer,
        )

        pd._preprocess_response(res_timestamps, res_outputs)
        assert res_outputs[0]["response"] == expected_response

    ###############################
    # SESSION MODE
    ###############################
    session_profile_data = {
        "service_kind": "openai",
        "endpoint": "v1/chat/completions",
        "experiments": [
            {
                "experiment": {
                    "mode": "request_rate",
                    "value": 0.0,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {
                            "payload": (
                                "{"
                                '  "model": "test_model",'
                                '  "messages": ['
                                '    {"role":"user","content":"I like dog"}'
                                "  ]"
                                "}"
                            ),
                            "session_id": "session-id-123",
                        },
                        "response_timestamps": [5],
                        "response_outputs": [
                            {
                                "response": (
                                    "{"
                                    '  "object": "chat.completion",'
                                    '  "model": "test_model",'
                                    '  "choices": ['
                                    '    {"message":{"role":"assistant","content":"I like dog too"}}'
                                    "  ]"
                                    "}"
                                )
                            }
                        ],
                    },
                    {
                        "timestamp": 14,
                        "request_inputs": {
                            "payload": (
                                "{"
                                '  "model": "test_model",'
                                '  "messages": ['
                                '    {"role":"user","content":"I like dog"},'
                                '    {"role":"assistant","content":"I like dog too"},'
                                '    {"role":"user","content":"I like cat"}'
                                "  ]"
                                "}"
                            ),
                            "session_id": "session-id-123",
                        },
                        "response_timestamps": [17],
                        "response_outputs": [
                            {
                                "response": (
                                    "{"
                                    '  "object": "chat.completion",'
                                    '  "model": "test_model",'
                                    '  "choices": ['
                                    '    {"message":{"role":"assistant","content":"I like cat too"}}'
                                    "  ]"
                                    "}"
                                )
                            }
                        ],
                    },
                ],
            },
        ],
    }

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=session_profile_data,
    )
    def test_session_metrics(self, mock_json) -> None:
        """Check if it handles session metrics."""
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("session_profile_export.json"),
            tokenizer=tokenizer,
        )

        expected_session_metrics = {
            # [5 - 1, 17 - 14]
            "request_latencies": [4, 3],
            # [2/(17 - 1)]
            "request_throughputs": [2 / ns_to_sec(16)],
            # Same as request_latencies
            "time_to_first_tokens": [4, 3],
            # N/A (not streaming)
            "time_to_second_tokens": [],
            "inter_token_latencies": [],
            "output_token_throughputs_per_user": [],
            # [(4 + 4)/(17 - 1)]
            "output_token_throughputs": [8 / ns_to_sec(16)],
            # [4, 4]
            "output_sequence_lengths": [4, 4],
            # [3, (3 + 4 + 3)]
            "input_sequence_lengths": [3, 10],
        }

        session_metrics = pd._session_metrics
        assert len(session_metrics) == 1
        assert "session-id-123" in session_metrics
        assert session_metrics["session-id-123"] == expected_session_metrics

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=openai_profile_data,
    )
    def test_no_session_metrics(self, mock_json) -> None:
        """Check if it handles profile export files without session metrics."""
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("openai_profile_export.json"),
            tokenizer=tokenizer,
        )

        session_metrics = pd._session_metrics
        assert session_metrics == {}

    ###############################
    # COMMON FUNCTIONALITY
    ###############################

    @pytest.mark.parametrize(
        "response_format,response_text,expected",
        [
            (
                ResponseFormat.HUGGINGFACE_GENERATE,
                '[{"generated_text":"Hello"}]',
                "Hello",
            ),
            (ResponseFormat.HUGGINGFACE_GENERATE, "[]", ""),
            (ResponseFormat.HUGGINGFACE_GENERATE, '["not a dict"]', None),
            (ResponseFormat.HUGGINGFACE_GENERATE, '{"generated_text":"Hello"}', None),
            (
                ResponseFormat.OPENAI_CHAT_COMPLETIONS,
                '{"object":"chat.completion","choices":[{"message":{"content":"Hi"}}]}',
                "Hi",
            ),
            (
                ResponseFormat.OPENAI_CHAT_COMPLETIONS,
                '{"object":"chat.completion.chunk","choices":[{"delta":{"content":"Stream"}}]}',
                "Stream",
            ),
            (
                ResponseFormat.OPENAI_CHAT_COMPLETIONS,
                '{"choices":[{"text":"Fallback"}]}',
                "Fallback",
            ),
            (
                ResponseFormat.OPENAI_COMPLETIONS,
                '{"object":"text_completion","choices":[{"text":"Complete"}]}',
                "Complete",
            ),
            (
                ResponseFormat.OPENAI_COMPLETIONS,
                '{"choices":[{"text":"Legacy"}]}',
                "Legacy",
            ),
            (ResponseFormat.TRITON_GENERATE, '{"text_output":"Triton"}', "Triton"),
            (ResponseFormat.OPENAI_CHAT_COMPLETIONS, "", ""),
            (ResponseFormat.OPENAI_CHAT_COMPLETIONS, "data: [DONE]", ""),
            (ResponseFormat.OPENAI_CHAT_COMPLETIONS, ": ping", ""),
            # Unknown format (should raise)
            ("UNKNOWN_FORMAT", "any response", None),
        ],
    )
    def test_extract_text_output(self, response_format, response_text, expected):
        parser = LLMProfileDataParser.__new__(LLMProfileDataParser)  # Bypass init
        parser._response_format = response_format  # type: ignore

        if expected is None:
            with pytest.raises(ValueError):
                parser._extract_text_output(response_text)
        else:
            assert parser._extract_text_output(response_text) == expected

    @pytest.mark.parametrize(
        "response_format, response_text, expected_is_empty",
        [
            (
                ResponseFormat.OPENAI_CHAT_COMPLETIONS,
                '{"object":"chat.completion","choices":[{"message":{"content":"Hello"}}]}',
                False,
            ),
            (ResponseFormat.OPENAI_CHAT_COMPLETIONS, "", True),
            (ResponseFormat.OPENAI_CHAT_COMPLETIONS, "data: [DONE]", True),
            (ResponseFormat.OPENAI_CHAT_COMPLETIONS, ": ping", True),
            (
                ResponseFormat.HUGGINGFACE_GENERATE,
                '[{"generated_text":"Hello"}]',
                False,
            ),
            (ResponseFormat.HUGGINGFACE_GENERATE, "[]", True),
        ],
    )
    def test_is_empty_response(self, response_format, response_text, expected_is_empty):
        parser = LLMProfileDataParser.__new__(LLMProfileDataParser)
        parser._response_format = response_format
        assert parser._is_empty_response(response_text) == expected_is_empty

    ###############################
    # HUGGINGFACE GENERATE
    ###############################
    huggingface_generate_profile_data = {
        "service_kind": "openai",
        "endpoint": "huggingface/generate",
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {
                            "payload": '{"inputs":"This is test","parameters":{"max_new_tokens":10}}',
                        },
                        "response_timestamps": [3, 5, 8],
                        "response_outputs": [
                            {"response": '[{"generated_text":" I"}]'},
                            {"response": '[{"generated_text":" like"}]'},
                            {"response": '[{"generated_text":" dogs"}]'},
                        ],
                    },
                    {
                        "timestamp": 2,
                        "request_inputs": {
                            "payload": '{"inputs":"This is test too","parameters":{"max_new_tokens":10}}',
                        },
                        "response_timestamps": [4, 7, 11],
                        "response_outputs": [
                            {"response": '[{"generated_text":" I"}]'},
                            {"response": '[{"generated_text":" like"}]'},
                            {"response": '[{"generated_text":" cats"}]'},
                        ],
                    },
                ],
            },
        ],
    }

    @patch(
        "genai_perf.profile_data_parser.profile_data_parser.load_json",
        return_value=huggingface_generate_profile_data,
    )
    @pytest.mark.parametrize(
        "infer_mode, load_level, expected_metrics",
        [
            (
                "concurrency",
                "10",
                {
                    "request_latencies": [7, 9],
                    "request_throughputs": [2 / ns_to_sec(10)],
                    "time_to_first_tokens": [2, 2],
                    "time_to_second_tokens": [2, 3],
                    "inter_token_latencies": [2, 2],
                    "output_token_throughputs_per_user": [
                        1 / ns_to_sec(2),
                        1 / ns_to_sec(2),
                    ],
                    "output_token_throughputs": [9 / ns_to_sec(10)],
                    "output_sequence_lengths": [4, 5],
                    "input_sequence_lengths": [3, 4],
                },
            ),
        ],
    )
    def test_huggingface_generate_llm_profile_data(
        self,
        mock_json,
        infer_mode,
        load_level,
        expected_metrics,
    ) -> None:
        """Collect LLM metrics from HuggingFace generate profile export data and check values.

        Metrics
        * request_latencies
            - experiment 1: [8 - 1, 11 - 2] = [7, 9]
        * request_throughputs
            - experiment 1: [2/(11 - 1)] = [2/10]
        * time to first tokens
            - experiment 1: [3 - 1, 4 - 2] = [2, 2]
        * time to second tokens
            - experiment 1: [5 - 3, 7 - 4] = [2, 3]
        * inter token latencies
            - experiment 1: [((8 - 1) - 2)/(4 - 1), ((11 - 2) - 2)/(5 - 1)]
                          : [5/3, 9/5]
                          : [2, 2]  # rounded
        * output token throughputs per user
            - experiment 1: [1 / ns_to_sec(2), 1 / ns_to_sec(2)]
        * output token throughputs
            - experiment 1: [(4 + 5)/(11 - 1)] = [9/10]
        * output sequence lengths
            - experiment 1: [4, 5]
        * input sequence lengths
            - experiment 1: [3, 4]
        """
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "hf-internal-testing/llama-tokenizer"
        tokenizer = get_tokenizer(config)
        pd = LLMProfileDataParser(
            filename=Path("huggingface_generate_profile_export.json"),
            tokenizer=tokenizer,
        )

        statistics = pd.get_statistics(infer_mode=infer_mode, load_level=load_level)
        metrics = cast(LLMMetrics, statistics.metrics)

        expected_metrics = LLMMetrics(**expected_metrics)
        expected_statistics = Statistics(expected_metrics)

        check_llm_metrics(metrics, expected_metrics)
        check_statistics(statistics, expected_statistics)

        # check non-existing profile data
        with pytest.raises(KeyError):
            pd.get_statistics(infer_mode="concurrency", load_level="40")
