# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections import namedtuple
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from genai_perf import tokenizer
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs import input_constants as ic
from genai_perf.inputs.input_constants import (
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
)
from genai_perf.inputs.inputs import Inputs
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.tokenizer import DEFAULT_TOKENIZER, get_tokenizer


class TestInputs:
    # Define service kind, backend or api, and output format combinations
    SERVICE_KIND_BACKEND_ENDPOINT_TYPE_FORMATS = [
        ("triton", "vllm", OutputFormat.VLLM),
        ("triton", "tensorrtllm", OutputFormat.TENSORRTLLM),
        ("openai", "v1/completions", OutputFormat.OPENAI_COMPLETIONS),
        ("openai", "v1/chat/completions", OutputFormat.OPENAI_CHAT_COMPLETIONS),
        ("openai", "v1/chat/completions", OutputFormat.OPENAI_VISION),
    ]

    # TODO (TMA-1754): Add tests that verify json schemas
    @pytest.fixture(scope="class")
    def default_tokenizer(self):
        yield tokenizer.get_tokenizer(tokenizer.DEFAULT_TOKENIZER)

    def test_input_type_synthetic_no_tokenizer(self):
        """
        Test for exception when input type is SYNTHETIC and no tokenizer
        """
        inputs = Inputs(
            InputsConfig(
                input_type=PromptSource.SYNTHETIC,
                tokenizer=None,
            )
        )
        with pytest.raises(GenAIPerfException):
            _ = inputs._check_for_tokenzier_if_input_type_is_synthetic()

    def test_illegal_starting_index(self):
        """
        Test for exceptions when illegal values are given for starting index
        """
        inputs = Inputs(
            InputsConfig(
                starting_index="foo",
            )
        )

        with pytest.raises(GenAIPerfException):
            _ = inputs._check_for_valid_starting_index()  # type: ignore

        inputs.config.starting_index = -1
        with pytest.raises(GenAIPerfException):
            _ = inputs._check_for_valid_starting_index()

    def test_illegal_length(self):
        """
        Test for exceptions when illegal values are given for length
        """
        inputs = Inputs(
            InputsConfig(
                length="foo",
            )
        )
        with pytest.raises(GenAIPerfException):
            _ = inputs._check_for_valid_length()  # type: ignore

        inputs.config.length = -1
        with pytest.raises(GenAIPerfException):
            _ = inputs._check_for_valid_length()

    # TODO (TPA-114) Refactor LLM inputs and testing
    # def test_inputs_with_non_default_length(self):
    #     """
    #     Test that non-default length works
    #     """
    #     configured_url = default_inputs._create_configured_url(
    #         ic.OPEN_ORCA_URL,
    #         ic.DEFAULT_STARTING_INDEX,
    #         (int(ic.DEFAULT_LENGTH / 2)),
    #     )
    #     dataset = default_inputs._download_dataset(
    #         configured_url,
    #     )
    #     dataset_json = default_inputs._convert_input_url_dataset_to_generic_json(
    #         dataset=dataset
    #     )

    #     assert dataset_json is not None
    #     assert len(dataset_json["rows"]) == ic.DEFAULT_LENGTH / 2

    # def test_convert_default_json_to_pa_format(self, default_configured_url):
    #     """
    #     Test that conversion to PA JSON format is correct
    #     """
    #     dataset = default_inputs._download_dataset(
    #         default_configured_url,
    #     )
    #     dataset_json = default_inputs._convert_input_url_dataset_to_generic_json(
    #         dataset=dataset
    #     )
    #     pa_json = default_inputs._convert_generic_json_to_output_format(
    #         output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
    #         generic_dataset=dataset_json,
    #         add_model_name=False,
    #         add_stream=False,
    #         extra_inputs={},
    #         output_tokens_mean=ic.DEFAULT_OUTPUT_TOKENS_MEAN,
    #         output_tokens_stddev=ic.DEFAULT_OUTPUT_TOKENS_STDDEV,
    #         output_tokens_deterministic=False,
    #         model_name=["test_model_A"],
    #     )

    #     assert pa_json is not None
    #     assert len(pa_json["data"]) == ic.DEFAULT_LENGTH

    # def test_create_openai_inputs_cnn_dailymail(self):
    #     """
    #     Test CNN_DAILYMAIL can be accessed
    #     """
    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.DATASET,
    #         dataset_name=CNN_DAILY_MAIL,
    #         output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
    #         model_name=["test_model_A"],
    #     )

    #     os.remove(DEFAULT_INPUT_DATA_JSON)

    #     assert pa_json is not None
    #     assert len(pa_json["data"]) == ic.DEFAULT_LENGTH

    # def test_write_to_file(self):
    #     """
    #     Test that write to file is working correctly
    #     """
    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.DATASET,
    #         dataset_name=ic.OPEN_ORCA,
    #         output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
    #         model_name="open_orca",
    #         add_model_name=True,
    #         add_stream=True,
    #     )
    #     try:
    #         with open(DEFAULT_INPUT_DATA_JSON, "r") as f:
    #             json_str = f.read()
    #     finally:
    #         os.remove(DEFAULT_INPUT_DATA_JSON)

    #     assert pa_json == json.loads(json_str)

    # def test_create_openai_to_vllm(self):
    #     """
    #     Test conversion of openai to vllm
    #     """
    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.DATASET,
    #         output_format=OutputFormat.VLLM,
    #         dataset_name=ic.OPEN_ORCA,
    #         add_model_name=False,
    #         add_stream=True,
    #         model_name=["test_model_A"],
    #     )

    #     os.remove(DEFAULT_INPUT_DATA_JSON)

    #     assert pa_json is not None
    #     assert len(pa_json["data"]) == ic.DEFAULT_LENGTH

    # def test_create_openai_to_completions(self):
    #     """
    #     Test conversion of openai to completions
    #     """
    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.DATASET,
    #         output_format=OutputFormat.OPENAI_COMPLETIONS,
    #         dataset_name=ic.OPEN_ORCA,
    #         add_model_name=False,
    #         add_stream=True,
    #         model_name=["test_model_A"],
    #     )

    #     os.remove(DEFAULT_INPUT_DATA_JSON)

    #     assert pa_json is not None
    #     assert len(pa_json["data"]) == ic.DEFAULT_LENGTH
    #     # NIM legacy completion endpoint only supports string and not
    #     # array of strings. Verify that the prompt is of type string
    #     # not list
    #     assert isinstance(pa_json["data"][0]["payload"][0]["prompt"], str)

    # def test_create_openai_to_trtllm(self):
    #     """
    #     Test conversion of openai to trtllm
    #     """
    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.DATASET,
    #         output_format=OutputFormat.TENSORRTLLM,
    #         dataset_name=ic.OPEN_ORCA,
    #         add_model_name=False,
    #         add_stream=True,
    #         model_name=["test_model_A"],
    #     )

    #     os.remove(DEFAULT_INPUT_DATA_JSON)

    #     assert pa_json is not None
    #     assert len(pa_json["data"]) == ic.DEFAULT_LENGTH

    # def test_random_synthetic_no_stddev(self, default_tokenizer):
    #     """
    #     Test that we can produce an exact number of random synthetic tokens
    #     """
    #     random.seed(1)

    #     def _subtest(token_length):
    #         synthetic_prompt = default_inputs._create_synthetic_prompt(
    #             tokenizer=default_tokenizer,
    #             prompt_tokens_mean=token_length,
    #             prompt_tokens_stddev=0,
    #         )

    #         actual_token_length = len(default_tokenizer.encode(synthetic_prompt))
    #         assert token_length == actual_token_length

    #     # Test all of 500-600 to make sure exact
    #     for i in range(500, 600):
    #         _subtest(i)

    #     # Test some larger values
    #     _subtest(1500)
    #     _subtest(10000)

    # def test_random_synthetic_stddev(self, default_tokenizer):
    #     """
    #     Test that we can produce random synthetic tokens within a requested stddev
    #     """
    #     random.seed(1)

    #     def _subtest(num_samples, mean, stddev):
    #         prompt_tokens = []
    #         for _ in range(num_samples):
    #             prompt = default_inputs._create_synthetic_prompt(
    #                 tokenizer=default_tokenizer,
    #                 prompt_tokens_mean=mean,
    #                 prompt_tokens_stddev=stddev,
    #             )
    #             prompt_tokens.append(len(default_tokenizer.encode(prompt)))

    #         assert statistics.mean(prompt_tokens) == pytest.approx(mean, rel=0.1)
    #         assert statistics.stdev(prompt_tokens) == pytest.approx(stddev, rel=0.2)

    #     _subtest(50, 200, 20)
    #     _subtest(50, 400, 10)
    #     _subtest(200, 50, 10)

    # def test_random_seed(self, default_tokenizer):
    #     """
    #     Test that when given the same seed, create_inputs will return the same result,
    #     and that when given a different seed, it will produce a different result
    #     """

    #     inputs_seed5_a = default_inputs.create_inputs(
    #         tokenizer=default_tokenizer,
    #         input_type=PromptSource.SYNTHETIC,
    #         output_format=OutputFormat.TENSORRTLLM,
    #         prompt_tokens_mean=300,
    #         prompt_tokens_stddev=20,
    #         num_prompts=5,
    #         random_seed=5,
    #         model_name=["test_model_A"],
    #     )

    #     inputs_seed5_b = default_inputs.create_inputs(
    #         tokenizer=default_tokenizer,
    #         input_type=PromptSource.SYNTHETIC,
    #         output_format=OutputFormat.TENSORRTLLM,
    #         prompt_tokens_mean=300,
    #         prompt_tokens_stddev=20,
    #         num_prompts=5,
    #         random_seed=5,
    #         model_name=["test_model_A"],
    #     )

    #     inputs_seed10 = default_inputs.create_inputs(
    #         tokenizer=default_tokenizer,
    #         input_type=PromptSource.SYNTHETIC,
    #         output_format=OutputFormat.TENSORRTLLM,
    #         prompt_tokens_mean=300,
    #         prompt_tokens_stddev=20,
    #         num_prompts=5,
    #         random_seed=10,
    #         model_name=["test_model_A"],
    #     )

    #     assert inputs_seed5_a == inputs_seed5_b
    #     assert inputs_seed5_a != inputs_seed10

    # def test_synthetic_to_vllm(self, default_tokenizer):
    #     """
    #     Test generating synthetic prompts and converting to vllm
    #     """
    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.SYNTHETIC,
    #         output_format=OutputFormat.VLLM,
    #         num_prompts=5,
    #         add_model_name=False,
    #         add_stream=True,
    #         tokenizer=default_tokenizer,
    #         model_name=["test_model_A"],
    #     )

    #     os.remove(DEFAULT_INPUT_DATA_JSON)

    #     assert pa_json is not None
    #     assert len(pa_json["data"]) == 5

    # def test_synthetic_to_trtllm(self, default_tokenizer):
    #     """
    #     Test generating synthetic prompts and converting to trtllm
    #     """
    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.SYNTHETIC,
    #         output_format=OutputFormat.TENSORRTLLM,
    #         num_prompts=5,
    #         add_model_name=False,
    #         add_stream=True,
    #         tokenizer=default_tokenizer,
    #         model_name=["test_model_A"],
    #     )

    #     os.remove(DEFAULT_INPUT_DATA_JSON)

    #     assert pa_json is not None
    #     assert len(pa_json["data"]) == 5

    # def test_synthetic_to_openai_chat_completions(self, default_tokenizer):
    #     """
    #     Test generating synthetic prompts and converting to OpenAI chat completions
    #     """
    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.SYNTHETIC,
    #         output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
    #         num_prompts=5,
    #         add_model_name=False,
    #         add_stream=True,
    #         tokenizer=default_tokenizer,
    #         model_name=["test_model_A"],
    #     )

    #     os.remove(DEFAULT_INPUT_DATA_JSON)

    #     assert pa_json is not None
    #     assert len(pa_json["data"]) == 5

    # def test_synthetic_to_openai_completions(self, default_tokenizer):
    #     """
    #     Test generating synthetic prompts and converting to OpenAI completions
    #     """
    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.SYNTHETIC,
    #         output_format=OutputFormat.OPENAI_COMPLETIONS,
    #         num_prompts=5,
    #         add_model_name=False,
    #         add_stream=True,
    #         tokenizer=default_tokenizer,
    #         model_name=["test_model_A"],
    #     )

    #     os.remove(DEFAULT_INPUT_DATA_JSON)

    #     assert pa_json is not None
    #     assert len(pa_json["data"]) == 5

    # @pytest.mark.parametrize(
    #     "output_format",
    #     [format[2] for format in SERVICE_KIND_BACKEND_ENDPOINT_TYPE_FORMATS],
    # )
    # def test_extra_inputs(
    #     self, default_tokenizer: Tokenizer, output_format: OutputFormat
    # ) -> None:
    #     input_name = "max_tokens"
    #     input_value = 5
    #     request_inputs = {input_name: input_value}

    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.SYNTHETIC,
    #         output_format=output_format,
    #         num_prompts=5,
    #         add_model_name=False,
    #         add_stream=True,
    #         tokenizer=default_tokenizer,
    #         extra_inputs=request_inputs,
    #         model_name=["test_model_A"],
    #     )

    #     assert len(pa_json["data"]) == 5

    #     if (
    #         output_format == OutputFormat.OPENAI_CHAT_COMPLETIONS
    #         or output_format == OutputFormat.OPENAI_COMPLETIONS
    #     ):
    #         for entry in pa_json["data"]:
    #             assert "payload" in entry, "Payload is missing in the request"
    #             payload = entry["payload"]
    #             for item in payload:
    #                 assert (
    #                     input_name in item
    #                 ), f"The input name {input_name} is not present in the request"
    #                 assert (
    #                     item[input_name] == input_value
    #                 ), f"The value of {input_name} is incorrect"
    #     elif (
    #         output_format == OutputFormat.TENSORRTLLM
    #         or output_format == OutputFormat.VLLM
    #     ):
    #         for entry in pa_json["data"]:
    #             assert (
    #                 input_name in entry
    #             ), f"The {input_name} is not present in the request"
    #             assert entry[input_name] == [
    #                 input_value
    #             ], f"The value of {input_name} is incorrect"
    #     else:
    #         assert False, f"Unsupported output format: {output_format}"

    @pytest.mark.parametrize(
        "generic_json, add_stream, output_tokens_mean, output_tokens_deterministic, expected_json",
        [
            (
                # generic_json
                {
                    "rows": [
                        {"text_input": "test input one"},
                        {"text_input": "test input two"},
                    ]
                },
                False,
                -1,
                False,
                # expected_json
                {
                    "data": [
                        {
                            "input_ids": {
                                "content": [1243, 1881, 697],
                                "shape": [3],
                            },
                            "input_lengths": [3],
                            "request_output_len": [ic.DEFAULT_TENSORRTLLM_MAX_TOKENS],
                        },
                        {
                            "input_ids": {
                                "content": [1243, 1881, 1023],
                                "shape": [3],
                            },
                            "input_lengths": [3],
                            "request_output_len": [ic.DEFAULT_TENSORRTLLM_MAX_TOKENS],
                        },
                    ],
                },
            ),
            (
                # generic_json
                {
                    "rows": [
                        {"text_input": "test input one"},
                        {"text_input": "test input two"},
                    ]
                },
                True,
                999,
                True,
                # expected_json
                {
                    "data": [
                        {
                            "input_ids": {
                                "content": [1243, 1881, 697],
                                "shape": [3],
                            },
                            "input_lengths": [3],
                            "request_output_len": [999],
                            "min_length": [999],
                            "streaming": [True],
                        },
                        {
                            "input_ids": {
                                "content": [1243, 1881, 1023],
                                "shape": [3],
                            },
                            "input_lengths": [3],
                            "request_output_len": [999],
                            "min_length": [999],
                            "streaming": [True],
                        },
                    ],
                },
            ),
        ],
    )
    def test_generic_json_to_trtllm_engine_format(
        self,
        generic_json,
        add_stream,
        output_tokens_mean,
        output_tokens_deterministic,
        expected_json,
    ) -> None:
        inputs = Inputs(
            InputsConfig(
                output_format=OutputFormat.TENSORRTLLM_ENGINE,
                tokenizer=get_tokenizer(DEFAULT_TOKENIZER),
                add_stream=add_stream,
                extra_inputs={},
                output_tokens_mean=output_tokens_mean,
                output_tokens_stddev=0,
                output_tokens_deterministic=output_tokens_deterministic,
            )
        )
        trtllm_json = inputs._convert_generic_json_to_output_format(generic_json)

        assert trtllm_json == expected_json

    @pytest.mark.parametrize(
        "row, expected_content",
        [
            # text and image
            (
                {"text_input": "test input one", "image": "test_image1"},
                [
                    {
                        "type": "text",
                        "text": "test input one",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "test_image1",
                        },
                    },
                ],
            ),
            # image only
            (
                {"image": "test_image1"},
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "test_image1",
                        },
                    },
                ],
            ),
        ],
    )
    def test_openai_multi_modal_json(self, row, expected_content) -> None:
        inputs = Inputs(
            InputsConfig(
                add_stream=True,
                extra_inputs={},
                output_tokens_mean=10,
                output_tokens_stddev=0,
                model_name=["test_model"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            )
        )

        generic_json = {"rows": [row]}
        pa_json = inputs._convert_generic_json_to_output_format(
            generic_json,
        )

        assert pa_json == {
            "data": [
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": expected_content,
                                }
                            ],
                            "stream": True,
                            "max_tokens": 10,
                        }
                    ]
                }
            ]
        }

    # def test_trtllm_default_max_tokens(self, default_tokenizer: Tokenizer) -> None:
    #     input_name = "max_tokens"
    #     input_value = 256

    #     pa_json = default_inputs.create_inputs(
    #         input_type=PromptSource.SYNTHETIC,
    #         output_format=OutputFormat.TENSORRTLLM,
    #         num_prompts=5,
    #         add_model_name=False,
    #         add_stream=True,
    #         tokenizer=default_tokenizer,
    #         model_name=["test_model_A"],
    #     )

    #     assert len(pa_json["data"]) == 5
    #     for entry in pa_json["data"]:
    #         assert (
    #             input_name in entry
    #         ), f"The {input_name} is not present in the request"
    #         assert entry[input_name] == [
    #             input_value
    #         ], f"The value of {input_name} is incorrect"

    # @pytest.mark.parametrize(
    #     "output_format",
    #     [format[2] for format in SERVICE_KIND_BACKEND_ENDPOINT_TYPE_FORMATS],
    # )
    # def test_output_tokens_mean(self, output_format, default_tokenizer):
    #     if (
    #         output_format != OutputFormat.VLLM
    #         and output_format != OutputFormat.TENSORRTLLM
    #     ):
    #         return

    #     output_tokens_mean = 100
    #     output_tokens_stddev = 0
    #     for deterministic in [True, False]:
    #         _ = default_inputs.create_inputs(
    #             input_type=PromptSource.SYNTHETIC,
    #             output_format=output_format,
    #             num_prompts=5,
    #             add_model_name=False,
    #             add_stream=True,
    #             tokenizer=default_tokenizer,
    #             output_tokens_mean=output_tokens_mean,
    #             output_tokens_stddev=output_tokens_stddev,
    #             output_tokens_deterministic=deterministic,
    #             model_name=["test_model_A"],
    #         )

    #         assert os.path.exists(
    #             DEFAULT_INPUT_DATA_JSON
    #         ), "inputs.json file is not created"

    #         with open(DEFAULT_INPUT_DATA_JSON, "r") as f:
    #             inputs_data = json.load(f)

    #         for entry in inputs_data["data"]:
    #             if output_format == OutputFormat.VLLM:
    #                 assert (
    #                     "sampling_parameters" in entry
    #                 ), "sampling_parameters is missing in inputs.json"
    #                 sampling_parameters = json.loads(entry["sampling_parameters"][0])
    #                 assert (
    #                     "max_tokens" in sampling_parameters
    #                 ), "max_tokens parameter is missing in sampling_parameters"
    #                 assert sampling_parameters["max_tokens"] == str(
    #                     output_tokens_mean
    #                 ), "max_tokens parameter is not properly set"
    #                 if deterministic:
    #                     assert (
    #                         "min_tokens" in sampling_parameters
    #                     ), "min_tokens parameter is missing in sampling_parameters"
    #                     assert sampling_parameters["min_tokens"] == str(
    #                         output_tokens_mean
    #                     ), "min_tokens parameter is not properly set"
    #                 else:
    #                     assert (
    #                         "min_tokens" not in sampling_parameters
    #                     ), "min_tokens parameter is present in sampling_parameters"
    #             elif output_format == OutputFormat.TENSORRTLLM:
    #                 assert (
    #                     "max_tokens" in entry
    #                 ), "max_tokens parameter is missing in inputs.json"
    #                 assert (
    #                     entry["max_tokens"][0] == output_tokens_mean
    #                 ), "max_tokens parameter is not properly set"
    #                 if deterministic:
    #                     assert (
    #                         "min_length" in entry
    #                     ), "min_length parameter is missing in inputs.json"
    #                     assert (
    #                         entry["min_length"][0] == output_tokens_mean
    #                     ), "min_length parameter is not properly set"
    #                 else:
    #                     assert (
    #                         "min_length" not in entry
    #                     ), "min_length parameter is present in inputs.json"
    #             else:
    #                 assert False, f"Unsupported output format: {output_format}"

    #         os.remove(DEFAULT_INPUT_DATA_JSON)
