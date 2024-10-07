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

from unittest.mock import patch

import pytest
import responses
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs import input_constants as ic
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.input_retriever_factory import InputRetrieverFactory
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.synthetic_image_generator import ImageFormat
from genai_perf.tokenizer import DEFAULT_TOKENIZER, get_tokenizer


class TestInputRetrieverFactory:

    @patch(
        "genai_perf.inputs.input_retriever_factory.InputRetrieverFactory._create_synthetic_prompt",
        return_value="This is test prompt",
    )
    @patch(
        "genai_perf.inputs.input_retriever_factory.InputRetrieverFactory._create_synthetic_image",
        return_value="test_image_base64",
    )
    @pytest.mark.parametrize(
        "output_format",
        [
            OutputFormat.OPENAI_CHAT_COMPLETIONS,
            OutputFormat.OPENAI_COMPLETIONS,
            OutputFormat.OPENAI_EMBEDDINGS,
            OutputFormat.RANKINGS,
            OutputFormat.OPENAI_VISION,
            OutputFormat.VLLM,
            OutputFormat.TENSORRTLLM,
            OutputFormat.TENSORRTLLM_ENGINE,
            OutputFormat.IMAGE_RETRIEVAL,
        ],
    )
    def test_get_input_dataset_from_synthetic(
        self, mock_prompt, mock_image, output_format
    ) -> None:
        _placeholder = 123  # dummy value
        num_prompts = 3

        input_retriever_factory = InputRetrieverFactory(
            InputsConfig(
                tokenizer=get_tokenizer(DEFAULT_TOKENIZER),
                prompt_tokens_mean=_placeholder,
                prompt_tokens_stddev=_placeholder,
                num_prompts=num_prompts,
                image_width_mean=_placeholder,
                image_width_stddev=_placeholder,
                image_height_mean=_placeholder,
                image_height_stddev=_placeholder,
                image_format=ImageFormat.PNG,
                output_format=output_format,
            )
        )
        dataset_json = input_retriever_factory._get_input_dataset_from_synthetic()

        assert len(dataset_json["rows"]) == num_prompts

        for i in range(num_prompts):
            row = dataset_json["rows"][i]["row"]

            if output_format == OutputFormat.OPENAI_VISION:
                assert row == {
                    "text_input": "This is test prompt",
                    "image": "test_image_base64",
                }
            else:
                assert row == {
                    "text_input": "This is test prompt",
                }
