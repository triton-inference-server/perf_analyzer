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

mocked_openorca_data = {
    "features": [
        {"feature_idx": 0, "name": "id", "type": {"dtype": "string", "_type": "Value"}},
        {
            "feature_idx": 1,
            "name": "system_prompt",
            "type": {"dtype": "string", "_type": "Value"},
        },
        {
            "feature_idx": 2,
            "name": "question",
            "type": {"dtype": "string", "_type": "Value"},
        },
        {
            "feature_idx": 3,
            "name": "response",
            "type": {"dtype": "string", "_type": "Value"},
        },
    ],
    "rows": [
        {
            "row_idx": 0,
            "row": {
                "id": "niv.242684",
                "system_prompt": "",
                "question": "You will be given a definition of a task first, then some input of the task.\\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\\n\\nAFC Ajax (amateurs)'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\\nOutput:",
                "response": '[\\n  ["AFC Ajax (amateurs)", "has ground", "Sportpark De Toekomst"],\\n  ["Ajax Youth Academy", "plays at", "Sportpark De Toekomst"]\\n]',
            },
            "truncated_cells": [],
        }
    ],
    "num_rows_total": 2914896,
    "num_rows_per_page": 100,
    "partial": True,
}

TEST_LENGTH = 1


class TestInputRetrieverFactory:

    @pytest.fixture
    def default_configured_url(self):
        input_retriever_factory = InputRetrieverFactory(
            InputsConfig(
                starting_index=ic.DEFAULT_STARTING_INDEX,
                length=ic.DEFAULT_LENGTH,
            )
        )
        default_configured_url = input_retriever_factory._create_configured_url(
            ic.OPEN_ORCA_URL,
        )

        yield default_configured_url

    def test_create_configured_url(self):
        """
        Test that we are appending and configuring the URL correctly
        """
        input_retriever_factory = InputRetrieverFactory(
            InputsConfig(
                starting_index=ic.DEFAULT_STARTING_INDEX,
                length=ic.DEFAULT_LENGTH,
            )
        )

        expected_configured_url = (
            "http://test-url.com"
            + f"&offset={ic.DEFAULT_STARTING_INDEX}"
            + f"&length={ic.DEFAULT_LENGTH}"
        )
        configured_url = input_retriever_factory._create_configured_url(
            "http://test-url.com"
        )

        assert configured_url == expected_configured_url

    def test_download_dataset_illegal_url(self):
        """
        Test for exception when URL is bad
        """
        input_retriever_factory = InputRetrieverFactory(InputsConfig())
        with pytest.raises(GenAIPerfException):
            _ = input_retriever_factory._download_dataset(
                "https://bad-url.zzz",
            )

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
                    "text": "This is test prompt",
                    "image": "test_image_base64",
                }
            else:
                assert row == {
                    "text": "This is test prompt",
                }

    @responses.activate
    def test_inputs_with_defaults(self, default_configured_url):
        """
        Test that default options work
        """
        responses.add(
            responses.GET,
            f"{default_configured_url}",
            json=mocked_openorca_data,
            status=200,
        )
        input_retriever_factory = InputRetrieverFactory(InputsConfig())
        dataset = input_retriever_factory._download_dataset(
            default_configured_url,
        )
        dataset_json = (
            input_retriever_factory._convert_input_url_dataset_to_generic_json(
                dataset=dataset
            )
        )

        assert dataset_json is not None
        assert len(dataset_json["rows"]) == TEST_LENGTH
