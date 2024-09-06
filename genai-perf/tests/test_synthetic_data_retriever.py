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
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.synthetic_data_retriever import SyntheticDataRetriever


class TestSyntheticDataRetriever:

    @patch(
        "genai_perf.inputs.synthetic_data_retriever.SyntheticPromptGenerator.create_synthetic_prompt",
        return_value="test prompt",
    )
    @pytest.mark.parametrize(
        "output_format",
        [
            (OutputFormat.OPENAI_COMPLETIONS),
            (OutputFormat.OPENAI_CHAT_COMPLETIONS),
            (OutputFormat.VLLM),
            (OutputFormat.TENSORRTLLM),
        ],
    )
    def test_synthetic_text(self, mock_prompt, output_format):
        config = InputsConfig(
            num_prompts=3,
            output_format=output_format,
        )
        synthetic_retriever = SyntheticDataRetriever(config)
        dataset = synthetic_retriever.retrieve_data()

        assert len(dataset) == 3
        assert dataset == [
            {"text_input": "test prompt"},
            {"text_input": "test prompt"},
            {"text_input": "test prompt"},
        ]

    @patch(
        "genai_perf.inputs.synthetic_data_retriever.SyntheticPromptGenerator.create_synthetic_prompt",
        return_value="test prompt",
    )
    @patch(
        "genai_perf.inputs.synthetic_data_retriever.SyntheticImageGenerator.create_synthetic_image",
        return_value="data:image/jpeg;base64,test_base64_encoding",
    )
    def test_synthetic_text_and_image(self, mock_prompt, mock_image):
        config = InputsConfig(
            num_prompts=3,
            output_format=OutputFormat.OPENAI_VISION,
        )
        synthetic_retriever = SyntheticDataRetriever(config)
        dataset = synthetic_retriever.retrieve_data()

        assert len(dataset) == 3
        assert dataset == [
            {
                "text_input": "test prompt",
                "image": "data:image/jpeg;base64,test_base64_encoding",
            },
            {
                "text_input": "test prompt",
                "image": "data:image/jpeg;base64,test_base64_encoding",
            },
            {
                "text_input": "test prompt",
                "image": "data:image/jpeg;base64,test_base64_encoding",
            },
        ]
