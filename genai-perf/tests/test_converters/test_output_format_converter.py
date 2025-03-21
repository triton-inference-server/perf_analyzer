# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import random

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.inputs.converters import *
from genai_perf.inputs.converters.output_format_converter_factory import (
    OutputFormatConverterFactory,
)
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat


class TestOutputFormatConverter:

    @pytest.mark.parametrize(
        "format, expected_converter",
        [
            (
                OutputFormat.OPENAI_CHAT_COMPLETIONS,
                OpenAIChatCompletionsConverter,
            ),
            (
                OutputFormat.OPENAI_COMPLETIONS,
                OpenAICompletionsConverter,
            ),
            (
                OutputFormat.OPENAI_EMBEDDINGS,
                OpenAIEmbeddingsConverter,
            ),
            (
                OutputFormat.IMAGE_RETRIEVAL,
                ImageRetrievalConverter,
            ),
            (
                OutputFormat.OPENAI_MULTIMODAL,
                OpenAIChatCompletionsConverter,
            ),
            (
                OutputFormat.RANKINGS,
                RankingsConverter,
            ),
            (
                OutputFormat.VLLM,
                VLLMConverter,
            ),
            (
                OutputFormat.TENSORRTLLM,
                TensorRTLLMConverter,
            ),
            (
                OutputFormat.TENSORRTLLM_ENGINE,
                TensorRTLLMEngineConverter,
            ),
        ],
    )
    def test_create(self, format, expected_converter):
        converter = OutputFormatConverterFactory.create(format, ConfigCommand())
        assert isinstance(converter, expected_converter)

    @pytest.mark.parametrize(
        "seed, model_name_list, index,model_selection_strategy,expected_model",
        [
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_A",
            ),
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                1,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_B",
            ),
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                2,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_C",
            ),
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                3,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_A",
            ),
            (
                100,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.RANDOM,
                "test_model_A",
            ),
            (
                100,
                ["test_model_A", "test_model_B", "test_model_C"],
                1,
                ModelSelectionStrategy.RANDOM,
                "test_model_A",
            ),
            (
                1652,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.RANDOM,
                "test_model_B",
            ),
            (
                95,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.RANDOM,
                "test_model_C",
            ),
        ],
    )
    def test_select_model_name(
        self,
        seed,
        model_name_list,
        index,
        model_selection_strategy,
        expected_model,
    ):
        """
        Test that model selection strategy controls the model selected
        """

        random.seed(seed)
        config = ConfigCommand({"model_names": model_name_list})
        config.endpoint.model_selection_strategy = model_selection_strategy
        config.endpoint.random_seed = seed

        converter = OutputFormatConverterFactory.create(
            OutputFormat.IMAGE_RETRIEVAL, config
        )
        actual_model = converter._select_model_name(index)
        assert actual_model == expected_model
