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

from pathlib import Path
from unittest.mock import patch

from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.inputs.input_constants import PromptSource
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.file_input_retriever import FileInputRetriever
from genai_perf.inputs.retrievers.input_retriever_factory import InputRetrieverFactory
from genai_perf.inputs.retrievers.payload_input_retriever import PayloadInputRetriever
from genai_perf.inputs.retrievers.synthetic_data_retriever import SyntheticDataRetriever
from genai_perf.tokenizer import get_empty_tokenizer


class TestInputRetrieverFactory:

    def test_create_file_retriever(self):
        config = ConfigCommand({"model_name": "test_model"})
        config.input.file = "input_data.jsonl"
        config.input.prompt_source = PromptSource.FILE

        inputs_config = InputsConfig(
            config=config,
            tokenizer=get_empty_tokenizer(),
            output_directory=Path("output"),
        )

        with patch(
            "genai_perf.inputs.retrievers.file_input_retriever.FileInputRetriever.__init__",
            return_value=None,
        ) as mock_init:
            retriever = InputRetrieverFactory.create(inputs_config)
            mock_init.assert_called_once_with(inputs_config)
            assert isinstance(
                retriever, FileInputRetriever
            ), "Should return a FileInputRetriever"

    def test_create_synthetic_retriever(self):
        """
        Test that SyntheticDataRetriever is created and passed the correct config.
        """
        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.prompt_source = PromptSource.SYNTHETIC
        config.input.num_dataset_entries = 10

        inputs_config = InputsConfig(
            config=config,
            tokenizer=get_empty_tokenizer(),
            output_directory=Path("output"),
        )

        with patch(
            "genai_perf.inputs.retrievers.synthetic_data_retriever.SyntheticDataRetriever.__init__",
            return_value=None,
        ) as mock_init:
            retriever = InputRetrieverFactory.create(inputs_config)
            mock_init.assert_called_once_with(inputs_config)
            assert isinstance(
                retriever, SyntheticDataRetriever
            ), "Should return a SyntheticDataRetriever"

    def test_create_payload_retriever(self):
        """
        Test that PayloadInputRetriever is created and passed the correct config.
        """
        config = ConfigCommand({"model_name": "test_model"})
        config.input.payload_file = "test_payload_data.jsonl"
        config.input.prompt_source = PromptSource.PAYLOAD

        inputs_config = InputsConfig(
            config=config,
            tokenizer=get_empty_tokenizer(),
            output_directory=Path("output"),
        )

        with patch(
            "genai_perf.inputs.retrievers.payload_input_retriever.PayloadInputRetriever.__init__",
            return_value=None,
        ) as mock_init:
            retriever = InputRetrieverFactory.create(inputs_config)
            mock_init.assert_called_once_with(inputs_config)
            assert isinstance(
                retriever, PayloadInputRetriever
            ), "Should return a PayloadInputRetriever"
