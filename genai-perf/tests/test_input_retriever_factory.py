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

from pathlib import Path
from unittest.mock import patch

from genai_perf.inputs.input_constants import PromptSource
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.file_input_retriever import FileInputRetriever
from genai_perf.inputs.retrievers.input_retriever_factory import InputRetrieverFactory
from genai_perf.inputs.retrievers.synthetic_data_retriever import SyntheticDataRetriever


class TestInputRetrieverFactory:

    def test_create_file_retrieverg(self):
        config = InputsConfig(
            input_type=PromptSource.FILE, input_filename=Path("input_data.jsonl")
        )
        with patch(
            "genai_perf.inputs.retrievers.file_input_retriever.FileInputRetriever.__init__",
            return_value=None,
        ) as mock_init:
            retriever = InputRetrieverFactory.create(config)
            mock_init.assert_called_once_with(config)
            assert isinstance(
                retriever, FileInputRetriever
            ), "Should return a FileInputRetriever"

    def test_create_synthetic_retriever(self):
        """
        Test that SyntheticDataRetriever is created and passed the correct config.
        """
        config = InputsConfig(input_type=PromptSource.SYNTHETIC, num_prompts=10)
        with patch(
            "genai_perf.inputs.retrievers.synthetic_data_retriever.SyntheticDataRetriever.__init__",
            return_value=None,
        ) as mock_init:
            retriever = InputRetrieverFactory.create(config)
            mock_init.assert_called_once_with(config)
            assert isinstance(
                retriever, SyntheticDataRetriever
            ), "Should return a SyntheticDataRetriever"
