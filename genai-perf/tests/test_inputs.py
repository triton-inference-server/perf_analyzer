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

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.inputs import Inputs
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.tokenizer import Tokenizer


class TestInputs:

    @patch("genai_perf.inputs.inputs.InputRetrieverFactory.create")
    @patch("genai_perf.inputs.inputs.OutputFormatConverterFactory.create")
    @patch("builtins.open", new_callable=mock_open)  # Mock the file writing
    @patch.object(
        Tokenizer, "encode", return_value=[1243, 1881, 697]
    )  # Mock Tokenizer encode method
    @patch("genai_perf.tokenizer.get_tokenizer")  # Mock the get_tokenizer function
    def test_data_retrieval_and_conversion(
        self,
        mock_get_tokenizer,
        mock_encode,
        mock_open_fn,
        mock_converter_factory,
        mock_retriever_factory,
    ):
        mock_tokenizer = MagicMock(spec=Tokenizer)
        mock_get_tokenizer.return_value = mock_tokenizer

        generic_dataset = GenericDataset(
            files_data={
                "file1.jsonl": FileData(
                    rows=[DataRow(texts=["test input"], images=[])],
                )
            }
        )

        mock_retriever_factory.return_value.retrieve_data.return_value = generic_dataset
        expected_output = {"data": "some converted data"}
        mock_converter = mock_converter_factory.return_value
        mock_converter.convert.return_value = expected_output

        inputs = Inputs(
            InputsConfig(
                output_format=OutputFormat.OPENAI_COMPLETIONS,
                tokenizer=mock_tokenizer,
            )
        )

        inputs.create_inputs()

        mock_retriever_factory.return_value.retrieve_data.assert_called_once()
        mock_converter.convert.assert_called_once_with(generic_dataset, inputs.config)

        mock_open_fn.assert_called_once_with(
            str(inputs.config.output_dir / "inputs.json"), "w"
        )
        mock_open_fn().write.assert_called_once_with(
            json.dumps(expected_output, indent=2)
        )

    @patch("genai_perf.inputs.inputs.InputRetrieverFactory.create")
    @patch("genai_perf.inputs.inputs.OutputFormatConverterFactory.create")
    @patch("builtins.open", new_callable=mock_open)
    @patch("genai_perf.tokenizer.get_tokenizer")
    def test_write_json_to_file(
        self,
        mock_get_tokenizer,
        mock_open_fn,
        mock_converter_factory,
        mock_retriever_factory,
    ):
        mock_tokenizer = MagicMock(spec=Tokenizer)
        mock_get_tokenizer.return_value = mock_tokenizer

        generic_dataset = GenericDataset(
            files_data={
                "file1.jsonl": FileData(
                    rows=[DataRow(texts=["test input one"], images=[])],
                )
            }
        )

        mock_retriever_factory.return_value.retrieve_data.return_value = generic_dataset
        expected_output = {"data": "some converted data"}
        mock_converter_factory.return_value.convert.return_value = expected_output

        inputs = Inputs(
            InputsConfig(
                output_format=OutputFormat.OPENAI_COMPLETIONS,
                tokenizer=mock_tokenizer,
                output_dir=Path("."),
            )
        )

        inputs.create_inputs()

        mock_retriever_factory.return_value.retrieve_data.assert_called_once()
        mock_converter_factory.return_value.convert.assert_called_once_with(
            generic_dataset, inputs.config
        )

        mock_open_fn.assert_called_once_with(
            str(inputs.config.output_dir / "inputs.json"), "w"
        )
        mock_open_fn().write.assert_called_once_with(
            json.dumps(expected_output, indent=2)
        )
