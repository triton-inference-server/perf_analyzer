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

from genai_perf.inputs.converters import OpenAIEmbeddingsConverter
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.retrievers.generic_dataset import GenericDataset, DataRow, FileData
import pytest


class TestOpenAIEmbeddingsConverter:

    def test_check_config_streaming_unsupported(self):
        config = InputsConfig(
            add_stream=True,
            output_format=OutputFormat.OPENAI_EMBEDDINGS,
        )
        converter = OpenAIEmbeddingsConverter()

        with pytest.raises(GenAIPerfException) as exc_info:
            converter.check_config(config)
        assert str(exc_info.value) == "The --streaming option is not supported for openai_embeddings."

    def test_convert_default(self):
        generic_dataset = GenericDataset(
            files_data={
                "file1": FileData(
                    filename="file1",
                    rows=[DataRow(texts=["text_1"]), DataRow(texts=["text_2"])]
                )
            }
        )
        config = InputsConfig(
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_EMBEDDINGS,
        )

        converter = OpenAIEmbeddingsConverter()
        result = converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": ["text_1"],
                            "model": "test_model",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "input": ["text_2"],
                            "model": "test_model",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_batched(self):
        generic_dataset = GenericDataset(
            files_data={
                "file1": FileData(
                    filename="file1",
                    rows=[DataRow(texts=["text_1", "text_2"]), DataRow(texts=["text_3", "text_4"])]
                )
            }
        )

        config = InputsConfig(
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_EMBEDDINGS,
            batch_size_text=2,
        )

        converter = OpenAIEmbeddingsConverter()
        result = converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": ["text_1", "text_2"],
                            "model": "test_model",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "input": ["text_3", "text_4"],
                            "model": "test_model",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_request_parameters(self):
        generic_dataset = GenericDataset(
            files_data={
                "file1": FileData(
                    filename="file1",
                    rows=[DataRow(texts=["text_1"]), DataRow(texts=["text_2"])]
                )
            }
        )

        extra_inputs = {
            "encoding_format": "base64",
            "truncate": "END",
            "additional_key": "additional_value",
        }

        config = InputsConfig(
            extra_inputs=extra_inputs,
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_EMBEDDINGS,
        )

        converter = OpenAIEmbeddingsConverter()
        result = converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": ["text_1"],
                            "model": "test_model",
                            "encoding_format": "base64",
                            "truncate": "END",
                            "additional_key": "additional_value",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "input": ["text_2"],
                            "model": "test_model",
                            "encoding_format": "base64",
                            "truncate": "END",
                            "additional_key": "additional_value",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_empty_dataset(self):
        generic_dataset = GenericDataset(files_data={})
        config = InputsConfig(
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_EMBEDDINGS,
        )

        converter = OpenAIEmbeddingsConverter()
        result = converter.convert(generic_dataset, config)

        expected_result = {"data": []}
        assert result == expected_result
