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

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters import OpenAIEmbeddingsConverter
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)


class TestOpenAIEmbeddingsConverter:

    def test_check_config_streaming_unsupported(self):
        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.streaming = True
        config.endpoint.output_format = OutputFormat.OPENAI_EMBEDDINGS

        converter = OpenAIEmbeddingsConverter(config)

        with pytest.raises(GenAIPerfException) as exc_info:
            converter.check_config()
        assert (
            str(exc_info.value)
            == "The --streaming option is not supported for openai_embeddings."
        )

    def test_convert_default(self):
        generic_dataset = GenericDataset(
            files_data={
                "file1": FileData(
                    rows=[DataRow(texts=["text_1"]), DataRow(texts=["text_2"])],
                )
            }
        )
        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.OPENAI_EMBEDDINGS

        converter = OpenAIEmbeddingsConverter(config)
        result = converter.convert(generic_dataset)

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
                    rows=[
                        DataRow(texts=["text_1", "text_2"]),
                        DataRow(texts=["text_3", "text_4"]),
                    ],
                )
            }
        )

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.OPENAI_EMBEDDINGS
        config.input.batch_size = 2

        converter = OpenAIEmbeddingsConverter(config)
        result = converter.convert(generic_dataset)

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
                    rows=[DataRow(texts=["text_1"]), DataRow(texts=["text_2"])],
                )
            }
        )

        extra_inputs = {
            "encoding_format": "base64",
            "truncate": "END",
            "additional_key": "additional_value",
        }

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.OPENAI_EMBEDDINGS
        config.input.extra = extra_inputs

        converter = OpenAIEmbeddingsConverter(config)
        result = converter.convert(generic_dataset)

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

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.OPENAI_EMBEDDINGS

        converter = OpenAIEmbeddingsConverter(config)
        result = converter.convert(generic_dataset)

        expected_result = {"data": []}
        assert result == expected_result

    def test_convert_with_payload_parameters(self):
        optional_data_1 = {
            "session_id": "abcd",
            "additional_key": "additional_value",
        }
        optional_data_2 = {
            "session_id": "cdef",
        }

        generic_dataset = GenericDataset(
            files_data={
                "file1": FileData(
                    rows=[
                        DataRow(
                            texts=["text_1"],
                            optional_data=optional_data_1,
                            payload_metadata={"timestamp": 0},
                        ),
                        DataRow(
                            texts=["text_2"],
                            optional_data=optional_data_2,
                            payload_metadata={"timestamp": 3047},
                        ),
                    ],
                )
            }
        )

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.OPENAI_EMBEDDINGS

        converter = OpenAIEmbeddingsConverter(config)
        result = converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": ["text_1"],
                            "model": "test_model",
                            "session_id": "abcd",
                            "additional_key": "additional_value",
                        }
                    ],
                    "timestamp": [0],
                },
                {
                    "payload": [
                        {
                            "input": ["text_2"],
                            "model": "test_model",
                            "session_id": "cdef",
                        }
                    ],
                    "timestamp": [3047],
                },
            ]
        }

        assert result == expected_result
