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

from genai_perf.inputs.converters import OpenAICompletionsConverter
from genai_perf.inputs.input_constants import (
    DEFAULT_OUTPUT_TOKENS_MEAN,
    ModelSelectionStrategy,
    OutputFormat,
)
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)


class TestOpenAICompletionsConverter:

    @staticmethod
    def create_generic_dataset() -> GenericDataset:
        """Helper method to create a standard generic dataset."""
        return GenericDataset(
            files_data={
                "file1": FileData(
                    filename="file1",
                    rows=[
                        DataRow(texts=["text input one"]),
                        DataRow(texts=["text input two"]),
                    ],
                )
            }
        )

    def test_convert_default(self):
        generic_dataset = self.create_generic_dataset()

        config = InputsConfig(
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_COMPLETIONS,
        )

        completions_converter = OpenAICompletionsConverter()
        result = completions_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "prompt": ["text input one"],
                            "model": "test_model",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "prompt": ["text input two"],
                            "model": "test_model",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_request_parameters(self):
        generic_dataset = self.create_generic_dataset()

        extra_inputs = {
            "ignore_eos": True,
            "additional_key": "additional_value",
        }

        config = InputsConfig(
            extra_inputs=extra_inputs,
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_COMPLETIONS,
            add_stream=True,
            output_tokens_mean=1234,
        )

        completions_converter = OpenAICompletionsConverter()
        result = completions_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "prompt": ["text input one"],
                            "model": "test_model",
                            "stream": True,
                            "ignore_eos": True,
                            "max_tokens": 1234,
                            "additional_key": "additional_value",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "prompt": ["text input two"],
                            "model": "test_model",
                            "stream": True,
                            "ignore_eos": True,
                            "max_tokens": 1234,
                            "additional_key": "additional_value",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_default_token_parameters(self):
        generic_dataset = self.create_generic_dataset()

        config = InputsConfig(
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_COMPLETIONS,
            output_tokens_mean=DEFAULT_OUTPUT_TOKENS_MEAN,
        )

        completions_converter = OpenAICompletionsConverter()
        result = completions_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "prompt": ["text input one"],
                            "model": "test_model",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "prompt": ["text input two"],
                            "model": "test_model",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_streaming(self):
        generic_dataset = self.create_generic_dataset()

        config = InputsConfig(
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_COMPLETIONS,
            add_stream=True,
        )

        completions_converter = OpenAICompletionsConverter()
        result = completions_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "prompt": ["text input one"],
                            "model": "test_model",
                            "stream": True,
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "prompt": ["text input two"],
                            "model": "test_model",
                            "stream": True,
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_multiple_models(self):
        generic_dataset = self.create_generic_dataset()

        config = InputsConfig(
            extra_inputs={},
            model_name=["model_a", "model_b"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_COMPLETIONS,
        )

        completions_converter = OpenAICompletionsConverter()
        result = completions_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "prompt": ["text input one"],
                            "model": "model_a",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "prompt": ["text input two"],
                            "model": "model_b",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result
