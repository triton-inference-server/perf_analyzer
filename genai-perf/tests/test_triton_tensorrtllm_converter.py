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

from genai_perf.inputs.converters import TensorRTLLMConverter
from genai_perf.inputs.input_constants import (
    DEFAULT_TENSORRTLLM_MAX_TOKENS,
    ModelSelectionStrategy,
    OutputFormat,
)
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)


class TestTensorRTLLMConverter:

    @staticmethod
    def create_generic_dataset():
        """Create a standard generic dataset for testing."""
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
            output_format=OutputFormat.TENSORRTLLM,
        )

        trtllm_converter = TensorRTLLMConverter()
        result = trtllm_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "model": "test_model",
                    "text_input": "text input one",
                    "max_tokens": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                },
                {
                    "model": "test_model",
                    "text_input": "text input two",
                    "max_tokens": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_request_parameters(self):
        generic_dataset = self.create_generic_dataset()

        extra_inputs = {
            "ignore_eos": True,
            "max_tokens": 1234,
            "additional_key": "additional_value",
        }

        config = InputsConfig(
            extra_inputs=extra_inputs,
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.TENSORRTLLM,
            add_stream=True,
        )

        trtllm_converter = TensorRTLLMConverter()
        result = trtllm_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "model": "test_model",
                    "text_input": "text input one",
                    "ignore_eos": [True],
                    "max_tokens": [1234],
                    "stream": [True],
                    "additional_key": ["additional_value"],
                },
                {
                    "model": "test_model",
                    "text_input": "text input two",
                    "ignore_eos": [True],
                    "max_tokens": [1234],
                    "stream": [True],
                    "additional_key": ["additional_value"],
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
            output_format=OutputFormat.TENSORRTLLM,
        )

        trtllm_converter = TensorRTLLMConverter()
        result = trtllm_converter.convert(generic_dataset, config)

        expected_result = {"data": []}
        assert result == expected_result
