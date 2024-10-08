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

from genai_perf.inputs.converters import TensorRTLLMEngineConverter
from genai_perf.inputs.input_constants import (
    DEFAULT_TENSORRTLLM_MAX_TOKENS,
    ModelSelectionStrategy,
    OutputFormat,
)
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.tokenizer import DEFAULT_TOKENIZER, get_tokenizer


class TestTensorRTLLMEngineConverter:

    def test_convert_default(self):

        generic_dataset = {
            "rows": [
                {"text": "text input one"},
                {"text": "text input two"},
            ]
        }

        config = InputsConfig(
            extra_inputs={},  # no extra inputs
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.TENSORRTLLM_ENGINE,
            tokenizer=get_tokenizer(DEFAULT_TOKENIZER),
        )

        trtllm_engine_converter = TensorRTLLMEngineConverter()
        result = trtllm_engine_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "input_ids": {
                        "content": [1426, 1881, 697],
                        "shape": [3],
                    },
                    "input_lengths": [3],
                    "request_output_len": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                },
                {
                    "input_ids": {
                        "content": [1426, 1881, 1023],
                        "shape": [3],
                    },
                    "input_lengths": [3],
                    "request_output_len": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_request_parameters(self):
        generic_dataset = {
            "rows": [
                {"text": "text input one"},
                {"text": "text input two"},
            ]
        }

        extra_inputs = {"additional_key": "additional_value"}

        config = InputsConfig(
            extra_inputs=extra_inputs,
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.TENSORRTLLM_ENGINE,
            tokenizer=get_tokenizer(DEFAULT_TOKENIZER),
            add_stream=True,
            output_tokens_mean=1234,
            output_tokens_deterministic=True,
        )

        trtllm_engine_converter = TensorRTLLMEngineConverter()
        result = trtllm_engine_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "input_ids": {
                        "content": [1426, 1881, 697],
                        "shape": [3],
                    },
                    "input_lengths": [3],
                    "request_output_len": [1234],
                    "streaming": [True],
                    "min_length": [1234],
                    "additional_key": ["additional_value"],
                },
                {
                    "input_ids": {
                        "content": [1426, 1881, 1023],
                        "shape": [3],
                    },
                    "input_lengths": [3],
                    "request_output_len": [1234],
                    "streaming": [True],
                    "min_length": [1234],
                    "additional_key": ["additional_value"],
                },
            ]
        }

        assert result == expected_result
