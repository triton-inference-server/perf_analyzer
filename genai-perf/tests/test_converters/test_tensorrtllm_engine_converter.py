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
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters import TensorRTLLMEngineConverter
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
from genai_perf.tokenizer import DEFAULT_TOKENIZER, get_empty_tokenizer, get_tokenizer


class TestTensorRTLLMEngineConverter:

    @staticmethod
    def create_generic_dataset() -> GenericDataset:
        """Helper method to create a standard generic dataset."""
        return GenericDataset(
            files_data={
                "file1": FileData(
                    rows=[
                        DataRow(texts=["text input one"]),
                        DataRow(texts=["text input two"]),
                    ],
                )
            }
        )

    @staticmethod
    def create_generic_dataset_with_payload_parameters() -> GenericDataset:
        optional_data_1 = {"session_id": "abcd"}
        optional_data_2 = {
            "session_id": "dfwe",
            "input_length": "6755",
            "output_length": "500",
        }
        return GenericDataset(
            files_data={
                "file1": FileData(
                    rows=[
                        DataRow(
                            texts=["text input one"],
                            timestamp="0",
                            optional_data=optional_data_1,
                        ),
                        DataRow(
                            texts=["text input two"],
                            timestamp="2345",
                            optional_data=optional_data_2,
                        ),
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

    def test_convert_with_chat_template(self):
        generic_dataset = self.create_generic_dataset()
        tokenizer = get_tokenizer(DEFAULT_TOKENIZER)
        config = InputsConfig(
            extra_inputs={"apply_chat_template": True},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.TENSORRTLLM_ENGINE,
            tokenizer=tokenizer,
        )

        trtllm_engine_converter = TensorRTLLMEngineConverter()

        result = trtllm_engine_converter.convert(generic_dataset, config)

        expected_texts = [
            config.tokenizer._tokenizer.apply_chat_template(
                [{"role": "user", "content": "text input one"}],
                tokenize=False,
                add_special_tokens=False,
            ),
            config.tokenizer._tokenizer.apply_chat_template(
                [{"role": "user", "content": "text input two"}],
                tokenize=False,
                add_special_tokens=False,
            ),
        ]
        expected_tokenized = [config.tokenizer.encode(text) for text in expected_texts]

        assert "data" in result
        assert isinstance(result["data"], list)
        assert len(result["data"]) == 2

        for i, payload in enumerate(result["data"]):
            assert "input_ids" in payload
            assert "content" in payload["input_ids"]
            assert payload["input_ids"]["content"] == expected_tokenized[i], (
                f"Mismatch in tokenized content for row {i}: "
                f"Expected {expected_tokenized[i]}, but got {payload['input_ids']['content']}"
            )

    def test_convert_with_request_parameters(self):
        generic_dataset = self.create_generic_dataset()

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

    def test_check_config_invalid_batch_size(self):
        config = InputsConfig(
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.TENSORRTLLM_ENGINE,
            batch_size_text=5,
            tokenizer=get_empty_tokenizer(),
        )

        trtllm_engine_converter = TensorRTLLMEngineConverter()

        with pytest.raises(GenAIPerfException) as exc_info:
            trtllm_engine_converter.check_config(config)

        assert str(exc_info.value) == (
            "The --batch-size-text flag is not supported for tensorrtllm_engine."
        )

    def test_convert_with_payload_parameters(self):
        generic_dataset = self.create_generic_dataset_with_payload_parameters()

        config = InputsConfig(
            extra_inputs={},
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
                    "session_id": "abcd",
                    "timestamp": ["0"],
                },
                {
                    "input_ids": {
                        "content": [1426, 1881, 1023],
                        "shape": [3],
                    },
                    "input_lengths": [3],
                    "request_output_len": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                    "session_id": "dfwe",
                    "input_length": "6755",
                    "output_length": "500",
                    "timestamp": ["2345"],
                },
            ]
        }

        assert result == expected_result
