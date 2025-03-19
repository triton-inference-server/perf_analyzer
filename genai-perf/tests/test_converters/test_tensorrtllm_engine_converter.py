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

from unittest.mock import Mock, patch

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters import TensorRTLLMEngineConverter
from genai_perf.inputs.input_constants import (
    DEFAULT_TENSORRTLLM_MAX_TOKENS,
    ModelSelectionStrategy,
    OutputFormat,
)
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.tokenizer import get_empty_tokenizer

# Mock tokenizer outputs
MOCK_TOKENIZED_ONE = [1426, 1881, 697]  # "text input one"
MOCK_TOKENIZED_TWO = [1426, 1881, 1023]  # "text input two"
MOCK_CHAT_TOKENIZED_ONE = [1111, 2222, 3333]  # chat template for "text input one"
MOCK_CHAT_TOKENIZED_TWO = [4444, 5555, 6666]  # chat template for "text input two"


class TestTensorRTLLMEngineConverter:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        # Create mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.encode.side_effect = lambda text, **kwargs: (
            MOCK_TOKENIZED_ONE if "one" in text else MOCK_TOKENIZED_TWO
        )

        # Create mock inner tokenizer for chat templates
        self.mock_inner_tokenizer = Mock()
        self.mock_inner_tokenizer.apply_chat_template.side_effect = (
            lambda messages, **kwargs: (
                "chat_one" if "one" in messages[0]["content"] else "chat_two"
            )
        )
        self.mock_tokenizer._tokenizer = self.mock_inner_tokenizer

        # Patch the get_tokenizer function
        self.get_tokenizer_patcher = patch(
            "genai_perf.tokenizer.get_tokenizer", return_value=self.mock_tokenizer
        )
        self.mock_get_tokenizer = self.get_tokenizer_patcher.start()
        yield
        self.get_tokenizer_patcher.stop()

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
                            optional_data=optional_data_1,
                            payload_metadata={"timestamp": 0},
                        ),
                        DataRow(
                            texts=["text input two"],
                            optional_data=optional_data_2,
                            payload_metadata={"timestamp": 2345},
                        ),
                    ],
                )
            }
        )

    def test_convert_default(self):
        generic_dataset = self.create_generic_dataset()

        config = ConfigCommand({"model_names": ["test_model"]})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.TENSORRTLLM_ENGINE

        trtllm_engine_converter = TensorRTLLMEngineConverter(
            config, self.mock_tokenizer
        )
        result = trtllm_engine_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "input_ids": {
                        "content": MOCK_TOKENIZED_ONE,
                        "shape": [3],
                    },
                    "input_lengths": [3],
                    "request_output_len": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                },
                {
                    "input_ids": {
                        "content": MOCK_TOKENIZED_TWO,
                        "shape": [3],
                    },
                    "input_lengths": [3],
                    "request_output_len": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_chat_template(self):
        config = ConfigCommand({"model_names": ["test_model"]})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.TENSORRTLLM_ENGINE
        config.input.extra = {"apply_chat_template": True}

        # Set up mock for chat template encoding
        self.mock_tokenizer.encode.side_effect = lambda text: (
            MOCK_CHAT_TOKENIZED_ONE if "chat_one" in text else MOCK_CHAT_TOKENIZED_TWO
        )

        trtllm_engine_converter = TensorRTLLMEngineConverter(
            config, self.mock_tokenizer
        )

        result = trtllm_engine_converter.convert(generic_dataset)

        expected_texts = [
            tokenizer._tokenizer.apply_chat_template(
                [{"role": "user", "content": "text input one"}],
                tokenize=False,
                add_special_tokens=False,
            ),
            tokenizer._tokenizer.apply_chat_template(
                [{"role": "user", "content": "text input two"}],
                tokenize=False,
                add_special_tokens=False,
            ),
        ]
        expected_tokenized = [tokenizer.encode(text) for text in expected_texts]

        assert "data" in result
        assert isinstance(result["data"], list)
        assert len(result["data"]) == 2

        expected_contents = [MOCK_CHAT_TOKENIZED_ONE, MOCK_CHAT_TOKENIZED_TWO]
        for i, payload in enumerate(result["data"]):
            assert "input_ids" in payload
            assert "content" in payload["input_ids"]
            assert payload["input_ids"]["content"] == expected_contents[i], (
                f"Mismatch in tokenized content for row {i}: "
                f"Expected {expected_contents[i]}, but got {payload['input_ids']['content']}"
            )

    def test_convert_with_request_parameters(self):
        generic_dataset = self.create_generic_dataset()

        config = ConfigCommand({"model_names": ["test_model"]})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.TENSORRTLLM_ENGINE
        config.endpoint.streaming = True
        config.input.output_tokens.mean = 1234
        config.input.output_tokens.deterministic = True
        config.input.extra = {"additional_key": "additional_value"}

        trtllm_engine_converter = TensorRTLLMEngineConverter(
            config, self.mock_tokenizer
        )
        result = trtllm_engine_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "input_ids": {
                        "content": MOCK_TOKENIZED_ONE,
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
                        "content": MOCK_TOKENIZED_TWO,
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
        config = ConfigCommand({"model_names": ["test_model"]})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.TENSORRTLLM_ENGINE
        config.input.batch_size = 5

        trtllm_engine_converter = TensorRTLLMEngineConverter()

        with pytest.raises(GenAIPerfException) as exc_info:
            trtllm_engine_converter.check_config(config)

        assert str(exc_info.value) == (
            "The --batch-size-text flag is not supported for tensorrtllm_engine."
        )

    def test_convert_with_payload_parameters(self):
        generic_dataset = self.create_generic_dataset_with_payload_parameters()

        config = ConfigCommand({"model_names": ["test_model"]})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.TENSORRTLLM_ENGINE

        tokenizer = get_tokenizer(config)
        trtllm_engine_converter = TensorRTLLMEngineConverter(
            config, self.mock_tokenizer
        )
        result = trtllm_engine_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "input_ids": {
                        "content": MOCK_TOKENIZED_ONE,
                        "shape": [3],
                    },
                    "input_lengths": [3],
                    "request_output_len": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                    "session_id": "abcd",
                    "timestamp": [0],
                },
                {
                    "input_ids": {
                        "content": MOCK_TOKENIZED_TWO,
                        "shape": [3],
                    },
                    "input_lengths": [3],
                    "request_output_len": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                    "session_id": "dfwe",
                    "input_length": "6755",
                    "output_length": "500",
                    "timestamp": [2345],
                },
            ]
        }

        assert result == expected_result
