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

from typing import Any, Dict, List

import pytest
from genai_perf.inputs.converters import OpenAIChatCompletionsConverter
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.tokenizer import get_empty_tokenizer


class TestOpenAIChatCompletionsConverter:

    @staticmethod
    def create_generic_dataset(rows: List[Dict[str, Any]]) -> GenericDataset:
        def clean_text(row):
            text = row.get("text", [])
            if isinstance(text, list):
                return [t for t in text if t]
            elif text:
                return [text]
            return []

        def clean_image(row):
            image = row.get("image", [])
            if isinstance(image, list):
                return [i for i in image if i]
            elif image:
                return [image]
            return []

        def clean_optional_data(row):
            optional_data = row.get("optional_data", {})
            if isinstance(optional_data, Dict):
                return optional_data
            return {}

        def clean_payload_metadata(row):
            payload_metadata = row.get("payload_metadata")
            if isinstance(payload_metadata, Dict):
                return payload_metadata
            return {}

        return GenericDataset(
            files_data={
                "file1": FileData(
                    rows=[
                        DataRow(
                            texts=clean_text(row),
                            images=clean_image(row),
                            optional_data=clean_optional_data(row),
                            payload_metadata=clean_payload_metadata(row),
                        )
                        for row in rows
                    ],
                )
            }
        )

    def test_convert_default(self):
        generic_dataset = self.create_generic_dataset(
            [{"text": "text input one"}, {"text": "text input two"}]
        )

        config = InputsConfig(
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            tokenizer=get_empty_tokenizer(),
        )

        chat_converter = OpenAIChatCompletionsConverter()
        result = chat_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "text input one",
                                }
                            ],
                            "model": "test_model",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "text input two",
                                }
                            ],
                            "model": "test_model",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_request_parameters(self):
        generic_dataset = self.create_generic_dataset(
            [{"text": "text input one"}, {"text": "text input two"}]
        )

        extra_inputs = {
            "ignore_eos": True,
            "max_tokens": 1234,
            "additional_key": "additional_value",
        }

        config = InputsConfig(
            extra_inputs=extra_inputs,
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            tokenizer=get_empty_tokenizer(),
            add_stream=True,
        )

        chat_converter = OpenAIChatCompletionsConverter()
        result = chat_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "text input one",
                                }
                            ],
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
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "text input two",
                                }
                            ],
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

    @pytest.mark.parametrize(
        "rows, output_format, first_content, second_content",
        [
            (
                [
                    {"text": "test input one", "image": "test_image_1"},
                    {"text": "test input two", "image": "test_image_2"},
                ],
                OutputFormat.OPENAI_VISION,
                [
                    {
                        "type": "text",
                        "text": "test input one",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "test_image_1",
                        },
                    },
                ],
                [
                    {
                        "type": "text",
                        "text": "test input two",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "test_image_2",
                        },
                    },
                ],
            ),
            (
                [
                    {"text": "test input A", "image": "test_image_A1"},
                    {"text": "test input B", "image": "test_image_B2"},
                ],
                OutputFormat.OPENAI_VISION,
                [
                    {
                        "type": "text",
                        "text": "test input A",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "test_image_A1",
                        },
                    },
                ],
                [
                    {
                        "type": "text",
                        "text": "test input B",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "test_image_B2",
                        },
                    },
                ],
            ),
        ],
    )
    def test_convert_multi_modal(
        self, rows, output_format, first_content, second_content
    ) -> None:
        """
        Test multi-modal format of OpenAI Chat API
        """
        generic_dataset = self.create_generic_dataset(rows)

        config = InputsConfig(
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=output_format,
            add_stream=True,
            tokenizer=get_empty_tokenizer(),
        )

        chat_converter = OpenAIChatCompletionsConverter()
        result = chat_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": first_content,
                                }
                            ],
                            "stream": True,
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": second_content,
                                }
                            ],
                            "stream": True,
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_payload_parameters(self):
        optional_data = {"session_id": "abcd"}
        generic_dataset = self.create_generic_dataset(
            [
                {
                    "text": "text input one",
                    "timestamp": 0,
                    "optional_data": optional_data,
                    "payload_metadata": {"timestamp": 0},
                }
            ]
        )

        config = InputsConfig(
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            tokenizer=get_empty_tokenizer(),
            add_stream=True,
        )

        chat_converter = OpenAIChatCompletionsConverter()
        result = chat_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "text input one",
                                }
                            ],
                            "model": "test_model",
                            "stream": True,
                            "session_id": "abcd",
                        }
                    ],
                    "timestamp": [0],
                },
            ]
        }

        assert result == expected_result
