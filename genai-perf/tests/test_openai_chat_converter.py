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

import pytest
from genai_perf.inputs.converters import OpenAIChatCompletionsConverter
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig


class TestOpenAIChatCompletionsConverter:

    def test_convert_default(self):
        generic_dataset = {
            "features": ["text_input"],
            "rows": [
                {"text_input": "text input one"},
                {"text_input": "text input two"},
            ],
        }

        config = InputsConfig(
            extra_inputs={},  # no extra inputs
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
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
        generic_dataset = {
            "features": ["text_input"],
            "rows": [
                {"text_input": "text input one"},
                {"text_input": "text input two"},
            ],
        }

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
            add_stream=True,  # set streaming
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
        "rows, first_content, second_content",
        [
            # both text and image
            (
                [
                    {"text_input": "test input one", "image": "test_image_1"},
                    {"text_input": "test input two", "image": "test_image_2"},
                ],
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
            # image only
            (
                [
                    {"image": "test_image_1"},
                    {"image": "test_image_2"},
                ],
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "test_image_1",
                        },
                    },
                ],
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "test_image_2",
                        },
                    },
                ],
            ),
            # text only
            (
                [
                    {"text_input": "test input one"},
                    {"text_input": "test input two"},
                ],
                "test input one",
                "test input two",
            ),
        ],
    )
    def test_convert_multi_modal(self, rows, first_content, second_content) -> None:
        """
        Test multi-modal format of OpenAI Chat API
        """
        generic_dataset = {
            "features": ["text_input"],
            "rows": rows,
        }

        config = InputsConfig(
            extra_inputs={},  # no extra inputs
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            add_stream=True,
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
