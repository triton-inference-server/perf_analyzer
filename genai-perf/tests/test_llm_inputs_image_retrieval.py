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

from pathlib import Path
from unittest.mock import patch

from genai_perf.inputs.inputs import LlmInputs, OutputFormat, PromptSource


class TestLlmInputsImageRetrieval:

    @patch(
        "genai_perf.inputs.inputs.LlmInputs._encode_image",
        return_value="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/",
    )
    @patch("genai_perf.inputs.inputs.LlmInputs._get_input_dataset_from_file")
    def test_image_retrieval(self, mock_get_input, mock_encode_image):
        mock_get_input.return_value = {
            "features": [{"name": "text_input"}],
            "rows": [
                {"row": [{"image": "genai_perf/inputs/source_images/image1.jpg"}]}
            ],
        }

        pa_json = LlmInputs.create_inputs(
            input_type=PromptSource.FILE,
            output_format=OutputFormat.IMAGE_RETRIEVAL,
            input_filename=Path("dummy.jsonl"),
            model_name=["test_model"],
            add_model_name=True,
        )

        expected_json = {
            "data": [
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
                                            },
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ]
        }

        assert pa_json == expected_json

    @patch("genai_perf.inputs.inputs.LlmInputs._get_input_dataset_from_file")
    @patch("genai_perf.inputs.inputs.LlmInputs._encode_image")
    def test_image_retrieval_batched(self, mock_encode_image, mock_get_input):
        mock_get_input.return_value = {
            "features": [{"name": "text_input"}],
            "rows": [
                {
                    "row": [
                        {"image": "genai_perf/inputs/source_images/image1.jpg"},
                        {"image": "genai_perf/inputs/source_images/image2.jpg"},
                    ]
                }
            ],
        }
        mock_encode_image.side_effect = [
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/",
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/",
        ]

        pa_json = LlmInputs.create_inputs(
            input_type=PromptSource.FILE,
            output_format=OutputFormat.IMAGE_RETRIEVAL,
            input_filename=Path("dummy.jsonl"),
            batch_size=2,
            num_of_output_prompts=1,
            model_name=["test_model"],
            add_model_name=True,
        )

        expected_json = {
            "data": [
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
                                            },
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
                                            },
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ]
        }

        assert pa_json == expected_json
