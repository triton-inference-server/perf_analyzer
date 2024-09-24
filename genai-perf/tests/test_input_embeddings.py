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

from pathlib import Path
from unittest.mock import mock_open, patch

from genai_perf.inputs.converters import OpenAIEmbeddingsConverter
from genai_perf.inputs.input_constants import (
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
)
from genai_perf.inputs.inputs import Inputs
from genai_perf.inputs.inputs_config import InputsConfig


class TestEmbeddingsConverter:

    def test_convert_generic_json_to_openai_embeddings_format(self):
        generic_dataset = {
            "rows": [
                {"input": ["text 1", "text 2"]},
                {"input": ["text 3", "text 4"]},
            ]
        }

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": ["text 1", "text 2"],
                            "model": "test_model",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "input": ["text 3", "text 4"],
                            "model": "test_model",
                        }
                    ]
                },
            ]
        }

        embedding_converter = OpenAIEmbeddingsConverter()
        result = embedding_converter.convert(
            generic_dataset=generic_dataset,
            config=InputsConfig(
                input_type=PromptSource.SYNTHETIC,
                extra_inputs={},
                model_name=["test_model"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                output_format=OutputFormat.OPENAI_EMBEDDINGS,
            ),
        )

        assert result is not None
        assert "data" in result
        assert len(result["data"]) == len(expected_result["data"])

        for i, item in enumerate(expected_result["data"]):
            assert "payload" in result["data"][i]
            assert result["data"][i]["payload"] == item["payload"]

    def test_convert_generic_json_to_openai_embeddings_format_with_extra_inputs(self):
        generic_dataset = {
            "rows": [
                {"input": ["text 1", "text 2"]},
                {"input": ["text 3", "text 4"]},
            ]
        }

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": ["text 1", "text 2"],
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
                            "input": ["text 3", "text 4"],
                            "model": "test_model",
                            "encoding_format": "base64",
                            "truncate": "END",
                            "additional_key": "additional_value",
                        }
                    ]
                },
            ]
        }

        extra_inputs = {
            "encoding_format": "base64",
            "truncate": "END",
            "additional_key": "additional_value",
        }

        inputs = Inputs(
            InputsConfig(
                input_type=PromptSource.SYNTHETIC,
                extra_inputs=extra_inputs,
                model_name=["test_model"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                output_format=OutputFormat.OPENAI_EMBEDDINGS,
            )
        )
        result = inputs._convert_generic_json_to_output_format(
            generic_dataset,
        )

        assert result is not None
        assert "data" in result
        assert len(result["data"]) == len(expected_result["data"])

        for i, item in enumerate(expected_result["data"]):
            assert "payload" in result["data"][i]
            assert result["data"][i]["payload"] == item["payload"]
