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

from copy import deepcopy

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters import NVClipConverter
from genai_perf.inputs.input_constants import ModelSelectionStrategy
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)


class TestNVClipConverter:

    @staticmethod
    def create_generic_dataset(rows) -> GenericDataset:
        return GenericDataset(files_data={"file1": FileData(rows)})

    @pytest.fixture
    def default_config(self):
        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN

        yield config

    def test_convert_default(self, default_config):
        generic_dataset = self.create_generic_dataset(
            [
                DataRow(texts=["text1"], images=["image1"]),
                DataRow(texts=["text2"], images=[]),
                DataRow(texts=[], images=["image2"]),
            ]
        )

        nv_clip_converter = NVClipConverter(default_config)
        result = nv_clip_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "input": ["text1", "image1"],
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "input": ["text2"],
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "input": ["image2"],
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_batched(self, default_config):
        generic_dataset = self.create_generic_dataset(
            [
                DataRow(texts=["text1", "text2"], images=["image1", "image2"]),
            ]
        )

        nv_clip_converter = NVClipConverter(default_config)
        result = nv_clip_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "input": ["text1", "text2", "image1", "image2"],
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_request_parameters(self, default_config):
        generic_dataset = self.create_generic_dataset(
            [
                DataRow(texts=["text1"], images=["image1"]),
            ]
        )

        extra_inputs = {"encoding_format": "base64"}
        config = default_config
        config.input.extra = extra_inputs

        nv_clip_converter = NVClipConverter(config)
        result = nv_clip_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "input": ["text1", "image1"],
                            "encoding_format": "base64",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_check_config_raises_exception_for_streaming(self):
        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.streaming = True

        nv_clip_converter = NVClipConverter(config)

        with pytest.raises(
            GenAIPerfException, match="The --streaming option is not supported"
        ):
            nv_clip_converter.check_config()

    def test_convert_with_payload_parameters(self, default_config):
        optional_data = {"session_id": "abcd"}
        generic_dataset = self.create_generic_dataset(
            [
                DataRow(
                    texts=["text1"],
                    images=["image1"],
                    optional_data=optional_data,
                    payload_metadata={"timestamp": 0},
                ),
            ]
        )

        nv_clip_converter = NVClipConverter(default_config)
        result = nv_clip_converter.convert(generic_dataset)
        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "model": "test_model",
                            "input": ["text1", "image1"],
                            "session_id": "abcd",
                        }
                    ],
                    "timestamp": [0],
                },
            ]
        }

        assert result == expected_result
