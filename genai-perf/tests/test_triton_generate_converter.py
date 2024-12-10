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
from genai_perf.inputs.converters import TritonGenerateConverter
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.tokenizer import get_empty_tokenizer


class TestTritonGenerateConverter:
    @staticmethod
    def create_generic_dataset(rows) -> GenericDataset:
        return GenericDataset(files_data={"file1": FileData(rows)})

    def test_check_config_raises_exception_for_deterministic_tokens(self):
        config = InputsConfig(
            output_tokens_deterministic=True,
            tokenizer=get_empty_tokenizer(),
        )

        converter = TritonGenerateConverter()

        with pytest.raises(
            ValueError,
            match="The --output-tokens-deterministic flag is not supported for Triton Generate.",
        ):
            converter.check_config(config)

    def test_convert_default(self):
        generic_dataset = self.create_generic_dataset(
            [DataRow(texts=["sample_prompt_1"]), DataRow(texts=["sample_prompt_2"])]
        )

        config = InputsConfig(
            extra_inputs={},
            tokenizer=get_empty_tokenizer(),
        )

        converter = TritonGenerateConverter()
        result = converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {"payload": [{"text_input": ["sample_prompt_1"]}]},
                {"payload": [{"text_input": ["sample_prompt_2"]}]},
            ]
        }

        assert result == expected_result

    def test_convert_with_extra_inputs(self):
        generic_dataset = self.create_generic_dataset(
            [DataRow(texts=["extra_input_prompt"])]
        )

        extra_inputs = {"temperature": 0.7, "top_p": 0.9}
        config = InputsConfig(
            extra_inputs=extra_inputs,
            tokenizer=get_empty_tokenizer(),
        )

        converter = TritonGenerateConverter()
        result = converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "text_input": ["extra_input_prompt"],
                            "temperature": 0.7,
                            "top_p": 0.9,
                        }
                    ]
                }
            ]
        }

        assert result == expected_result

    def test_convert_with_streaming(self):
        generic_dataset = self.create_generic_dataset(
            [DataRow(texts=["streaming_prompt"])]
        )

        config = InputsConfig(
            add_stream=True,
            tokenizer=get_empty_tokenizer(),
        )

        converter = TritonGenerateConverter()
        result = converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "text_input": ["streaming_prompt"],
                            "stream": True,
                        }
                    ]
                }
            ]
        }

        assert result == expected_result

    def test_convert_with_output_tokens_mean(self, monkeypatch):
        generic_dataset = self.create_generic_dataset(
            [DataRow(texts=["tokens_mean_prompt"])]
        )

        config = InputsConfig(
            output_tokens_mean=100,
            output_tokens_stddev=10,
            tokenizer=get_empty_tokenizer(),
        )

        def mock_sample_bounded_normal(mean, stddev, lower):
            assert mean == 100
            assert stddev == 10
            assert lower == 1
            return 95

        monkeypatch.setattr(
            "genai_perf.inputs.converters.triton_generate_converter.sample_bounded_normal",
            mock_sample_bounded_normal,
        )

        converter = TritonGenerateConverter()
        result = converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "text_input": ["tokens_mean_prompt"],
                            "max_tokens": 95,
                        }
                    ]
                }
            ]
        }

        assert result == expected_result

    def test_convert_empty_dataset(self):
        generic_dataset = GenericDataset(files_data={})

        config = InputsConfig(
            extra_inputs={},
            tokenizer=get_empty_tokenizer(),
        )

        converter = TritonGenerateConverter()
        result = converter.convert(generic_dataset, config)

        expected_result = {"data": []}
        assert result == expected_result
