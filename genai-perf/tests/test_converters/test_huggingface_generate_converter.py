# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.inputs.converters.huggingface_generate_converter import (
    HuggingFaceGenerateConverter,
)
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)


class TestHuggingFaceGenerateConverter:

    def test_convert_default(self):
        expected_texts = ["Hello world.", "Test prompt."]
        dataset = GenericDataset(
            files_data={
                "file1": FileData(
                    rows=[
                        DataRow(texts=[expected_texts[0]]),
                        DataRow(texts=[expected_texts[1]]),
                    ]
                )
            }
        )
        config = ConfigCommand({"model_name": "hf_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.HUGGINGFACE_GENERATE

        converter = HuggingFaceGenerateConverter(config)
        result = converter.convert(dataset)

        assert isinstance(result, dict)
        assert "data" in result
        assert len(result["data"]) == 2
        for i, item in enumerate(result["data"]):
            assert "payload" in item
            payload_list = item["payload"]
            assert isinstance(payload_list, list)
            payload = payload_list[0]
            assert "model" in payload
            assert "inputs" in payload
            assert payload["inputs"] == expected_texts[i]

    def test_convert_with_extra_params(self):
        expected_temperature = 0.7
        expected_max_tokens = 100
        expected_texts = ["Hello again."]
        dataset = GenericDataset(
            files_data={
                "file1": FileData(
                    rows=[
                        DataRow(
                            texts=expected_texts,
                            optional_data={"max_tokens": expected_max_tokens},
                        )
                    ]
                )
            }
        )
        config = ConfigCommand({"model_name": "hf_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.HUGGINGFACE_GENERATE
        config.input.extra = {"temperature": expected_temperature}

        converter = HuggingFaceGenerateConverter(config)
        result = converter.convert(dataset)

        payload_list = result["data"][0]["payload"]
        payload = payload_list[0]
        assert payload["model"] == "hf_model"
        assert "inputs" in payload
        assert "parameters" in payload
        assert payload["inputs"] == expected_texts[0]
        assert payload["temperature"] == expected_temperature
        assert payload["max_tokens"] == expected_max_tokens

    def test_convert_empty_dataset(self):
        dataset = GenericDataset(files_data={})

        config = ConfigCommand({"model_name": "hf_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.HUGGINGFACE_GENERATE

        converter = HuggingFaceGenerateConverter(config)
        result = converter.convert(dataset)

        assert result == {"data": []}
