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

from genai_perf.inputs.converters.rankings_converter import RankingsConverter
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig


class TestRankingsConverter:

    def test_convert_default(self):
        generic_dataset = {
            "rows": [
                {
                    "query": {"text": "query 1"},
                    "passages": [{"text": "passage 1"}, {"text": "passage 2"}],
                },
                {
                    "query": {"text": "query 2"},
                    "passages": [{"text": "passage 3"}, {"text": "passage 4"}],
                },
            ]
        }

        config = InputsConfig(
            extra_inputs={},  # no extra inputs
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.RANKINGS,
        )

        rankings_converter = RankingsConverter()
        result = rankings_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "query": {"text": "query 1"},
                            "passages": [{"text": "passage 1"}, {"text": "passage 2"}],
                            "model": "test_model",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "query": {"text": "query 2"},
                            "passages": [{"text": "passage 3"}, {"text": "passage 4"}],
                            "model": "test_model",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_request_parameters(self):
        generic_dataset = {
            "rows": [
                {
                    "query": {"text": "query 1"},
                    "passages": [{"text": "passage 1"}, {"text": "passage 2"}],
                },
                {
                    "query": {"text": "query 2"},
                    "passages": [{"text": "passage 3"}, {"text": "passage 4"}],
                },
            ]
        }

        extra_inputs = {
            "encoding_format": "base64",
            "truncate": "END",
            "additional_key": "additional_value",
        }

        config = InputsConfig(
            extra_inputs=extra_inputs,
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.RANKINGS,
        )

        rankings_converter = RankingsConverter()
        result = rankings_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "query": {"text": "query 1"},
                            "passages": [{"text": "passage 1"}, {"text": "passage 2"}],
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
                            "query": {"text": "query 2"},
                            "passages": [{"text": "passage 3"}, {"text": "passage 4"}],
                            "model": "test_model",
                            "encoding_format": "base64",
                            "truncate": "END",
                            "additional_key": "additional_value",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_huggingface_tei(self):
        generic_dataset = {
            "rows": [
                {
                    "query": {"text": "query 1"},
                    "passages": [{"text": "passage 1"}, {"text": "passage 2"}],
                },
                {
                    "query": {"text": "query 2"},
                    "passages": [{"text": "passage 3"}, {"text": "passage 4"}],
                },
            ]
        }

        extra_inputs = {
            "rankings": "tei",  # specify HF TGI
            "additional_key": "additional_value",
        }

        config = InputsConfig(
            extra_inputs=extra_inputs,
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            output_format=OutputFormat.RANKINGS,
        )

        rankings_converter = RankingsConverter()
        result = rankings_converter.convert(generic_dataset, config)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "query": "query 1",
                            "texts": ["passage 1", "passage 2"],
                            "additional_key": "additional_value",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "query": "query 2",
                            "texts": ["passage 3", "passage 4"],
                            "additional_key": "additional_value",
                        }
                    ]
                },
            ]
        }

        assert result == expected_result
