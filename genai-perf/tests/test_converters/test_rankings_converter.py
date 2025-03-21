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

from typing import Any, Dict, List, Optional

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.inputs.converters import RankingsConverter
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)


class TestRankingsConverter:

    @staticmethod
    def create_generic_dataset(
        queries_data: Optional[List[List[str]]] = None,
        passages_data: Optional[List[List[str]]] = None,
    ) -> GenericDataset:
        files_data = {}

        if queries_data is not None:
            files_data["queries"] = FileData(
                rows=[DataRow(texts=query) for query in queries_data],
            )

        if passages_data is not None:
            files_data["passages"] = FileData(
                rows=[DataRow(texts=passage) for passage in passages_data],
            )

        return GenericDataset(files_data=files_data)

    @staticmethod
    def create_generic_dataset_payload_parameters(
        queries_data: Optional[List[List[str]]] = None,
        passages_data: Optional[List[List[str]]] = None,
        optional_data: Optional[List[Dict[Any, Any]]] = None,
        payload_metadata: Optional[List[Dict[Any, Any]]] = None,
    ) -> GenericDataset:
        files_data = {}

        if queries_data is not None:
            files_data["queries"] = FileData(
                rows=[DataRow(texts=query) for query in queries_data],
            )

        if passages_data is not None:
            files_data["passages"] = FileData(
                rows=[
                    DataRow(
                        texts=passage,
                        optional_data=(
                            optional_data[index]
                            if optional_data and index < len(optional_data)
                            else {}
                        ),
                        payload_metadata=(
                            payload_metadata[index]
                            if payload_metadata and index < len(payload_metadata)
                            else {}
                        ),
                    )
                    for index, passage in enumerate(passages_data)
                ],
            )

        return GenericDataset(files_data=files_data)

    def test_convert_default(self):
        generic_dataset = self.create_generic_dataset(
            queries_data=[["query 1"], ["query 2"]],
            passages_data=[["passage 1", "passage 2"], ["passage 3", "passage 4"]],
        )

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.RANKINGS

        rankings_converter = RankingsConverter(config)
        result = rankings_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "query": {"text": "query 1"},
                            "passages": [
                                {"text": "passage 1"},
                                {"text": "passage 2"},
                            ],
                            "model": "test_model",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "query": {"text": "query 2"},
                            "passages": [
                                {"text": "passage 3"},
                                {"text": "passage 4"},
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
            queries_data=[["query 1"], ["query 2"]],
            passages_data=[["passage 1", "passage 2"], ["passage 3", "passage 4"]],
        )

        extra_inputs = {
            "encoding_format": "base64",
            "truncate": "END",
            "additional_key": "additional_value",
        }

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.RANKINGS
        config.input.extra = extra_inputs

        rankings_converter = RankingsConverter(config)
        result = rankings_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "query": {"text": "query 1"},
                            "passages": [
                                {"text": "passage 1"},
                                {"text": "passage 2"},
                            ],
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
                            "passages": [
                                {"text": "passage 3"},
                                {"text": "passage 4"},
                            ],
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
        generic_dataset = self.create_generic_dataset(
            queries_data=[["query 1"], ["query 2"]],
            passages_data=[["passage 1", "passage 2"], ["passage 3", "passage 4"]],
        )

        extra_inputs = {
            "rankings": "tei",
            "additional_key": "additional_value",
        }

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.RANKINGS
        config.input.extra = extra_inputs

        rankings_converter = RankingsConverter(config)
        result = rankings_converter.convert(generic_dataset)

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

    @pytest.mark.parametrize(
        "queries_data, passages_data, expected_error",
        [
            (
                None,
                [["passage 1"], ["passage 2"]],
                "Both 'queries.jsonl' and 'passages.jsonl' must be present in the input datasets.",
            ),
            (
                [["query 1"], ["query 2"]],
                None,
                "Both 'queries.jsonl' and 'passages.jsonl' must be present in the input datasets.",
            ),
            (
                None,
                None,
                "Both 'queries.jsonl' and 'passages.jsonl' must be present in the input datasets.",
            ),
        ],
    )
    def test_convert_missing_files(self, queries_data, passages_data, expected_error):
        generic_dataset = self.create_generic_dataset(
            queries_data=queries_data, passages_data=passages_data
        )

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.RANKINGS

        rankings_converter = RankingsConverter(config)

        with pytest.raises(ValueError) as excinfo:
            rankings_converter.convert(generic_dataset)

        assert str(excinfo.value) == expected_error

    @pytest.mark.parametrize(
        "queries_data, passages_data, expected_result",
        [
            # More queries than passages
            (
                [["query 1"], ["query 2"], ["query 3"]],
                [["passage 1"], ["passage 2"]],
                {
                    "data": [
                        {
                            "payload": [
                                {
                                    "query": {"text": "query 1"},
                                    "passages": [{"text": "passage 1"}],
                                    "model": "test_model",
                                }
                            ]
                        },
                        {
                            "payload": [
                                {
                                    "query": {"text": "query 2"},
                                    "passages": [{"text": "passage 2"}],
                                    "model": "test_model",
                                }
                            ]
                        },
                    ]
                },
            ),
            # More passages than queries
            (
                [["query 1"]],
                [["passage 1"], ["passage 2"]],
                {
                    "data": [
                        {
                            "payload": [
                                {
                                    "query": {"text": "query 1"},
                                    "passages": [{"text": "passage 1"}],
                                    "model": "test_model",
                                }
                            ]
                        }
                    ]
                },
            ),
        ],
    )
    def test_convert_mismatched_queries_and_passages(
        self, queries_data, passages_data, expected_result
    ):
        generic_dataset = self.create_generic_dataset(
            queries_data=queries_data, passages_data=passages_data
        )

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.RANKINGS

        rankings_converter = RankingsConverter(config)

        result = rankings_converter.convert(generic_dataset)

        assert result == expected_result

    def test_convert_with_payload_parameters(self):
        optional_data_1 = {"session_id": "abcd"}
        optional_data_2 = {
            "session_id": "dfwe",
            "input_length": "6755",
            "output_length": "500",
        }
        generic_dataset = self.create_generic_dataset_payload_parameters(
            queries_data=[["query 1"], ["query 2"]],
            passages_data=[["passage 1", "passage 2"], ["passage 3", "passage 4"]],
            optional_data=[optional_data_1, optional_data_2],
            payload_metadata=[{"timestamp": 0}, {"timestamp": 2345}],
        )

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.model_selection_strategy = ModelSelectionStrategy.ROUND_ROBIN
        config.endpoint.output_format = OutputFormat.RANKINGS

        rankings_converter = RankingsConverter(config)
        result = rankings_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "query": {"text": "query 1"},
                            "passages": [
                                {"text": "passage 1"},
                                {"text": "passage 2"},
                            ],
                            "model": "test_model",
                            "session_id": "abcd",
                        }
                    ],
                    "timestamp": [0],
                },
                {
                    "payload": [
                        {
                            "query": {"text": "query 2"},
                            "passages": [
                                {"text": "passage 3"},
                                {"text": "passage 4"},
                            ],
                            "model": "test_model",
                            "session_id": "dfwe",
                            "input_length": "6755",
                            "output_length": "500",
                        }
                    ],
                    "timestamp": [2345],
                },
            ]
        }

        assert result == expected_result
