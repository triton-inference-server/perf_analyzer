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

import io
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import ModelSelectionStrategy
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.inputs.retrievers.payload_input_retriever import PayloadInputRetriever
from genai_perf.inputs.retrievers.synthetic_prompt_generator import (
    SyntheticPromptGenerator,
)
from genai_perf.tokenizer import get_empty_tokenizer


class TestPayloadInputRetriever:
    @pytest.fixture
    def mock_config(self):
        class MockConfig:
            def __init__(self):
                self.config = ConfigCommand({"model_name": "test_model"})
                self.config.endpoint.model_selection_strategy = (
                    ModelSelectionStrategy.ROUND_ROBIN
                )
                self.config.input.synthetic_tokens.mean = 10
                self.config.input.synthetic_tokens.stddev = 2
                self.config.input.sessions.turn_delay.ratio = 1.0
                self.config.input.payload_file = Path("test_input.jsonl")
                self.tokenizer = get_empty_tokenizer()
                self.output_directory = Path("test_output")

        return MockConfig()

    @pytest.fixture
    def retriever(self, mock_config):
        return PayloadInputRetriever(mock_config)

    @pytest.mark.parametrize(
        "input_data, expected_prompts, expected_payload_metadata_list, expected_optional_data_list",
        [
            (
                '{"text": "What is AI?", "timestamp": 123, "session_id": "abc"}\n'
                '{"text": "How does ML work?", "timestamp": 0, "custom_field": "value"}\n',
                ["What is AI?", "How does ML work?"],
                [{"timestamp": 123, "session_id": "abc"}, {"timestamp": 0}],
                [{}, {"custom_field": "value"}],
            ),
            (
                '{"text_input": "Legacy prompt", "timestamp": 456}\n'
                '{"text": "New prompt", "timestamp": 432, "session_id": "def"}\n',
                ["Legacy prompt", "New prompt"],
                [{"timestamp": 456}, {"timestamp": 432, "session_id": "def"}],
                [{}, {}],
            ),
            (
                '{"text": "What is AI?", "timestamp": 123, "session_id": "abc"}',
                ["What is AI?"],
                [{"timestamp": 123, "session_id": "abc"}],
                [{}],
            ),
            (
                '{"text_input": "Legacy prompt", "timestamp": 456}',
                ["Legacy prompt"],
                [{"timestamp": 456}],
                [{}],
            ),
            ('{"timestamp": 789}\n', ["Synthetic prompt"], [{"timestamp": 789}], [{}]),
            (
                '{"text": "What is AI?", "delay": 123, "session_id": "abc"}\n'
                '{"text": "How does ML work?", "delay": 0, "custom_field": "value"}\n',
                ["What is AI?", "How does ML work?"],
                [{"delay": 123, "session_id": "abc"}, {"delay": 0}],
                [{}, {"custom_field": "value"}],
            ),
        ],
    )
    @patch("builtins.open")
    @patch.object(SyntheticPromptGenerator, "create_synthetic_prompt")
    def test_get_content_from_input_file(
        self,
        mock_synthetic_prompt,
        mock_file,
        retriever,
        input_data,
        expected_prompts,
        expected_payload_metadata_list,
        expected_optional_data_list,
    ):
        mock_file.return_value = io.StringIO(input_data)
        mock_synthetic_prompt.return_value = "Synthetic prompt"

        data_dict = retriever._get_content_from_input_file(Path("test_input.jsonl"))

        assert data_dict["prompts"] == expected_prompts
        assert data_dict["payload_metadata_list"] == expected_payload_metadata_list
        assert data_dict["optional_data_list"] == expected_optional_data_list

        if "text" not in input_data and "text_input" not in input_data:
            mock_synthetic_prompt.assert_called_once()

    def test_convert_content_to_data_file(self, retriever):
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        optional_data_list = [{"session_id": "123"}, {"custom_field": "value"}, {}]
        payload_metadata_list = [{"timestamp": 1}, {"timestamp": 2}, {"delay": 3}]
        data_dict = {
            "prompts": prompts,
            "payload_metadata_list": payload_metadata_list,
            "optional_data_list": optional_data_list,
        }

        file_data = retriever._convert_content_to_data_file(data_dict)

        assert len(file_data.rows) == 3
        assert file_data.rows[0].texts == ["Prompt 1"]
        assert file_data.rows[0].payload_metadata
        assert file_data.rows[0].payload_metadata.get("timestamp")
        assert file_data.rows[0].payload_metadata.get("timestamp") == 1
        assert file_data.rows[0].optional_data == {"session_id": "123"}
        assert file_data.rows[1].texts == ["Prompt 2"]
        assert file_data.rows[1].payload_metadata
        assert file_data.rows[1].payload_metadata.get("timestamp")
        assert file_data.rows[1].payload_metadata.get("timestamp") == 2
        assert file_data.rows[1].optional_data == {"custom_field": "value"}
        assert file_data.rows[2].texts == ["Prompt 3"]
        assert file_data.rows[2].payload_metadata
        assert file_data.rows[2].payload_metadata.get("delay")
        assert file_data.rows[2].payload_metadata.get("delay") == 3

    @pytest.mark.parametrize(
        "payload_metadata",
        [
            {"timestamp": 123, "session_id": "abc"},
            {"delay": 456, "session_id": "def"},
        ],
    )
    @patch.object(PayloadInputRetriever, "_get_input_dataset_from_file")
    def test_retrieve_data(self, mock_get_input, retriever, payload_metadata):
        mock_file_data = FileData(
            [
                DataRow(
                    texts=["Test prompt"],
                    optional_data={"key": "value"},
                    payload_metadata=payload_metadata,
                )
            ]
        )
        mock_get_input.return_value = mock_file_data

        dataset = retriever.retrieve_data()

        assert isinstance(dataset, GenericDataset)
        assert str(retriever.config.input.payload_file) in dataset.files_data
        assert (
            dataset.files_data[str(retriever.config.input.payload_file)]
            == mock_file_data
        )

    @patch("builtins.open", new_callable=mock_open)
    def test_conflicting_keys_error(self, mock_file, retriever):
        conflicting_data = '{"text": "Prompt", "text_input": "Conflicting prompt"}\n'
        mock_file.return_value = io.StringIO(conflicting_data)

        with pytest.raises(
            ValueError,
            match="Each data entry must have only one of 'text_input' or 'text' key name.",
        ):
            retriever._get_content_from_input_file(Path("test_input.jsonl"))

    @pytest.mark.parametrize(
        "delay, delay_ratio, expected_delay",
        [
            (1000, 1.0, 1000),  # No change with ratio 1.0
            (1000, 0.5, 500),  # Half the delay
            (1000, 2.0, 2000),  # Double the delay
            (0, 1.5, 0),  # Zero delay stays zero
        ],
    )
    def test_delay_ratio_application(
        self, retriever, delay, delay_ratio, expected_delay
    ):
        # Set the delay ratio in the config
        retriever.config.input.sessions.turn_delay.ratio = delay_ratio

        # Create test data with a delay
        data = {"delay": delay}

        # Get the payload metadata
        metadata = retriever._get_payload_metadata(data)

        # Check that the delay was properly scaled
        assert metadata["delay"] == expected_delay
