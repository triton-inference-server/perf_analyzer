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

from typing import cast
from unittest.mock import patch

import pytest
from genai_perf.inputs.input_constants import DEFAULT_SYNTHETIC_FILENAME, OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.synthetic_data_retriever import SyntheticDataRetriever
from genai_perf.tokenizer import get_empty_tokenizer


class TestSyntheticDataRetriever:
    @patch(
        "genai_perf.inputs.retrievers.synthetic_data_retriever.SyntheticPromptGenerator.create_synthetic_prompt",
        return_value="test prompt",
    )
    @pytest.mark.parametrize(
        "batch_size_text, num_dataset_entries",
        [
            (1, 3),
            (2, 2),
        ],
    )
    def test_synthetic_text(self, mock_prompt, batch_size_text, num_dataset_entries):
        config = InputsConfig(
            num_dataset_entries=num_dataset_entries,
            batch_size_text=batch_size_text,
            output_format=OutputFormat.OPENAI_COMPLETIONS,
            synthetic_input_filenames=[DEFAULT_SYNTHETIC_FILENAME],
            tokenizer=get_empty_tokenizer(),
        )
        synthetic_retriever = SyntheticDataRetriever(config)
        dataset = synthetic_retriever.retrieve_data()

        synthetic_input_filenames = cast(list[str], config.synthetic_input_filenames)
        assert (
            len(dataset.files_data[synthetic_input_filenames[0]].rows)
            == num_dataset_entries
        )
        for row in dataset.files_data[synthetic_input_filenames[0]].rows:
            assert len(row.texts) == batch_size_text
            assert all(text == "test prompt" for text in row.texts)

    @patch(
        "genai_perf.inputs.retrievers.synthetic_data_retriever.SyntheticPromptGenerator.create_synthetic_prompt",
        return_value="test prompt",
    )
    @patch(
        "genai_perf.inputs.retrievers.synthetic_data_retriever.SyntheticImageGenerator.create_synthetic_image",
        return_value="data:image/jpeg;base64,test_base64_encoding",
    )
    @pytest.mark.parametrize(
        "batch_size_text, batch_size_image, num_dataset_entries",
        [
            (1, 1, 3),
            (2, 1, 2),
            (1, 2, 1),
        ],
    )
    def test_synthetic_text_and_image(
        self,
        mock_prompt,
        mock_image,
        batch_size_text,
        batch_size_image,
        num_dataset_entries,
    ):
        config = InputsConfig(
            num_dataset_entries=num_dataset_entries,
            batch_size_text=batch_size_text,
            batch_size_image=batch_size_image,
            output_format=OutputFormat.OPENAI_VISION,
            synthetic_input_filenames=[DEFAULT_SYNTHETIC_FILENAME],
            tokenizer=get_empty_tokenizer(),
        )
        synthetic_retriever = SyntheticDataRetriever(config)
        dataset = synthetic_retriever.retrieve_data()

        synthetic_input_filenames = cast(list[str], config.synthetic_input_filenames)
        assert (
            len(dataset.files_data[synthetic_input_filenames[0]].rows)
            == num_dataset_entries
        )

        for row in dataset.files_data[synthetic_input_filenames[0]].rows:
            assert len(row.texts) == batch_size_text
            assert len(row.images) == batch_size_image
            assert all(text == "test prompt" for text in row.texts)
            assert all(
                image == "data:image/jpeg;base64,test_base64_encoding"
                for image in row.images
            )

    @patch(
        "genai_perf.inputs.retrievers.synthetic_data_retriever.SyntheticPromptGenerator.create_synthetic_prompt",
        return_value="test prompt",
    )
    @patch(
        "genai_perf.inputs.retrievers.synthetic_data_retriever.SyntheticImageGenerator.create_synthetic_image",
        return_value="data:image/jpeg;base64,test_base64_encoding",
    )
    def test_synthetic_multiple_files(self, mock_prompt, mock_image):
        """
        Test synthetic data generation when multiple synthetic files are specified.
        """
        config = InputsConfig(
            num_dataset_entries=2,
            batch_size_text=1,
            batch_size_image=1,
            synthetic_input_filenames=["file1.jsonl", "file2.jsonl"],
            output_format=OutputFormat.OPENAI_VISION,
            tokenizer=get_empty_tokenizer(),
        )
        synthetic_retriever = SyntheticDataRetriever(config)
        dataset = synthetic_retriever.retrieve_data()

        assert len(dataset.files_data) == 2
        assert "file1.jsonl" in dataset.files_data
        assert "file2.jsonl" in dataset.files_data

        for file_data in dataset.files_data.values():
            assert len(file_data.rows) == 2

            for row in file_data.rows:
                assert len(row.texts) == 1
                assert len(row.images) == 1
                assert row.texts[0] == "test prompt"
                assert row.images[0] == "data:image/jpeg;base64,test_base64_encoding"

    @patch(
        "genai_perf.inputs.retrievers.synthetic_data_retriever.SyntheticPromptGenerator.create_synthetic_prompt",
        return_value="test prompt",
    )
    @patch(
        "genai_perf.inputs.retrievers.synthetic_data_retriever.SyntheticPromptGenerator.create_prefix_prompts_pool"
    )
    @patch(
        "genai_perf.inputs.retrievers.synthetic_data_retriever.SyntheticPromptGenerator.get_random_prefix_prompt",
        return_value="prompt prefix",
    )
    def test_synthetic_with_prefix_prompts(
        self,
        mock_random_prefix_prompt,
        mock_create_prefix_prompts_pool,
        mock_create_synthetic_prompt,
    ):
        config = InputsConfig(
            num_dataset_entries=3,
            num_prefix_prompts=3,
            prefix_prompt_length=20,
            batch_size_text=1,
            output_format=OutputFormat.OPENAI_COMPLETIONS,
            synthetic_input_filenames=[DEFAULT_SYNTHETIC_FILENAME],
            tokenizer=get_empty_tokenizer(),
        )

        synthetic_retriever = SyntheticDataRetriever(config)
        dataset = synthetic_retriever.retrieve_data()

        # Validate the number of rows in the retrieved dataset matches the expected count
        synthetic_input_filenames = cast(list[str], config.synthetic_input_filenames)
        expected_row_count = config.num_dataset_entries
        actual_row_count = len(dataset.files_data[synthetic_input_filenames[0]].rows)

        assert (
            actual_row_count == expected_row_count
        ), f"Expected {expected_row_count} rows, got {actual_row_count}"

        # Ensure the prompt prefixess pool was created exactly once
        mock_create_prefix_prompts_pool.assert_called_once()

        # Validate that every text in the dataset has the right prefix
        for row_index, row in enumerate(
            dataset.files_data[synthetic_input_filenames[0]].rows
        ):
            expected_prefix = "prompt prefix "
            for text_index, text in enumerate(row.texts):
                assert text.startswith(
                    expected_prefix
                ), f"Row {row_index}, text {text_index}: text does not start with '{expected_prefix}'. Actual: '{text}'"
