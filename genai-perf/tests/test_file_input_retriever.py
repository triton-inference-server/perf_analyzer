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

import pytest
from genai_perf.inputs.input_constants import ModelSelectionStrategy
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.file_input_retriever import FileInputRetriever
from genai_perf.tokenizer import get_empty_tokenizer
from PIL import Image


class TestFileInputRetriever:

    @staticmethod
    def open_side_effect(filepath, *args, **kwargs):
        single_prompt = '{"text": "What is the capital of France?"}\n'
        multiple_prompts = (
            '{"text": "What is the capital of France?"}\n'
            '{"text": "Who wrote 1984?"}\n'
            '{"text": "What is quantum computing?"}\n'
        )
        single_image = '{"image": "image1.png"}\n'
        multiple_images = (
            '{"image": "image1.png"}\n'
            '{"image": "image2.png"}\n'
            '{"image": "image3.png"}\n'
        )
        multi_modal = (
            '{"text": "What is this image?", "image": "image1.png"}\n'
            '{"text": "Who is this person?", "image": "image2.png"}\n'
        )
        deprecated_text_input = (
            '{"text_input": "Who is Albert Einstein?"}\n'
            '{"text_input": "What is the speed of light?"}\n'
        )
        conflicting_key = '{"text": "This is a conflicting key", "text_input": "This is a conflicting key"}\n'

        file_contents = {
            "single_prompt.jsonl": single_prompt,
            "multiple_prompts.jsonl": multiple_prompts,
            "single_image.jsonl": single_image,
            "multiple_images.jsonl": multiple_images,
            "multi_modal.jsonl": multi_modal,
            "deprecated_text_input.jsonl": deprecated_text_input,
            "conflicting_key.jsonl": conflicting_key,
        }
        filename = Path(filepath).name
        return mock_open(read_data=file_contents.get(filename))()

    @staticmethod
    @patch(
        "genai_perf.inputs.retrievers.file_input_retriever.FileInputRetriever._encode_image",
        return_value="mock_base64_image",
    )
    def mock_encode_image(mock_encode_image):
        return mock_encode_image

    @patch("pathlib.Path.exists", return_value=True)
    @patch("PIL.Image.open", return_value=Image.new("RGB", (10, 10)))
    @patch(
        "genai_perf.inputs.retrievers.file_input_retriever.FileInputRetriever._encode_image",
        return_value="mock_base64_image",
    )
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_single_image(
        self, mock_file, mock_image, mock_encode_image, mock_exists
    ):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(),
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("single_image.jsonl"),
            )
        )
        file_data = file_retriever._get_input_dataset_from_file(
            Path("single_image.jsonl")
        )

        assert file_data is not None
        assert file_data.filename == "single_image.jsonl"
        assert len(file_data.rows) == 1
        assert file_data.rows[0].images[0] == "mock_base64_image"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("PIL.Image.open", return_value=Image.new("RGB", (10, 10)))
    @patch(
        "genai_perf.inputs.retrievers.file_input_retriever.FileInputRetriever._encode_image",
        side_effect=["mock_base64_image1", "mock_base64_image2", "mock_base64_image3"],
    )
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_multiple_images(
        self, mock_file, mock_image_open, mock_encode_image, mock_exists
    ):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(),
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("multiple_images.jsonl"),
            )
        )
        file_data = file_retriever._get_input_dataset_from_file(
            Path("multiple_images.jsonl")
        )

        assert file_data is not None
        assert file_data.filename == "multiple_images.jsonl"
        assert len(file_data.rows) == 3
        expected_images = [
            "mock_base64_image1",
            "mock_base64_image2",
            "mock_base64_image3",
        ]
        for i, image in enumerate(expected_images):
            assert file_data.rows[i].images[0] == image

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_single_prompt(self, mock_file, mock_exists):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(),
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("single_prompt.jsonl"),
            )
        )
        file_data = file_retriever._get_input_dataset_from_file(
            Path("single_prompt.jsonl")
        )

        assert file_data is not None
        assert file_data.filename == "single_prompt.jsonl"
        assert len(file_data.rows) == 1
        assert file_data.rows[0].texts[0] == "What is the capital of France?"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_multiple_prompts(self, mock_file, mock_exists):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(),
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("multiple_prompts.jsonl"),
            )
        )
        file_data = file_retriever._get_input_dataset_from_file(
            Path("multiple_prompts.jsonl")
        )

        assert file_data is not None
        assert file_data.filename == "multiple_prompts.jsonl"
        assert len(file_data.rows) == 3
        expected_prompts = [
            "What is the capital of France?",
            "Who wrote 1984?",
            "What is quantum computing?",
        ]
        for i, prompt in enumerate(expected_prompts):
            assert file_data.rows[i].texts[0] == prompt

    @patch("pathlib.Path.exists", return_value=True)
    @patch("PIL.Image.open", return_value=Image.new("RGB", (10, 10)))
    @patch(
        "genai_perf.inputs.retrievers.file_input_retriever.FileInputRetriever._encode_image",
        return_value="mock_base64_image",
    )
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_multi_modal(
        self, mock_file, mock_image, mock_encode_image, mock_exists
    ):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(),
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("multi_modal.jsonl"),
            )
        )
        file_data = file_retriever._get_input_dataset_from_file(
            Path("multi_modal.jsonl")
        )

        assert file_data is not None
        assert file_data.filename == "multi_modal.jsonl"
        assert len(file_data.rows) == 2
        assert file_data.rows[0].texts[0] == "What is this image?"
        assert file_data.rows[0].images[0] == "mock_base64_image"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_deprecated_text_input(self, mock_file, mock_exists):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(),
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("deprecated_text_input.jsonl"),
            )
        )
        file_data = file_retriever._get_input_dataset_from_file(
            Path("deprecated_text_input.jsonl")
        )

        assert file_data is not None
        assert file_data.filename == "deprecated_text_input.jsonl"
        assert len(file_data.rows) == 2
        assert file_data.rows[0].texts[0] == "Who is Albert Einstein?"
        assert file_data.rows[1].texts[0] == "What is the speed of light?"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_conflicting_key(self, mock_file, mock_exists):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(),
                batch_size_image=1,
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("conflicting_key.jsonl"),
            )
        )
        with pytest.raises(
            ValueError,
            match="Each data entry must have only one of 'text_input' or 'text' key name.",
        ):
            file_retriever._get_input_dataset_from_file(Path("conflicting_key.jsonl"))

    def test_get_input_file_without_file_existing(self):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(),
                input_filename=Path("nonexistent_file.jsonl"),
            )
        )
        with pytest.raises(FileNotFoundError):
            file_retriever._get_input_dataset_from_file(Path("nonexistent_file.jsonl"))

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.glob", return_value=[])
    def test_get_input_datasets_from_dir_no_jsonl_files(
        self, mock_exists, mock_is_dir, mock_glob
    ):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(), input_filename=Path("empty_dir")
            )
        )
        with pytest.raises(ValueError, match="No JSONL files found in directory"):
            _ = file_retriever._get_input_datasets_from_dir()

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch(
        "pathlib.Path.glob",
        return_value=[
            Path("single_prompt.jsonl"),
            Path("multiple_prompts.jsonl"),
            Path("single_image.jsonl"),
            Path("multi_modal.jsonl"),
        ],
    )
    @patch("PIL.Image.open", return_value=Image.new("RGB", (10, 10)))
    @patch(
        "genai_perf.inputs.retrievers.file_input_retriever.FileInputRetriever._encode_image",
        return_value="mock_base64_image",
    )
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_datasets_from_dir(
        self,
        mock_file,
        mock_image_open,
        mock_encode_image,
        mock_glob,
        mock_is_dir,
        mock_exists,
    ):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(),
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("test_dir"),
            )
        )

        file_data = file_retriever._get_input_datasets_from_dir()

        assert len(file_data) == 4

        assert file_data["single_prompt"].filename == "single_prompt.jsonl"
        assert len(file_data["single_prompt"].rows) == 1
        assert (
            file_data["single_prompt"].rows[0].texts[0]
            == "What is the capital of France?"
        )

        assert file_data["multiple_prompts"].filename == "multiple_prompts.jsonl"
        assert len(file_data["multiple_prompts"].rows) == 3
        expected_prompts = [
            "What is the capital of France?",
            "Who wrote 1984?",
            "What is quantum computing?",
        ]
        for i, prompt in enumerate(expected_prompts):
            assert file_data["multiple_prompts"].rows[i].texts[0] == prompt

        assert file_data["single_image"].filename == "single_image.jsonl"
        assert len(file_data["single_image"].rows) == 1
        assert file_data["single_image"].rows[0].images[0] == "mock_base64_image"

        assert file_data["multi_modal"].filename == "multi_modal.jsonl"
        assert len(file_data["multi_modal"].rows) == 2
        assert file_data["multi_modal"].rows[0].texts[0] == "What is this image?"
        assert file_data["multi_modal"].rows[0].images[0] == "mock_base64_image"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.glob", return_value=[])
    def test_get_input_datasets_from_empty_dir(
        self, mock_exists, mock_is_dir, mock_glob
    ):
        file_retriever = FileInputRetriever(
            InputsConfig(
                tokenizer=get_empty_tokenizer(), input_filename=Path("empty_dir")
            )
        )
        with pytest.raises(ValueError, match="No JSONL files found in directory"):
            _ = file_retriever._get_input_datasets_from_dir()
