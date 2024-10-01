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

from collections import namedtuple
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from genai_perf.inputs.file_input_retriever import FileInputRetriever
from genai_perf.inputs.input_constants import ModelSelectionStrategy
from genai_perf.inputs.inputs_config import InputsConfig
from PIL import Image


class TestFileInputRetriever:

    def open_side_effects(self, filepath, *args, **kwargs):
        queries_content = "\n".join(
            [
                '{"text": "What production company co-owned by Kevin Loader and Rodger Michell produced My Cousin Rachel?"}',
                '{"text": "Who served as the 1st Vice President of Colombia under El Libertador?"}',
                '{"text": "Are the Barton Mine and Hermiston-McCauley Mine located in The United States of America?"}',
            ]
        )
        passages_content = "\n".join(
            [
                '{"text": "Eric Anderson (sociologist) Eric Anderson (born January 18, 1968) is an American sociologist"}',
                '{"text": "Kevin Loader is a British film and television producer. "}',
                '{"text": "Barton Mine, also known as Net Lake Mine, is an abandoned surface and underground mine in Northeastern Ontario"}',
            ]
        )

        file_contents = {
            "queries.jsonl": queries_content,
            "passages.jsonl": passages_content,
        }
        return mock_open(
            read_data=file_contents.get(filepath, file_contents["queries.jsonl"])
        )()

    mock_open_obj = mock_open()
    mock_open_obj.side_effect = open_side_effects

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", mock_open_obj)
    def test_read_rankings_input_files(self, mock_file):
        queries_filename = Path("queries.jsonl")
        passages_filename = Path("passages.jsonl")
        batch_size = 2
        config = InputsConfig(
            batch_size=batch_size,
            num_prompts=100,
        )
        file_retriever = FileInputRetriever(config)
        dataset = file_retriever._read_rankings_input_files(
            queries_filename=queries_filename, passages_filename=passages_filename
        )

        assert dataset is not None
        assert len(dataset["rows"]) == 100
        for row in dataset["rows"]:
            assert "row" in row
            payload = row["row"]
            assert "query" in payload
            assert "passages" in payload
            assert isinstance(payload["passages"], list)
            assert len(payload["passages"]) == batch_size

        # Try error case where batch size is larger than the number of available texts
        with pytest.raises(
            ValueError,
            match="Batch size cannot be larger than the number of available passages",
        ):
            config.batch_size = 5
            file_retriever._read_rankings_input_files(
                queries_filename, passages_filename
            )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_read_rankings_input_files_wrong_key(self, mock_file, mock_exists):
        queries = (
            '{"random_key": "test query one"}\n'  # random/wrong key
            '{"text": "text query two"}\n'
        )
        passages = '{"text": "test passage one"}\n' '{"text": "text passage two"}\n'

        mock_file.side_effect = [
            mock_open(read_data=queries).return_value,
            mock_open(read_data=passages).return_value,
        ]

        file_retriever = FileInputRetriever(InputsConfig(num_prompts=3))
        with pytest.raises(ValueError):
            _ = file_retriever._read_rankings_input_files(
                queries_filename=Path("queries.jsonl"),
                passages_filename=Path("passages.jsonl"),
            )

    def test_get_input_file_without_file_existing(self):
        file_retriever = FileInputRetriever(
            InputsConfig(input_filename=Path("prompt.txt"))
        )
        with pytest.raises(FileNotFoundError):
            file_retriever._get_input_dataset_from_file()

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"text_input": "single prompt"}\n',
    )
    def test_get_input_file_with_single_prompt(self, mock_file, mock_exists):
        expected_prompts = ["single prompt"]
        file_retriever = FileInputRetriever(
            InputsConfig(
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("prompt.txt"),
            )
        )
        dataset = file_retriever._get_input_dataset_from_file()

        assert dataset is not None
        assert len(dataset["rows"]) == len(expected_prompts)
        for i, prompt in enumerate(expected_prompts):
            assert dataset["rows"][i]["row"]["text_input"] == prompt

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"text_input": "prompt1"}\n{"text_input": "prompt2"}\n{"text_input": "prompt3"}\n',
    )
    def test_get_input_file_with_multiple_prompts(self, mock_file, mock_exists):
        expected_prompts = ["prompt1", "prompt2", "prompt3"]
        file_retriever = FileInputRetriever(
            InputsConfig(
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("prompt.txt"),
            )
        )
        dataset = file_retriever._get_input_dataset_from_file()

        assert dataset is not None
        assert len(dataset["rows"]) == len(expected_prompts)
        for i, prompt in enumerate(expected_prompts):
            assert dataset["rows"][i]["row"]["text_input"] == prompt

    @patch("pathlib.Path.exists", return_value=True)
    @patch("PIL.Image.open", return_value=Image.new("RGB", (10, 10)))
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=(
            '{"text_input": "prompt1", "image": "image1.png"}\n'
            '{"text_input": "prompt2", "image": "image2.png"}\n'
            '{"text_input": "prompt3", "image": "image3.png"}\n'
        ),
    )
    def test_get_input_file_with_multi_modal_data(
        self, mock_exists, mock_image, mock_file
    ):
        file_retriever = FileInputRetriever(
            InputsConfig(
                model_name=["test_model_A"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                input_filename=Path("prompt.txt"),
            )
        )
        Data = namedtuple("Data", ["text_input", "image"])
        expected_data = [
            Data(text_input="prompt1", image="image1.png"),
            Data(text_input="prompt2", image="image2.png"),
            Data(text_input="prompt3", image="image3.png"),
        ]
        dataset = file_retriever._get_input_dataset_from_file()

        assert dataset is not None
        assert len(dataset["rows"]) == len(expected_data)
        for i, data in enumerate(expected_data):
            assert dataset["rows"][i]["row"]["text_input"] == data.text_input
            assert dataset["rows"][i]["row"]["image"] == data.image
