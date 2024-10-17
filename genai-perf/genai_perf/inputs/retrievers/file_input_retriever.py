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

import random
from pathlib import Path
from typing import Dict, List, Tuple, cast

from genai_perf import utils
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import DEFAULT_BATCH_SIZE
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.base_input_retriever import BaseInputRetriever
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.inputs.retrievers.synthetic_image_generator import ImageFormat
from genai_perf.utils import load_json_str
from PIL import Image


class FileInputRetriever(BaseInputRetriever):
    """
    A input retriever class that handles input data provided by the user through
    file and directories.
    """

    def retrieve_data(self) -> GenericDataset:
        """
        Retrieves the dataset from a file or directory.

        Returns
        -------
        GenericDataset
            The dataset containing file data.
        """

        files_data: Dict[str, FileData] = {}
        self.config.input_filename = cast(Path, self.config.input_filename)
        if self.config.input_filename.is_dir():
            files_data = self._get_input_datasets_from_dir()
        else:
            file_data = self._get_input_dataset_from_file(self.config.input_filename)
            files_data = {file_data.filename: file_data}

        return GenericDataset(files_data)

    def _get_input_datasets_from_dir(self) -> Dict[str, FileData]:
        """
        Retrieves the dataset from a directory containing multiple JSONL files.

        Args
        ----------
        directory : Path
            The directory path to process.

        Returns
        -------
        Dict[str, FileData]
            The dataset in the required format with the content
            read from the files.
        """
        self.config.input_filename = cast(Path, self.config.input_filename)
        jsonl_files = list(self.config.input_filename.glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(
                f"No JSONL files found in directory '{self.config.input_filename}'."
            )

        files_data: Dict[str, FileData] = {}
        for file in jsonl_files:
            file_data = self._get_input_dataset_from_file(file)
            files_data[file.stem] = file_data
        return files_data

    def _get_input_dataset_from_file(self, filename: Path) -> FileData:
        """
        Retrieves the dataset from a specific JSONL file.

        Args
        ----------
        filename : Path
            The path of the file to process.

        Returns
        -------
        Dict
            The dataset in the required format with the content
            read from the file.
        """
        self._verify_file(filename)
        prompts, images = self._get_content_from_input_file(filename)
        return self._convert_content_to_data_file(prompts, images, filename)

    def _verify_file(self, filename: Path) -> None:
        """
        Verifies that the file exists.

        Args
        ----------
        filename : Path
            The file path to verify.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not filename.exists():
            raise FileNotFoundError(f"The file '{filename}' does not exist.")

    def _get_content_from_input_file(
        self, filename: Path
    ) -> Tuple[List[str], List[str]]:
        """
        Reads the content from a JSONL file and returns lists of each content type.

        Args
        ----------
        filename : Path
            The file path from which to read the content.

        Returns
        -------
        Tuple[List[str], List[str]]
            A list of prompts and images read from the file.
        """
        prompts = []
        images = []
        with open(filename, mode="r", newline=None) as file:
            for line in file:
                if line.strip():
                    data = load_json_str(line)
                    # None if not provided
                    prompt = data.get("text")
                    prompt_alt = data.get("text_input")
                    if prompt and prompt_alt:
                        raise ValueError(
                            "Each data entry must have only one of 'text_input' or 'text' key name."
                        )
                    prompt = prompt if prompt else prompt_alt
                    prompts.append(prompt.strip() if prompt else prompt)
                    image = data.get("image")
                    if image is not None:
                        image = self._encode_image(image.strip())
                        images.append(image)
        return prompts, images

    def _encode_image(self, filename: str) -> str:
        """
        Encodes the image file from a given filepath to
        the base64 format of the image.

        Args
        ----------
        filename : str
            The file path of the image to encode.

        Returns
        -------
        str
            The base64-encoded image string.
        """
        try:
            img = Image.open(filename)
        except FileNotFoundError:
            raise GenAIPerfException(f"Failed to open image '{filename}'.")
        if img.format is None:
            raise GenAIPerfException(
                f"Failed to determine image format of '{filename}'."
            )

        if img.format.lower() not in utils.get_enum_names(ImageFormat):
            raise GenAIPerfException(
                f"Unsupported image format '{img.format}' of "
                f"the image '{filename}'."
            )

        img_base64 = utils.encode_image(img, img.format)
        payload = f"data:image/{img.format.lower()};base64,{img_base64}"
        return payload

    def _convert_content_to_data_file(
        self, prompts: List[str], images: List[str], filename: Path
    ) -> FileData:
        """
        Converts the content to a DataFile.

        Args
        ----------
        prompts : List[str]
            The list of prompts to convert.
        images : List[str]
            The list of images to convert.
        filename : Path
            The filename to use for the DataFile.

        Returns
        -------
        FileData
            The DataFile containing the converted data.
        """
        data_rows: List[DataRow] = []

        if prompts and images:
            if self.config.batch_size_text > len(prompts):
                raise ValueError(
                    "Batch size for texts cannot be larger than the number of available texts"
                )
            if self.config.batch_size_image > len(images):
                raise ValueError(
                    "Batch size for images cannot be larger than the number of available images"
                )
            if (
                self.config.batch_size_image > DEFAULT_BATCH_SIZE
                or self.config.batch_size_text > DEFAULT_BATCH_SIZE
            ):
                for _ in range(self.config.num_prompts):
                    sampled_texts = random.sample(prompts, self.config.batch_size_text)
                    sampled_images = random.sample(images, self.config.batch_size_image)
                    data_rows.append(
                        DataRow(texts=sampled_texts, images=sampled_images)
                    )
            else:
                for prompt, image in zip(prompts, images):
                    data_rows.append(DataRow(texts=[prompt], images=[image]))
        elif prompts:
            if self.config.batch_size_text > len(prompts):
                raise ValueError(
                    "Batch size for texts cannot be larger than the number of available texts"
                )
            if self.config.batch_size_text > DEFAULT_BATCH_SIZE:
                for _ in range(self.config.num_prompts):
                    sampled_texts = random.sample(prompts, self.config.batch_size_text)
                    data_rows.append(DataRow(texts=sampled_texts, images=[]))
            else:
                for prompt in prompts:
                    data_rows.append(DataRow(texts=[prompt], images=[]))
        elif images:
            if self.config.batch_size_image > len(images):
                raise ValueError(
                    "Batch size for images cannot be larger than the number of available images"
                )

            if self.config.batch_size_image > DEFAULT_BATCH_SIZE:
                for _ in range(self.config.num_prompts):
                    sampled_images = random.sample(images, self.config.batch_size_image)
                    data_rows.append(DataRow(texts=[], images=sampled_images))
            else:
                for image in images:
                    data_rows.append(DataRow(texts=[], images=[image]))

        return FileData(str(filename), data_rows)
