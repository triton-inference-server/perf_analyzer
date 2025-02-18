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

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.retrievers.base_file_input_retriever import (
    BaseFileInputRetriever,
)
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.inputs.retrievers.synthetic_prompt_generator import (
    SyntheticPromptGenerator,
)
from genai_perf.utils import load_json_str


class PayloadInputRetriever(BaseFileInputRetriever):
    """
    A input retriever class that handles payload level input data provided by the user
    through a file.
    """

    def retrieve_data(self) -> GenericDataset:
        """
        Retrieves the dataset from a file.

        Returns
        -------
        GenericDataset
            The dataset containing file data.
        """

        files_data: Dict[str, FileData] = {}
        input_file = self.config.payload_input_filename
        if input_file is None:
            raise ValueError("Input file cannot be None")
        file_data = self._get_input_dataset_from_file(input_file)
        files_data = {str(input_file): file_data}

        return GenericDataset(files_data)

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
        prompts, timestamps, optional_datas = self._get_content_from_input_file(
            filename
        )
        return self._convert_content_to_data_file(prompts, timestamps, optional_datas)

    def _get_content_from_input_file(
        self, filename: Path
    ) -> Tuple[List[str], List[int], List[Dict[Any, Any]]]:
        """
        Reads the content from a JSONL file and returns lists of each content type.

        Args
        ----------
        filename : Path
            The file path from which to read the content.

        Returns
        -------
        Tuple[List[str], Dict, str]
            A list of prompts, and optional data.
        """
        prompts = []
        optional_datas = []
        timestamps = []
        with open(filename, mode="r", newline=None) as file:
            for line in file:
                if line.strip():
                    data = load_json_str(line)
                    hash_ids = data.get("hash_ids")
                    prompt = self._get_prompt(data, hash_ids)
                    prompts.append(prompt.strip() if prompt else prompt)
                    timestamp = self._get_valid_timestamp(data)
                    timestamps.append(timestamp)
                    optional_data = self._check_for_optional_data(data)
                    optional_datas.append(optional_data)
        return prompts, timestamps, optional_datas

    def _get_prompt(self, data: Dict[str, Any], hash_ids: Optional[List[int]]) -> str:
        prompt = data.get("text")
        prompt_alt = data.get("text_input")
        # Check if only one of the keys is provided
        if prompt and prompt_alt:
            raise ValueError(
                "Each data entry must have only one of 'text_input' or 'text' key name."
            )
        # If none of the keys are provided, generate a synthetic prompt
        if not prompt and not prompt_alt:
            prompt = SyntheticPromptGenerator.create_synthetic_prompt(
                self.config.tokenizer,
                self.config.prompt_tokens_mean,
                self.config.prompt_tokens_stddev,
                hash_ids,
            )
            prompt = prompt if prompt else prompt_alt
        return str(prompt)

    def _get_valid_timestamp(self, data: Dict[str, Any]) -> int:
        """
        Retrieves and validates timestamp from input data
        """
        timestamp = data.get("timestamp")
        if timestamp is None:
            raise GenAIPerfException("Each data entry must have a 'timestamp' field.")
        try:
            timestamp = int(timestamp)
        except Exception:
            raise GenAIPerfException(
                f"Invalid timestamp: Expecting an integer but received '{timestamp}'."
            )

        return timestamp

    def _check_for_optional_data(self, data: Dict[str, Any]) -> Dict[Any, Any]:
        """
        Checks if there is any optional data in the file to pass in the payload.
        """
        excluded_keys = {
            "text",
            "text_input",
            "timestamp",
            "hash_ids",
            "input_length",
            "output_length",
        }
        optional_data = {k: v for k, v in data.items() if k not in excluded_keys}
        return optional_data

    def _convert_content_to_data_file(
        self,
        prompts: List[str],
        timestamps: List[int],
        optional_datas: List[Dict[Any, Any]] = [{}],
    ) -> FileData:
        """
        Converts the content to a DataFile.

        Args
        ----------
        prompts : List[str]
            The list of prompts to convert.
        timestamps: int
            The timestamp at which the request should be sent.
        optional_data : Dict
            The optional data included in every payload.

        Returns
        -------
        FileData
            The DataFile containing the converted data.
        """
        data_rows: List[DataRow] = [
            DataRow(
                texts=[prompt],
                timestamp=timestamps[index],
                optional_data=optional_datas[index],
            )
            for index, prompt in enumerate(prompts)
        ]

        return FileData(data_rows)
