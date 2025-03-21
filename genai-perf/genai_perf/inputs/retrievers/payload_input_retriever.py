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
from typing import Any, Dict, List

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import (
    PAYLOAD_METADATA_FIELDS,
    PAYLOAD_METADATA_INT_FIELDS,
)
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
        input_file = self.config.input.payload_file
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
        data_dict = self._get_content_from_input_file(filename)
        return self._convert_content_to_data_file(data_dict)

    def _get_content_from_input_file(self, filename: Path) -> Dict[str, Any]:
        """
        Reads the content from a JSONL file and returns lists of each content type.

        Args
        ----------
        filename : Path
            The file path from which to read the content.

        Returns
        -------
        A dictionary containing extracted data:
            - "prompts": List[str] - Extracted prompts
            - "payload_metadata_list": List[Dict[str, Any]] - Corresponding payload metadata list
            - "optional_data_list": List[Dict[Any, Any]] - Optional data list
        """
        prompts = []
        optional_data_list = []
        payload_metadata_list = []
        with open(filename, mode="r", newline=None) as file:
            for line in file:
                if line.strip():
                    data = load_json_str(line)
                    prompt = self._get_prompt(data)
                    prompts.append(prompt.strip() if prompt else prompt)
                    try:
                        payload_metadata_list.append(self._get_payload_metadata(data))
                    except Exception as e:
                        raise GenAIPerfException(
                            f"Error while processing payload metadata: {e}"
                        )
                    optional_data = self._get_optional_data(data)
                    optional_data_list.append(optional_data)
        return {
            "prompts": prompts,
            "payload_metadata_list": payload_metadata_list,
            "optional_data_list": optional_data_list,
        }

    def _get_prompt(self, data: Dict[str, Any]) -> str:
        """
        Extracts or generates a prompt from the input data.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary containing input data.

        Returns
        -------
        str
            The extracted or generated prompt.

        Raises
        ------
        ValueError
            If both "text" and "text_input" fields are present.
        """
        input_length = data.get("input_length")
        prompt_tokens_mean = (
            input_length if input_length else self.config.input.synthetic_tokens.mean
        )
        prompt_tokens_stddev = (
            0 if input_length else self.config.input.synthetic_tokens.stddev
        )
        hash_ids = data.get("hash_ids", None)
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
                self.tokenizer,
                prompt_tokens_mean,
                prompt_tokens_stddev,
                hash_ids,
            )
        prompt = prompt if prompt else prompt_alt
        return str(prompt)

    def _get_payload_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves and payload metadata from input data
        and scales the delay if the delay_ratio is not 1.0
        """

        metadata = {
            key: (int(data[key]) if key in PAYLOAD_METADATA_INT_FIELDS else data[key])
            for key in PAYLOAD_METADATA_FIELDS
            if key in data
        }

        if "delay" in metadata and self.config.input.sessions.turn_delay.ratio != 1.0:
            metadata["delay"] = int(
                metadata["delay"] * self.config.input.sessions.turn_delay.ratio
            )

        return metadata

    def _get_optional_data(self, data: Dict[str, Any]) -> Dict[Any, Any]:
        """
        Extracts optional data from the input data. If "output_length" is present,
        it is explicitly renamed to "max_tokens".

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary containing input data.

        Returns
        -------
        Dict[Any, Any]
            A dictionary containing extracted optional data,
            with "output_length" renamed to "max_tokens" if present.
        """
        excluded_keys = {
            "text",
            "text_input",
            "hash_ids",
            "input_length",
            "output_length",
        }
        excluded_keys.update(PAYLOAD_METADATA_FIELDS)
        optional_data = {k: v for k, v in data.items() if k not in excluded_keys}
        max_tokens = data.get("output_length")
        if max_tokens:
            optional_data["max_tokens"] = max_tokens

        return optional_data

    def _convert_content_to_data_file(
        self,
        data_dict: Dict[str, Any],
    ) -> FileData:
        """
        Converts the content to a DataFile.

        Args
        data_dict : Dict[str, Any]
            A dictionary containing extracted lists of prompts, payload metadata,
            and optional data.

        Returns
        -------
        FileData
            The DataFile containing the converted data.
        """
        prompt_list = data_dict["prompts"]
        payload_metadata_list = data_dict["payload_metadata_list"]
        optional_data_list = data_dict["optional_data_list"]

        data_rows: List[DataRow] = [
            DataRow(
                texts=[prompt],
                optional_data=optional_data_list[index],
                payload_metadata=payload_metadata_list[index],
            )
            for index, prompt in enumerate(prompt_list)
        ]

        return FileData(data_rows)
