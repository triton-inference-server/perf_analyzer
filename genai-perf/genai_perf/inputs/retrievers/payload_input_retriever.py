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
        data_dict = self._get_content_from_input_file(filename)
        return self._convert_content_to_data_file(data_dict)
        prompts, optional_data_list, payload_metadata_list = (
            self._get_content_from_input_file(filename)
        )
        return self._convert_content_to_data_file(
            prompts, optional_data_list, payload_metadata_list
        )

    def _get_content_from_input_file(
        self, filename: Path
    ) -> Tuple[List[str], List[Dict[Any, Any]], List[Dict[str, Any]]]:
        """
        Reads the content from a JSONL file and returns lists of each content type.

        Args
        ----------
        filename : Path
            The file path from which to read the content.

        Returns
        -------
        Tuple[List[str], List[Dict[Any, Any]], List[Dict[str, Any]]]
            The lists of prompts, optional data, and payload metadata.
        """
        prompts = []
        optional_data_list = []
        payload_metadata_list = []
        with open(filename, mode="r", newline=None) as file:
            for line in file:
                if line.strip():
                    data = load_json_str(line)
                    # None if not provided
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
                        )
                    prompt = prompt if prompt else prompt_alt
                    prompts.append(prompt.strip() if prompt else prompt)
                    try:
                        payload_metadata_list.append(self._get_payload_metadata(data))
                    except Exception as e:
                        raise GenAIPerfException(
                            f"Error while processing payload metadata: {e}"
                        )
                    optional_data = self._get_optional_data(data)
                    optional_data_list.append(optional_data)
        return prompts, optional_data_list, payload_metadata_list

    def _get_payload_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves and payload metadata from input data
        """

        return {
            key: (int(data[key]) if key in PAYLOAD_METADATA_INT_FIELDS else data[key])
            for key in PAYLOAD_METADATA_FIELDS
            if key in data
        }

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
        prompts: List[str],
        optional_data_list: List[Dict[Any, Any]] = [{}],
        payload_metadata_list: List[Dict[str, Any]] = [{}],
    ) -> FileData:
        """
        Converts the content to a DataFile.

        Args
        ----------
        prompts : List[str]
            The list of prompts to convert.
        optional_data : Dict
            The optional data included in every payload.
        payload_metadata_list : Dict
            The metadata about the payload.

        Returns
        -------
        FileData
            The DataFile containing the converted data.
        """
        prompt_list = data_dict["prompts"]
        timestamp_list = data_dict["timestamps"]
        optional_data_list = data_dict["optional_datas"]

        data_rows: List[DataRow] = [
            DataRow(
                texts=[prompt],
                optional_data=optional_data_list[index],
                payload_metadata=payload_metadata_list[index],
            )
            for index, prompt in enumerate(prompt_list)
        ]

        return FileData(data_rows)
