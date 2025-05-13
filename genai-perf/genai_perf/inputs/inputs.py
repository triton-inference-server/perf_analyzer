# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import random
from typing import Dict

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters.output_format_converter_factory import (
    OutputFormatConverterFactory,
)
from genai_perf.inputs.input_constants import (
    DEFAULT_INPUT_DATA_JSON,
    MINIMUM_LENGTH,
    PromptSource,
)
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.input_retriever_factory import InputRetrieverFactory


class Inputs:
    """
    A class that generates the request input payloads for the target endpoint.
    """

    def __init__(self, inputs_config: InputsConfig):
        self.inputs_config = inputs_config
        self.config = inputs_config.config
        self.tokenizer = inputs_config.tokenizer
        self.output_directory = inputs_config.output_directory

        self.converter = OutputFormatConverterFactory.create(
            self.config.endpoint.output_format, self.config, self.tokenizer
        )
        self.converter.check_config()

        random.seed(self.config.input.random_seed)

    def create_inputs(self) -> None:
        """
        Given an input type, input format, and output type. Output a string of LLM Inputs
        (in a JSON dictionary) to a file.
        """
        self._check_for_valid_args()
        input_retriever = InputRetrieverFactory.create(self.inputs_config)
        generic_dataset = input_retriever.retrieve_data()

        json_in_pa_format = self._convert_generic_dataset_to_output_format(
            generic_dataset,
        )
        self._write_json_to_file(json_in_pa_format)

    def _check_for_valid_args(self) -> None:
        self._check_for_tokenzier_if_input_type_is_synthetic()
        self._check_for_valid_length()

    def _convert_generic_dataset_to_output_format(self, generic_dataset) -> Dict:
        return self.converter.convert(generic_dataset)

    def _write_json_to_file(self, json_in_pa_format: Dict) -> None:
        filename = self.output_directory / DEFAULT_INPUT_DATA_JSON
        with open(str(filename), "w") as f:
            f.write(json.dumps(json_in_pa_format, indent=2))

    def _check_for_tokenzier_if_input_type_is_synthetic(self) -> None:
        if (
            self.config.input.prompt_source == PromptSource.SYNTHETIC
            and not self.tokenizer
        ):
            raise GenAIPerfException(
                "Input type is SYNTHETIC, but a tokenizer was not specified."
            )

    def _check_for_valid_length(self) -> None:
        if not isinstance(self.config.input.num_dataset_entries, int):
            raise GenAIPerfException(
                f"length: {self.config.input.num_dataset_entries} must be an integer."
            )

        if self.config.input.num_dataset_entries < MINIMUM_LENGTH:
            raise GenAIPerfException(
                f"starting_index: {self.config.input.num_dataset_entries} must be larger than {MINIMUM_LENGTH}."
            )
