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

from typing import Any, Dict, List

import requests
from genai_perf import utils
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.file_input_retriever import FileInputRetriever
from genai_perf.inputs.input_constants import OutputFormat, PromptSource
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.synthetic_image_generator import (
    ImageFormat,
    SyntheticImageGenerator,
)
from genai_perf.inputs.synthetic_prompt_generator import SyntheticPromptGenerator
from PIL import Image
from requests import Response


class InputRetrieverFactory:
    def __init__(self, config: InputsConfig):
        self.config = config

    def get_input_data(self) -> Dict:
        """
        Retrieve and convert the dataset based on the input type.

        Returns
        -------
        Dict:
            The generic dataset JSON
        """

        if self.config.output_format in [
            OutputFormat.OPENAI_EMBEDDINGS,
            OutputFormat.RANKINGS,
            OutputFormat.IMAGE_RETRIEVAL,
        ]:
            # TODO: remove once the factory fully integrates retrievers
            file_retriever = FileInputRetriever(self.config)
            input_data = file_retriever.retrieve_data()

            if self.config.output_format == OutputFormat.IMAGE_RETRIEVAL:
                input_data = self._encode_images_in_input_dataset(input_data)

            generic_dataset_json = (
                self._convert_input_synthetic_or_file_dataset_to_generic_json(
                    input_data
                )
            )
        else:
            if self.config.input_type == PromptSource.SYNTHETIC:
                synthetic_dataset = self._get_input_dataset_from_synthetic()
                generic_dataset_json = (
                    self._convert_input_synthetic_or_file_dataset_to_generic_json(
                        synthetic_dataset
                    )
                )
            elif self.config.input_type == PromptSource.FILE:
                # TODO: remove once the factory fully integrates retrievers
                file_retriever = FileInputRetriever(self.config)
                input_data = file_retriever.retrieve_data()

                input_file_dataset = self._encode_images_in_input_dataset(input_data)
                generic_dataset_json = (
                    self._convert_input_synthetic_or_file_dataset_to_generic_json(
                        input_file_dataset
                    )
                )
            else:
                raise GenAIPerfException("Input source is not recognized.")

        return generic_dataset_json

    def _convert_input_synthetic_or_file_dataset_to_generic_json(
        self, dataset: Dict
    ) -> Dict[str, List[Dict]]:
        generic_dataset_json = self._convert_dataset_to_generic_input_json(dataset)

        return generic_dataset_json

    def _encode_images_in_input_dataset(self, input_file_dataset: Dict) -> Dict:
        for row in input_file_dataset["rows"]:
            if isinstance(row["row"], list):
                for content in row["row"]:
                    filename = content.get("image")
                    if filename:
                        payload = self._encode_image(filename)
                        content["image"] = payload
            else:
                filename = row["row"].get("image")
                if filename:
                    payload = self._encode_image(filename)
                    row["row"]["image"] = payload

        return input_file_dataset

    def _convert_input_url_dataset_to_generic_json(self, dataset: Response) -> Dict:
        dataset_json = dataset.json()
        try:
            self._check_for_error_in_json_of_dataset(dataset_json)
        except Exception as e:
            raise GenAIPerfException(e)

        generic_dataset_json = self._convert_dataset_to_generic_input_json(dataset_json)

        return generic_dataset_json

    def _get_input_dataset_from_synthetic(self) -> Dict[str, Any]:
        dataset_json: Dict[str, Any] = {}
        dataset_json["features"] = [{"name": "text_input"}]
        dataset_json["rows"] = []
        for _ in range(self.config.num_prompts):
            row: Dict["str", Any] = {"row": {}}
            synthetic_prompt = self._create_synthetic_prompt()
            row["row"]["text_input"] = synthetic_prompt

            if self.config.output_format == OutputFormat.OPENAI_VISION:
                synthetic_image = self._create_synthetic_image()
                row["row"]["image"] = synthetic_image

            dataset_json["rows"].append(row)

        return dataset_json

    def _convert_dataset_to_generic_input_json(
        self, dataset_json: Dict
    ) -> Dict[str, List[Dict]]:
        generic_input_json = self._add_features_to_generic_json({}, dataset_json)
        generic_input_json = self._add_rows_to_generic_json(
            generic_input_json, dataset_json
        )

        return generic_input_json

    def _add_features_to_generic_json(
        self, generic_input_json: Dict, dataset_json: Dict
    ) -> Dict:
        if "features" in dataset_json.keys():
            generic_input_json["features"] = []
            for feature in dataset_json["features"]:
                generic_input_json["features"].append(feature["name"])

        return generic_input_json

    def _add_rows_to_generic_json(
        self, generic_input_json: Dict, dataset_json: Dict
    ) -> Dict[str, List[Dict]]:
        generic_input_json["rows"] = []
        for row in dataset_json["rows"]:
            generic_input_json["rows"].append(row["row"])

        return generic_input_json

    def _encode_image(self, filename: str) -> str:
        img = Image.open(filename)
        if img is None:
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

    def _check_for_error_in_json_of_dataset(self, dataset_json: Dict) -> None:
        if "error" in dataset_json:
            raise GenAIPerfException(dataset_json["error"])

    def _create_synthetic_prompt(self) -> str:
        return SyntheticPromptGenerator.create_synthetic_prompt(
            self.config.tokenizer,
            self.config.prompt_tokens_mean,
            self.config.prompt_tokens_stddev,
        )

    def _create_synthetic_image(self) -> str:
        return SyntheticImageGenerator.create_synthetic_image(
            image_width_mean=self.config.image_width_mean,
            image_width_stddev=self.config.image_width_stddev,
            image_height_mean=self.config.image_height_mean,
            image_height_stddev=self.config.image_height_stddev,
            image_format=self.config.image_format,
        )
