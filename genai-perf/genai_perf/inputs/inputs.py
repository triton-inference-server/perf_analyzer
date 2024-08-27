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

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from genai_perf import utils
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_DATA_JSON,
    MINIMUM_LENGTH,
    MINIMUM_STARTING_INDEX,
    OutputFormat,
    PromptSource,
    dataset_url_map,
)
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.output_format_converter import OutputFormatConverterFactory
from genai_perf.inputs.synthetic_image_generator import (
    ImageFormat,
    SyntheticImageGenerator,
)
from genai_perf.inputs.synthetic_prompt_generator import SyntheticPromptGenerator
from genai_perf.utils import load_json_str
from PIL import Image
from requests import Response


class Inputs:
    """
    A class that generates the request input payloads for the target endpoint.
    """

    def __init__(self, config: InputsConfig):
        self.config = config
        if self.config.extra_inputs is None:
            self.config.extra_inputs = {}

        random.seed(self.config.random_seed)

    def create_inputs(self) -> Dict:
        """
        Given an input type, input format, and output type. Output a string of LLM Inputs
        (in a JSON dictionary) to a file.
        """
        generic_dataset_json = self._get_generic_dataset_json()
        json_in_pa_format = self._convert_generic_json_to_output_format(
            generic_dataset_json,
        )
        self._write_json_to_file(json_in_pa_format)
        return json_in_pa_format

    def _get_generic_dataset_json(self) -> Dict:
        """
        Retrieve and convert the dataset based on the input type.

        Returns
        -------
        Dict:
            The generic dataset JSON
        """

        if self.config.output_format == OutputFormat.OPENAI_EMBEDDINGS:
            if self.config.input_type != PromptSource.FILE:
                raise GenAIPerfException(
                    f"{OutputFormat.OPENAI_EMBEDDINGS.to_lowercase()} only supports a file as input."
                )
            input_file_dataset = self._get_input_dataset_from_embeddings_file()
            generic_dataset_json = (
                self._convert_input_synthetic_or_file_dataset_to_generic_json(
                    input_file_dataset
                )
            )
        elif self.config.output_format == OutputFormat.RANKINGS:
            if self.config.input_type != PromptSource.FILE:
                raise GenAIPerfException(
                    f"{OutputFormat.RANKINGS.to_lowercase()} only supports a directory as input."
                )
            queries_filename = self.config.input_filename / "queries.jsonl"
            passages_filename = self.config.input_filename / "passages.jsonl"
            input_file_dataset = self._get_input_dataset_from_rankings_files(
                queries_filename, passages_filename
            )
            generic_dataset_json = (
                self._convert_input_synthetic_or_file_dataset_to_generic_json(
                    input_file_dataset
                )
            )
        elif self.config.output_format == OutputFormat.IMAGE_RETRIEVAL:
            if self.config.input_type != PromptSource.FILE:
                raise GenAIPerfException(
                    f"{OutputFormat.IMAGE_RETRIEVAL.to_lowercase()} only supports a file as input."
                )
            input_file_dataset = self._get_input_dataset_from_file()
            input_file_dataset = self._encode_images_in_input_dataset(
                input_file_dataset
            )
            generic_dataset_json = (
                self._convert_input_synthetic_or_file_dataset_to_generic_json(
                    input_file_dataset
                )
            )
        else:
            if self.config.input_type == PromptSource.DATASET:
                # (TMA-1990) support VLM input from public dataset
                if self.config.output_format == OutputFormat.OPENAI_VISION:
                    raise GenAIPerfException(
                        f"{OutputFormat.OPENAI_VISION.to_lowercase()} currently "
                        "does not support dataset as input."
                    )
                dataset = self._get_input_dataset_from_url()
                generic_dataset_json = self._convert_input_url_dataset_to_generic_json(
                    dataset
                )
            elif self.config.input_type == PromptSource.SYNTHETIC:
                synthetic_dataset = self._get_input_dataset_from_synthetic()
                generic_dataset_json = (
                    self._convert_input_synthetic_or_file_dataset_to_generic_json(
                        synthetic_dataset
                    )
                )
            elif self.config.input_type == PromptSource.FILE:
                input_file_dataset = self._get_input_dataset_from_file()
                input_file_dataset = self._encode_images_in_input_dataset(
                    input_file_dataset
                )
                generic_dataset_json = (
                    self._convert_input_synthetic_or_file_dataset_to_generic_json(
                        input_file_dataset
                    )
                )
            else:
                raise GenAIPerfException("Input source is not recognized.")

        return generic_dataset_json

    def _get_input_dataset_from_embeddings_file(self) -> Dict[str, Any]:
        with open(self.config.input_filename, "r") as file:
            file_content = [load_json_str(line) for line in file]

        texts = [item["text"] for item in file_content]

        if self.config.batch_size > len(texts):
            raise ValueError(
                "Batch size cannot be larger than the number of available texts"
            )

        dataset_json: Dict[str, Any] = {}
        dataset_json["features"] = [{"name": "input"}]
        dataset_json["rows"] = []

        for _ in range(self.config.num_prompts):
            sampled_texts = random.sample(texts, self.config.batch_size)
            dataset_json["rows"].append({"row": {"payload": {"input": sampled_texts}}})

        return dataset_json

    def _get_input_dataset_from_rankings_files(
        self,
        queries_filename: Path,
        passages_filename: Path,
    ) -> Dict[str, Any]:

        with open(queries_filename, "r") as file:
            queries_content = [load_json_str(line) for line in file]
        queries_texts = [item for item in queries_content]

        with open(passages_filename, "r") as file:
            passages_content = [load_json_str(line) for line in file]
        passages_texts = [item for item in passages_content]

        if self.config.batch_size > len(passages_texts):
            raise ValueError(
                "Batch size cannot be larger than the number of available passages"
            )

        dataset_json: Dict[str, Any] = {}
        dataset_json["features"] = [{"name": "input"}]
        dataset_json["rows"] = []

        for _ in range(self.config.num_prompts):
            sampled_texts = random.sample(passages_texts, self.config.batch_size)
            query_sample = random.choice(queries_texts)
            entry_dict: Dict = {}
            entry_dict["query"] = query_sample
            entry_dict["passages"] = sampled_texts
            dataset_json["rows"].append({"row": {"payload": entry_dict}})
        return dataset_json

    def _check_for_valid_args(self) -> None:
        try:
            self._check_for_dataset_name_if_input_type_is_url()
            self._check_for_tokenzier_if_input_type_is_synthetic()
            self._check_for_valid_starting_index()
            self._check_for_valid_length()

        except Exception as e:
            raise GenAIPerfException(e)

    def _get_input_dataset_from_url(self) -> Response:
        url = self._resolve_url()
        configured_url = self._create_configured_url(url)
        dataset = self._download_dataset(configured_url)

        return dataset

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

    def _resolve_url(self) -> str:
        if self.config.dataset_name in dataset_url_map:
            return dataset_url_map[self.config.dataset_name]
        else:
            raise GenAIPerfException(
                f"{self.config.dataset_name} does not have a corresponding URL in the dataset_url_map."
            )

    def _create_configured_url(self, url: str) -> str:
        starting_index_str = str(self.config.starting_index)
        length_str = str(self.config.length)
        configured_url = url + f"&offset={starting_index_str}&length={length_str}"

        return configured_url

    def _download_dataset(self, configured_url: str) -> Response:
        dataset = self._query_server(configured_url)

        return dataset

    def _convert_input_url_dataset_to_generic_json(self, dataset: Response) -> Dict:
        dataset_json = dataset.json()
        try:
            self._check_for_error_in_json_of_dataset(dataset_json)
        except Exception as e:
            raise GenAIPerfException(e)

        generic_dataset_json = self._convert_dataset_to_generic_input_json(dataset_json)

        return generic_dataset_json

    def _convert_input_synthetic_or_file_dataset_to_generic_json(
        self, dataset: Dict
    ) -> Dict[str, List[Dict]]:
        generic_dataset_json = self._convert_dataset_to_generic_input_json(dataset)

        return generic_dataset_json

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

    def _get_input_dataset_from_file(self) -> Dict:
        """
        Returns
        -------
        Dict
            The dataset in the required format with the prompts and/or images
            read from the file.
        """
        self._verify_file()
        prompts, images = self._get_prompts_from_input_file()
        if self.config.batch_size > len(prompts):
            raise ValueError(
                "Batch size cannot be larger than the number of available texts"
            )
        dataset_json: Dict[str, Any] = {}
        dataset_json["features"] = [{"name": "text_input"}]
        dataset_json["rows"] = []

        if self.config.batch_size == DEFAULT_BATCH_SIZE:
            for prompt, image in zip(prompts, images):
                content = {}
                if prompt is not None:
                    content["text_input"] = prompt
                if image is not None:
                    content["image"] = image
                dataset_json["rows"].append({"row": content})
        else:
            for _ in range(self.config.num_prompts):
                content_array = []
                sampled_indices = random.sample(
                    range(len(prompts)), self.config.batch_size
                )
                sampled_texts_images = [
                    (prompts[i], images[i]) for i in sampled_indices
                ]

                for prompt, image in sampled_texts_images:
                    content = {}
                    if prompt is not None:
                        content["text_input"] = prompt
                    if image is not None:
                        content["image"] = image
                    content_array.append(content)
                dataset_json["rows"].append({"row": content_array})

        return dataset_json

    def _get_prompts_from_input_file(self) -> Tuple[List[str], List[str]]:
        """
        Reads the input prompts from a JSONL file and returns a list of prompts.

        Returns
        -------
        Tuple[List[str], List[str]]
            A list of prompts and images read from the file.
        """
        prompts = []
        images = []
        with open(self.config.input_filename, mode="r", newline=None) as file:
            for line in file:
                if line.strip():
                    # None if not provided
                    prompt = load_json_str(line).get("text_input")
                    image = load_json_str(line).get("image")
                    prompts.append(prompt.strip() if prompt else prompt)
                    images.append(image.strip() if image else image)
        return prompts, images

    def _verify_file(self) -> None:
        if not self.config.input_filename.exists():
            raise FileNotFoundError(
                f"The file '{self.config.input_filename}' does not exist."
            )

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

    def _convert_generic_json_to_output_format(self, generic_dataset) -> Dict:
        converter = OutputFormatConverterFactory.create(self.config.output_format)
        return converter.convert(generic_dataset, self.config)

    def _write_json_to_file(self, json_in_pa_format: Dict) -> None:
        filename = self.config.output_dir / DEFAULT_INPUT_DATA_JSON
        with open(str(filename), "w") as f:
            f.write(json.dumps(json_in_pa_format, indent=2))

    def _check_for_dataset_name_if_input_type_is_url(self) -> None:
        if (
            self.config.input_type == PromptSource.DATASET
            and not self.config.dataset_name
        ):
            raise GenAIPerfException(
                "Input type is dataset, but dataset_name is not specified."
            )

    def _check_for_tokenzier_if_input_type_is_synthetic(self) -> None:
        if (
            self.config.input_type == PromptSource.SYNTHETIC
            and not self.config.tokenizer
        ):
            raise GenAIPerfException(
                "Input type is SYNTHETIC, but a tokenizer was not specified."
            )

    def _check_for_valid_starting_index(self) -> None:
        if not isinstance(self.config.starting_index, int):
            raise GenAIPerfException(
                f"starting_index: {self.config.starting_index} must be an integer."
            )

        if self.config.starting_index < MINIMUM_STARTING_INDEX:
            raise GenAIPerfException(
                f"starting_index: {self.config.starting_index} must be larger than {MINIMUM_STARTING_INDEX}."
            )

    def _check_for_valid_length(self) -> None:
        if not isinstance(self.config.length, int):
            raise GenAIPerfException(
                f"length: {self.config.length} must be an integer."
            )

        if self.config.length < MINIMUM_LENGTH:
            raise GenAIPerfException(
                f"starting_index: {self.config.length} must be larger than {MINIMUM_LENGTH}."
            )

    def _query_server(self, configured_url: str) -> Response:
        try:
            response = requests.get(configured_url)
        except Exception as e:
            error_message = self._create_error_message(e)
            raise GenAIPerfException(error_message)

        return response

    def _create_error_message(self, exception: Exception) -> str:
        url_str = exception.args[0].args[0]
        url_start = url_str.find("'")
        url_end = url_str.find("'", url_start + 1) + 1
        error_message = f"Invalid URL: {url_str[url_start:url_end]}"

        return error_message

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
