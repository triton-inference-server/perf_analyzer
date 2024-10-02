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

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from genai_perf import utils
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import (
    DEFAULT_BATCH_SIZE,
    OutputFormat,
    PromptSource,
    dataset_url_map,
)
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.synthetic_image_generator import (
    ImageFormat,
    SyntheticImageGenerator,
)
from genai_perf.inputs.synthetic_prompt_generator import SyntheticPromptGenerator
from genai_perf.utils import load_json_str
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

    def _convert_input_synthetic_or_file_dataset_to_generic_json(
        self, dataset: Dict
    ) -> Dict[str, List[Dict]]:
        generic_dataset_json = self._convert_dataset_to_generic_input_json(dataset)

        return generic_dataset_json

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

    def _get_input_dataset_from_url(self) -> Response:
        url = self._resolve_url()
        configured_url = self._create_configured_url(url)
        dataset = self._download_dataset(configured_url)

        return dataset

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
        prompt_desc = []
        if self.config.schedule_file is not None:
            with open(self.config.schedule_file, "r") as f:
                for j, line in enumerate(f):
                    if j == self.config.num_prompts:
                        break
                    prompt_desc.append(json.loads(line))
            assert (
                j == self.config.num_prompts
            ), "not enough prompts in the schedule-file."
        for i in range(self.config.num_prompts):
            row: Dict["str", Any] = {"row": {}}
            if prompt_desc:
                synthetic_prompt = self._create_synthetic_prompt(
                    self.config.tokenizer,
                    prompt_desc[i]["input_length"],
                    0,
                    prompt_hash_list=prompt_desc[i].get("hash_ids", None),
                    block_size=self.config.block_size,
                )
            else:
                synthetic_prompt = self._create_synthetic_prompt(
                    self.config.tokenizer,
                    self.config.prompt_tokens_mean,
                    self.config.prompt_tokens_stddev,
                )
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

    def _check_for_error_in_json_of_dataset(self, dataset_json: Dict) -> None:
        if "error" in dataset_json:
            raise GenAIPerfException(dataset_json["error"])

    def _create_synthetic_prompt(
        self,
        tokenizer,
        prompt_tokens_mean,
        prompt_tokens_stddev,
        prompt_hash_list=None,
        block_size=None,
    ) -> str:
        return SyntheticPromptGenerator.create_synthetic_prompt(
            tokenizer,
            prompt_tokens_mean,
            prompt_tokens_stddev,
            prompt_hash_list,
            block_size,
        )

    def _create_synthetic_image(self) -> str:
        return SyntheticImageGenerator.create_synthetic_image(
            image_width_mean=self.config.image_width_mean,
            image_width_stddev=self.config.image_width_stddev,
            image_height_mean=self.config.image_height_mean,
            image_height_stddev=self.config.image_height_stddev,
            image_format=self.config.image_format,
        )

    def _verify_file(self) -> None:
        if not self.config.input_filename.exists():
            raise FileNotFoundError(
                f"The file '{self.config.input_filename}' does not exist."
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
