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
from genai_perf.inputs.config import InputsConfig
from genai_perf.inputs.input_constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_DATA_JSON,
    DEFAULT_OUTPUT_TOKENS_MEAN,
    DEFAULT_TENSORRTLLM_MAX_TOKENS,
    EMPTY_JSON_IN_OPENAI_PA_FORMAT,
    EMPTY_JSON_IN_TENSORRTLLM_PA_FORMAT,
    EMPTY_JSON_IN_VLLM_PA_FORMAT,
    MINIMUM_LENGTH,
    MINIMUM_STARTING_INDEX,
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
    dataset_url_map,
)
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
        if (
            self.config.output_format == OutputFormat.OPENAI_CHAT_COMPLETIONS
            or self.config.output_format == OutputFormat.OPENAI_VISION
            or self.config.output_format == OutputFormat.IMAGE_RETRIEVAL
        ):
            output_json = self._convert_generic_json_to_openai_chat_completions_format(
                generic_dataset,
            )
        elif self.config.output_format == OutputFormat.OPENAI_COMPLETIONS:
            output_json = self._convert_generic_json_to_openai_completions_format(
                generic_dataset,
            )
        elif self.config.output_format == OutputFormat.OPENAI_EMBEDDINGS:
            output_json = self._convert_generic_json_to_openai_embeddings_format(
                generic_dataset
            )
        elif self.config.output_format == OutputFormat.RANKINGS:
            output_json = self._convert_generic_json_to_rankings_format(generic_dataset)
        elif self.config.output_format == OutputFormat.VLLM:
            output_json = self._convert_generic_json_to_vllm_format(generic_dataset)
        elif self.config.output_format == OutputFormat.TENSORRTLLM:
            output_json = self._convert_generic_json_to_trtllm_format(generic_dataset)
        elif self.config.output_format == OutputFormat.TENSORRTLLM_ENGINE:
            output_json = self._convert_generic_json_to_trtllm_engine_format(
                generic_dataset
            )
        else:
            raise GenAIPerfException(
                f"Output format {self.config.output_format} is not currently supported"
            )

        return output_json

    def _convert_generic_json_to_openai_chat_completions_format(
        self,
        dataset_json: Dict,
    ) -> Dict:
        # TODO (TMA-1757): Implement a way to select a role for `text_input`
        (
            system_role_headers,
            user_role_headers,
            _,
        ) = self._determine_json_feature_roles(dataset_json)
        pa_json = self._populate_openai_chat_completions_output_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
        )

        return pa_json

    def _convert_generic_json_to_openai_completions_format(
        self,
        dataset_json: Dict,
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            text_input_headers,
        ) = self._determine_json_feature_roles(dataset_json)
        pa_json = self._populate_openai_completions_output_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
            text_input_headers,
        )

        return pa_json

    def _convert_generic_json_to_openai_embeddings_format(
        self,
        generic_dataset: Dict,
    ) -> Dict[str, Any]:
        pa_json: Dict[str, Any] = {"data": []}

        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = self._select_model_name(index)
            payload = entry.get("payload", {})
            input_values = payload.get("input")

            if input_values is None:
                raise ValueError("Missing required fields 'input' in dataset entry")
            if not isinstance(input_values, list):
                raise ValueError(
                    f"Required field 'input' must be a list (actual: {type(input_values)})"
                )

            payload = {
                "input": input_values,
                "model": iter_model_name,
            }

            for key, value in self.config.extra_inputs.items():
                payload[key] = value

            pa_json["data"].append({"payload": [payload]})

        return pa_json

    def _contains_rankings_tei(self) -> bool:
        """
        Check if user specified that they are using the Hugging Face
        Text Embeddings Interface for ranking models
        """
        if (
            self.config.extra_inputs
            and self.config.extra_inputs.get("rankings") == "tei"
        ):
            return True
        return False

    def _convert_generic_json_to_rankings_format(
        self,
        generic_dataset: Dict,
    ) -> Dict[str, Any]:
        pa_json: Dict[str, Any] = {"data": []}
        use_tei_format = self._contains_rankings_tei()

        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = self._select_model_name(index)

            payload = entry.get("payload", {})
            query_values = payload.get("query")

            if use_tei_format:
                passage_values = payload.get("passages", [])
                passage_values = [item.get("text", "") for item in passage_values]
            else:
                passage_values = payload.get("passages")

            if query_values is None:
                raise ValueError("Missing required fields 'query' in dataset entry")
            if passage_values is None:
                raise ValueError(
                    f"Missing required fields '{'texts' if use_tei_format else 'passages'}' in dataset entry"
                )
            if not isinstance(passage_values, list):
                raise ValueError(
                    f"Required field '{'texts' if use_tei_format else 'passages'}' must be a list (actual: {type(passage_values)})"
                )

            if use_tei_format:
                payload = {"query": query_values["text"], "texts": passage_values}
            else:
                payload = {
                    "query": query_values,
                    "passages": passage_values,
                    "model": iter_model_name,
                }

            for key, value in self.config.extra_inputs.items():
                if not (key == "rankings" and value == "tei"):
                    payload[key] = value

            pa_json["data"].append({"payload": [payload]})

        return pa_json

    def _convert_generic_json_to_vllm_format(
        self,
        dataset_json: Dict,
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            text_input_headers,
        ) = self._determine_json_feature_roles(dataset_json)

        pa_json = self._populate_vllm_output_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
            text_input_headers,
        )

        return pa_json

    def _convert_generic_json_to_trtllm_format(
        self,
        dataset_json: Dict,
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            text_input_headers,
        ) = self._determine_json_feature_roles(dataset_json)

        pa_json = self._populate_trtllm_output_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
            text_input_headers,
        )

        return pa_json

    def _convert_generic_json_to_trtllm_engine_format(
        self,
        dataset_json: Dict,
    ) -> Dict:
        pa_json = self._populate_trtllm_engine_output_json(
            dataset_json,
        )
        return pa_json

    def _write_json_to_file(self, json_in_pa_format: Dict) -> None:
        filename = self.config.output_dir / DEFAULT_INPUT_DATA_JSON
        with open(str(filename), "w") as f:
            f.write(json.dumps(json_in_pa_format, indent=2))

    def _determine_json_feature_roles(
        self, dataset_json: Dict
    ) -> Tuple[List[str], List[str], List[str]]:
        SYSTEM_ROLE_LIST = ["system_prompt"]
        USER_ROLE_LIST = ["question", "article"]
        TEXT_INPUT_LIST = ["text_input"]

        system_role_headers: List[str] = []
        user_role_headers: List[str] = []
        text_input_headers: List[str] = []

        if "features" in dataset_json.keys():
            for feature in dataset_json["features"]:
                if feature in SYSTEM_ROLE_LIST:
                    system_role_headers.append(feature)
                if feature in USER_ROLE_LIST:
                    user_role_headers.append(feature)
                if feature in TEXT_INPUT_LIST:
                    user_role_headers.append(feature)

        assert (
            system_role_headers is not None
            or user_role_headers is not None
            or text_input_headers is not None
        )

        return system_role_headers, user_role_headers, text_input_headers

    def _select_model_name(self, index):
        if self.config.model_selection_strategy == ModelSelectionStrategy.ROUND_ROBIN:
            return self.config.model_name[index % len(self.config.model_name)]
        elif self.config.model_selection_strategy == ModelSelectionStrategy.RANDOM:
            return random.choice(self.config.model_name)
        else:
            raise GenAIPerfException(
                f"Model selection strategy '{self.config.model_selection_strategy}' is unsupported"
            )

    def _populate_openai_chat_completions_output_json(
        self,
        dataset_json: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
    ) -> Dict:
        pa_json = self._create_empty_openai_pa_json()

        for index, entry in enumerate(dataset_json["rows"]):
            iter_model_name = self._select_model_name(index)
            openai_json: Dict = {"payload": [{"messages": []}]}

            # Check if the entry is a list (batched entries) or a single entry
            if isinstance(entry, list):
                for item in entry:
                    self._process_row_content(
                        item, system_role_headers, user_role_headers, openai_json
                    )
            elif isinstance(entry, dict):
                self._process_row_content(
                    entry, system_role_headers, user_role_headers, openai_json
                )
            else:
                raise GenAIPerfException(f"Unexpected data type in rows: {type(entry)}")

            self._add_optional_tags_to_openai_json(
                openai_json,
                iter_model_name,
            )
            pa_json["data"].append(openai_json)

        return pa_json

    def _process_row_content(
        self,
        entry: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        openai_json: Dict,
    ) -> None:
        if "image" in entry:
            contents = self._extract_chat_contents(entry)
            if openai_json["payload"][0]["messages"]:
                openai_json["payload"][0]["messages"][0]["content"].extend(contents)
            else:
                openai_json["payload"][0]["messages"].append(
                    {"role": "user", "content": contents}
                )
        else:
            for header, content in entry.items():
                message = self._create_new_openai_chat_completions_message(
                    header, system_role_headers, user_role_headers, content
                )
                self._add_message_to_json(openai_json, message)

    def _extract_chat_contents(self, entry: Dict) -> List[Dict]:
        contents: List = []
        if isinstance(entry, list):
            for item in entry:
                for content_type, content in item.items():
                    self._add_content(contents, content_type, content)
        else:
            for content_type, content in entry.items():
                self._add_content(contents, content_type, content)
        return contents

    def _add_content(self, contents: List[Dict], content_type: str, content: str):
        if content_type == "text_input":
            contents.append({"type": "text", "text": content})
        elif content_type == "image":
            contents.append({"type": "image_url", "image_url": {"url": content}})
        else:
            raise GenAIPerfException(
                "Failed to construct OpenAI chat completions message "
                f"contents. Unknown content type: '{content_type}'."
            )

    def _populate_openai_completions_output_json(
        self,
        dataset_json: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
    ) -> Dict:
        pa_json = self._create_empty_openai_pa_json()

        for index, entry in enumerate(dataset_json["rows"]):
            iter_model_name = self._select_model_name(index)
            pa_json["data"].append({"payload": []})
            pa_json["data"][index]["payload"].append({"prompt": ""})

            for header, content in entry.items():
                new_prompt = self._create_new_prompt(
                    header,
                    system_role_headers,
                    user_role_headers,
                    text_input_headers,
                    content,
                )

                pa_json = self._add_new_prompt_to_json(pa_json, index, new_prompt)

            self._add_optional_tags_to_openai_json(
                pa_json["data"][index],
                iter_model_name,
            )

        return pa_json

    def _populate_vllm_output_json(
        self,
        dataset_json: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
    ) -> Dict:
        pa_json = self._create_empty_vllm_pa_json()

        for index, entry in enumerate(dataset_json["rows"]):
            iter_model_name = self._select_model_name(index)
            pa_json["data"].append({"text_input": [""]})

            for header, content in entry.items():
                new_text_input = self._create_new_text_input(
                    header,
                    system_role_headers,
                    user_role_headers,
                    text_input_headers,
                    content,
                )

                pa_json = self._add_new_text_input_to_json(
                    pa_json, index, new_text_input
                )

            pa_json = self._add_optional_tags_to_vllm_json(
                pa_json,
                index,
                iter_model_name,
            )

        return pa_json

    def _populate_trtllm_output_json(
        self,
        dataset_json: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
    ) -> Dict:
        pa_json = self._create_empty_trtllm_pa_json()
        default_max_tokens = (
            "max_tokens" not in self.config.extra_inputs
            or self.config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN
        )

        for index, entry in enumerate(dataset_json["rows"]):
            iter_model_name = self._select_model_name(index)
            pa_json["data"].append({"text_input": [""]})

            for header, content in entry.items():
                new_text_input = self._create_new_text_input(
                    header,
                    system_role_headers,
                    user_role_headers,
                    text_input_headers,
                    content,
                )

                pa_json = self._add_new_text_input_to_json(
                    pa_json, index, new_text_input
                )

            pa_json = self._add_required_tags_to_trtllm_json(
                pa_json, index, default_max_tokens
            )
            pa_json = self._add_optional_tags_to_trtllm_json(
                pa_json,
                index,
                iter_model_name,
            )

        return pa_json

    def _populate_trtllm_engine_output_json(
        self,
        dataset_json: Dict,
    ) -> Dict:
        pa_json = self._create_empty_trtllm_pa_json()

        for index, entry in enumerate(dataset_json["rows"]):
            token_ids = self.config.tokenizer.encode(entry["text_input"])
            pa_json["data"].append(
                {
                    "input_ids": {
                        "content": token_ids,
                        "shape": [len(token_ids)],
                    },
                    "input_lengths": [len(token_ids)],
                    "request_output_len": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                }
            )

            pa_json = self._add_optional_tags_to_trtllm_engine_json(
                pa_json,
                index,
            )
        return pa_json

    def _create_empty_openai_pa_json(self) -> Dict:
        empty_pa_json = deepcopy(EMPTY_JSON_IN_OPENAI_PA_FORMAT)

        return empty_pa_json

    def _create_empty_vllm_pa_json(self) -> Dict:
        empty_pa_json = deepcopy(EMPTY_JSON_IN_VLLM_PA_FORMAT)

        return empty_pa_json

    def _create_empty_trtllm_pa_json(self) -> Dict:
        empty_pa_json = deepcopy(EMPTY_JSON_IN_TENSORRTLLM_PA_FORMAT)

        return empty_pa_json

    def _create_new_openai_chat_completions_message(
        self,
        header: str,
        system_role_headers: List[str],
        user_role_headers: List[str],
        content: str,
    ) -> Optional[Dict]:
        # Do not add messages with blank content
        if not content:
            return {}

        if header in system_role_headers:
            message = {
                "role": "system",
                "content": content,
            }
        elif header in user_role_headers:
            message = {
                "role": "user",
                "content": content,
            }
        else:
            message = {}

        return message

    def _create_new_prompt(
        self,
        header: str,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        content: str,
    ) -> str:
        new_prompt = ""

        if (
            header in system_role_headers
            or header in user_role_headers
            or header in text_input_headers
        ):
            new_prompt = content

        return new_prompt

    def _create_new_text_input(
        self,
        header: str,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        content: str,
    ) -> str:
        new_text_input = ""

        if (
            header in system_role_headers
            or header in user_role_headers
            or header in text_input_headers
        ):
            new_text_input = content

        return new_text_input

    def _add_message_to_json(self, openai_json: Dict, message: Optional[Dict]) -> Dict:
        if message:
            openai_json["payload"][0]["messages"].append(message)

        return openai_json

    def _add_new_text_input_to_json(
        self, pa_json: Dict, index: int, new_text_input: str
    ) -> Dict:
        if new_text_input:
            if pa_json["data"][index]["text_input"][0]:
                pa_json["data"][index]["text_input"][0] = (
                    pa_json["data"][index]["text_input"][0] + f" {new_text_input}"
                )
            else:
                pa_json["data"][index]["text_input"][0] = new_text_input

        return pa_json

    def _add_new_prompt_to_json(
        self,
        pa_json: Dict,
        index: int,
        new_prompt: str,
    ) -> Dict:
        if new_prompt:
            if pa_json["data"][index]["payload"][0]["prompt"]:
                pa_json["data"][index]["payload"][0]["prompt"] += f" {new_prompt}"
            else:
                pa_json["data"][index]["payload"][0]["prompt"] = new_prompt

        return pa_json

    def _add_optional_tags_to_openai_json(
        self,
        openai_json: Dict,
        model_name: str = "",
    ) -> None:
        payload = openai_json["payload"][0]
        payload["model"] = model_name
        if self.config.add_stream:
            payload["stream"] = True
        if self.config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            payload["max_tokens"] = int(
                random.gauss(
                    self.config.output_tokens_mean, self.config.output_tokens_stddev
                )
            )
        for key, value in self.config.extra_inputs.items():
            payload[key] = value

    def _add_optional_tags_to_vllm_json(
        self,
        pa_json: Dict,
        index: int,
        model_name: str = "",
    ) -> Dict:
        row = pa_json["data"][index]
        row["model"] = model_name
        if self.config.add_stream:
            row["stream"] = [True]
        if self.config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            number_of_tokens = str(
                int(
                    max(
                        0,
                        random.gauss(
                            self.config.output_tokens_mean,
                            self.config.output_tokens_stddev,
                        ),
                    )
                )
            )
            sampling_parameters = {
                "max_tokens": number_of_tokens,
            }
            if self.config.output_tokens_deterministic:
                sampling_parameters["min_tokens"] = number_of_tokens
            sampling_parameters_str = json.dumps(sampling_parameters)
            row["sampling_parameters"] = [sampling_parameters_str]
        for key, value in self.config.extra_inputs.items():
            row[key] = [value]
        if "exclude_input_in_output" not in row:
            row["exclude_input_in_output"] = [True]

        return pa_json

    def _add_optional_tags_to_trtllm_json(
        self,
        pa_json: Dict,
        index: int,
        model_name: str = "",
    ) -> Dict:
        row = pa_json["data"][index]
        row["model"] = model_name
        if self.config.add_stream:
            row["stream"] = [True]
        if self.config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            number_of_tokens = int(
                random.gauss(
                    self.config.output_tokens_mean, self.config.output_tokens_stddev
                )
            )
            if self.config.output_tokens_deterministic:
                row["min_length"] = [number_of_tokens]
            row["max_tokens"] = [number_of_tokens]
        for key, value in self.config.extra_inputs.items():
            row[key] = [value]

        return pa_json

    def _add_optional_tags_to_trtllm_engine_json(
        self, pa_json: Dict, index: int
    ) -> Dict:
        row = pa_json["data"][index]
        if self.config.add_stream:
            row["streaming"] = [True]
        if self.config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            num_tokens = int(
                random.gauss(
                    self.config.output_tokens_mean, self.config.output_tokens_stddev
                )
            )
            row["request_output_len"] = [num_tokens]
            if self.config.output_tokens_deterministic:
                row["min_length"] = [num_tokens]

        for key, value in self.config.extra_inputs.items():
            row[key] = [value]

        return pa_json

    def _add_required_tags_to_trtllm_json(
        self,
        pa_json: Dict,
        index: int,
        default_max_tokens: bool,
    ) -> Dict:
        row = pa_json["data"][index]
        if default_max_tokens:
            row["max_tokens"] = [DEFAULT_TENSORRTLLM_MAX_TOKENS]

        return pa_json

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
