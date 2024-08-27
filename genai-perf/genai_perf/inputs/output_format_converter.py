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
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import (
    DEFAULT_OUTPUT_TOKENS_MEAN,
    DEFAULT_TENSORRTLLM_MAX_TOKENS,
    EMPTY_JSON_IN_OPENAI_PA_FORMAT,
    EMPTY_JSON_IN_TENSORRTLLM_PA_FORMAT,
    EMPTY_JSON_IN_VLLM_PA_FORMAT,
    ModelSelectionStrategy,
    OutputFormat,
)
from genai_perf.inputs.inputs_config import InputsConfig


class OutputFormatConverterFactory:
    """
    This class converts the generic JSON to the specific format
    used by a given endpoint.
    """

    @staticmethod
    def create(output_format: OutputFormat):
        converters = {
            OutputFormat.OPENAI_CHAT_COMPLETIONS: OpenAIChatCompletionsConverter,
            OutputFormat.OPENAI_COMPLETIONS: OpenAICompletionsConverter,
            OutputFormat.OPENAI_EMBEDDINGS: OpenAIEmbeddingsConverter,
            OutputFormat.IMAGE_RETRIEVAL: OpenAIChatCompletionsConverter,
            OutputFormat.OPENAI_VISION: OpenAIChatCompletionsConverter,
            OutputFormat.RANKINGS: RankingsConverter,
            OutputFormat.VLLM: VLLMConverter,
            OutputFormat.TENSORRTLLM: TensorRTLLMConverter,
            OutputFormat.TENSORRTLLM_ENGINE: TensorRTLLMEngineConverter,
        }
        if output_format not in converters:
            raise GenAIPerfException(f"Output format {output_format} is not supported")
        return converters[output_format]()


class BaseConverter:
    def convert(
        self,
        generic_dataset: Dict,
        config: InputsConfig,
    ) -> Dict:
        raise NotImplementedError

    def _select_model_name(
        self,
        config: InputsConfig,
        index: int,
    ) -> str:
        if config.model_selection_strategy == ModelSelectionStrategy.ROUND_ROBIN:
            return config.model_name[index % len(config.model_name)]
        elif config.model_selection_strategy == ModelSelectionStrategy.RANDOM:
            return random.choice(config.model_name)
        else:
            raise GenAIPerfException(
                f"Model selection strategy '{config.model_selection_strategy}' is unsupported"
            )

    def _determine_json_feature_roles(
        self, generic_dataset: Dict
    ) -> Tuple[List[str], List[str], List[str]]:
        SYSTEM_ROLE_LIST = ["system_prompt"]
        USER_ROLE_LIST = ["question", "article"]
        TEXT_INPUT_LIST = ["text_input"]

        system_role_headers: List[str] = []
        user_role_headers: List[str] = []
        text_input_headers: List[str] = []

        if "features" in generic_dataset.keys():
            for feature in generic_dataset["features"]:
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


class OpenAIChatCompletionsConverter(BaseConverter):
    def convert(
        self,
        generic_dataset: Dict,
        config: InputsConfig,
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            _,
        ) = self._determine_json_feature_roles(generic_dataset)
        pa_json = self._populate_openai_chat_completions_output_json(
            generic_dataset,
            system_role_headers,
            user_role_headers,
            config,
        )

        return pa_json

    def _populate_openai_chat_completions_output_json(
        self,
        generic_dataset: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        config: InputsConfig,
    ) -> Dict:
        pa_json = deepcopy(EMPTY_JSON_IN_OPENAI_PA_FORMAT)

        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = self._select_model_name(config, index)
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
                config,
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

    def _add_optional_tags_to_openai_json(
        self,
        openai_json: Dict,
        config: InputsConfig,
        model_name: str = "",
    ) -> None:
        payload = openai_json["payload"][0]
        payload["model"] = model_name
        if config.add_stream:
            payload["stream"] = True
        if config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            payload["max_tokens"] = int(
                random.gauss(config.output_tokens_mean, config.output_tokens_stddev)
            )
        for key, value in config.extra_inputs.items():
            payload[key] = value


class OpenAICompletionsConverter(BaseConverter):
    def convert(
        self,
        generic_dataset: Dict,
        config: InputsConfig,
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            text_input_headers,
        ) = self._determine_json_feature_roles(generic_dataset)

        pa_json = self._populate_openai_completions_output_json(
            generic_dataset,
            system_role_headers,
            user_role_headers,
            text_input_headers,
            config,
        )

        return pa_json

    def _populate_openai_completions_output_json(
        self,
        generic_dataset: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        config: InputsConfig,
    ) -> Dict:
        pa_json = deepcopy(EMPTY_JSON_IN_OPENAI_PA_FORMAT)

        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = self._select_model_name(config, index)
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
                config,
                iter_model_name,
            )

        return pa_json

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
        config,
        model_name: str = "",
    ) -> None:
        payload = openai_json["payload"][0]
        payload["model"] = model_name
        if config.add_stream:
            payload["stream"] = True
        if config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            payload["max_tokens"] = int(
                random.gauss(config.output_tokens_mean, config.output_tokens_stddev)
            )
        for key, value in config.extra_inputs.items():
            payload[key] = value


class OpenAIEmbeddingsConverter(BaseConverter):
    def convert(
        self,
        generic_dataset: Dict,
        config: InputsConfig,
    ) -> Dict:
        pa_json: Dict[str, Any] = {"data": []}

        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = self._select_model_name(config, index)
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

            for key, value in config.extra_inputs.items():
                payload[key] = value

            pa_json["data"].append({"payload": [payload]})

        return pa_json


class RankingsConverter(BaseConverter):
    def convert(
        self,
        generic_dataset: Dict,
        config: InputsConfig,
    ) -> Dict:
        pa_json: Dict[str, Any] = {"data": []}
        use_tei_format = self._contains_rankings_tei(config)

        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = self._select_model_name(config, index)

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

            for key, value in config.extra_inputs.items():
                if not (key == "rankings" and value == "tei"):
                    payload[key] = value

            pa_json["data"].append({"payload": [payload]})

        return pa_json

    def _contains_rankings_tei(self, config: InputsConfig) -> bool:
        """
        Check if user specified that they are using the Hugging Face
        Text Embeddings Interface for ranking models
        """
        if config.extra_inputs and config.extra_inputs.get("rankings") == "tei":
            return True
        return False


class VLLMConverter(BaseConverter):
    def convert(
        self,
        generic_dataset: Dict,
        config: InputsConfig,
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            text_input_headers,
        ) = self._determine_json_feature_roles(generic_dataset)

        pa_json = self._populate_vllm_output_json(
            generic_dataset,
            system_role_headers,
            user_role_headers,
            text_input_headers,
            config,
        )

        return pa_json

    def _populate_vllm_output_json(
        self,
        generic_dataset: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        config: InputsConfig,
    ) -> Dict:
        pa_json = deepcopy(EMPTY_JSON_IN_VLLM_PA_FORMAT)

        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = self._select_model_name(config, index)
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
                config,
                iter_model_name,
            )

        return pa_json

    def _add_optional_tags_to_vllm_json(
        self,
        pa_json: Dict,
        index: int,
        config: InputsConfig,
        model_name: str = "",
    ) -> Dict:
        row = pa_json["data"][index]
        row["model"] = model_name
        if config.add_stream:
            row["stream"] = [True]
        if config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            number_of_tokens = str(
                int(
                    max(
                        0,
                        random.gauss(
                            config.output_tokens_mean,
                            config.output_tokens_stddev,
                        ),
                    )
                )
            )
            sampling_parameters = {
                "max_tokens": number_of_tokens,
            }
            if config.output_tokens_deterministic:
                sampling_parameters["min_tokens"] = number_of_tokens
            sampling_parameters_str = json.dumps(sampling_parameters)
            row["sampling_parameters"] = [sampling_parameters_str]
        for key, value in config.extra_inputs.items():
            row[key] = [value]
        if "exclude_input_in_output" not in row:
            row["exclude_input_in_output"] = [True]

        return pa_json


class TensorRTLLMConverter(BaseConverter):
    def convert(
        self,
        generic_dataset: Dict,
        config: InputsConfig,
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            text_input_headers,
        ) = self._determine_json_feature_roles(generic_dataset)

        pa_json = self._populate_trtllm_output_json(
            generic_dataset,
            system_role_headers,
            user_role_headers,
            text_input_headers,
            config,
        )

        return pa_json

    def _populate_trtllm_output_json(
        self,
        generic_dataset: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        config,
    ) -> Dict:
        pa_json = deepcopy(EMPTY_JSON_IN_TENSORRTLLM_PA_FORMAT)
        default_max_tokens = (
            "max_tokens" not in config.extra_inputs
            or config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN
        )

        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = self._select_model_name(config, index)
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

    def _add_optional_tags_to_trtllm_json(
        self,
        pa_json: Dict,
        index: int,
        config,
        model_name: str = "",
    ) -> Dict:
        row = pa_json["data"][index]
        row["model"] = model_name
        if config.add_stream:
            row["stream"] = [True]
        if config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            number_of_tokens = int(
                random.gauss(config.output_tokens_mean, config.output_tokens_stddev)
            )
            if config.output_tokens_deterministic:
                row["min_length"] = [number_of_tokens]
            row["max_tokens"] = [number_of_tokens]
        for key, value in config.extra_inputs.items():
            row[key] = [value]

        return pa_json


class TensorRTLLMEngineConverter(BaseConverter):
    def convert(
        self,
        generic_dataset: Dict,
        config: InputsConfig,
    ) -> Dict:
        pa_json = deepcopy(EMPTY_JSON_IN_TENSORRTLLM_PA_FORMAT)

        for index, entry in enumerate(generic_dataset["rows"]):
            token_ids = config.tokenizer.encode(entry["text_input"])
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
                pa_json, index, config
            )
        return pa_json

    def _add_optional_tags_to_trtllm_engine_json(
        self,
        pa_json: Dict,
        index: int,
        config,
    ) -> Dict:
        row = pa_json["data"][index]
        if config.add_stream:
            row["streaming"] = [True]
        if config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            num_tokens = int(
                random.gauss(config.output_tokens_mean, config.output_tokens_stddev)
            )
            row["request_output_len"] = [num_tokens]
            if config.output_tokens_deterministic:
                row["min_length"] = [num_tokens]

        for key, value in config.extra_inputs.items():
            row[key] = [value]

        return pa_json
