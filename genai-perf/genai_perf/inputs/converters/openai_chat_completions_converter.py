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
from typing import Any, Dict, List, Optional, Tuple

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.input_constants import DEFAULT_OUTPUT_TOKENS_MEAN
from genai_perf.inputs.inputs_config import InputsConfig


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

    def _populate_openai_chat_completions_output_json(
        self,
        generic_dataset: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        config: InputsConfig,
    ) -> Dict:
        pa_json: Dict[str, Any] = {"data": []}

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
