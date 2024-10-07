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
from typing import Dict, List, Optional

from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.input_constants import (
    DEFAULT_OUTPUT_TOKENS_MEAN,
    EMPTY_JSON_IN_OPENAI_PA_FORMAT,
)
from genai_perf.inputs.inputs_config import InputsConfig


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
        output_tokens = []
        if config.schedule_file is not None:
            with open(config.schedule_file, "r") as f:
                for j, line in enumerate(f):
                    if j == config.num_prompts:
                        break
                    output_tokens.append(json.loads(line)["output_length"])
        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = self._select_model_name(config, index)
            pa_json["data"].append({"payload": []})
            pa_json["data"][index]["payload"].append({"prompt": ""})

            for header, content in entry.items():
                new_prompt = self._create_new_text_input(
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
                output_tokens[index] if output_tokens else None,
            )

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
        config,
        model_name: str = "",
        output_tokens: Optional[int] = None,
    ) -> None:
        payload = openai_json["payload"][0]
        payload["model"] = model_name
        if config.add_stream:
            payload["stream"] = True
        if config.output_tokens_mean != DEFAULT_OUTPUT_TOKENS_MEAN:
            payload["max_tokens"] = int(
                random.gauss(config.output_tokens_mean, config.output_tokens_stddev)
            )
        payload["max_tokens"] = output_tokens
        for key, value in config.extra_inputs.items():
            payload[key] = value
