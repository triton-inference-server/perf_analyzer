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
from copy import deepcopy
from typing import Dict, List

from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.input_constants import (
    DEFAULT_OUTPUT_TOKENS_MEAN,
    DEFAULT_TENSORRTLLM_MAX_TOKENS,
    EMPTY_JSON_IN_TENSORRTLLM_PA_FORMAT,
)
from genai_perf.inputs.inputs_config import InputsConfig


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
