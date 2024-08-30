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

from typing import Any, Dict

from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.inputs_config import InputsConfig


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
