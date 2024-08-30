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
