# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from genai_perf.config.input.config_defaults import OutputTokenDefaults
from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.retrievers.generic_dataset import DataRow, GenericDataset


class HuggingFaceGenerateConverter(BaseConverter):

    def convert(
        self,
        generic_dataset: GenericDataset,
    ) -> Dict[Any, Any]:
        request_body: Dict[str, Any] = {"data": []}

        for file_data in generic_dataset.files_data.values():
            for index, row in enumerate(file_data.rows):
                model_name = self._select_model_name(index)
                prompt = self._create_prompt(row)

                payload = {
                    "model": model_name,
                    "inputs": prompt,
                }
                request_body["data"].append(self._finalize_payload(payload, row))

        return request_body

    def _create_prompt(self, row: DataRow) -> str:
        prompt_parts = []

        for image in row.images:
            prompt_parts.append(f"![]({image})")

        if row.texts:
            prompt_parts.extend(row.texts)

        return " ".join(prompt_parts)

    def _add_request_params(self, payload: Dict, optional_data: Dict[Any, Any]) -> None:
        number_of_tokens = self._get_max_tokens(optional_data)
        if number_of_tokens != OutputTokenDefaults.MEAN:
            payload.setdefault("parameters", {})["max_new_tokens"] = number_of_tokens
        if self.config.input.extra:
            for key, value in self.config.input.extra.items():
                if key in payload["parameters"]:
                    payload["parameters"][key] = value
                else:
                    payload[key] = value
