# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict, List, Union

from genai_perf.config.input.config_defaults import InputDefaults, OutputTokenDefaults
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.retrievers.generic_dataset import DataRow, GenericDataset


class OpenAIChatCompletionsConverter(BaseConverter):

    def check_config(self) -> None:
        if (
            self.config.endpoint.output_format == OutputFormat.OPENAI_CHAT_COMPLETIONS
            or self.config.endpoint.output_format == OutputFormat.OPENAI_MULTIMODAL
        ):
            if self.config.input.batch_size != InputDefaults.BATCH_SIZE:
                raise GenAIPerfException(
                    f"The --batch-size-text flag is not supported for {self.config.endpoint.output_format.to_lowercase()}."
                )

    def convert(
        self,
        generic_dataset: GenericDataset,
    ) -> Dict[Any, Any]:
        request_body: Dict[str, Any] = {"data": []}

        for file_data in generic_dataset.files_data.values():
            for index, row in enumerate(file_data.rows):
                payload = self._create_payload(index, row)
                request_body["data"].append(self._finalize_payload(payload, row))

        return request_body

    def _create_payload(self, index: int, row: DataRow) -> Dict[Any, Any]:
        model_name = self._select_model_name(index)
        content = self._retrieve_content(row)

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }

        return payload

    def _retrieve_content(self, row: DataRow) -> Union[str, List[Dict[Any, Any]]]:
        content: Union[str, List[Dict[Any, Any]]] = ""
        if self.config.endpoint.output_format == OutputFormat.OPENAI_CHAT_COMPLETIONS:
            content = row.texts[0]
        elif self.config.endpoint.output_format == OutputFormat.OPENAI_MULTIMODAL:
            content = self._add_multi_modal_content(row)
        else:
            raise GenAIPerfException(
                f"Output format {self.config.endpoint.output_format} is not supported"
            )
        return content

    def _add_multi_modal_content(self, entry: DataRow) -> List[Dict[Any, Any]]:
        content: List[Dict[Any, Any]] = []
        for text in entry.texts:
            content.append(
                {
                    "type": "text",
                    "text": text,
                }
            )
        for image in entry.images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image,
                    },
                }
            )
        for audio in entry.audios:
            format, b64_audio = audio.split(",")
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": b64_audio,
                        "format": format,
                    },
                }
            )
        return content

    def _add_request_params(self, payload: Dict, optional_data: Dict[Any, Any]) -> None:
        if self.config.endpoint.streaming:
            payload["stream"] = True
        max_tokens = self._get_max_tokens(optional_data)
        if max_tokens != OutputTokenDefaults.MEAN:
            payload["max_tokens"] = max_tokens
        if self.config.input.extra:
            for key, value in self.config.input.extra.items():
                payload[key] = value
