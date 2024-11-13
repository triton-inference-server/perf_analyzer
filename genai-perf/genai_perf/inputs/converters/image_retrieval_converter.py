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

from typing import Any, Dict, List, Union

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.generic_dataset import DataRow, GenericDataset


class ImageRetrievalConverter(BaseConverter):

    def check_config(self, config: InputsConfig) -> None:
        if config.output_format == OutputFormat.IMAGE_RETRIEVAL:
            if config.add_stream:
                raise GenAIPerfException(
                    f"The --streaming option is not supported for {config.output_format.to_lowercase()}."
                )

    def convert(
        self, generic_dataset: GenericDataset, config: InputsConfig
    ) -> Dict[Any, Any]:
        request_body: Dict[str, Any] = {"input": []}

        for file_data in generic_dataset.files_data.values():
            for index, row in enumerate(file_data.rows):
                payload = self._create_payload(row, config)
                request_body["data"].append({"payload": [payload]})

        return request_body

    def _create_payload(
        self, row: DataRow, config: InputsConfig
    ) -> Dict[Any, Any]:
        content = self._retrieve_content(row, config)
        return content

    def _retrieve_content(
        self, row: DataRow, config: InputsConfig
    ) -> Union[str, List[Dict[Any, Any]]]:
        content: Union[str, List[Dict[Any, Any]]] = ""
        if config.output_format == OutputFormat.IMAGE_RETRIEVAL:
            content = self._add_multi_modal_content(row)
        else:
            raise GenAIPerfException(
                f"Output format {config.output_format} is not supported"
            )
        return content

    def _add_multi_modal_content(self, entry: DataRow) -> List[Dict[Any, Any]]:
        content: List[Dict[Any, Any]] = []
        for image in entry.images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image,
                    },
                }
            )
        return content