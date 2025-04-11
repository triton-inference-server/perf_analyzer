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

from typing import Any, Dict

from genai_perf.config.input.config_defaults import InputDefaults
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.retrievers.generic_dataset import GenericDataset


class DynamicGRPCConverter(BaseConverter):

    def check_config(self) -> None:
        if self.config.input.batch_size != InputDefaults.BATCH_SIZE:
            raise GenAIPerfException(
                f"The --batch-size-text flag is not supported for {self.config.endpoint.output_format.to_lowercase()}."
            )
        if self.config.input.file == "":
            raise GenAIPerfException(
                f"The dynamic GRPC converter only supports the input file path."
            )

    def convert(
        self,
        generic_dataset: GenericDataset,
    ) -> Dict[Any, Any]:
        request_body: Dict[str, Any] = {"data": []}

        for file_data in generic_dataset.files_data.values():
            for index, row in enumerate(file_data.rows):
                payload = {"message_generator": row.texts[0]}
                self._add_request_params(payload, {})
                request_body["data"].append(payload)

        return request_body
