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

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.retrievers.generic_dataset import GenericDataset


class RankingsConverter(BaseConverter):

    def check_config(self) -> None:
        if self.config.endpoint.streaming:
            raise GenAIPerfException(
                f"The --streaming option is not supported for {self.config.endpoint.output_format.to_lowercase()}."
            )

    def convert(
        self,
        generic_dataset: GenericDataset,
    ) -> Dict[Any, Any]:
        provided_filenames = list(generic_dataset.files_data.keys())
        if "queries" not in provided_filenames or "passages" not in provided_filenames:
            raise ValueError(
                "Both 'queries.jsonl' and 'passages.jsonl' must be present in the input datasets."
            )

        queries_data = generic_dataset.files_data["queries"]
        passages_data = generic_dataset.files_data["passages"]

        rows_of_passage_data = len(passages_data.rows)

        request_body: Dict[str, Any] = {"data": []}

        for query_index, query_row in enumerate(queries_data.rows):
            if query_index >= rows_of_passage_data:
                break

            model_name = self._select_model_name(query_index)
            query = query_row.texts[0]

            passage_entry = passages_data.rows[query_index]

            passages: Union[List[str], List[Dict[str, str]]]
            payload: Dict[str, Any]

            if self._is_rankings_tei():
                passages = passage_entry.texts
                payload = {"query": query, "texts": passages}
            else:
                passages = [{"text": p} for p in passage_entry.texts if p is not None]
                payload = {
                    "query": {"text": query},
                    "passages": passages,
                    "model": model_name,
                }

            request_body["data"].append(self._finalize_payload(payload, passage_entry))

        return request_body

    def _is_rankings_tei(self) -> bool:
        """
        Check if user specified that they are using the Hugging Face
        Text Embeddings Interface for ranking models
        """
        if self.config.input.extra and self.config.input.extra.get("rankings") == "tei":
            return True
        return False

    def _add_request_params(self, payload: Dict, optional_data: Dict[Any, Any]) -> None:
        if self.config.input.extra:
            for key, value in self.config.input.extra.items():
                if not (key == "rankings" and value == "tei"):
                    payload[key] = value
