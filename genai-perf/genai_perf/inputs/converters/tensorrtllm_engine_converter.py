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

from typing import Any, Dict, List

from genai_perf.config.input.config_defaults import InputDefaults
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.input_constants import DEFAULT_TENSORRTLLM_MAX_TOKENS
from genai_perf.inputs.retrievers.generic_dataset import DataRow, GenericDataset
from genai_perf.utils import sample_bounded_normal


class TensorRTLLMEngineConverter(BaseConverter):
    def check_config(self) -> None:
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
            for row in file_data.rows:
                token_ids = self._encode_tokens(row.texts[0])
                payload = {
                    "input_ids": {
                        "content": token_ids,
                        "shape": [len(token_ids)],
                    },
                    "input_lengths": [len(token_ids)],
                    "request_output_len": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
                }

                request_body["data"].append(
                    self._finalize_payload(payload, row, triton_format=True)
                )

        return request_body

    def _add_request_params(self, payload: Dict, optional_data: Dict[Any, Any]) -> None:
        if self.config.endpoint.streaming:
            payload["streaming"] = [True]
        if self.config.input.output_tokens.get_field("mean").is_set_by_user:
            num_tokens = int(
                sample_bounded_normal(
                    mean=self.config.input.output_tokens.mean,
                    stddev=self.config.input.output_tokens.stddev,
                    lower=1,  # output token must be >= 1
                )
            )
            payload["request_output_len"] = [num_tokens]
            if self.config.input.output_tokens.deterministic:
                payload["min_length"] = [num_tokens]
        if self.config.input.extra:
            for key, value in self.config.input.extra.items():
                if key == "set_end_id":
                    payload["end_id"] = [self.tokenizer._tokenizer.eos_token_id]
                elif key == "apply_chat_template":
                    pass
                else:
                    payload[key] = [value]

    def _encode_tokens(self, prompt: str) -> List[int]:
        assert self.tokenizer is not None, "Tokenizer is required for encoding tokens"

        if self.config.input.extra and "apply_chat_template" in self.config.input.extra:
            token_ids = self._encode_with_chat_template(prompt)
        else:
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return token_ids

    def _encode_with_chat_template(self, prompt: str) -> List[int]:
        """
        Apply the default TRT-LLM engine chat template to the prompt
        """
        import jinja2

        default_template = self._construct_default_template(prompt)
        return self.tokenizer.encode(
            self.tokenizer._tokenizer.apply_chat_template(
                default_template, tokenize=False, add_special_tokens=False
            )
        )

    def _construct_default_template(self, prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def _add_payload_optional_data(self, payload: Dict[Any, Any], row: DataRow) -> None:
        for key, value in row.optional_data.items():
            if key == "max_tokens":
                payload["request_output_len"] = [value]
            else:
                payload[key] = value
