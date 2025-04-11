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

import random
from typing import Any, Dict, Optional, Union

from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.input.config_defaults import OutputTokenDefaults
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import ModelSelectionStrategy
from genai_perf.inputs.retrievers.generic_dataset import DataRow, GenericDataset
from genai_perf.tokenizer import Tokenizer, get_empty_tokenizer
from genai_perf.utils import sample_bounded_normal


class BaseConverter:
    def __init__(self, config: ConfigCommand, tokenizer: Optional[Tokenizer] = None):
        self.config = config
        self.tokenizer = tokenizer if tokenizer else get_empty_tokenizer()

    """
    Base class for all converters that take generic JSON payloads
    and convert them to endpoint-specific payloads.
    """

    def check_config(self) -> None:
        """
        Check whether the provided configuration is valid for this converter.

        Throws a GenAIPerfException if the configuration is invalid.
        """
        pass

    def convert(self, generic_dataset: GenericDataset) -> Dict[Any, Any]:
        """
        Construct a request body using the endpoint specific request format.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _select_model_name(self, index: int) -> str:
        if (
            self.config.endpoint.model_selection_strategy
            == ModelSelectionStrategy.ROUND_ROBIN
        ):
            return self.config.model_names[index % len(self.config.model_names)]
        elif (
            self.config.endpoint.model_selection_strategy
            == ModelSelectionStrategy.RANDOM
        ):
            return random.choice(self.config.model_names)
        else:
            raise GenAIPerfException(
                f"Model selection strategy '{self.config.endpoint.model_selection_strategy}' is unsupported"
            )

    def _get_max_tokens(self, optional_data: Dict[Any, Any]) -> Union[int, None]:
        """
        Return the `max_tokens` value to be added in the payload.
        If `max_tokens` is present in `optional_data`, that value is used.
        Otherwise, `max_tokens` is sampled from a bounded normal
        distribution with a minimum value of 1.
        """
        if "max_tokens" in optional_data:
            return optional_data["max_tokens"]
        elif self.config.input.output_tokens.get_field("mean").is_set_by_user:
            return int(
                sample_bounded_normal(
                    mean=self.config.input.output_tokens.mean,
                    stddev=self.config.input.output_tokens.stddev,
                    lower=1,  # output token must be >= 1
                )
            )
        return OutputTokenDefaults.MEAN

    def _add_request_params(
        self,
        payload: Dict[Any, Any],
        optional_data: Dict[Any, Any],
    ) -> None:
        if self.config.input.extra:
            for key, value in self.config.input.extra.items():
                payload[key] = value

    def _add_payload_optional_data(self, payload: Dict[Any, Any], row: DataRow) -> None:
        for key, value in row.optional_data.items():
            payload[key] = value

    def _add_payload_metadata(self, record: Dict[str, Any], row: DataRow) -> None:
        for key, value in row.payload_metadata.items():
            record[key] = [value]

    def _finalize_payload(
        self,
        payload: Dict[Any, Any],
        row: DataRow,
        triton_format=False,
    ) -> Dict[str, Any]:
        self._add_request_params(payload, row.optional_data)
        self._add_payload_optional_data(payload, row)
        record: Dict[str, Any] = {}
        if not triton_format:
            record["payload"] = [payload]
        else:
            record.update(payload)
        self._add_payload_metadata(record, row)

        return record
