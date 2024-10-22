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
from typing import Any, Dict

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import ModelSelectionStrategy
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.generic_dataset import GenericDataset


class BaseConverter:
    """
    Base class for all converters that take generic JSON payloads
    and convert them to endpoint-specific payloads.
    """

    def check_config(self, config: InputsConfig) -> None:
        """
        Check whether the provided configuration is valid for this converter.

        Throws a GenAIPerfException if the configuration is invalid.
        """
        pass

    def convert(
        self, generic_dataset: GenericDataset, config: InputsConfig
    ) -> Dict[Any, Any]:
        """
        Construct a request body using the endpoint specific request format.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _select_model_name(self, config: InputsConfig, index: int) -> str:
        if config.model_selection_strategy == ModelSelectionStrategy.ROUND_ROBIN:
            return config.model_name[index % len(config.model_name)]
        elif config.model_selection_strategy == ModelSelectionStrategy.RANDOM:
            return random.choice(config.model_name)
        else:
            raise GenAIPerfException(
                f"Model selection strategy '{config.model_selection_strategy}' is unsupported"
            )

    def _add_request_params(
        self, payload: Dict[Any, Any], config: InputsConfig
    ) -> None:
        for key, value in config.extra_inputs.items():
            payload[key] = value
