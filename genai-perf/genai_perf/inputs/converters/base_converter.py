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
from typing import Dict, List, Union, cast

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import ModelSelectionStrategy
from genai_perf.inputs.inputs_config import InputsConfig


class BaseConverter:

    _CONTENT_NAMES: List[str]

    def convert(self, generic_dataset: Dict, config: InputsConfig) -> Dict:
        """
        Construct a request body using the endpoint specific request format.
        """
        raise NotImplementedError

    def _select_model_name(self, config: InputsConfig, index: int) -> str:
        if config.model_selection_strategy == ModelSelectionStrategy.ROUND_ROBIN:
            return config.model_name[index % len(config.model_name)]
        elif config.model_selection_strategy == ModelSelectionStrategy.RANDOM:
            return random.choice(config.model_name)
        else:
            raise GenAIPerfException(
                f"Model selection strategy '{config.model_selection_strategy}' is unsupported"
            )

    def _construct_text_payload_batch_agnostic(
        self, batch_size_text: int, input_data: Union[Dict, List]
    ) -> Union[str, List]:
        """
        Construct text payload content for non-chat based LLM converters.
        Allow batched and unbatched input data.
        """
        if batch_size_text == 1:
            input_data = cast(Dict, input_data)
            return self._construct_text_payload(input_data)
        else:
            input_data = cast(List, input_data)
            return self._construct_batched_text_payload(input_data)

    def _construct_text_payload(self, input_data: Dict) -> str:
        """
        Construct text payload content for non-chat based LLM converters.
        Since there are no roles or turns in non-chat LLM endpoints, all the
        (pre-defined) text contents are concatenated into a single text prompt.
        """
        contents = [v for k, v in input_data.items() if k in self._CONTENT_NAMES]
        return " ".join(contents)

    def _construct_batched_text_payload(self, input_data: List) -> List:
        """
        Construct batched text payload content for non-chat based LLM converters.
        """
        contents = [item["text_input"] for item in input_data]
        return contents
