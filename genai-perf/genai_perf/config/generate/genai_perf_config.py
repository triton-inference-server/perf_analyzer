# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

from genai_perf.config.generate.search_parameter import SearchUsage
from genai_perf.config.input.config_command import (
    ConfigCommand,
    ConfigInput,
    ConfigOutputTokens,
    ConfigSyntheticTokens,
)
from genai_perf.types import ModelObjectiveParameters


@dataclass
class GenAIPerfConfig:
    """
    Creates a config that can be used by be
    used by other modules to configure the various
    options of GenAI-Perf
    """

    def __init__(
        self,
        config: ConfigCommand,
        model_objective_parameters: ModelObjectiveParameters,
    ):
        self._set_options_based_on_config(config)
        self._set_options_based_on_objective(model_objective_parameters)

    ###########################################################################
    # Set Options Methods
    ###########################################################################
    def _set_options_based_on_config(self, config: ConfigCommand) -> None:
        self.input: ConfigInput = config.input
        self.output_tokens: ConfigOutputTokens = config.output_tokens

    def _set_options_based_on_objective(
        self, model_objective_parameters: ModelObjectiveParameters
    ) -> None:
        for objective in model_objective_parameters.values():
            for name, parameter in objective.items():
                if parameter.usage == SearchUsage.RUNTIME_GAP:
                    if hasattr(self.input, name):
                        self.input.__setattr__(
                            name, parameter.get_value_based_on_category()
                        )

    ###########################################################################
    # Checkpoint Methods
    ###########################################################################
    def write_to_checkpoint(self) -> Dict[str, Any]:
        """
        Converts the class data into a dictionary that can be written to
        the checkpoint file
        """
        genai_perf_config_dict = deepcopy(self.__dict__)

        return genai_perf_config_dict

    @classmethod
    def read_from_checkpoint(
        cls, genai_perf_config_dict: Dict[str, Any]
    ) -> "GenAIPerfConfig":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of a GenAIPerfConfig
        """
        genai_perf_config = GenAIPerfConfig(
            config=ConfigCommand([""]),
            model_objective_parameters={},
        )
        genai_perf_config.input = ConfigInput(**genai_perf_config_dict["input"])
        genai_perf_config.input.synthetic_tokens = ConfigSyntheticTokens(
            **genai_perf_config_dict["input"]["synthetic_tokens"]
        )
        genai_perf_config.output_tokens = ConfigOutputTokens(
            **genai_perf_config_dict["output_tokens"]
        )

        return genai_perf_config
