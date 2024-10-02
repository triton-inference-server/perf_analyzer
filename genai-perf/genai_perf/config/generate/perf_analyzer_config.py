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

from typing import Any, Dict

from genai_perf.config.generate.objective_parameter import ObjectiveCategory
from genai_perf.config.generate.search_parameter import SearchUsage
from genai_perf.config.input.config_command import ConfigPerfAnalyzer


class PerfAnalyzerConfig:
    """
    Contains the all the methods necessary for handling calls
    to PerfAnalzyer
    """

    from genai_perf.config.input.config_command import ConfigCommand
    from genai_perf.types import ModelObjectiveParameters

    def __init__(
        self,
        config: ConfigCommand,
        model_objective_parameters: ModelObjectiveParameters,
    ):
        self._set_options_based_on_config(config)
        self._set_options_based_on_objective(model_objective_parameters)

    def _set_options_based_on_config(self, config: ConfigCommand) -> None:
        self._config: ConfigPerfAnalyzer = config.perf_analzyer

    def _set_options_based_on_objective(
        self, model_objective_parameters: ModelObjectiveParameters
    ) -> None:
        self._parameters: Dict[str, Any] = {}
        foo = model_objective_parameters.values()
        for objective in model_objective_parameters.values():
            for name, parameter in objective.items():
                if parameter.usage == SearchUsage.RUNTIME:
                    if (
                        parameter.category == ObjectiveCategory.INTEGER
                        or parameter.category == ObjectiveCategory.STR
                    ):
                        self._parameters[name] = parameter.value
                    elif parameter.category == ObjectiveCategory.EXPONENTIAL:
                        self._parameters[name] = 2**parameter.value
