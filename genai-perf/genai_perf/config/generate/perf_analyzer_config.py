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
from genai_perf.config.input.config_command import ConfigCommand, ConfigPerfAnalyzer
from genai_perf.types import ModelName, ModelObjectiveParameters


class PerfAnalyzerConfig:
    """
    Contains the all the methods necessary for handling calls
    to PerfAnalzyer
    """

    def __init__(
        self,
        config: ConfigCommand,
        model_objective_parameters: ModelObjectiveParameters,
        model_name: ModelName,
    ):
        self._model_name = model_name
        self._set_options_based_on_config(config)
        self._set_options_based_on_objective(model_objective_parameters)

    def _set_options_based_on_config(self, config: ConfigCommand) -> None:
        self._config: ConfigPerfAnalyzer = config.perf_analzyer

    def _set_options_based_on_objective(
        self, model_objective_parameters: ModelObjectiveParameters
    ) -> None:
        self._parameters: Dict[str, Any] = {}
        for objective in model_objective_parameters.values():
            for name, parameter in objective.items():
                if parameter.usage == SearchUsage.RUNTIME:
                    self._parameters[name] = parameter.get_value_based_on_category()

    def create_cli_string(self) -> str:
        # Required args - from config
        cli_args = [
            self._config.path,
            "-model-name",
            self._model_name,
            "--stability-percentage",
            str(self._config.stability_threshold),
        ]

        # Parameter args
        for name, value in self._parameters.items():
            cli_args.append(self._convert_objective_to_cli_option(name))
            cli_args.append(str(value))

        cli_string = " ".join(cli_args)
        return cli_string

    def _convert_objective_to_cli_option(self, objective_name: str) -> str:
        obj_to_cli_dict = {
            "runtime_batch_size": "batch-size",
            "concurrency": "concurrency-range",
            "request-rate": "request-rate-range",
        }

        return obj_to_cli_dict[objective_name]
