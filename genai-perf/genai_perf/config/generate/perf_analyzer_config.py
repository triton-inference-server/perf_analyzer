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
from typing import Any, Dict, List

from genai_perf.config.generate.search_parameter import SearchUsage
from genai_perf.config.input.config_command import ConfigCommand, ConfigPerfAnalyzer
from genai_perf.exceptions import GenAIPerfException
from genai_perf.types import CheckpointObject, ModelName, ModelObjectiveParameters


@dataclass
class PerfAnalyzerConfig:
    """
    Contains all the methods necessary for handling calls
    to PerfAnalyzer
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

    ###########################################################################
    # Set Options Methods
    ###########################################################################
    def _set_options_based_on_config(self, config: ConfigCommand) -> None:
        self._config: ConfigPerfAnalyzer = config.perf_analyzer

    def _set_options_based_on_objective(
        self, model_objective_parameters: ModelObjectiveParameters
    ) -> None:
        self._parameters: Dict[str, Any] = {}
        for objective in model_objective_parameters.values():
            for name, parameter in objective.items():
                if parameter.usage == SearchUsage.RUNTIME_PA:
                    self._parameters[name] = parameter.get_value_based_on_category()

    ###########################################################################
    # Get Accessor Methods
    ###########################################################################
    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters and their values
        """
        return self._parameters

    ###########################################################################
    # CLI String Creation Methods
    ###########################################################################
    def create_command(self) -> List[str]:
        """
        Returns the PA command a list of strings
        """
        cli_args = self._create_required_args()
        cli_args += self._create_parameter_args()

        return cli_args

    def create_cli_string(self) -> str:
        """
        Returns the PA command as a string
        """
        cli_args = self.create_command()

        cli_string = " ".join(cli_args)
        return cli_string

    def _create_required_args(self) -> List[str]:
        # These come from the config and are always used
        required_args = [
            self._config.path,
            "-m",
            self._model_name,
            "--stability-percentage",
            str(self._config.stability_threshold),
        ]

        return required_args

    def _create_parameter_args(self) -> List[str]:
        parameter_args = []
        for name, value in self._parameters.items():
            parameter_args.append(self._convert_objective_to_cli_option(name))
            parameter_args.append(str(value))

        return parameter_args

    def _convert_objective_to_cli_option(self, objective_name: str) -> str:
        obj_to_cli_dict = {
            "runtime_batch_size": "--batch-size",
            "concurrency": "--concurrency-range",
            "request-rate": "--request-rate-range",
        }

        try:
            return obj_to_cli_dict[objective_name]
        except:
            raise GenAIPerfException(f"{objective_name} not found")

    ###########################################################################
    # Representation Methods
    ###########################################################################
    def representation(self) -> str:
        """
        A string representation of the PA command that removes values which
        can vary between runs, but should be ignored when determining
        if a previous (checkpointed) run can be used
        """
        options_with_arg_to_remove = [
            "--url",
            "--metrics-url",
            "--latency-report-file",
            "--measurement-request-count",
        ]
        options_only_to_remove = ["--verbose", "--extra-verbose", "--verbose-csv"]

        representation = self.create_cli_string()

        # Remove the PA call path which is always the first token
        representation_list = representation.split(" ")
        representation_list.pop(0)
        representation = " ".join(representation_list)

        for option_with_arg in options_with_arg_to_remove:
            representation = self._remove_option_from_cli_string(
                option_with_arg, representation, with_arg=True
            )

        for option_only in options_only_to_remove:
            representation = self._remove_option_from_cli_string(
                option_only, representation, with_arg=False
            )

        return representation

    def _remove_option_from_cli_string(
        self, option_to_remove: str, cli_string: str, with_arg: bool
    ) -> str:
        if option_to_remove not in cli_string:
            return cli_string

        cli_str_tokens = cli_string.split(" ")

        removal_index = cli_str_tokens.index(option_to_remove)
        cli_str_tokens.pop(removal_index)

        if with_arg:
            cli_str_tokens.pop(removal_index)

        return " ".join(cli_str_tokens)

    ###########################################################################
    # Checkpoint Methods
    ###########################################################################
    def create_checkpoint_object(self) -> CheckpointObject:
        """
        Converts the class data into a dictionary that can be written to
        the checkpoint file
        """
        pa_config_dict = deepcopy(self.__dict__)

        return pa_config_dict

    @classmethod
    def create_class_from_checkpoint(
        cls, perf_analyzer_config_dict: CheckpointObject
    ) -> "PerfAnalyzerConfig":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of a PerfAnalyzerConfig
        """
        perf_analyzer_config = PerfAnalyzerConfig(
            model_name=perf_analyzer_config_dict["_model_name"],
            config=ConfigCommand([""]),
            model_objective_parameters={},
        )
        perf_analyzer_config._config = ConfigPerfAnalyzer(
            **perf_analyzer_config_dict["_config"]
        )
        perf_analyzer_config._parameters = perf_analyzer_config_dict["_parameters"]

        return perf_analyzer_config