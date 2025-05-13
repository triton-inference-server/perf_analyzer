# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass

from genai_perf.config.generate.search_parameter import SearchUsage
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.types import CheckpointObject, ModelObjectiveParameters, Parameters


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
        self._parameters = self._set_parameters_based_on_config(config)
        self._parameters |= self._set_parameters_based_on_objective(
            model_objective_parameters
        )

    ###########################################################################
    # Set Options Methods
    ###########################################################################
    def _set_parameters_based_on_config(self, config: ConfigCommand) -> Parameters:
        """
        Store values set in the config that should be checked when
        determining if a previously checkpointed run can be used
        """
        parameters: Parameters = {}

        # ENDPOINT
        parameters["endpoint"] = config.endpoint.to_json_dict()

        # Remove any fields that have no bearing on the
        # values of metrics being measured
        del parameters["endpoint"]["server_metrics_urls"]
        del parameters["endpoint"]["url"]

        # INPUT
        parameters["input"] = config.input.to_json_dict()

        # TOKENIZER
        parameters["tokenizer"] = config.tokenizer.to_json_dict()

        return parameters

    def _set_parameters_based_on_objective(
        self, model_objective_parameters: ModelObjectiveParameters
    ) -> Parameters:
        parameters: Parameters = {}
        for objective in model_objective_parameters.values():
            for name, parameter in objective.items():
                if parameter.usage == SearchUsage.RUNTIME_GAP:
                    parameters[name] = parameter.get_value_based_on_category()

        return parameters

    ###########################################################################
    # Get Accessor Methods
    ###########################################################################
    def get_parameters(self) -> Parameters:
        """
        Returns a dictionary of parameters and their values
        """
        return self._parameters

    ###########################################################################
    # Representation Methods
    ###########################################################################
    def representation(self) -> str:
        """
        A string representation of the GAP options which will be
        used when determining if a previous (checkpointed) run can be used
        """
        representation = " ".join([self._parameters.__str__()])

        return representation

    ###########################################################################
    # Checkpoint Methods
    ###########################################################################
    def create_checkpoint_object(self) -> CheckpointObject:
        """
        Converts the class data into a dictionary that can be written to
        the checkpoint file
        """
        genai_perf_config_dict = deepcopy(self.__dict__)

        return genai_perf_config_dict

    @classmethod
    def create_class_from_checkpoint(
        cls, genai_perf_config_dict: CheckpointObject
    ) -> "GenAIPerfConfig":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of a GenAIPerfConfig
        """
        genai_perf_config = GenAIPerfConfig(
            config=ConfigCommand(skip_inferencing_and_checking=True),
            model_objective_parameters={},
        )

        genai_perf_config._parameters = genai_perf_config_dict["_parameters"]

        return genai_perf_config
