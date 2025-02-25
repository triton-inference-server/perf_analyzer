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
from itertools import product
from typing import Any, Dict, Generator, List, TypeAlias

import genai_perf.logging as logging
from genai_perf.config.generate.objective_parameter import ObjectiveParameter
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.types import ModelName, ModelObjectiveParameters, ModelSearchParameters

logger = logging.getLogger(__name__)


###########################################################################
# Type Aliases
###########################################################################
ParameterCombination: TypeAlias = Dict[str, Any]
ParameterCombinations: TypeAlias = List[ParameterCombination]

ModelParameterCombination: TypeAlias = Dict[ModelName, ParameterCombination]
ModelParameterCombinations: TypeAlias = Dict[ModelName, ParameterCombinations]

AllParameterCombinations: TypeAlias = List[ModelParameterCombination]


class SweepObjectiveGenerator:
    """
    Generates the next set of objectives to profile when
    exhauastively the search space
    """

    def __init__(
        self, config: ConfigCommand, model_search_parameters: ModelSearchParameters
    ):
        self._config = config
        self._model_search_parameters = model_search_parameters

    ###########################################################################
    # Sweep (Generator) Method
    ###########################################################################
    def get_objectives(self) -> Generator[ModelObjectiveParameters, None, None]:
        """
        Generates objectives that will be used to create the next
        RunConfig to be profiled
        """

        if logger.level == logging.logging.DEBUG:
            self._print_debug_search_space_info()

        yield from self._create_objectives()

    ###########################################################################
    # Objectives Methods
    ###########################################################################
    def _create_objectives(self) -> Generator[ModelObjectiveParameters, None, None]:
        # First create the dictionary of PER MODEL parameter combinations
        model_all_search_parameter_combinations: ModelParameterCombinations = {}
        for model_name in self._config.model_names:
            model_all_search_parameter_combinations[model_name] = (
                self._create_list_of_model_search_parameter_combinations(model_name)
            )

        # Next combine the per model combinations to create a ALL MODEL combination list
        all_search_parameter_combinations = (
            self._create_list_of_all_search_parameter_combinations(
                model_all_search_parameter_combinations
            )
        )

        # Then iterate through the all model list to create a single set of
        # objectives (per model) to profile
        yield from self._create_model_objective_parameters(
            all_search_parameter_combinations
        )

    def _create_model_objective_parameters(
        self, all_search_parameter_combinations: AllParameterCombinations
    ) -> Generator[ModelObjectiveParameters, None, None]:
        model_objective_parameters: ModelObjectiveParameters = {}
        for all_search_parameter_combination in all_search_parameter_combinations:
            for (
                model_name,
                model_search_parameter_combination,
            ) in all_search_parameter_combination.items():

                model_objective_parameters[model_name] = {}
                for objective_name, value in model_search_parameter_combination.items():
                    category = self._model_search_parameters[
                        model_name
                    ].get_objective_category(objective_name)
                    usage = self._model_search_parameters[model_name].get_type(
                        objective_name
                    )

                    model_objective_parameters[model_name][objective_name] = (
                        ObjectiveParameter(usage, category, value)
                    )

            yield model_objective_parameters

    ###########################################################################
    # Parameter Combination Methods
    ###########################################################################
    def _create_list_of_model_search_parameter_combinations(
        self, model_name: ModelName
    ) -> ParameterCombinations:
        search_parameters = self._model_search_parameters[model_name]

        parameter_lists = {}
        for parameter_name in search_parameters.get_parameter_names():
            parameter_lists[parameter_name] = search_parameters.get_list(parameter_name)

        parameter_names, parameter_combinations = zip(*parameter_lists.items())

        model_search_parameter_combinations = [
            dict(zip(parameter_names, parameter_combination))
            for parameter_combination in product(*parameter_combinations)
        ]

        return model_search_parameter_combinations

    def _create_list_of_all_search_parameter_combinations(
        self,
        model_search_parameter_combinations: ModelParameterCombinations,
    ) -> AllParameterCombinations:
        model_names, model_parameter_combinations = zip(
            *model_search_parameter_combinations.items()
        )

        all_search_parameter_combinations = []
        for model_parameter_combination in product(*model_parameter_combinations):
            model_combination_dict = {}
            for count, model_name in enumerate(model_names):
                model_combination_dict[model_name] = model_parameter_combination[count]

            all_search_parameter_combinations.append(model_combination_dict)

        return all_search_parameter_combinations

    ###########################################################################
    # General Search Space Methods
    ###########################################################################
    def _calculate_num_of_configs_in_search_space(self) -> int:
        num_of_configs_in_search_space = 1
        for model_name in self._config.model_names:
            num_of_configs_in_search_space *= self._model_search_parameters[
                model_name
            ].number_of_total_possible_configurations()

        return num_of_configs_in_search_space

    ###########################################################################
    # Info/Debug Methods
    ###########################################################################
    def _print_debug_search_space_info(self) -> None:
        logger.info("")
        num_of_configs_in_search_space = (
            self._calculate_num_of_configs_in_search_space()
        )
        logger.debug(
            f"Number of configs in search space: {num_of_configs_in_search_space}"
        )
        self._print_debug_model_search_space_info()
        logger.info("")

    def _print_debug_model_search_space_info(self) -> None:
        for model_name in self._config.model_names:
            logger.debug(f"Model - {model_name}:")
            for search_parameter_name in self._model_search_parameters[
                model_name
            ].get_parameter_names():
                logger.debug(
                    self._model_search_parameters[model_name].print_info(
                        search_parameter_name
                    )
                )
