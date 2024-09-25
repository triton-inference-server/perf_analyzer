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
from dataclasses import dataclass
from random import randint
from typing import Generator, Optional

import genai_perf.logging as logging
import optuna
from genai_perf.config.generate.objective_parameter import ObjectiveParameters
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.measurements.run_config_measurement import RunConfigMeasurement
from genai_perf.types import ModelSearchParameters

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptunaObjectiveGeneratorDefaults:
    BASELINE_SCORE = 0


class OptunaObjectiveGenerator:
    """
    Generates the next set of objectives to profile using Optuna's hyperparameter
    algorithm
    """

    # This list represents all possible parameters Optuna can currently search for
    optuna_parameter_list = [
        "batch_sizes",
        "max_batch_size",
        "instance_group",
        "concurrency",
        "max_queue_delay_microseconds",
        "request_rate",
    ]

    def __init__(
        self,
        config: ConfigCommand,
        model_search_parameters: ModelSearchParameters,
        baseline_measurement: RunConfigMeasurement,
        user_seed: Optional[int] = None,
    ):
        self._config = config
        self._model_search_parameters = model_search_parameters
        self._seed = self._create_seed(user_seed)

        self._baseline_measurement = baseline_measurement
        self._last_measurement: Optional[RunConfigMeasurement] = None
        self._best_score: Optional[float] = (
            OptunaObjectiveGeneratorDefaults.BASELINE_SCORE
        )
        self._best_trial_number: Optional[int] = None

        self._sampler = optuna.samplers.TPESampler(seed=self._seed)
        self._study_name = "_".join(config.model_names)
        self._study = optuna.create_study(
            study_name=self._study_name,
            direction="maximize",
            sampler=self._sampler,
        )

        # FIXME: OPTIMIZE - will be used once checkpoint logic is created
        # self._init_state()

    # FIXME: OPTIMIZE - will be used once checkpoint logic is created
    # def _get_seed(self) -> int:
    #     return self._state_manager.get_state_variable("OptunaRunConfigGenerator.seed")

    def _create_seed(self, user_seed: Optional[int]) -> int:
        # FIXME: OPTIMIZE - will be used once checkpoint logic is created
        # if self._state_manager.starting_fresh_run():
        #     seed = randint(0, 10000) if user_seed is None else user_seed
        # else:
        # seed = self._get_seed() if user_seed is None else user_seed

        seed = randint(0, 10000) if user_seed is None else user_seed
        return seed

    # FIXME: OPTIMIZE - will be used once checkpoint logic is created
    # def _init_state(self) -> None:
    #     self._state_manager.set_state_variable("OptunaRunConfigGenerator.seed", self._seed)

    def set_last_measurement(self, measurement: Optional[RunConfigMeasurement]):
        self._last_measurement = measurement

    # def get_next_objective(self) -> Generator[ObjectiveParameters, None, None]:
    #     NotImplemented

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
    # Minimum Search Space Methods
    ###########################################################################
    def _determine_minimum_number_of_configs_to_search(self) -> int:
        min_trials_based_on_percentage_of_search_space = (
            self._determine_trials_based_on_min_percentage_of_search_space()
        )

        min_configs_to_search = self._decide_min_between_percentage_and_trial_count(
            min_trials_based_on_percentage_of_search_space
        )

        return min_configs_to_search

    def _determine_trials_based_on_min_percentage_of_search_space(self) -> int:
        total_num_of_possible_configs = self._calculate_num_of_configs_in_search_space()
        min_trials_based_on_percentage_of_search_space = int(
            total_num_of_possible_configs
            * self._config.optimize.search_space_percentage.min
            / 100
        )

        return min_trials_based_on_percentage_of_search_space

    def _decide_min_between_percentage_and_trial_count(
        self, min_trials_based_on_percentage_of_search_space: int
    ) -> int:
        # By default we will search based on percentage of search space
        # If the user specifies a number of trials we will use that instead
        # If both are specified we will use the larger number
        min_trials_set_by_user = self._config.optimize.is_set_by_user(
            "number_of_trials"
        )
        min_percentage_set_by_user = self._config.optimize.is_set_by_user(
            "search_space_percentage"
        )

        if min_trials_set_by_user and min_percentage_set_by_user:
            if (
                self._config.optimize.number_of_trials.min
                > min_trials_based_on_percentage_of_search_space
            ):
                logger.debug(
                    f"Minimum number of trials: {self._config.optimize.number_of_trials.min} (optuna_min_trials)"
                )
                min_configs_to_search = self._config.optimize.number_of_trials.min
            else:
                logger.debug(
                    f"Minimum number of trials: {min_trials_based_on_percentage_of_search_space} "
                    f"({self._config.optimize.search_space_percentage.min}% of search space)"
                )
                min_configs_to_search = min_trials_based_on_percentage_of_search_space
        elif min_trials_set_by_user:
            logger.debug(
                f"Minimum number of trials: {self._config.optimize.number_of_trials.min} (optuna_min_trials)"
            )
            min_configs_to_search = self._config.optimize.number_of_trials.min
        else:
            logger.debug(
                f"Minimum number of trials: {min_trials_based_on_percentage_of_search_space} "
                f"({self._config.optimize.search_space_percentage.min}% of search space)"
            )
            min_configs_to_search = min_trials_based_on_percentage_of_search_space

        return min_configs_to_search

    ###########################################################################
    # Maximum Search Space Methods
    ###########################################################################
    def _determine_maximum_number_of_configs_to_search(self) -> int:
        max_trials_based_on_percentage_of_search_space = (
            self._determine_trials_based_on_max_percentage_of_search_space()
        )

        max_configs_to_search = self._decide_max_between_percentage_and_trial_count(
            max_trials_based_on_percentage_of_search_space
        )

        return max_configs_to_search

    def _determine_trials_based_on_max_percentage_of_search_space(self) -> int:
        total_num_of_possible_configs = self._calculate_num_of_configs_in_search_space()
        max_trials_based_on_percentage_of_search_space = int(
            total_num_of_possible_configs
            * self._config.optimize.search_space_percentage.max
            / 100
        )

        return max_trials_based_on_percentage_of_search_space

    def _decide_max_between_percentage_and_trial_count(
        self, max_trials_based_on_percentage_of_search_space: int
    ) -> int:
        # By default we will search based on percentage of search space
        # If the user specifies a number of trials we will use that instead
        # If both are specified we will use the smaller number
        max_trials_set_by_user = self._config.optimize.is_set_by_user(
            "number_of_trials"
        )
        max_percentage_set_by_user = self._config.optimize.is_set_by_user(
            "search_space_percentage"
        )

        if max_trials_set_by_user and max_percentage_set_by_user:
            if (
                self._config.optimize.number_of_trials.max
                < max_trials_based_on_percentage_of_search_space
            ):
                logger.debug(
                    f"Maximum number of trials: {self._config.optimize.number_of_trials.max} (optuna_max_trials)"
                )
                max_configs_to_search = self._config.optimize.number_of_trials.max
            else:
                logger.debug(
                    f"Maximum number of trials: {max_trials_based_on_percentage_of_search_space} "
                    f"({self._config.optimize.search_space_percentage.max}% of search space)"
                )
                max_configs_to_search = max_trials_based_on_percentage_of_search_space
        elif max_trials_set_by_user:
            logger.debug(
                f"Maximum number of trials: {self._config.optimize.number_of_trials.max} (optuna_max_trials)"
            )
            max_configs_to_search = self._config.optimize.number_of_trials.max
        else:
            logger.debug(
                f"Maximum number of trials: {max_trials_based_on_percentage_of_search_space} "
                f"({self._config.optimize.search_space_percentage.max}% of search space)"
            )
            max_configs_to_search = max_trials_based_on_percentage_of_search_space

        # FIXME: OPTIMIZE: Why doesn't this work?
        # if logging.DEBUG:
        #     logger.info("")
        return max_configs_to_search
