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
from math import log2
from random import randint
from typing import Dict, Generator, Optional, TypeAlias, Union

import genai_perf.logging as logging
import optuna
from genai_perf.config.generate.objective_parameter import ObjectiveParameter
from genai_perf.config.generate.search_parameter import SearchCategory, SearchParameter
from genai_perf.config.generate.search_parameters import SearchParameters
from genai_perf.config.input.config_command import ConfigCommand, RunConfigDefaults
from genai_perf.measurements.run_config_measurement import RunConfigMeasurement
from genai_perf.types import ModelName, ModelObjectiveParameters, ModelSearchParameters

logger = logging.getLogger(__name__)


###########################################################################
# Type Aliases
###########################################################################
ParameterName: TypeAlias = str
ObjectiveName: TypeAlias = str

TrialObjective: TypeAlias = Union[str | int]
TrialObjectives: TypeAlias = Dict[ParameterName, TrialObjective]
ModelTrialObjectives: TypeAlias = Dict[ModelName, TrialObjectives]


###########################################################################
# Defaults
###########################################################################
@dataclass(frozen=True)
class OptunaObjectiveGeneratorDefaults:
    BASELINE_SCORE = 0
    NO_MEASUREMENT_SCORE = -1


class OptunaObjectiveGenerator:
    """
    Generates the next set of objectives to profile using Optuna's hyperparameter
    algorithm
    """

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
        self._measurement: Optional[RunConfigMeasurement] = None

        self._best_score: float = OptunaObjectiveGeneratorDefaults.BASELINE_SCORE
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

    ###########################################################################
    # Objective (Generator) Method
    ###########################################################################
    def get_objectives(self) -> Generator[ModelObjectiveParameters, None, None]:
        """
        Generates objectives that will be used to create the next
        RunConfig to be profiled

        After profiling completes the RunConfigMeasurement must be passed back
        to the class (via set_last_measurement) so that a score can be calculated
        and Optuna can make a suggestion for the next set of parameters
        """

        if logger.level == logging.logging.DEBUG:
            self._print_debug_search_space_info()

        min_configs_to_search = self._determine_minimum_number_of_configs_to_search()
        max_configs_to_search = self._determine_maximum_number_of_configs_to_search()

        for trial_number in range(1, max_configs_to_search + 1):
            trial = self._study.ask()
            trial_objectives = self._create_trial_objectives(trial)
            logger.debug(f"Trial {trial_number} of {max_configs_to_search}:")

            objective_parameters = self._create_objective_parameters(trial_objectives)

            self._measurement_set = False
            yield objective_parameters

            # A measurement must be set before the next call to the generator is made
            assert self._measurement_set
            score = self._calculate_score()
            self._set_best_measurement(score, trial_number)

            if logger.level == logging.logging.DEBUG:
                self._print_debug_score_info(trial_number, score)

            if self._should_terminate_early(min_configs_to_search, trial_number):
                logger.debug("Early termination threshold reached")
                break
            self._study.tell(trial, score)

    ###########################################################################
    # Measurement Methods
    ###########################################################################
    def set_measurement(self, measurement: Optional[RunConfigMeasurement]):
        """
        After profiling completes you must pass in the results
        """
        self._measurement_set = True
        self._measurement = measurement

    def _set_best_measurement(self, score: float = 0, trial_number: int = 0) -> None:
        if not self._best_trial_number or score > self._best_score:
            self._best_score = score
            self._best_trial_number = trial_number

    def _calculate_score(self) -> float:
        if self._measurement:
            score = self._measurement.get_score(self._measurement)
        else:
            score = OptunaObjectiveGeneratorDefaults.NO_MEASUREMENT_SCORE

        return score

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

        if logger.level == logging.logging.DEBUG:
            logger.info("")

        return max_configs_to_search

    ###########################################################################
    # Trial Objective Methods
    ###########################################################################
    def _create_objective_parameters(
        self, trial_objectives: ModelTrialObjectives
    ) -> ModelObjectiveParameters:
        model_objective_parameters: ModelObjectiveParameters = {}
        for model_name, model_objectives in trial_objectives.items():
            model_objective_parameters[model_name] = {}
            for model_objective, value in model_objectives.items():
                usage = self._model_search_parameters[model_name].get_type(
                    model_objective
                )
                category = self._model_search_parameters[
                    model_name
                ].get_objective_category(model_objective)

                model_objective_parameters[model_name][model_objective] = (
                    ObjectiveParameter(usage, category, value)
                )

        return model_objective_parameters

    def _create_trial_objective_name(
        self, model_name: ModelName, parameter_name: ParameterName
    ) -> ObjectiveName:
        # This ensures that Optuna has a unique name
        # for each objective we are searching
        objective_name = f"{model_name}::{parameter_name}"

        return objective_name

    def _create_trial_objectives(self, trial: optuna.Trial) -> ModelTrialObjectives:
        trial_objectives: ModelTrialObjectives = {}

        for model_name in self._config.model_names:
            trial_objectives[model_name] = {}

            for parameter_name in SearchParameters.all_parameters:
                parameter = self._model_search_parameters[model_name].get_parameter(
                    parameter_name
                )
                if parameter:
                    objective_name = self._create_trial_objective_name(
                        model_name=model_name, parameter_name=parameter_name
                    )

                    trial_objectives[model_name][parameter_name] = (
                        self._create_trial_objective(trial, objective_name, parameter)
                    )

            if self._config.optimize.perf_analyzer.use_concurrency_formula:
                trial_objectives[model_name]["concurrency"] = (
                    self._get_objective_concurrency(trial_objectives[model_name])
                )

        return trial_objectives

    def _create_trial_objective(
        self, trial: optuna.Trial, name: ObjectiveName, parameter: SearchParameter
    ) -> TrialObjective:
        if (
            parameter.category is SearchCategory.INTEGER
            or parameter.category is SearchCategory.EXPONENTIAL
        ):
            objective = trial.suggest_int(
                name, parameter.min_range, parameter.max_range
            )
        elif parameter.category is SearchCategory.INT_LIST:
            objective = int(trial.suggest_categorical(name, parameter.enumerated_list))
        elif parameter.category is SearchCategory.STR_LIST:
            objective = trial.suggest_categorical(name, parameter.enumerated_list)

        return objective

    def _get_objective_concurrency(self, trial_objectives: TrialObjectives) -> int:
        model_batch_size = int(
            trial_objectives.get(
                "model_batch_size", RunConfigDefaults.MIN_MODEL_BATCH_SIZE
            )
        )
        concurrency = int(
            2 * int(trial_objectives["instance_count"]) * model_batch_size**2
        )

        if concurrency > self._config.get_max(
            self._config.optimize.perf_analyzer.concurrency
        ):
            concurrency = self._config.get_max(
                self._config.optimize.perf_analyzer.concurrency
            )
        elif concurrency < self._config.get_min(
            self._config.optimize.perf_analyzer.concurrency
        ):
            concurrency = self._config.get_min(
                self._config.optimize.perf_analyzer.concurrency
            )

        return int(log2(concurrency))

    ###########################################################################
    # Early Termination Methods
    ###########################################################################
    def _should_terminate_early(
        self, min_configs_to_search: int, trial_number: int
    ) -> bool:
        number_of_trials_since_best = trial_number - self._best_trial_number  # type: ignore
        if trial_number < min_configs_to_search:
            should_terminate_early = False
        elif number_of_trials_since_best >= self._config.optimize.early_exit_threshold:
            should_terminate_early = True
        else:
            should_terminate_early = False

        return should_terminate_early

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

    def _print_debug_score_info(
        self,
        trial_number: int,
        score: float,
    ) -> None:
        if score != OptunaObjectiveGeneratorDefaults.NO_MEASUREMENT_SCORE:
            logger.debug(
                f"Objective score for {trial_number}: {int(score * 100)} --- "  # type: ignore
                f"Best: {self._best_trial_number} ({int(self._best_score * 100)})"  # type: ignore
            )

        logger.info("")
