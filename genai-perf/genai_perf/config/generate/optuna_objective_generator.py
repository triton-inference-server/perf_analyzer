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

import optuna
from genai_perf.config.generate.objective_parameter import ObjectiveParameters
from genai_perf.config.generate.search_parameters import SearchParameters
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.measurements.run_config_measurement import RunConfigMeasurement


@dataclass(frozen=True)
class OptunaObjectiveGeneratorDefaults:
    BASELINE_SCORE = 0


class OptunaObjectiveGenerator:
    """
    Generates the next set of objectives to try using Optuna's hyperparameter
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
        search_parameters: SearchParameters,
        baseline_measurement: RunConfigMeasurement,
        user_seed: Optional[int],
    ):
        self._config = config
        self._search_parameters = search_parameters
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

    # def _calculate_num_of_configs_in_search_space(self) -> int:
    #     NotImplemented
