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

import json
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from genai_perf.config.generate.optuna_objective_generator import (
    OptunaObjectiveGenerator,
)
from genai_perf.config.generate.search_parameters import SearchParameters
from genai_perf.config.input.config_command import ConfigCommand
from tests.test_utils import (
    create_model_config_measurement,
    create_run_config_measurement,
)


class TestModelConfigMeasurement(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._config = ConfigCommand(model_names=["test_model"])
        self._model_search_parameters = {"test_model": SearchParameters(self._config)}

        self._baseline_mcm = create_model_config_measurement(
            throughput=1000, latency=50
        )
        self._baseline_rcm = create_run_config_measurement(
            gpu_power=80, gpu_utilization=70
        )
        self._baseline_rcm.add_model_config_measurement(
            "test_model", self._baseline_mcm
        )

        self._optuna_obj_gen = OptunaObjectiveGenerator(
            self._config,
            self._model_search_parameters,
            self._baseline_rcm,
            user_seed=100,
        )

        self._num_configs_in_search_space = (
            self._optuna_obj_gen._calculate_num_of_configs_in_search_space()
        )

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Minimum Search Space Testing
    ###########################################################################
    def test_min_number_of_configs_to_search_percentage(self):
        """
        Test percentage based min num of configs to search
        """
        min_configs_to_search = (
            self._optuna_obj_gen._determine_minimum_number_of_configs_to_search()
        )
        expected_min_configs = int(
            (
                self._num_configs_in_search_space
                * self._config.optimize.search_space_percentage.min
            )
            / 100
        )

        self.assertEqual(expected_min_configs, min_configs_to_search)

    @patch(
        "genai_perf.config.input.config_command.ConfigOptimize.is_set_by_user",
        MagicMock(return_value=True),
    )
    def test_min_number_of_configs_to_search_count(self):
        """
        Test count based min num of configs to search
        """
        optuna_obj_gen = deepcopy(self._optuna_obj_gen)
        optuna_obj_gen._config.optimize.number_of_trials.min = 12

        min_configs_to_search = (
            optuna_obj_gen._determine_minimum_number_of_configs_to_search()
        )

        self.assertEqual(12, min_configs_to_search)

    ###########################################################################
    # Maximum Search Space Testing
    ###########################################################################
    def test_max_number_of_configs_to_search_percentage(self):
        """
        Test percentage based max num of configs to search
        """
        max_configs_to_search = (
            self._optuna_obj_gen._determine_maximum_number_of_configs_to_search()
        )
        expected_max_configs = int(
            (
                self._num_configs_in_search_space
                * self._config.optimize.search_space_percentage.max
            )
            / 100
        )

        self.assertEqual(expected_max_configs, max_configs_to_search)

    @patch(
        "genai_perf.config.input.config_command.ConfigOptimize.is_set_by_user",
        MagicMock(return_value=True),
    )
    def test_max_number_of_configs_to_search_count(self):
        """
        Test count based max num of configs to search
        """
        optuna_obj_gen = deepcopy(self._optuna_obj_gen)
        optuna_obj_gen._config.optimize.number_of_trials.max = 30
        optuna_obj_gen._config.optimize.search_space_percentage.max = 100

        max_configs_to_search = (
            optuna_obj_gen._determine_maximum_number_of_configs_to_search()
        )

        self.assertEqual(30, max_configs_to_search)


if __name__ == "__main__":
    unittest.main()
