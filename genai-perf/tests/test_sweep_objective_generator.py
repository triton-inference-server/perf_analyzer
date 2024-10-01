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

import unittest
from math import log2
from unittest.mock import patch

from genai_perf.config.generate.search_parameters import SearchParameters
from genai_perf.config.generate.sweep_objective_generator import SweepObjectiveGenerator
from genai_perf.config.input.config_command import ConfigCommand, RunConfigDefaults


class TestSweepObjectiveGenerator(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._config = ConfigCommand(model_names=["test_modelA", "test_modelB"])
        self._model_search_parameters = {
            "test_modelA": SearchParameters(self._config),
            "test_modelB": SearchParameters(self._config),
        }

        self._sweep_obj_gen = SweepObjectiveGenerator(
            self._config,
            self._model_search_parameters,
        )

        self._expected_model_search_parameter_combination_count = len(
            range(
                int(log2(RunConfigDefaults.MIN_MODEL_BATCH_SIZE)),
                int(log2(RunConfigDefaults.MAX_MODEL_BATCH_SIZE)) + 1,
            )
        ) * len(
            range(
                RunConfigDefaults.MIN_INSTANCE_COUNT,
                RunConfigDefaults.MAX_INSTANCE_COUNT + 1,
            )
        )

        self._expected_all_search_parameter_combination_count = (
            self._expected_model_search_parameter_combination_count
            * self._expected_model_search_parameter_combination_count
        )

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    #  Objective List Testing
    ###########################################################################
    def test_list_of_per_model_search_parameter_combinations(self):
        """
        Test that we can properly generate the list of per model
        search parameter combinations
        """
        model_search_parameter_combinations = (
            self._sweep_obj_gen._create_list_of_model_search_parameter_combinations(
                "test_modelA"
            )
        )

        self.assertEqual(
            self._expected_model_search_parameter_combination_count,
            len(model_search_parameter_combinations),
        )

    def test_list_of_all_search_parameter_combinations(self):
        """
        Test that we can properly generate the list of all
        search parameter combinations
        """
        model_search_parameter_combinations = {}
        model_search_parameter_combinations["test_modelA"] = (
            self._sweep_obj_gen._create_list_of_model_search_parameter_combinations(
                "test_modelA"
            )
        )
        model_search_parameter_combinations["test_modelB"] = (
            self._sweep_obj_gen._create_list_of_model_search_parameter_combinations(
                "test_modelB"
            )
        )

        all_search_parameter_combinations = (
            self._sweep_obj_gen._create_list_of_all_search_parameter_combinations(
                model_search_parameter_combinations
            )
        )

        self.assertEqual(
            self._expected_all_search_parameter_combination_count,
            len(all_search_parameter_combinations),
        )

    ###########################################################################
    #  Generator Testing
    ###########################################################################
    def test_objectives_generator(self):
        """
        Test that the objectives generator is working (end-to-end)
        """

        count = len(list(self._sweep_obj_gen.get_objectives()))
        self.assertEqual(self._expected_all_search_parameter_combination_count, count)

    if __name__ == "__main__":
        unittest.main()
