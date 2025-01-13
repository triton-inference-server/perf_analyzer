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
from unittest.mock import patch

from genai_perf.checkpoint.checkpoint import checkpoint_encoder
from genai_perf.config.generate.genai_perf_config import GenAIPerfConfig
from genai_perf.config.generate.objective_parameter import (
    ObjectiveCategory,
    ObjectiveParameter,
)
from genai_perf.config.generate.search_parameters import SearchUsage
from genai_perf.config.input.config_command import (
    ConfigCommand,
    ConfigInput,
    ConfigOutputTokens,
)


class TestGenAIPerfConfig(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._config = ConfigCommand(user_config={})
        self._config.model_names = ["test_model"]

        self._objective_parameters = {
            "test_model": {
                "num_dataset_entries": ObjectiveParameter(
                    SearchUsage.RUNTIME_GAP,
                    ObjectiveCategory.INTEGER,
                    50,
                ),
            }
        }

        self._default_genai_perf_config = GenAIPerfConfig(
            config=self._config,
            model_objective_parameters=self._objective_parameters,
        )

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Test Config and Objective Capture
    ###########################################################################
    def test_default_config_and_objective_capture(self):
        """
        Test that we capture the config and objective parameters correctly
        at __init__
        """
        expected_input_config = ConfigInput()
        expected_input_config.num_dataset_entries = 50

        self.assertEqual(expected_input_config, self._default_genai_perf_config._input)

    ###########################################################################
    # Test Representation
    ###########################################################################
    def test_representation(self):
        """
        Test that the representation is created correctly
        """
        expected_representation = " ".join(
            [
                ConfigInput(num_dataset_entries=50).__str__(),
            ]
        )
        representation = self._default_genai_perf_config.representation()

        self.assertEqual(expected_representation, representation)

    ###########################################################################
    # Checkpoint Tests
    ###########################################################################
    def test_checkpoint_methods(self):
        """
        Checks to ensure checkpoint methods work as intended
        """
        genai_perf_config_json = json.dumps(
            self._default_genai_perf_config, default=checkpoint_encoder
        )

        genai_perf_config_from_checkpoint = (
            GenAIPerfConfig.create_class_from_checkpoint(
                json.loads(genai_perf_config_json)
            )
        )

        actual_input = genai_perf_config_from_checkpoint._input.__str__()
        expected_input = self._default_genai_perf_config._input.__str__()

        self.assertEqual(
            genai_perf_config_from_checkpoint._input,
            self._default_genai_perf_config._input,
        )


if __name__ == "__main__":
    unittest.main()
