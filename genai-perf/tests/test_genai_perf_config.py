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
from genai_perf.config.input.config_command import ConfigCommand


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
    # Test Parameters
    ###########################################################################
    def test_parameters(self):
        """
        Test that we capture the config and objective parameters correctly
        at __init__
        """
        expected_parameters = {"num_dataset_entries": 50}

        expected_parameters["endpoint"] = self._config.endpoint.to_json_dict()
        del expected_parameters["endpoint"]["server_metrics_urls"]
        del expected_parameters["endpoint"]["url"]

        expected_parameters["input"] = self._config.input.to_json_dict()
        expected_parameters["tokenizer"] = self._config.tokenizer.to_json_dict()

        actual_parameters = self._default_genai_perf_config.get_parameters()
        for key, value in expected_parameters.items():
            self.assertEqual(value, actual_parameters[key])

    ###########################################################################
    # Test Representation
    ###########################################################################
    def test_representation(self):
        """
        Test that the representation is created correctly
        """
        expected_parameters = {}
        expected_parameters["endpoint"] = self._config.endpoint.to_json_dict()
        del expected_parameters["endpoint"]["server_metrics_urls"]
        del expected_parameters["endpoint"]["url"]

        expected_parameters["input"] = self._config.input.to_json_dict()
        expected_parameters["tokenizer"] = self._config.tokenizer.to_json_dict()

        expected_parameters = {"num_dataset_entries": 50}

        representation = self._default_genai_perf_config.representation()

        for key, value in expected_parameters.items():
            self.assertIn(key, representation)
            self.assertIn(value.__str__(), representation)

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

        self.assertEqual(
            genai_perf_config_from_checkpoint,
            self._default_genai_perf_config,
        )


if __name__ == "__main__":
    unittest.main()
