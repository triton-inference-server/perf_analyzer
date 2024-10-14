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
from genai_perf.utils import checkpoint_encoder


class TestGenAIPerfConfig(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._config = ConfigCommand(model_names=["test_model"])

        self._objective_parameters = {
            "test_model": {
                "num_prompts": ObjectiveParameter(
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
        expected_input_config.num_prompts = 50

        self.assertEqual(expected_input_config, self._default_genai_perf_config.input)

        expected_output_tokens_config = ConfigOutputTokens()

        self.assertEqual(
            expected_output_tokens_config, self._default_genai_perf_config.output_tokens
        )

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

        genai_perf_config_from_checkpoint = GenAIPerfConfig.read_from_checkpoint(
            json.loads(genai_perf_config_json)
        )

        self.assertEqual(
            genai_perf_config_from_checkpoint.input,
            self._default_genai_perf_config.input,
        )
        self.assertEqual(
            genai_perf_config_from_checkpoint.output_tokens,
            self._default_genai_perf_config.output_tokens,
        )


if __name__ == "__main__":
    unittest.main()
