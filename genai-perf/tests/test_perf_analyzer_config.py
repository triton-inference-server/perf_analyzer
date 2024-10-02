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
from copy import deepcopy
from unittest.mock import MagicMock, patch

from genai_perf.config.generate.objective_parameter import (
    ObjectiveCategory,
    ObjectiveParameter,
)
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.generate.search_parameters import SearchParameters, SearchUsage
from genai_perf.config.input.config_command import ConfigCommand, ConfigPerfAnalyzer


class TestPerfAnalyzerConfig(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._config = ConfigCommand(model_names=["test_model"])

        self._objective_parameters = {
            "test_model": {
                "model_batch_size": ObjectiveParameter(
                    SearchUsage.MODEL, ObjectiveCategory.EXPONENTIAL, 4
                ),
                "runtime_batch_size": ObjectiveParameter(
                    SearchUsage.RUNTIME, ObjectiveCategory.INTEGER, 1
                ),
                "instance_count": ObjectiveParameter(
                    SearchUsage.MODEL, ObjectiveCategory.INTEGER, 2
                ),
                "concurrency": ObjectiveParameter(
                    SearchUsage.RUNTIME, ObjectiveCategory.EXPONENTIAL, 6
                ),
            }
        }

        self._default_perf_analyzer_config = PerfAnalyzerConfig(
            config=self._config,
            model_objective_parameters=self._objective_parameters,
            model_name="test_model",
        )

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Test Config and Objective Capture
    ###########################################################################
    def test_config_and_objective_capture(self):
        """
        Test that we capture the config and objective parameters correctly
        at __init__
        """
        expected_config_options = ConfigPerfAnalyzer()
        expected_parameters = {"runtime_batch_size": 1, "concurrency": 64}

        self.assertEqual("test_model", self._default_perf_analyzer_config._model_name)
        self.assertEqual(
            expected_config_options, self._default_perf_analyzer_config._config
        )
        self.assertEqual(
            expected_parameters, self._default_perf_analyzer_config._parameters
        )

    ###########################################################################
    # Test CLI String Creation
    ###########################################################################
    def test_default_cli_string_creation(self):
        """
        Test that the default CLI string is created correctly
        """
        expected_cli_string = " ".join(
            [
                self._config.perf_analzyer.path,
                "--model-name",
                "test_model",
                "--stability-percentage",
                str(self._config.perf_analzyer.stability_threshold),
                "--batch-size",
                "1",
                "--concurrency-range",
                "64",
            ]
        )
        cli_string = self._default_perf_analyzer_config.create_cli_string()

        self.assertEqual(expected_cli_string, cli_string)

    ###########################################################################
    # Test Representation
    ###########################################################################
    def test_default_representation(self):
        """
        Test that the representation is created correctly in the default case
        """
        expected_representation = " ".join(
            [
                "--model-name",
                "test_model",
                "--stability-percentage",
                str(self._config.perf_analzyer.stability_threshold),
                "--batch-size",
                "1",
                "--concurrency-range",
                "64",
            ]
        )
        representation = self._default_perf_analyzer_config.representation()

        self.assertEqual(expected_representation, representation)

    @patch(
        "genai_perf.config.generate.perf_analyzer_config.PerfAnalyzerConfig.create_cli_string",
        MagicMock(
            return_value=" ".join(
                [
                    "perf_analyzer",
                    "--url",
                    "url_string",
                    "--metrics-url",
                    "url_string",
                    "--latency-report-file",
                    "file_string",
                    "--measurement-request-count",
                    "mrc_string",
                    "--verbose",
                    "--extra-verbose",
                    "--verbose-csv",
                ]
            )
        ),
    )
    def test_with_removal_representation(self):
        """
        Test that the representation is created correctly when every
        possible value that should be removed is added
        """
        representation = self._default_perf_analyzer_config.representation()

        expected_representation = ""
        self.assertEqual(expected_representation, representation)


if __name__ == "__main__":
    unittest.main()
