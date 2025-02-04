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

from genai_perf import parser
from genai_perf.checkpoint.checkpoint import checkpoint_encoder
from genai_perf.config.generate.objective_parameter import (
    ObjectiveCategory,
    ObjectiveParameter,
)
from genai_perf.config.generate.perf_analyzer_config import (
    InferenceType,
    PerfAnalyzerConfig,
)
from genai_perf.config.generate.search_parameters import SearchUsage
from genai_perf.config.input.config_command import ConfigCommand, ConfigPerfAnalyzer


class TestPerfAnalyzerConfig(unittest.TestCase):

    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._config = ConfigCommand({"model_name": "test_model"})

        self._objective_parameters = {
            "test_model": {
                "model_batch_size": ObjectiveParameter(
                    SearchUsage.MODEL, ObjectiveCategory.EXPONENTIAL, 4
                ),
                "runtime_batch_size": ObjectiveParameter(
                    SearchUsage.RUNTIME_PA, ObjectiveCategory.INTEGER, 1
                ),
                "instance_count": ObjectiveParameter(
                    SearchUsage.MODEL, ObjectiveCategory.INTEGER, 2
                ),
                "concurrency": ObjectiveParameter(
                    SearchUsage.RUNTIME_PA, ObjectiveCategory.EXPONENTIAL, 6
                ),
            }
        }

        self._default_perf_analyzer_config = PerfAnalyzerConfig(
            config=self._config,
            model_objective_parameters=self._objective_parameters,
        )

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Test Config and Objective Capture
    ###########################################################################
    def test_objective_capture(self):
        """
        Test that we capture the objective parameters correctly
        at __init__
        """
        expected_parameters = {"runtime_batch_size": 1, "concurrency": 64}

        self.assertEqual(
            expected_parameters, self._default_perf_analyzer_config._parameters
        )

    ###########################################################################
    # Test Command Creation
    ###########################################################################
    def test_default_command_creation(self):
        """
        Test that the default CLI string is created correctly
        """
        expected_command = [
            self._config.perf_analyzer.path,
            "-m",
            "test_model",
            "--async",
            "-i",
            "grpc",
            "--streaming",
            "-u",
            "localhost:8001",
            "--shape",
            "max_tokens:1",
            "--shape",
            "text_input:1",
            "--service-kind",
            "triton",
            "--measurement-interval",
            "10000",
            "--stability-percentage",
            "999",
            "--endpoint",
            "tensorrtllm",
            "--input-data",
            "artifacts/test_model-triton-tensorrtllm-runtime_batch_size1-concurrency64/inputs.json",
            "--profile-export-file",
            "artifacts/test_model-triton-tensorrtllm-runtime_batch_size1-concurrency64/profile_export.json",
            "-b",
            "1",
            "--concurrency-range",
            "64",
        ]
        command = self._default_perf_analyzer_config.create_command()

        for field in expected_command:
            self.assertIn(field, command)

        self.assertEqual(len(expected_command), len(command))

    ###########################################################################
    # Test Representation
    ###########################################################################
    def test_default_representation(self):
        """
        Test that the representation is created correctly in the default case
        """
        expected_representation = " ".join(
            [
                "-m",
                "test_model",
                "--async",
                "--stability-percentage",
                "999",
                "--measurement-interval",
                "10000",
                "--streaming",
                "--shape",
                "max_tokens:1",
                "--shape",
                "text_input:1",
                "--concurrency-range",
                "64",
                "--service-kind",
                "triton",
                "--endpoint",
                "tensorrtllm",
                "-b",
                "1",
            ]
        )
        representation = self._default_perf_analyzer_config.representation()

        for field in expected_representation:
            self.assertIn(field, representation)

        self.assertEqual(len(expected_representation), len(representation))

    ###########################################################################
    # Test Inference Methods
    ###########################################################################
    def test_get_inference_type(self):
        infer_type = self._default_perf_analyzer_config.get_inference_type()
        expected_infer_type = InferenceType.CONCURRENCY

        self.assertEqual(expected_infer_type, infer_type)

    def test_get_inference_value(self):
        infer_value = self._default_perf_analyzer_config.get_inference_value()
        expected_infer_value = 64

        self.assertEqual(expected_infer_value, infer_value)

    ###########################################################################
    # Checkpoint Tests
    ###########################################################################
    def test_checkpoint_methods(self):
        """
        Checks to ensure checkpoint methods work as intended
        """
        pa_config_json = json.dumps(
            self._default_perf_analyzer_config, default=checkpoint_encoder
        )

        pa_config_from_checkpoint = PerfAnalyzerConfig.create_class_from_checkpoint(
            json.loads(pa_config_json)
        )

        self.assertEqual(
            pa_config_from_checkpoint._model_name,
            self._default_perf_analyzer_config._model_name,
        )
        self.assertEqual(
            pa_config_from_checkpoint._parameters,
            self._default_perf_analyzer_config._parameters,
        )
        self.assertEqual(
            pa_config_from_checkpoint._cli_args,
            self._default_perf_analyzer_config._cli_args,
        )

        # Catchall in case something new is added
        self.assertEqual(pa_config_from_checkpoint, self._default_perf_analyzer_config)


if __name__ == "__main__":
    unittest.main()
