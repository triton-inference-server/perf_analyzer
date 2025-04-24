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
from pathlib import Path
from unittest.mock import patch

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
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.inputs.input_constants import (
    OutputFormat,
    PerfAnalyzerMeasurementMode,
    PromptSource,
)


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
    def test_command_creation(self):
        """
        Test that the CLI string is created correctly
        """
        expected_command = {
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
            "--stability-percentage",
            "999",
            "--request-count",
            "128",
            "--warmup-request-count",
            "0",
            "--input-data",
            "artifacts/test_model-triton-tensorrtllm-runtime_batch_size1-concurrency64/inputs.json",
            "--profile-export-file",
            "artifacts/test_model-triton-tensorrtllm-runtime_batch_size1-concurrency64/profile_export.json",
            "-b",
            "1",
            "--concurrency-range",
            "64",
        }
        actual_command = set(self._default_perf_analyzer_config.create_command())

        for field in expected_command:
            self.assertIn(field, actual_command)

        self.assertEqual(len(expected_command), len(actual_command))

    ###########################################################################
    # Test Representation
    ###########################################################################
    def test_representation(self):
        """
        Test that the representation is created correctly
        """
        expected_representation = " ".join(
            [
                "-m",
                "test_model",
                "--async",
                "--stability-percentage",
                "999",
                "--request-count",
                "128",
                "--warmup-request-count",
                "0",
                "--streaming",
                "--shape",
                "max_tokens:1",
                "--shape",
                "text_input:1",
                "--concurrency-range",
                "64",
                "--backend",
                "tensorrtllm",
                "-b",
                "1",
            ]
        )
        representation = self._default_perf_analyzer_config.representation()
        actual_representation = set(representation.split())

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

    ###########################################################################
    # Test Multi-model artifact paths
    ###########################################################################
    def test_multi_model_artifact_paths(self):
        """
        Test that the artifact paths are created correctly
        for multi-model scenarios
        """
        config = ConfigCommand({"model_names": "test_model_A,test_model_B"})

        perf_analyzer_config = PerfAnalyzerConfig(
            config=config, model_objective_parameters={}
        )

        artifact_directory = perf_analyzer_config.get_artifact_directory()
        expected_artifact_directory = Path(
            "artifacts/test_model_A_multi-triton-tensorrtllm-concurrency1"
        )
        self.assertEqual(expected_artifact_directory, artifact_directory)

    ###########################################################################
    # Test _add_required_args
    ###########################################################################
    def test_add_required_args_with_dynamic_grpc(self):
        """
        Test that _add_required_args returns the correct arguments
        when service_kind is 'dynamic_grpc'
        """
        self._config.endpoint.service_kind = "dynamic_grpc"
        expected_args = [f"{self._config.perf_analyzer.path}"]

        actual_args = self._default_perf_analyzer_config._add_required_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    def test_add_required_args_with_non_dynamic_grpc(self):
        """
        Test that _add_required_args returns the correct arguments
        when service_kind is not 'dynamic_grpc'
        """
        self._config.endpoint.service_kind = "triton"
        expected_args = [
            f"{self._config.perf_analyzer.path}",
            "-m",
            "test_model",
            "--async",
        ]

        actual_args = self._default_perf_analyzer_config._add_required_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    ###########################################################################
    # Test _add_perf_analyzer_args
    ###########################################################################
    def test_add_perf_analyzer_args_with_payload_prompt_source(self):
        """
        Test that _add_perf_analyzer_args returns an empty list
        when prompt_source is PAYLOAD
        """
        self._config.input.prompt_source = PromptSource.PAYLOAD
        expected_args = []

        actual_args = self._default_perf_analyzer_config._add_perf_analyzer_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    def test_add_perf_analyzer_args_with_non_payload_prompt_source(self):
        """
        Test that _add_perf_analyzer_args returns the correct arguments
        when prompt_source is not PAYLOAD
        """
        self._config.input.prompt_source = PromptSource.FILE
        self._config.perf_analyzer.stability_percentage = 567
        self._config.perf_analyzer.measurement.mode = (
            PerfAnalyzerMeasurementMode.INTERVAL
        )
        self._config.perf_analyzer.measurement.num = 5000
        self._config.perf_analyzer.warmup_request_count = 30

        expected_args = [
            "--stability-percentage",
            "567",
            "--warmup-request-count",
            "30",
            "--measurement-interval",
            "5000",
        ]

        actual_args = self._default_perf_analyzer_config._add_perf_analyzer_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    ###########################################################################
    # Test _add_protocol_args
    ###########################################################################
    def test_add_protocol_args_with_triton(self):
        """
        Test that _add_protocol_args returns the correct arguments
        when service_kind is 'triton'
        """
        self._config.endpoint.service_kind = "triton"
        self._config.endpoint.backend = OutputFormat.TENSORRTLLM
        self._config.endpoint.get_field("url").is_set_by_user = False

        expected_args = [
            "-i",
            "grpc",
            "--streaming",
            "--shape",
            "max_tokens:1",
            "--shape",
            "text_input:1",
        ]

        actual_args = self._default_perf_analyzer_config._add_protocol_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    def test_add_protocol_args_with_openai(self):
        """
        Test that _add_protocol_args returns the correct arguments
        when service_kind is 'openai'
        """
        self._config.endpoint.service_kind = "openai"

        expected_args = ["-i", "http"]

        actual_args = self._default_perf_analyzer_config._add_protocol_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    def test_add_protocol_args_with_tensorrtllm_engine(self):
        """
        Test that _add_protocol_args returns the correct arguments
        when service_kind is 'tensorrtllm_engine'
        """
        self._config.endpoint.service_kind = "tensorrtllm_engine"

        expected_args = ["--service-kind", "triton_c_api", "--streaming"]

        actual_args = self._default_perf_analyzer_config._add_protocol_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    ###########################################################################
    # Test _add_inference_load_args
    ###########################################################################
    def test_add_inference_load_args_with_no_parameters_and_set_stimulus(self):
        """
        Test that _add_inference_load_args returns the correct arguments
        when there are no parameters and stimulus is set by user
        """
        self._default_perf_analyzer_config._parameters = {}
        self._config.perf_analyzer.get_field("stimulus").is_set_by_user = True
        self._config.perf_analyzer.stimulus = {"concurrency": 10}

        expected_args = ["--concurrency-range", "10"]

        actual_args = self._default_perf_analyzer_config._add_inference_load_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    def test_add_inference_load_args_with_no_parameters_and_request_rate(self):
        """
        Test that _add_inference_load_args returns the correct arguments
        when there are no parameters and request_rate is set by user
        """
        self._default_perf_analyzer_config._parameters = {}
        self._config.perf_analyzer.get_field("stimulus").is_set_by_user = True
        self._config.perf_analyzer.stimulus = {"request_rate": 5}

        expected_args = ["--request-rate-range", "5"]

        actual_args = self._default_perf_analyzer_config._add_inference_load_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    def test_add_inference_load_args_with_no_parameters_and_session_concurrency(self):
        """
        Test that _add_inference_load_args returns the correct arguments
        when there are no parameters and session_concurrency is set by user
        """
        self._default_perf_analyzer_config._parameters = {}
        self._config.perf_analyzer.get_field("stimulus").is_set_by_user = True
        self._config.perf_analyzer.stimulus = {"session_concurrency": 3}

        expected_args = ["--session-concurrency", "3"]

        actual_args = self._default_perf_analyzer_config._add_inference_load_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    def test_add_inference_load_args_with_parameters(self):
        """
        Test that _add_inference_load_args returns the correct arguments
        when parameters are set
        """
        self._default_perf_analyzer_config._parameters = {
            "concurrency": 10,
            "request_rate": 5,
            "runtime_batch_size": 2,
        }

        expected_args = [
            "--concurrency-range",
            "10",
            "--request-rate-range",
            "5",
            "-b",
            "2",
        ]

        actual_args = self._default_perf_analyzer_config._add_inference_load_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    ###########################################################################
    # Test _add_prompt_source_args
    ###########################################################################
    def test_add_prompt_source_args_with_payload_and_no_session_concurrency(self):
        """
        Test that _add_prompt_source_args returns the correct arguments
        when prompt_source is PAYLOAD and session_concurrency is not in stimulus
        """
        self._config.input.prompt_source = PromptSource.PAYLOAD
        self._config.perf_analyzer.stimulus = {}

        expected_args = ["--fixed-schedule"]

        actual_args = self._default_perf_analyzer_config._add_prompt_source_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    def test_add_prompt_source_args_with_payload_and_session_concurrency(self):
        """
        Test that _add_prompt_source_args returns an empty list
        when prompt_source is PAYLOAD and session_concurrency is in stimulus
        """
        self._config.input.prompt_source = PromptSource.PAYLOAD
        self._config.perf_analyzer.stimulus = {"session_concurrency": 3}

        expected_args = []

        actual_args = self._default_perf_analyzer_config._add_prompt_source_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    def test_add_prompt_source_args_with_non_payload(self):
        """
        Test that _add_prompt_source_args returns an empty list
        when prompt_source is not PAYLOAD
        """
        self._config.input.prompt_source = PromptSource.FILE

        expected_args = []

        actual_args = self._default_perf_analyzer_config._add_prompt_source_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    ###########################################################################
    # Test _add_endpoint_args
    ###########################################################################
    def test_add_endpoint_args_with_custom_endpoint(self):
        """
        Test that _add_endpoint_args returns the correct arguments
        when custom endpoint is set
        """
        self._config.endpoint.custom = "custom_endpoint"

        expected_args = ["--service-kind", "triton", "--endpoint", "custom_endpoint"]

        actual_args = self._default_perf_analyzer_config._add_endpoint_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    def test_add_endpoint_args_without_custom_endpoint(self):
        """
        Test that _add_endpoint_args returns the correct arguments
        when custom endpoint is not set
        """
        self._config.endpoint.custom = None

        expected_args = ["--service-kind", "triton"]

        actual_args = self._default_perf_analyzer_config._add_endpoint_args(
            self._config
        )

        self.assertEqual(expected_args, actual_args)

    ###########################################################################
    # Test _add_url_args
    ###########################################################################
    def test_add_url_args_with_user_set_url(self):
        """
        Test that _add_url_args returns the correct arguments
        when the URL is set by the user
        """
        self._config.endpoint.get_field("url").is_set_by_user = True
        self._config.endpoint.url = "http://custom-url:8000"

        expected_args = ["-u", "http://custom-url:8000"]

        actual_args = self._default_perf_analyzer_config._add_url_args(self._config)

        self.assertEqual(expected_args, actual_args)

    def test_add_url_args_with_triton_service_kind(self):
        """
        Test that _add_url_args returns the correct arguments
        when service_kind is 'triton'
        """
        self._config.endpoint.get_field("url").is_set_by_user = False
        self._config.endpoint.service_kind = "triton"
        self._config.endpoint.url = "http://triton-url:8001"

        expected_args = ["-u", "http://triton-url:8001"]

        actual_args = self._default_perf_analyzer_config._add_url_args(self._config)

        self.assertEqual(expected_args, actual_args)

    def test_add_url_args_with_dynamic_grpc_service_kind(self):
        """
        Test that _add_url_args returns the correct arguments
        when service_kind is 'dynamic_grpc'
        """
        self._config.endpoint.get_field("url").is_set_by_user = False
        self._config.endpoint.service_kind = "dynamic_grpc"
        self._config.endpoint.url = "http://dynamic-grpc-url:9000"

        expected_args = ["-u", "http://dynamic-grpc-url:9000"]

        actual_args = self._default_perf_analyzer_config._add_url_args(self._config)

        self.assertEqual(expected_args, actual_args)

    def test_add_url_args_with_no_url_and_openai_service_kind(self):
        """
        Test that _add_url_args returns an empty list
        when the URL is not set and service_kind is 'openai'
        """
        self._config.endpoint.service_kind = "openai"

        expected_args = []

        actual_args = self._default_perf_analyzer_config._add_url_args(self._config)

        self.assertEqual(expected_args, actual_args)


if __name__ == "__main__":
    unittest.main()
