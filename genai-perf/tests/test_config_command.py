# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Skip type checking to avoid mypy error
# Issue: https://github.com/python/mypy/issues/10632
import yaml  # type: ignore
from genai_perf.config.input.config_command import ConfigCommand, Range
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat


class TestConfigCommand(unittest.TestCase):

    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        pass

    def tearDown(self):
        patch.stopall()

    def test_model_name_only(self):
        """
        Test that a configuration with only a model name is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.model_names, ["gpt2"])

    def test_analyze_subcommand(self):
        """
        Test that the analyze subcommand is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            analyze:
                concurrency:
                    start: 4
                    stop: 32

                num_dataset_entries:
                    start: 1
                    stop: 5

                input_sequence_length:
                    start: 100
                    stop: 200
                    step: 20
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.analyze.sweep_parameters["concurrency"], Range(4, 32))
        self.assertEqual(
            config.analyze.sweep_parameters["num_dataset_entries"], Range(1, 5)
        )
        self.assertEqual(
            config.analyze.sweep_parameters["input_sequence_length"],
            [100, 120, 140, 160, 180, 200],
        )

    def test_endpoint(self):
        """
        Test that the endpoint configuration is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            endpoint:
                model_selection_strategy: random
                backend: vllm
                custom: custom_endpoint
                type: endpoint_type
                service_kind: test_service
                streaming: True
                server_metrics_url: "test_server_metrics_url"
                url: "test_url"
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(
            config.endpoint.model_selection_strategy, ModelSelectionStrategy.RANDOM
        )
        self.assertEqual(config.endpoint.backend, OutputFormat.VLLM)
        self.assertEqual(config.endpoint.custom, "custom_endpoint")
        self.assertEqual(config.endpoint.type, "endpoint_type")
        self.assertEqual(config.endpoint.service_kind, "test_service")
        self.assertEqual(config.endpoint.streaming, True)
        self.assertEqual(config.endpoint.server_metrics_url, "test_server_metrics_url")
        self.assertEqual(config.endpoint.url, "test_url")


if __name__ == "__main__":
    unittest.main()
