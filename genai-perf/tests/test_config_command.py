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

    def test_model_name(self):
        """
        Test that a configuration with a model name is parsed correctly
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

    def test_perf_analyzer(self):
        """
        Test that the perf analyzer configuration is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            perf_analyzer:
                path: test_path
                stimulus:
                    test_stimulus: 1
                stability_percentage: 500
                measurement_interval: 1000


            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.perf_analyzer.path, "test_path")
        self.assertEqual(config.perf_analyzer.stimulus, {"test_stimulus": 1})
        self.assertEqual(config.perf_analyzer.stability_percentage, 500)
        self.assertEqual(config.perf_analyzer.measurement_interval, 1000)

    def test_input(self):
        """
        Test that the input configuration is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            input:
                batch_size: 4
                extra:
                    test_extra: 1
                goodput:
                    test_goodput : 2
                header:
                    test_header: 3
                file:
                    test_file: 4
                num_dataset_entries: 50
                random_seed: 100

                image:
                    batch_size: 8
                    width_mean: 9
                    width_stddev: 10
                    height_mean: 11
                    height_stddev: 12
                    format: "test_format"

                output_tokens:
                    mean: 13
                    deterministic: True
                    stddev: 14

                synthetic_tokens:
                    mean: 15
                    stddev: 16

                prefix_prompt:
                    num: 17
                    length: 18

                request_count:
                    num: 19
                    warmup: 20
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.input.batch_size, 4)
        self.assertEqual(config.input.extra, {"test_extra": 1})
        self.assertEqual(config.input.goodput, {"test_goodput": 2})
        self.assertEqual(config.input.header, {"test_header": 3})
        self.assertEqual(config.input.file, {"test_file": 4})
        self.assertEqual(config.input.num_dataset_entries, 50)
        self.assertEqual(config.input.random_seed, 100)
        self.assertEqual(config.input.image.batch_size, 8)
        self.assertEqual(config.input.image.width_mean, 9)
        self.assertEqual(config.input.image.width_stddev, 10)
        self.assertEqual(config.input.image.height_mean, 11)
        self.assertEqual(config.input.image.height_stddev, 12)
        self.assertEqual(config.input.image.format, "test_format")
        self.assertEqual(config.input.output_tokens.mean, 13)
        self.assertEqual(config.input.output_tokens.deterministic, True)
        self.assertEqual(config.input.output_tokens.stddev, 14)
        self.assertEqual(config.input.synthetic_tokens.mean, 15)
        self.assertEqual(config.input.synthetic_tokens.stddev, 16)
        self.assertEqual(config.input.prefix_prompt.num, 17)
        self.assertEqual(config.input.prefix_prompt.length, 18)
        self.assertEqual(config.input.request_count.num, 19)
        self.assertEqual(config.input.request_count.warmup, 20)

    def test_output(self):
        """
        Test that the output configuration is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            output:
                artifact_directory: "test_artifact_directory"
                checkpoint_directory: "test_checkpoint_directory"
                profile_export_file: "test_profile_export_file"
                generate_plots: True
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.output.artifact_directory, "test_artifact_directory")
        self.assertEqual(
            config.output.checkpoint_directory, "test_checkpoint_directory"
        )
        self.assertEqual(config.output.profile_export_file, "test_profile_export_file")
        self.assertEqual(config.output.generate_plots, True)

    def test_tokenizer(self):
        """
        Test that the tokenizer configuration is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            tokenizer:
                name: "test_name"
                revision: "test_revision"
                trust_remote_code: True
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.tokenizer.name, "test_name")
        self.assertEqual(config.tokenizer.revision, "test_revision")
        self.assertEqual(config.tokenizer.trust_remote_code, True)


if __name__ == "__main__":
    unittest.main()
