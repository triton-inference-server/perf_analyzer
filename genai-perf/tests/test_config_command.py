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

import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

# Skip type checking to avoid mypy error
# Issue: https://github.com/python/mypy/issues/10632
import yaml  # type: ignore
from genai_perf.config.input.config_command import ConfigCommand, ConfigInput, Range
from genai_perf.config.input.config_defaults import PerfAnalyzerDefaults
from genai_perf.inputs.input_constants import (
    ModelSelectionStrategy,
    OutputFormat,
    PerfAnalyzerMeasurementMode,
    PromptSource,
)
from genai_perf.inputs.retrievers.synthetic_audio_generator import AudioFormat
from genai_perf.inputs.retrievers.synthetic_image_generator import ImageFormat


class TestConfigCommand(unittest.TestCase):

    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        pass

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Test Model Name
    ###########################################################################
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
        self.assertEqual(config.get_field("model_names").required, True)

    def test_multi_model_name_string(self):
        """
        Test that a configuration with a string of multiple model names is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: "gpt2,bert"
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.model_names, ["gpt2_multi"])

    def test_multi_model_name_list(self):
        """
        Test that a configuration with a list of multiple model names is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_names:
                - gpt2
                - bert
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.model_names, ["gpt2_multi"])

    ###########################################################################
    # Test Analyze Subcommand
    ###########################################################################
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

    ###########################################################################
    # Test Endpoint Configuration
    ###########################################################################
    def test_endpoint_config(self):
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
                type: kserve
                service_kind: triton
                streaming: True
                server_metrics_url: "http://test_server_metrics_url:8002/metrics"
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
        self.assertEqual(config.endpoint.type, "kserve")
        self.assertEqual(config.endpoint.service_kind, "triton")
        self.assertEqual(config.endpoint.streaming, True)
        self.assertEqual(
            config.endpoint.server_metrics_urls,
            ["http://test_server_metrics_url:8002/metrics"],
        )
        self.assertEqual(config.endpoint.url, "test_url")

    ###########################################################################
    # Test Perf Analyzer Config
    ###########################################################################
    def test_perf_analyzer_config(self):
        """
        Test that the perf analyzer configuration is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            perf_analyzer:
                path: test_path
                verbose: True
                stimulus:
                    concurrency: 64
                stability_percentage: 500
                warmup_request_count: 200

                measurement:
                  mode: request_count
                  num: 100
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.perf_analyzer.path, "test_path")
        self.assertEqual(config.perf_analyzer.verbose, True)
        self.assertEqual(config.perf_analyzer.stimulus, {"concurrency": 64})
        self.assertEqual(config.perf_analyzer.stability_percentage, 500)
        self.assertEqual(
            config.perf_analyzer.measurement.mode,
            PerfAnalyzerMeasurementMode.REQUEST_COUNT,
        )
        self.assertEqual(config.perf_analyzer.measurement.num, 100)
        self.assertEqual(config.perf_analyzer.warmup_request_count, 200)

    def test_session_turn_delay_ratio(self):
        """
        Test that the session turn delay ratio is parsed correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            input:
                sessions:
                    turn_delay:
                        ratio: 0.5
                        mean: 100
                        stddev: 20
                    turns:
                        mean: 3
                        stddev: 1
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.input.sessions.turn_delay.ratio, 0.5)
        self.assertEqual(config.input.sessions.turn_delay.mean, 100)
        self.assertEqual(config.input.sessions.turn_delay.stddev, 20)
        self.assertEqual(config.input.sessions.turns.mean, 3)
        self.assertEqual(config.input.sessions.turns.stddev, 1)

    ###########################################################################
    # Test Input Config
    ###########################################################################
    def test_input_config(self):
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
                    test_goodput: 2
                header:
                    test_header: 3
                file: "synthetic:test_file"
                num_dataset_entries: 50
                random_seed: 100

                audio:
                    length:
                        mean: 24
                        stddev: 25

                    format: MP3
                    depths: 26, 27,  28
                    sample_rates: 29
                    num_channels: 2

                image:
                    batch_size: 8

                    width:
                      mean: 9
                      stddev: 10

                    height:
                      mean: 11
                      stddev: 12

                    format: PNG

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

                sessions:
                  num: 19

                  turns:
                    mean: 20
                    stddev: 21

                  turn_delay:
                    mean: 22
                    stddev: 23
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.input.batch_size, 4)
        self.assertEqual(config.input.extra, {"test_extra": 1})
        self.assertEqual(config.input.goodput, {"test_goodput": 2})
        self.assertEqual(config.input.header, {"test_header": 3})
        self.assertEqual(config.input.file, Path("synthetic:test_file"))
        self.assertEqual(config.input.num_dataset_entries, 50)
        self.assertEqual(config.input.random_seed, 100)
        self.assertEqual(config.input.image.batch_size, 8)
        self.assertEqual(config.input.image.width.mean, 9)
        self.assertEqual(config.input.image.width.stddev, 10)
        self.assertEqual(config.input.image.height.mean, 11)
        self.assertEqual(config.input.image.height.stddev, 12)
        self.assertEqual(config.input.image.format, ImageFormat.PNG)
        self.assertEqual(config.input.output_tokens.mean, 13)
        self.assertEqual(config.input.output_tokens.deterministic, True)
        self.assertEqual(config.input.output_tokens.stddev, 14)
        self.assertEqual(config.input.synthetic_tokens.mean, 15)
        self.assertEqual(config.input.synthetic_tokens.stddev, 16)
        self.assertEqual(config.input.prefix_prompt.num, 17)
        self.assertEqual(config.input.prefix_prompt.length, 18)
        self.assertEqual(config.input.sessions.num, 19)
        self.assertEqual(config.input.sessions.turns.mean, 20)
        self.assertEqual(config.input.sessions.turns.stddev, 21)
        self.assertEqual(config.input.sessions.turn_delay.mean, 22)
        self.assertEqual(config.input.sessions.turn_delay.stddev, 23)
        self.assertEqual(config.input.audio.length.mean, 24)
        self.assertEqual(config.input.audio.length.stddev, 25)
        self.assertEqual(config.input.audio.format, AudioFormat.MP3)
        self.assertEqual(config.input.audio.depths, [26, 27, 28])
        self.assertEqual(config.input.audio.sample_rates, [29])
        self.assertEqual(config.input.audio.num_channels, 2)

    ###########################################################################
    # Test Output Config
    ###########################################################################
    def test_output_config(self):
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

        self.assertEqual(
            config.output.artifact_directory, Path("test_artifact_directory")
        )
        self.assertEqual(
            config.output.checkpoint_directory, Path("test_checkpoint_directory")
        )
        self.assertEqual(
            config.output.profile_export_file,
            Path("test_profile_export_file"),
        )
        self.assertEqual(config.output.generate_plots, True)

    ###########################################################################
    # Test Tokenizer Config
    ###########################################################################
    def test_tokenizer_config(self):
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

    ###########################################################################
    # Test Deepcopy
    ###########################################################################
    def test_deepcopy_config_input(self):
        """
        Test that the input configuration can be deepcopied
        """
        config_input = ConfigInput()
        config_input.batch_size = 16

        copied_config_input = deepcopy(config_input)

        # Check that the copied object is not the same object
        self.assertNotEqual(id(config_input), id(copied_config_input))

        # Check that the copied object is equal to the original object
        self.assertEqual(config_input.batch_size, copied_config_input.batch_size)

        # Check that the copied object can be modified without affecting the original object
        config_input.batch_size = 32
        self.assertNotEqual(config_input.batch_size, copied_config_input.batch_size)

    ###########################################################################
    # Test Input Inference Methods
    ###########################################################################
    def test_infer_prompt_source_file(self):
        """
        Test that the prompt source is inferred correctly for non-synthetic names
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            input:
              file: "test_file"
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)

        with patch("genai_perf.config.input.config_input.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.is_file.return_value = True
            mock_path.return_value = mock_path_instance

            config = ConfigCommand(user_config)

            self.assertEqual(config.input.prompt_source, PromptSource.FILE)

    def test_infer_prompt_source_synthetic(self):
        """
        Test that the prompt source is inferred correctly for synthetic names
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            input:
              file: "synthetic:test_file"
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.input.prompt_source, PromptSource.SYNTHETIC)

    def test_infer_synthetic_files(self):
        """
        Test that the synthetic files are inferred correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            input:
              file: "synthetic:test_file1,test_file2"
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.input.synthetic_files, ["test_file1", "test_file2"])

    ###########################################################################
    # Test EndPoint Inference Methods
    ###########################################################################
    def test_infer_type_service_kind_triton(self):
        """
        Test that the type is inferred correctly for triton service kind
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            endpoint:
                service_kind: triton
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.endpoint.type, "kserve")

    def test_infer_type_service_kind_tensorrtllm_engine(self):
        """
        Test that the type is inferred correctly for tensorrtllm_engine service kind
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            endpoint:
                service_kind: tensorrtllm_engine
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.endpoint.type, "tensorrtllm_engine")

    def test_infer_service_kind(self):
        """
        Test that the service kind is inferred correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            endpoint:
                service_kind: triton
                type: generate
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.endpoint.service_kind, "openai")

    def test_infer_output_format_triton(self):
        """
        Test that the output format is inferred correctly with service kind of triton
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            endpoint:
                service_kind: triton
                type: kserve
                backend: vllm
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.endpoint.output_format, OutputFormat.VLLM)

    def test_infer_output_format_tensorrtllm_engine(self):
        """
        Test that the output format is inferred correctly with service kind of tensorrtllm_engine
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            endpoint:
                service_kind: tensorrtllm_engine
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.endpoint.output_format, OutputFormat.TENSORRTLLM_ENGINE)

    def test_infer_custom(self):
        """
        Test that the custom endpoint is inferred correctly
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2

            endpoint:
                service_kind: openai
                type: embeddings
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)

        self.assertEqual(config.endpoint.custom, "v1/embeddings")

    def test_infer_tokenizer(self):
        """
        Test that tokenizer is inferred from model name when not explicitly set
        """
        # Test case 1: No tokenizer set, should infer from model name
        yaml_str = """
            model_name: gpt2
            """
        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)
        self.assertEqual(config.tokenizer.name, "gpt2")

        # Test case 2: Tokenizer explicitly set, should not infer
        yaml_str = """
            model_name: gpt2
            tokenizer:
                name: t5-small
            """
        user_config = yaml.safe_load(yaml_str)
        config = ConfigCommand(user_config)
        self.assertEqual(config.tokenizer.name, "t5-small")


if __name__ == "__main__":
    unittest.main()
