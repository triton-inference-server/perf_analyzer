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


class TestConfigParser(unittest.TestCase):

    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        pass

    def tearDown(self):
        patch.stopall()

    def test_yaml_parsing(self):
        """
        Test that I know how to parse a yaml string
        """
        # yapf: disable
        yaml_str = ("""
            model_name: gpt2
            checkpoint_directory: /path/to/checkpoints

            endpoint:
                service_kind: triton
                backend: vllm
                streaming: True

            input:
                num_dataset_entries: 100

            analyze:
                concurrency:
                    start: 4
                    stop: 32

                input_sequence_length:
                    start: 100
                    stop: 200
                    step: 20
            """)
        # yapf: enable

        user_config = yaml.safe_load(yaml_str)

        config = ConfigCommand(user_config)

        self.assertEqual(config.model_names, ["gpt2"])
        self.assertEqual(config.checkpoint_directory, "/path/to/checkpoints")
        self.assertEqual(config.analyze.sweep_parameters["concurrency"], Range(4, 32))
        self.assertEqual(
            config.analyze.sweep_parameters["input_sequence_length"],
            [100, 120, 140, 160, 180, 200],
        )


if __name__ == "__main__":
    unittest.main()
