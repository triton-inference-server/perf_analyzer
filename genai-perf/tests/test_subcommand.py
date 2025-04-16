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

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.input.config_field import ConfigField
from genai_perf.subcommand.subcommand import Subcommand


class TestCommon:
    @patch("genai_perf.subcommand.subcommand.subprocess.run")
    def test_stdout_verbose(self, mock_subprocess_run):
        config = ConfigCommand(user_config={"model_name": "test_model"})
        config.verbose = ConfigField(default=False, value=True)
        perf_analyzer_config = PerfAnalyzerConfig(config)
        subcommand = Subcommand(config)
        subcommand._run_perf_analyzer(
            perf_analyzer_config=perf_analyzer_config,
        )

        # Check that standard output was not redirected.
        for call_args in mock_subprocess_run.call_args_list:
            _, kwargs = call_args
            assert (
                "stdout" not in kwargs or kwargs["stdout"] is None
            ), "With the verbose flag, stdout should not be redirected."

    @patch("genai_perf.subcommand.subcommand.subprocess.run")
    def test_stdout_not_verbose(self, mock_subprocess_run):
        config = ConfigCommand(user_config={"model_name": "test_model"})
        config.verbose = ConfigField(default=False)
        perf_analyzer_config = PerfAnalyzerConfig(config)
        subcommand = Subcommand(config)
        subcommand._run_perf_analyzer(
            perf_analyzer_config=perf_analyzer_config,
        )

        # Check that standard output was redirected.
        for call_args in mock_subprocess_run.call_args_list:
            _, kwargs = call_args
            assert (
                kwargs["stdout"] is subprocess.DEVNULL
            ), "When the verbose flag is not passed, stdout should be redirected to /dev/null."
