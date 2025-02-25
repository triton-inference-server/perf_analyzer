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

import subprocess
from unittest.mock import MagicMock, patch

from genai_perf.subcommand.common import run_perf_analyzer


class TestCommon:
    @patch("genai_perf.subcommand.common.subprocess.run")
    def test_stdout_verbose(self, mock_subprocess_run):
        args = MagicMock()
        args.model = "test_model"
        args.verbose = True
        run_perf_analyzer(
            args=args,
            extra_args=None,
        )

        # Check that standard output was not redirected.
        for call_args in mock_subprocess_run.call_args_list:
            _, kwargs = call_args
            assert (
                "stdout" not in kwargs or kwargs["stdout"] is None
            ), "With the verbose flag, stdout should not be redirected."

    @patch("genai_perf.subcommand.common.subprocess.run")
    def test_stdout_not_verbose(self, mock_subprocess_run):
        args = MagicMock()
        args.model = "test_model"
        args.verbose = False
        run_perf_analyzer(
            args=args,
            extra_args=None,
        )

        # Check that standard output was redirected.
        for call_args in mock_subprocess_run.call_args_list:
            _, kwargs = call_args
            assert (
                kwargs["stdout"] is subprocess.DEVNULL
            ), "When the verbose flag is not passed, stdout should be redirected to /dev/null."
