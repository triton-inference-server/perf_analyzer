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

from genai_perf.checkpoint.checkpoint import checkpoint_encoder
from genai_perf.config.generate.genai_perf_config import GenAIPerfConfig
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.run.run_config import RunConfig
from tests.test_utils import create_perf_metrics, create_run_config_measurement


class TestRunConfig(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._config = ConfigCommand(model_names=["test_model"])

        self._baseline_genai_perf_config = GenAIPerfConfig(
            config=self._config, model_objective_parameters={}
        )
        self._baseline_perf_analyzer_config = PerfAnalyzerConfig(
            model_name="test_model", config=self._config, model_objective_parameters={}
        )

        self._perf_metrics = create_perf_metrics(throughput=1000, latency=50)
        self._baseline_rcm = create_run_config_measurement(
            gpu_power=80, gpu_utilization=70
        )
        self._baseline_rcm.add_perf_metrics("test_model", self._perf_metrics)

        self._run_config = RunConfig(
            name="test_run_config",
            genai_perf_config=self._baseline_genai_perf_config,
            perf_analyzer_config=self._baseline_perf_analyzer_config,
            measurement=self._baseline_rcm,
        )

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Checkpoint Tests
    ###########################################################################
    def test_checkpoint_methods(self):
        """
        Checks to ensure checkpoint methods work as intended
        """
        run_config_json = json.dumps(self._run_config, default=checkpoint_encoder)

        run_config_from_checkpoint = RunConfig.read_from_checkpoint(
            json.loads(run_config_json)
        )

        self.assertEqual(run_config_from_checkpoint.name, self._run_config.name)
        self.assertEqual(
            run_config_from_checkpoint.genai_perf_config,
            self._run_config.genai_perf_config,
        )
        self.assertEqual(
            run_config_from_checkpoint.perf_analyzer_config,
            self._run_config.perf_analyzer_config,
        )
        self.assertEqual(
            run_config_from_checkpoint.measurement, self._run_config.measurement
        )

        # Catchall in case something new is added
        self.assertEqual(run_config_from_checkpoint, self._run_config)


if __name__ == "__main__":
    unittest.main()
