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
from genai_perf.config.run.results import Results
from genai_perf.measurements.run_constraints import ModelConstraints, RunConstraints
from genai_perf.record.types.gpu_power_usage import GPUPowerUsage
from genai_perf.record.types.perf_latency_p99 import PerfLatencyP99
from tests.test_utils import create_run_config


class TestResults(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._results = Results()

        for i in range(10):
            run_config_name = "test_run_config_" + str(i)
            run_config = create_run_config(
                run_config_name,
                gpu_power=500 + 10 * i,
                gpu_utilization=50 - i,
                throughput=300 - 10 * i,
                latency=100 - 5 * i,
            )
            self._results.add_run_config(run_config)

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Checkpoint Tests
    ###########################################################################
    def test_checkpoint_methods(self):
        """
        Checks to ensure checkpoint methods work as intended
        """
        results_json = json.dumps(self._results, default=checkpoint_encoder)

        results_from_checkpoint = Results.read_from_checkpoint(json.loads(results_json))

        self.assertEqual(results_from_checkpoint, self._results)

    ###########################################################################
    # Accessor Method Tests
    ###########################################################################
    def test_objective_setting(self):
        """
        Check to ensure that we sort correctly when changing the objectives
        """

        # The default objective is throughput and for this config_0 will be best
        self.assertEqual("test_run_config_0", self._results.run_configs[0].name)

        # Changing the objective to latency will result in config_9 being best
        self._results.set_perf_metric_objectives(
            {"test_model": {PerfLatencyP99.tag: 1}}
        )
        self.assertEqual("test_run_config_9", self._results.run_configs[0].name)

        # Changing the objective to GPU Power will result in config_0 being best
        self._results.set_gpu_metric_objectives({"test_model": {GPUPowerUsage.tag: 1}})
        self._results.set_perf_metric_objectives({"test_model": {}})
        self.assertEqual("test_run_config_0", self._results.run_configs[0].name)

    def test_constraint_setting(self):
        """
        Check to ensure that constraints work
        """

        # GPU Power ranges from 510 -> 590, this will make the first 5 pass
        model_constraints = ModelConstraints({GPUPowerUsage.tag: 550})
        run_constraints = RunConstraints({"test_model": model_constraints})
        self._results.set_constraints(run_constraints)

        passing_results = self._results.get_results_passing_constraints()
        self.assertEqual(5, len(passing_results.run_configs))
        self.assertEqual("test_run_config_0", passing_results.run_configs[0].name)

        failing_results = self._results.get_results_failing_constraints()
        self.assertEqual(5, len(failing_results.run_configs))
        self.assertEqual("test_run_config_5", failing_results.run_configs[0].name)


if __name__ == "__main__":
    unittest.main()
