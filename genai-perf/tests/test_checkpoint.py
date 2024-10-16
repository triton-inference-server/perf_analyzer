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

import os
import unittest
from unittest.mock import patch

from genai_perf.checkpoint.checkpoint import Checkpoint
from genai_perf.config.generate.search_parameters import SearchParameters
from genai_perf.config.generate.sweep_objective_generator import SweepObjectiveGenerator
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.run.results import Results
from tests.test_utils import create_run_config


class TestCheckpoint(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._config = ConfigCommand(model_names=["test_model"])
        self._model_search_parameters = {
            "test_model": SearchParameters(self._config.analyze)
        }

        self._sweep_obj_gen = SweepObjectiveGenerator(
            self._config, self._model_search_parameters
        )

        self._results = Results()
        for count, objective in enumerate(self._sweep_obj_gen.get_objectives()):
            run_config_name = "test_model_run_config_" + str(count)
            run_config = create_run_config(
                run_config_name=run_config_name,
                model_objective_parameters=objective,
                model_name="test_model",
                gpu_power=500 + 10 * count,
                gpu_utilization=50 - count,
                throughput=300 - 10 * count,
                latency=100 - 5 * count,
                input_seq_length=20 + 10 * count,
                output_seq_length=50 + 5 * count,
            )
            self._results.add_run_config(run_config)

        self._checkpoint = Checkpoint(
            config=ConfigCommand("test_model"), results=self._results
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

        # First ensure that there is no state when a checkpoint doesn't exist
        self.assertEqual({}, self._checkpoint._state)

        # Then write and read back the Results
        self._checkpoint.write_to_checkpoint()
        self._checkpoint._read_from_checkpoint()
        os.remove(self._checkpoint._create_checkpoint_filename())

        self.assertEqual(self._results, self._checkpoint._state["Results"])


if __name__ == "__main__":
    unittest.main()
