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

from genai_perf.measurements.model_config_measurement import (
    ModelConfigMeasurement,
    ModelConfigMeasurementDefaults,
)
from genai_perf.record.types.perf_latency_p99 import PerfLatencyP99
from genai_perf.record.types.perf_throughput import PerfThroughput
from genai_perf.record.types.time_to_first_token_avg import TimeToFirstTokenAvg
from genai_perf.utils import checkpoint_encoder


class TestModelConfigMeasurement(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):

        self.throughput_recordA = PerfThroughput(1000)
        self.latency_recordA = PerfLatencyP99(20)

        self.perf_metricsA = {
            PerfThroughput.tag: self.throughput_recordA,
            PerfLatencyP99.tag: self.latency_recordA,
        }

        self.mcmA = ModelConfigMeasurement(self.perf_metricsA)

        self.throughput_recordB = PerfThroughput(500)
        self.latency_recordB = PerfLatencyP99(10)

        self.perf_metricsB = {
            PerfThroughput.tag: self.throughput_recordB,
            PerfLatencyP99.tag: self.latency_recordB,
        }

        self.mcmB = ModelConfigMeasurement(self.perf_metricsB)

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Accessor Tests
    ###########################################################################
    def test_basic_accessor_methods(self):
        """
        Test that values are properly initialized
        """
        self.assertEqual(self.mcmA.get_perf_metrics(), self.perf_metricsA)
        self.assertEqual(
            self.mcmA.get_perf_metric(PerfLatencyP99.tag), self.latency_recordA
        )
        self.assertEqual(
            self.mcmA.get_perf_metric_value(PerfThroughput.tag, return_value=-1),
            self.throughput_recordA.value(),
        )
        self.assertEqual(
            self.mcmA.get_perf_metric_value(TimeToFirstTokenAvg.tag, return_value=-1),
            -1,
        )

    def test_set_metric_objective(self):
        """
        Test that metric objective weighting is set correctly
        """
        # Default
        self.assertEqual(
            ModelConfigMeasurementDefaults.METRIC_OBJECTIVE,
            self.mcmA._metric_objectives,
        )

        self.mcmA.set_metric_objectives({PerfThroughput.tag: 2, PerfLatencyP99.tag: 3})
        expected_mw = {PerfThroughput.tag: 2 / 5, PerfLatencyP99.tag: 3 / 5}
        self.assertEqual(expected_mw, self.mcmA._metric_objectives)

    def test_get_weighted_score(self):
        """
        Test that weighted score is returned correctly
        """

        # In the default case we are comparing throughputs with mcmA = 1000, mcmB = 500
        # scoreA will be positive (2/3), and scoreB be will be its negative
        scoreA = self.mcmA.get_weighted_score(self.mcmB)
        scoreB = self.mcmB.get_weighted_score(self.mcmA)

        self.assertEqual(2 / 3, scoreA)
        self.assertEqual(-2 / 3, scoreB)

        # In this case we will change the objective to be latency, with mcmA = 20, mcmB = 5
        # since latency is a decreasing record (lower is better), scoreB will be positive
        self.mcmA.set_metric_objectives({PerfLatencyP99.tag: 1})
        self.mcmB.set_metric_objectives({PerfLatencyP99.tag: 1})
        scoreA = self.mcmA.get_weighted_score(self.mcmB)
        scoreB = self.mcmB.get_weighted_score(self.mcmA)

        self.assertEqual(-2 / 3, scoreA)
        self.assertEqual(2 / 3, scoreB)

    ###########################################################################
    # Checkpoint Tests
    ###########################################################################
    def test_checkpoint_methods(self):
        """
        Checks to ensure checkpoint methods work as intended
        """
        mcmA_json = json.dumps(self.mcmA, default=checkpoint_encoder)

        mcmA_from_checkpoint = ModelConfigMeasurement.read_from_checkpoint(
            json.loads(mcmA_json)
        )

        self.assertEqual(
            mcmA_from_checkpoint.get_perf_metrics(), self.mcmA.get_perf_metrics()
        )

        # Catchall in case something new is added
        self.assertEqual(mcmA_from_checkpoint, self.mcmA)

    ###########################################################################
    # Calculation Tests
    ###########################################################################
    def test_calculate_weighted_percentage_gain(self):
        """
        Test that weighted percentages are returned correctly
        """

        # throughput: mcmA: 1000, mcmB: 500
        self.assertEqual(self.mcmA.calculate_weighted_percentage_gain(self.mcmB), 100)
        self.assertEqual(self.mcmB.calculate_weighted_percentage_gain(self.mcmA), -50)

        self.mcmA.set_metric_objectives({PerfLatencyP99.tag: 1})
        self.mcmB.set_metric_objectives({PerfLatencyP99.tag: 1})

        # latency: mcmA: 20, mcmB: 10
        self.assertEqual(self.mcmA.calculate_weighted_percentage_gain(self.mcmB), -50)
        self.assertEqual(self.mcmB.calculate_weighted_percentage_gain(self.mcmA), 100)

        # This illustrates why we need to use score, not percentages to determine
        # which model is better. In both cases we will (correctly) report that
        # mcmA/B is 25% better than the other, even though they are equal
        #
        # mcmA has 50% worse throughput, but 100% better latency
        # mcmB has 100% better latency, but 50% worse throughput
        self.mcmA.set_metric_objectives({PerfThroughput.tag: 1, PerfLatencyP99.tag: 1})
        self.mcmB.set_metric_objectives({PerfThroughput.tag: 1, PerfLatencyP99.tag: 1})
        self.assertEqual(self.mcmA, self.mcmB)
        self.assertEqual(self.mcmA.calculate_weighted_percentage_gain(self.mcmB), 25)
        self.assertEqual(self.mcmB.calculate_weighted_percentage_gain(self.mcmA), 25)

    ###########################################################################
    # Comparison Tests
    ###########################################################################
    def test_is_better_than(self):
        """
        Test that individual metric comparison works as expected
        """
        self.mcmA.set_metric_objectives({PerfThroughput.tag: 1})

        # throughput: 1000 is better than 500
        self.assertTrue(self.mcmA.is_better_than(self.mcmB))
        self.assertGreater(self.mcmA, self.mcmB)

        self.mcmA.set_metric_objectives({PerfLatencyP99.tag: 1})

        # latency: 20 is worse than 10
        self.assertFalse(self.mcmA.is_better_than(self.mcmB))
        self.assertLess(self.mcmA, self.mcmB)

    def test_is_better_than_combo(self):
        """
        Test that combination metric comparison works as expected
        """
        # throuhput: 2000 vs. 1000 (better), latency: 20 vs. 10 (worse)
        # with latency bias mcmB is better
        self.mcmA.set_metric_objectives({PerfThroughput.tag: 1, PerfLatencyP99.tag: 3})

        self.assertFalse(self.mcmA.is_better_than(self.mcmB))

    def test_is_better_than_empty(self):
        """
        Test for correct return values when comparing for/against an empty set
        """
        empty_mcm0 = ModelConfigMeasurement({})
        empty_mcm1 = ModelConfigMeasurement({})

        self.assertTrue(self.mcmA.is_better_than(empty_mcm0))
        self.assertFalse(empty_mcm0.is_better_than(self.mcmA))
        self.assertEqual(empty_mcm0, empty_mcm1)


if __name__ == "__main__":
    unittest.main()
