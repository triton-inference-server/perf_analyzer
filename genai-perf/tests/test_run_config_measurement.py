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
from genai_perf.measurements.model_config_measurement import ModelConfigMeasurement
from genai_perf.measurements.model_constraints import ModelConstraints
from genai_perf.measurements.run_config_measurement import RunConfigMeasurement
from genai_perf.measurements.run_constraints import RunConstraints
from genai_perf.record.types.gpu_power_usage import GPUPowerUsage
from genai_perf.record.types.gpu_utilization import GPUUtilization
from genai_perf.record.types.perf_latency_p99 import PerfLatencyP99
from genai_perf.record.types.perf_throughput import PerfThroughput


class TestRunConfigMeasurement(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._create_gpu_metrics()
        self._create_perf_metrics()

    def tearDown(self):
        patch.stopall()

    def _create_gpu_metrics(self) -> None:
        #
        # Record A
        self.gpu_power_recordA = GPUPowerUsage(120)
        self.gpu_util_recordA = GPUUtilization(50)

        self.gpu_metricsA = {
            "0": {
                GPUPowerUsage.tag: self.gpu_power_recordA,
                GPUUtilization.tag: self.gpu_util_recordA,
            }
        }

        #
        # Record B
        self.gpu_power_recordB = GPUPowerUsage(60)
        self.gpu_util_recordB = GPUUtilization(50)

        self.gpu_metricsB = {
            "0": {
                GPUPowerUsage.tag: self.gpu_power_recordB,
                GPUUtilization.tag: self.gpu_util_recordB,
            }
        }

        #
        # Record MM
        self.gpu_power_recordMM = GPUPowerUsage(120)
        self.gpu_util_recordMM = GPUUtilization(50)

        self.gpu_metricsMM = {
            "0": {
                GPUPowerUsage.tag: self.gpu_power_recordMM,
                GPUUtilization.tag: self.gpu_util_recordMM,
            }
        }

    def _create_perf_metrics(self) -> None:
        #
        # Record A
        self.throughput_recordA = PerfThroughput(1000)
        self.latency_recordA = PerfLatencyP99(40)

        self.perf_metricsA = {
            PerfThroughput.tag: self.throughput_recordA,
            PerfLatencyP99.tag: self.latency_recordA,
        }

        #
        # Record B
        self.throughput_recordB = PerfThroughput(500)
        self.latency_recordB = PerfLatencyP99(30)

        self.perf_metricsB = {
            PerfThroughput.tag: self.throughput_recordB,
            PerfLatencyP99.tag: self.latency_recordB,
        }

        #
        # Record MM
        self.throughput_recordMM_0 = PerfThroughput(1000)
        self.latency_recordMM_0 = PerfLatencyP99(20)

        self.throughput_recordMM_1 = PerfThroughput(2000)
        self.latency_recordMM_1 = PerfLatencyP99(30)

        self.perf_metricsMM_0 = {
            PerfThroughput.tag: self.throughput_recordMM_0,
            PerfLatencyP99.tag: self.latency_recordMM_0,
        }

        self.perf_metricsMM_1 = {
            PerfThroughput.tag: self.throughput_recordMM_1,
            PerfLatencyP99.tag: self.latency_recordMM_1,
        }

    def _create_rcmA(self) -> RunConfigMeasurement:
        rcmA = RunConfigMeasurement(self.gpu_metricsA)
        rcmA.add_perf_metrics(model_name="test_model", perf_metrics=self.perf_metricsA)

        return rcmA

    def _create_rcmB(self) -> RunConfigMeasurement:
        rcmB = RunConfigMeasurement(self.gpu_metricsB)
        rcmB.add_perf_metrics(model_name="test_model", perf_metrics=self.perf_metricsB)

        return rcmB

    def _create_multi_model_rcm(self) -> RunConfigMeasurement:
        rcmMM = RunConfigMeasurement(self.gpu_metricsMM)
        rcmMM.add_perf_metrics(
            model_name="modelMM_0", perf_metrics=self.perf_metricsMM_0
        )
        rcmMM.add_perf_metrics(
            model_name="modelMM_1", perf_metrics=self.perf_metricsMM_1
        )

        return rcmMM

    ###########################################################################
    # Accessor Tests
    ###########################################################################
    def test_basic_accessor_methods(self):
        """
        Test that values are properly initialized
        """
        rcmA = self._create_rcmA()

        #
        # GPU Metrics accessors
        self.assertEqual(rcmA.get_all_gpu_metrics(), self.gpu_metricsA)

        expected_gpu_metric = {
            "0": {GPUPowerUsage.tag: self.gpu_metricsA["0"].get(GPUPowerUsage.tag)}
        }
        self.assertEqual(expected_gpu_metric, rcmA.get_gpu_metric(GPUPowerUsage.tag))

        #
        # MCM accessors
        self.assertIsNone(rcmA.get_model_config_measurement("ModelNotPresent"))

        expected_mcmA = ModelConfigMeasurement(self.perf_metricsA)
        expected_mcm_dict = {"test_model": expected_mcmA}
        self.assertEqual(rcmA.get_model_config_measurements(), expected_mcm_dict)
        self.assertEqual(rcmA.get_model_config_measurement("test_model"), expected_mcmA)

        #
        # Perf Metrics accessors
        expected_all_perf_metrics_dict = {"test_model": self.perf_metricsA}
        self.assertEqual(expected_all_perf_metrics_dict, rcmA.get_all_perf_metrics())
        self.assertEqual(self.perf_metricsA, rcmA.get_model_perf_metrics("test_model"))
        self.assertEqual(
            self.perf_metricsA[PerfThroughput.tag],
            rcmA.get_model_perf_metric("test_model", PerfThroughput.tag),
        )
        self.assertEqual(
            self.perf_metricsA[PerfThroughput.tag].value(),
            rcmA.get_model_perf_metric_value("test_model", PerfThroughput.tag),
        )
        self.assertEqual(
            10,
            rcmA.get_model_perf_metric_value(
                "test_model", "MetricNotPresent", return_value=10
            ),
        )

        #
        # Weighted Perf Metrics accessor
        rcmMM = self._create_multi_model_rcm()
        model_weights = {"modelMM_0": 0.8, "modelMM_1": 0.2}
        rcmMM.set_model_weighting(model_weights)

        expected_weighted_perf_metric_values = {
            "modelMM_0": self.perf_metricsMM_0[PerfThroughput.tag].value() * 0.8,
            "modelMM_1": self.perf_metricsMM_1[PerfThroughput.tag].value() * 0.2,
        }

        self.assertEqual(
            expected_weighted_perf_metric_values,
            rcmMM.get_weighted_perf_metric_values(PerfThroughput.tag),
        )

    def test_set_gpu_metric_objectives(self):
        """
        Test that GPU metric objectives can be set in a multi-model setting
        """
        rcmMM = self._create_multi_model_rcm()
        gpu_metric_objectives = {
            "modelMM_0": {GPUPowerUsage.tag: 1},
            "modelMM_1": {GPUUtilization.tag: 1},
        }

        rcmMM.set_gpu_metric_objectives(gpu_metric_objectives)
        self.assertEqual(gpu_metric_objectives, rcmMM._gpu_metric_objectives)

    ###########################################################################
    # Checkpoint Tests
    ###########################################################################
    def test_checkpoint_methods(self):
        """
        Checks to ensure checkpoint methods work as intended
        """
        rcmA = self._create_rcmA()
        rcmA_json = json.dumps(rcmA, default=checkpoint_encoder)

        rcmA_from_checkpoint = RunConfigMeasurement.read_from_checkpoint(
            json.loads(rcmA_json)
        )

        self.assertEqual(
            rcmA_from_checkpoint.get_all_gpu_metrics(), rcmA.get_all_gpu_metrics()
        )
        self.assertEqual(
            rcmA_from_checkpoint.get_model_config_measurements(),
            rcmA.get_model_config_measurements(),
        )

        # Catchall in case something new is added
        self.assertEqual(rcmA_from_checkpoint, rcmA)

    ###########################################################################
    # Comparison Tests
    ###########################################################################
    def test_is_better_than_perf_metric(self):
        """
        Test to ensure measurement perf metric comparison is working as intended
        """
        # RCMA: 1000, 40    RCMB: 500, 30
        # RCMA's throughput is better than RCMB
        # RCMB's latency is worse than RCMA
        # Factoring in perf metric objectives (equal)
        # tips this is favor of RCMA (2x throughput, 33% worse latency)
        rcmA = self._create_rcmA()
        rcmB = self._create_rcmB()

        self.assertTrue(rcmA.is_better_than(rcmB))
        self.assertFalse(rcmB.is_better_than(rcmA))

        # Changing the metric objectives to bias latency
        # this tips the scale in the favor of RCMB
        latency_bias_objectives = {
            "test_model": {PerfThroughput.tag: 1, PerfLatencyP99.tag: 4}
        }
        rcmA.set_perf_metric_objectives(latency_bias_objectives)
        rcmB.set_perf_metric_objectives(latency_bias_objectives)

        self.assertFalse(rcmA.is_better_than(rcmB))
        self.assertTrue(rcmB.is_better_than(rcmA))

    def test_is_better_than_gpu_metric(self):
        """
        Test to ensure measurement GPU metric comparison is working as intended
        """
        # RCMA's power is higher, therefore RCMB is better
        rcmA = self._create_rcmA()
        rcmB = self._create_rcmB()

        rcmA.set_gpu_metric_objectives({"test_model": {GPUPowerUsage.tag: 1}})
        rcmB.set_gpu_metric_objectives({"test_model": {GPUPowerUsage.tag: 1}})
        rcmA.set_perf_metric_objectives({"test_model": {}})
        rcmB.set_perf_metric_objectives({"test_model": {}})

        self.assertFalse(rcmA.is_better_than(rcmB))
        self.assertTrue(rcmB.is_better_than(rcmA))

    ###########################################################################
    # Calculation Tests
    ###########################################################################
    def test_calculate_weighted_percentage_gain(self):
        """
        Test that weighted percentages are returned correctly
        """
        rcmA = self._create_rcmA()
        rcmB = self._create_rcmB()

        # Default is throughput, rcmA = 1000, rcmB = 500
        self.assertEqual(rcmA.calculate_weighted_percentage_gain(rcmB), 100)
        self.assertEqual(rcmB.calculate_weighted_percentage_gain(rcmA), -50)

        # Now we'll add in GPU Power, this has equal weighting with perf metrics
        # rcmA = 120, rcmB = 60
        rcmA.set_gpu_metric_objectives({"test_model": {GPUPowerUsage.tag: 1}})
        rcmB.set_gpu_metric_objectives({"test_model": {GPUPowerUsage.tag: 1}})
        self.assertEqual(rcmA.calculate_weighted_percentage_gain(rcmB), 25)
        self.assertEqual(rcmB.calculate_weighted_percentage_gain(rcmA), 25)

        # You'll note that they both report being 25% better because one metric
        # has a 100% gain, while the other has a -50% gain. This illustrates why
        # percentage gain is not reliable when multiple objectives are present

    ###########################################################################
    # Constraint Tests
    ###########################################################################
    def test_is_passing_constraints_none(self):
        """
        Test to ensure constraints are reported as passing
        if none were specified
        """
        rcmA = self._create_rcmA()
        self.assertTrue(rcmA.is_passing_constraints())

        rcmA.set_constraints(RunConstraints({"modelA": None}))
        self.assertTrue(rcmA.is_passing_constraints())

    def test_is_passing_gpu_constraints(self):
        """
        Test to ensure GPU constraints are reported as
        passing/failing if model is above/below
        GPU Power threshold
        """
        rcmA = self._create_rcmA()

        # RCMA's power is 120
        model_constraints = ModelConstraints({GPUPowerUsage.tag: 50})
        run_constraints = RunConstraints({"test_model": model_constraints})
        rcmA.set_constraints(run_constraints)
        self.assertFalse(rcmA.is_passing_constraints())

        model_constraints = ModelConstraints({GPUPowerUsage.tag: 150})
        run_constraints = RunConstraints({"test_model": model_constraints})
        rcmA.set_constraints(run_constraints)
        self.assertTrue(rcmA.is_passing_constraints())

    def test_is_passing_perf_constraints(self):
        """
        Test to ensure perf constraints are reported as
        passing/failing if model is above/below
        latency threshold
        """
        rcmA = self._create_rcmA()

        # RCMA's latency is 40
        model_constraints = ModelConstraints({PerfLatencyP99.tag: 50})
        run_constraints = RunConstraints({"test_model": model_constraints})
        rcmA.set_constraints(run_constraints)
        self.assertTrue(rcmA.is_passing_constraints())

        model_constraints = ModelConstraints({PerfLatencyP99.tag: 20})
        run_constraints = RunConstraints({"test_model": model_constraints})
        rcmA.set_constraints(run_constraints)
        self.assertFalse(rcmA.is_passing_constraints())


if __name__ == "__main__":
    unittest.main()
