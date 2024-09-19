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
from statistics import mean
from unittest.mock import patch

from genai_perf.measurements.model_config_measurement import ModelConfigMeasurement
from genai_perf.measurements.run_config_measurement import (
    RunConfigMeasurement,
    RunConfigMeasurementDefaults,
)
from genai_perf.record.types.gpu_power_usage import GPUPowerUsage
from genai_perf.record.types.gpu_utilization import GPUUtilization
from genai_perf.record.types.perf_latency_p99 import PerfLatencyP99
from genai_perf.record.types.perf_throughput import PerfThroughput
from genai_perf.utils import checkpoint_encoder


class TestRunConfigMeasurement(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        self._create_rcmA()
        self._create_multi_model_rcm()

    def tearDown(self):
        patch.stopall()

    def _create_rcmA(self) -> None:
        self.gpu_power_recordA = GPUPowerUsage(120)
        self.gpu_util_recordA = GPUUtilization(50)

        self.gpu_metricsA = {
            "0": {
                GPUPowerUsage.tag: self.gpu_power_recordA,
                GPUUtilization.tag: self.gpu_util_recordA,
            }
        }

        self.throughput_recordA = PerfThroughput(1000)
        self.latency_recordA = PerfLatencyP99(20)

        self.perf_metricsA = {
            PerfThroughput.tag: self.throughput_recordA,
            PerfLatencyP99.tag: self.latency_recordA,
        }

        self.rcmA = RunConfigMeasurement(self.gpu_metricsA)
        self.rcmA.add_model_config_measurement(
            model_name="modelA", perf_metrics=self.perf_metricsA
        )

    def _create_multi_model_rcm(self) -> None:
        self.gpu_power_recordB = GPUPowerUsage(120)
        self.gpu_util_recordB = GPUUtilization(50)

        self.gpu_metricsB = {
            "0": {
                GPUPowerUsage.tag: self.gpu_power_recordB,
                GPUUtilization.tag: self.gpu_util_recordB,
            }
        }

        self.throughput_recordB_0 = PerfThroughput(1000)
        self.latency_recordB_0 = PerfLatencyP99(20)

        self.throughput_recordB_1 = PerfThroughput(2000)
        self.latency_recordB_1 = PerfLatencyP99(30)

        self.perf_metricsB_0 = {
            PerfThroughput.tag: self.throughput_recordB_0,
            PerfLatencyP99.tag: self.latency_recordB_0,
        }

        self.perf_metricsB_1 = {
            PerfThroughput.tag: self.throughput_recordB_1,
            PerfLatencyP99.tag: self.latency_recordB_1,
        }

        self.rcmB = RunConfigMeasurement(self.gpu_metricsB)
        self.rcmB.add_model_config_measurement(
            model_name="modelB_0", perf_metrics=self.perf_metricsB_0
        )
        self.rcmB.add_model_config_measurement(
            model_name="modelB_1", perf_metrics=self.perf_metricsB_1
        )

    ###########################################################################
    # Accessor Tests
    ###########################################################################
    def test_basic_accessor_methods(self):
        """
        Test that values are properly initialized
        """

        #
        # GPU Metrics accessors
        self.assertEqual(self.rcmA.get_all_gpu_metrics(), self.gpu_metricsA)

        expected_gpu_metric = {
            "0": {GPUPowerUsage.tag: self.gpu_metricsA["0"].get(GPUPowerUsage.tag)}
        }
        self.assertEqual(
            expected_gpu_metric, self.rcmA.get_gpu_metric(GPUPowerUsage.tag)
        )

        #
        # MCM accessors
        self.assertIsNone(self.rcmA.get_model_config_measurement("ModelNotPresent"))

        expected_mcmA = ModelConfigMeasurement(self.perf_metricsA)
        expected_mcm_dict = {"modelA": expected_mcmA}
        self.assertEqual(self.rcmA.get_model_config_measurements(), expected_mcm_dict)
        self.assertEqual(
            self.rcmA.get_model_config_measurement("modelA"), expected_mcmA
        )

        #
        # Perf Metrics accessors
        expected_all_perf_metrics_dict = {"modelA": self.perf_metricsA}
        self.assertEqual(
            expected_all_perf_metrics_dict, self.rcmA.get_all_perf_metrics()
        )
        self.assertEqual(self.perf_metricsA, self.rcmA.get_model_perf_metrics("modelA"))
        self.assertEqual(
            self.perf_metricsA[PerfThroughput.tag],
            self.rcmA.get_model_perf_metric("modelA", PerfThroughput.tag),
        )
        self.assertEqual(
            self.perf_metricsA[PerfThroughput.tag].value(),
            self.rcmA.get_model_perf_metric_value("modelA", PerfThroughput.tag),
        )
        self.assertEqual(
            10,
            self.rcmA.get_model_perf_metric_value(
                "modelA", "MetricNotPresent", return_value=10
            ),
        )

        #
        # Weighted Perf Metrics accessor
        model_weights = {"modelB_0": 0.8, "modelB_1": 0.2}
        self.rcmB.set_model_weighting(model_weights)

        expected_weighted_perf_metric_values = {
            "modelB_0": self.perf_metricsB_0[PerfThroughput.tag].value() * 0.8,
            "modelB_1": self.perf_metricsB_1[PerfThroughput.tag].value() * 0.2,
        }

        self.assertEqual(
            expected_weighted_perf_metric_values,
            self.rcmB.get_weighted_perf_metric_values(PerfThroughput.tag),
        )

    ###########################################################################
    # Checkpoint Tests
    ###########################################################################
    def test_checkpoint_methods(self):
        """
        Checks to ensure checkpoint methods work as intended
        """
        rcmA_json = json.dumps(self.rcmA, default=checkpoint_encoder)

        rcmA_from_checkpoint = RunConfigMeasurement.read_from_checkpoint(
            json.loads(rcmA_json)
        )

        foo = rcmA_from_checkpoint.get_all_gpu_metrics()
        goo = self.rcmA.get_all_gpu_metrics()

        self.assertEqual(
            rcmA_from_checkpoint.get_all_gpu_metrics(), self.rcmA.get_all_gpu_metrics()
        )
        self.assertEqual(
            rcmA_from_checkpoint.get_model_config_measurements(),
            self.rcmA.get_model_config_measurements(),
        )

        # Catchall in case something new is added
        self.assertEqual(rcmA_from_checkpoint, self.rcmA)

    # def test_is_better_than(self):
    #     """
    #     Test to ensure measurement comparison is working as intended
    #     """
    #     # RCM0: 1000, 40    RCM1: 500, 30  weights:[1,3]
    #     # RCM0-A's throughput is better than RCM1-A (0.5)
    #     # RCM0-B's latency is worse than RCM1-B (-0.25)
    #     # Factoring in model config weighting
    #     # tips this is favor of RCM1 (0.125, -0.1875)
    #     self.assertFalse(self.rcm0.is_better_than(self.rcm1))

    #     # This tips the scale in the favor of RCM0 (0.2, -0.15)
    #     self.rcm0.set_model_config_weighting([2, 3])
    #     self.assertTrue(self.rcm0.is_better_than(self.rcm1))
    #     self.assertGreater(self.rcm0, self.rcm1)

    # def test_is_better_than_consistency(self):
    #     """
    #     Test to ensure measurement comparison is working correctly
    #     when the percentage gain is very close
    #     """
    #     # Throughputs
    #     #   RCM2{A/B}: {336,223}   RCM3{A/B}: {270,272}
    #     #   RCM2-A is 21.8% better than RCM3-A
    #     #   RMC2-B is 19.8% worse than RCM3-B
    #     # Therefore, RCM2 is (very slightly) better than RCM3
    #     self.assertTrue(self.rcm2.is_better_than(self.rcm3))
    #     self.assertFalse(self.rcm3.is_better_than(self.rcm2))

    # def test_compare_measurements(self):
    #     """
    #     Test to ensure compare measurement function returns
    #     the correct magnitude
    #     # RCM4's throughput is 1000
    #     # RCM5's throughput is 2000
    #     # Therefore, the magnitude is (RCM5 - RCM4) / avg throughput
    #     #                             (2000 - 1000) / 1500
    #     """
    #     rcm4_vs_rcm5 = self.rcm4.compare_measurements(self.rcm5)
    #     self.assertEqual(rcm4_vs_rcm5, 1000 / 1500)

    #     rcm5_vs_rcm4 = self.rcm5.compare_measurements(self.rcm4)
    #     self.assertEqual(rcm5_vs_rcm4, -1000 / 1500)

    # def test_is_passing_constraints_none(self):
    #     """
    #     Test to ensure constraints are reported as passing
    #     if none were specified
    #     """
    #     self.rcm5.set_constraint_manager(
    #         construct_constraint_manager(
    #             """
    #         profile_models:
    #           modelA
    #         """
    #         )
    #     )
    #     self.assertTrue(self.rcm5.is_passing_constraints())

    # def test_is_passing_constraints(self):
    #     """
    #     Test to ensure constraints are reported as
    #     passing/failing if model is above/below
    #     throughput threshold
    #     """
    #     constraint_manager = construct_constraint_manager(
    #         """
    #         profile_models:
    #           modelA:
    #             constraints:
    #               perf_throughput:
    #                 min: 500
    #         """
    #     )
    #     self.rcm5.set_constraint_manager(constraint_manager)

    #     self.assertTrue(self.rcm5.is_passing_constraints())

    #     constraint_manager = construct_constraint_manager(
    #         """
    #         profile_models:
    #           modelA:
    #             constraints:
    #               perf_throughput:
    #                 min: 3000
    #         """
    #     )
    #     self.rcm5.set_constraint_manager(constraint_manager)

    #     self.assertFalse(self.rcm5.is_passing_constraints())

    # def test_compare_constraints_none(self):
    #     """
    #     Checks case where either self or other is passing constraints
    #     """
    #     # RCM4's throughput is 1000
    #     # RCM5's throughput is 2000
    #     constraint_manager = construct_constraint_manager(
    #         """
    #         profile_models:
    #           modelA:
    #             constraints:
    #               perf_throughput:
    #                 min: 500
    #         """
    #     )
    #     self.rcm4.set_constraint_manager(constraint_manager)

    #     constraint_manager = construct_constraint_manager(
    #         """
    #         profile_models:
    #           modelA:
    #             constraints:
    #               perf_throughput:
    #                 min: 2500
    #         """
    #     )
    #     self.rcm5.set_constraint_manager(constraint_manager)

    #     self.assertEqual(self.rcm4.compare_constraints(self.rcm5), None)
    #     self.assertEqual(self.rcm5.compare_constraints(self.rcm4), None)

    # def test_compare_constraints_equal(self):
    #     """
    #     Test to ensure compare constraints reports zero when both
    #     RCMs are missing constraints by the same amount
    #     """
    #     # RCM4's throughput is 1000
    #     # RCM5's throughput is 2000
    #     constraint_manager = construct_constraint_manager(
    #         """
    #         profile_models:
    #           modelA:
    #             constraints:
    #               perf_throughput:
    #                 min: 1250
    #         """
    #     )
    #     self.rcm4.set_constraint_manager(constraint_manager)

    #     constraint_manager = construct_constraint_manager(
    #         """
    #         profile_models:
    #           modelA:
    #             constraints:
    #               perf_throughput:
    #                 min: 2500
    #         """
    #     )
    #     self.rcm5.set_constraint_manager(constraint_manager)

    #     # RCM4 is failing by 20%, RCM5 is failing by 20%
    #     self.assertEqual(self.rcm4.compare_constraints(self.rcm5), 0)
    #     self.assertEqual(self.rcm5.compare_constraints(self.rcm4), 0)

    # def test_compare_constraints_unequal(self):
    #     """
    #     Test to ensure compare constraints reports the correct
    #     value when the RCMs are both failing constraints by different
    #     amounts
    #     """
    #     # RCM4's throughput is 1000
    #     # RCM5's throughput is 2000
    #     constraint_manager = construct_constraint_manager(
    #         """
    #         profile_models:
    #           modelA:
    #             constraints:
    #               perf_throughput:
    #                 min: 2000
    #         """
    #     )
    #     self.rcm4.set_constraint_manager(constraint_manager)

    #     constraint_manager = construct_constraint_manager(
    #         """
    #         profile_models:
    #           modelA:
    #             constraints:
    #               perf_throughput:
    #                 min: 2500
    #         """
    #     )
    #     self.rcm5.set_constraint_manager(constraint_manager)

    #     # RCM4 is failing by 50%, RCM5 is failing by 20%
    #     self.assertEqual(self.rcm4.compare_constraints(self.rcm5), 0.30)
    #     self.assertEqual(self.rcm5.compare_constraints(self.rcm4), -0.30)

    # def test_calculate_weighted_percentage_gain(self):
    #     """
    #     Test to ensure weighted percentage gain is being calculated correctly
    #     """

    #     # RCM0: 1000, 40    RCM1: 500, 30  weights:[1,3]
    #     # RCM0-A's throughput is better than RCM1-A (0.5)
    #     # RCM0-B's latency is worse than RCM1-B (-0.25)
    #     # Factoring in model config weighting
    #     # tips this is favor of RCM1 (0.125, -0.1875)
    #     # However, by percentage RCM0 will be evaluated as slightly better
    #     # 100% on throughput, -25% on latency
    #     # Factoring in weighting, RCM0 is slightly better (100 - 75) / 4 = 6.25%
    #     self.assertEqual(self.rcm0.calculate_weighted_percentage_gain(self.rcm1), 6.25)

    #     # Changing the weighting tips the scale in the favor of RCM0 (0.2, -0.15)
    #     # And, from a percentage standpoint we get: (200 - 75) / 5 = 25%
    #     self.rcm0.set_model_config_weighting([2, 3])
    #     self.assertEqual(self.rcm0.calculate_weighted_percentage_gain(self.rcm1), 25.0)

    # def test_from_dict(self):
    #     """
    #     Test to ensure class can be correctly restored from a dictionary
    #     """
    #     rcm0_json = json.dumps(self.rcm0, default=default_encode)

    #     rcm0_from_dict = RunConfigMeasurement.from_dict(json.loads(rcm0_json))

    #     self.assertEqual(
    #         rcm0_from_dict.model_variants_name(), self.rcm0.model_variants_name()
    #     )
    #     self.assertEqual(rcm0_from_dict.gpu_data(), self.rcm0.gpu_data())
    #     self.assertEqual(rcm0_from_dict.non_gpu_data(), self.rcm0.non_gpu_data())
    #     self.assertEqual(
    #         list(rcm0_from_dict.data().values()), list(self.rcm0.data().values())
    #     )
    #     self.assertEqual(
    #         rcm0_from_dict._model_config_measurements,
    #         self.rcm0._model_config_measurements,
    #     )
    #     self.assertEqual(rcm0_from_dict._model_config_weights, [])

    # def _construct_rcm0(self):
    #     self.model_name = "modelA,modelB"
    #     self.model_config_name = ["modelA_config_0", "modelB_config_1"]
    #     self.model_variants_name = "".join(self.model_config_name)
    #     self.model_specific_pa_params = [
    #         {"batch_size": 1, "concurrency": 1},
    #         {"batch_size": 2, "concurrency": 2},
    #     ]

    #     self.gpu_metric_values = {
    #         "0": {"gpu_used_memory": 6000, "gpu_utilization": 60},
    #         "1": {"gpu_used_memory": 10000, "gpu_utilization": 20},
    #     }
    #     self.avg_gpu_metric_values = {"gpu_used_memory": 8000, "gpu_utilization": 40}

    #     self.rcm0_non_gpu_metric_values = [
    #         {
    #             # modelA_config_0
    #             "perf_throughput": 1000,
    #             "perf_latency_p99": 20,
    #             "cpu_used_ram": 1000,
    #         },
    #         {
    #             # modelB_config_1
    #             "perf_throughput": 2000,
    #             "perf_latency_p99": 40,
    #             "cpu_used_ram": 1500,
    #         },
    #     ]

    #     self.metric_objectives = [{"perf_throughput": 1}, {"perf_latency_p99": 1}]

    #     self.weights = [1, 3]

    #     self.rcm0_weighted_non_gpu_metric_values = []
    #     for index, non_gpu_metric_values in enumerate(self.rcm0_non_gpu_metric_values):
    #         self.rcm0_weighted_non_gpu_metric_values.append(
    #             {
    #                 objective: value * self.weights[index] / sum(self.weights)
    #                 for (objective, value) in non_gpu_metric_values.items()
    #             }
    #         )

    #     self.rcm0 = construct_run_config_measurement(
    #         self.model_name,
    #         self.model_config_name,
    #         self.model_specific_pa_params,
    #         self.gpu_metric_values,
    #         self.rcm0_non_gpu_metric_values,
    #         MagicMock(),
    #         self.metric_objectives,
    #         self.weights,
    #     )

    # def _construct_rcm1(self):
    #     model_name = "modelA,modelB"
    #     model_config_name = ["modelA_config_2", "modelB_config_3"]
    #     model_specific_pa_params = [
    #         {"batch_size": 3, "concurrency": 3},
    #         {"batch_size": 4, "concurrency": 4},
    #     ]

    #     gpu_metric_values = {
    #         "0": {"gpu_used_memory": 7000, "gpu_utilization": 40},
    #         "1": {"gpu_used_memory": 12000, "gpu_utilization": 30},
    #     }

    #     self.rcm1_non_gpu_metric_values = [
    #         {
    #             # modelA_config_2
    #             "perf_throughput": 500,
    #             "perf_latency_p99": 20,
    #             "cpu_used_ram": 1000,
    #         },
    #         {
    #             # modelB_config_3
    #             "perf_throughput": 1200,
    #             "perf_latency_p99": 30,
    #             "cpu_used_ram": 1500,
    #         },
    #     ]

    #     metric_objectives = [{"perf_throughput": 1}, {"perf_throughput": 1}]

    #     weights = [1, 3]

    #     self.rcm1_weighted_non_gpu_metric_values = []
    #     for index, non_gpu_metric_values in enumerate(self.rcm1_non_gpu_metric_values):
    #         self.rcm1_weighted_non_gpu_metric_values.append(
    #             {
    #                 objective: value * self.weights[index] / sum(weights)
    #                 for (objective, value) in non_gpu_metric_values.items()
    #             }
    #         )

    #     self.rcm1 = construct_run_config_measurement(
    #         model_name,
    #         model_config_name,
    #         model_specific_pa_params,
    #         gpu_metric_values,
    #         self.rcm1_non_gpu_metric_values,
    #         MagicMock(),
    #         metric_objectives,
    #         weights,
    #     )

    # def _construct_rcm2(self):
    #     model_name = "modelA,modelB"
    #     model_config_name = ["modelA_config_1", "modelB_config_2"]
    #     model_specific_pa_params = [
    #         {"batch_size": 3, "concurrency": 3},
    #         {"batch_size": 4, "concurrency": 4},
    #     ]

    #     gpu_metric_values = {
    #         "0": {"gpu_used_memory": 7000, "gpu_utilization": 40},
    #         "1": {"gpu_used_memory": 12000, "gpu_utilization": 30},
    #     }

    #     self.rcm2_non_gpu_metric_values = [
    #         {
    #             # modelA_config_1
    #             "perf_throughput": 336,
    #             "perf_latency_p99": 20,
    #             "cpu_used_ram": 1000,
    #         },
    #         {
    #             # modelB_config_2
    #             "perf_throughput": 223,
    #             "perf_latency_p99": 30,
    #             "cpu_used_ram": 1500,
    #         },
    #     ]

    #     metric_objectives = [{"perf_throughput": 1}, {"perf_throughput": 1}]

    #     weights = [1, 1]

    #     self.rcm2_weighted_non_gpu_metric_values = []
    #     for index, non_gpu_metric_values in enumerate(self.rcm2_non_gpu_metric_values):
    #         self.rcm2_weighted_non_gpu_metric_values.append(
    #             {
    #                 objective: value * self.weights[index] / sum(weights)
    #                 for (objective, value) in non_gpu_metric_values.items()
    #             }
    #         )

    #     self.rcm2 = construct_run_config_measurement(
    #         model_name,
    #         model_config_name,
    #         model_specific_pa_params,
    #         gpu_metric_values,
    #         self.rcm2_non_gpu_metric_values,
    #         MagicMock(),
    #         metric_objectives,
    #         weights,
    #     )

    # def _construct_rcm3(self):
    #     model_name = "modelA,modelB"
    #     model_config_name = ["modelA_config_1", "modelB_config_2"]
    #     model_specific_pa_params = [
    #         {"batch_size": 3, "concurrency": 3},
    #         {"batch_size": 4, "concurrency": 4},
    #     ]

    #     gpu_metric_values = {
    #         "0": {"gpu_used_memory": 7000, "gpu_utilization": 40},
    #         "1": {"gpu_used_memory": 12000, "gpu_utilization": 30},
    #     }

    #     self.rcm3_non_gpu_metric_values = [
    #         {
    #             # modelA_config_1
    #             "perf_throughput": 270,
    #             "perf_latency_p99": 20,
    #             "cpu_used_ram": 1000,
    #         },
    #         {
    #             # modelB_config_2
    #             "perf_throughput": 272,
    #             "perf_latency_p99": 30,
    #             "cpu_used_ram": 1500,
    #         },
    #     ]

    #     metric_objectives = [{"perf_throughput": 1}, {"perf_throughput": 1}]

    #     weights = [1, 1]

    #     self.rcm3_weighted_non_gpu_metric_values = []
    #     for index, non_gpu_metric_values in enumerate(self.rcm3_non_gpu_metric_values):
    #         self.rcm3_weighted_non_gpu_metric_values.append(
    #             {
    #                 objective: value * self.weights[index] / sum(weights)
    #                 for (objective, value) in non_gpu_metric_values.items()
    #             }
    #         )

    #     self.rcm3 = construct_run_config_measurement(
    #         model_name,
    #         model_config_name,
    #         model_specific_pa_params,
    #         gpu_metric_values,
    #         self.rcm3_non_gpu_metric_values,
    #         MagicMock(),
    #         metric_objectives,
    #         weights,
    #     )

    # def _construct_rcm4(self):
    #     model_name = "modelA"
    #     model_config_name = ["modelA_config_0"]
    #     model_specific_pa_params = [
    #         {"batch_size": 1, "concurrency": 1},
    #         {"batch_size": 2, "concurrency": 2},
    #     ]

    #     gpu_metric_values = {
    #         "0": {"gpu_used_memory": 6000, "gpu_utilization": 60},
    #         "1": {"gpu_used_memory": 10000, "gpu_utilization": 20},
    #     }

    #     self.rcm4_non_gpu_metric_values = [
    #         {
    #             # modelA_config_0
    #             "perf_throughput": 1000,
    #             "perf_latency_p99": 20,
    #             "cpu_used_ram": 1000,
    #         },
    #     ]

    #     metric_objectives = [{"perf_throughput": 1}]

    #     weights = [1]

    #     self.rcm4_weighted_non_gpu_metric_values = []
    #     for index, non_gpu_metric_values in enumerate(self.rcm4_non_gpu_metric_values):
    #         self.rcm4_weighted_non_gpu_metric_values.append(
    #             {
    #                 objective: value * self.weights[index] / sum(self.weights)
    #                 for (objective, value) in non_gpu_metric_values.items()
    #             }
    #         )

    #     self.rcm4 = construct_run_config_measurement(
    #         model_name,
    #         model_config_name,
    #         model_specific_pa_params,
    #         gpu_metric_values,
    #         self.rcm4_non_gpu_metric_values,
    #         MagicMock(),
    #         metric_objectives,
    #         weights,
    #     )

    # def _construct_rcm5(self):
    #     model_name = "modelA"
    #     model_config_name = ["modelA_config_0"]
    #     model_specific_pa_params = [
    #         {"batch_size": 1, "concurrency": 1},
    #         {"batch_size": 2, "concurrency": 2},
    #     ]

    #     gpu_metric_values = {
    #         "0": {"gpu_used_memory": 6000, "gpu_utilization": 60},
    #         "1": {"gpu_used_memory": 10000, "gpu_utilization": 20},
    #     }

    #     self.rcm5_non_gpu_metric_values = [
    #         {
    #             # modelA_config_0
    #             "perf_throughput": 2000,
    #             "perf_latency_p99": 20,
    #             "cpu_used_ram": 1000,
    #         },
    #     ]

    #     metric_objectives = [{"perf_throughput": 1}]

    #     weights = [1]

    #     self.rcm5_weighted_non_gpu_metric_values = []
    #     for index, non_gpu_metric_values in enumerate(self.rcm5_non_gpu_metric_values):
    #         self.rcm5_weighted_non_gpu_metric_values.append(
    #             {
    #                 objective: value * self.weights[index] / sum(self.weights)
    #                 for (objective, value) in non_gpu_metric_values.items()
    #             }
    #         )

    #     self.rcm5 = construct_run_config_measurement(
    #         model_name,
    #         model_config_name,
    #         model_specific_pa_params,
    #         gpu_metric_values,
    #         self.rcm5_non_gpu_metric_values,
    #         MagicMock(),
    #         metric_objectives,
    #         weights,
    #     )


if __name__ == "__main__":
    unittest.main()
