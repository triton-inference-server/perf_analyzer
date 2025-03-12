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

import unittest
from unittest.mock import patch

from genai_perf.record.record import RecordType


class TestRecord(unittest.TestCase):
    """
    The record types in the genai_perf.record.types package are contextual
        when it uses 'less than' (<) and 'greater than' (>) operators.

    The 'less than' and 'greater than' operators are overloaded to
        mean 'worse than' and 'better than' respectively.

    Some record types treat MORE as better
        (eg, gpu_free_memory, cpu_available_ram)
    Other record types treat LESS as better
        (eg, gpu_used_memory, cpu_used_ram)

    So, when comparing two objects of type 'cpu_used_ram'
        12 > 13 is actually true, since 12 'is better than' 13.
    """

    def setUp(self):
        ###########################################################################
        # Setup & Teardown
        ###########################################################################
        record_types = RecordType.get_all_record_types()
        self.all_record_types = record_types.values()

        self.less_is_better_types = {
            record_types[t]
            for t in [
                "request_latency_min",
                "request_latency_max",
                "request_latency_avg",
                "request_latency_std",
                "request_latency_p1",
                "request_latency_p5",
                "request_latency_p10",
                "request_latency_p25",
                "request_latency_p50",
                "request_latency_p75",
                "request_latency_p90",
                "request_latency_p95",
                "request_latency_p99",
                "inter_token_latency_min",
                "inter_token_latency_max",
                "inter_token_latency_avg",
                "inter_token_latency_std",
                "inter_token_latency_p1",
                "inter_token_latency_p5",
                "inter_token_latency_p10",
                "inter_token_latency_p25",
                "inter_token_latency_p50",
                "inter_token_latency_p75",
                "inter_token_latency_p90",
                "inter_token_latency_p95",
                "inter_token_latency_p99",
                "time_to_first_token_min",
                "time_to_first_token_max",
                "time_to_first_token_avg",
                "time_to_first_token_std",
                "time_to_first_token_p1",
                "time_to_first_token_p5",
                "time_to_first_token_p10",
                "time_to_first_token_p25",
                "time_to_first_token_p50",
                "time_to_first_token_p75",
                "time_to_first_token_p90",
                "time_to_first_token_p95",
                "time_to_first_token_p99",
                "time_to_second_token_min",
                "time_to_second_token_max",
                "time_to_second_token_avg",
                "time_to_second_token_std",
                "time_to_second_token_p1",
                "time_to_second_token_p5",
                "time_to_second_token_p10",
                "time_to_second_token_p25",
                "time_to_second_token_p50",
                "time_to_second_token_p75",
                "time_to_second_token_p90",
                "time_to_second_token_p95",
                "time_to_second_token_p99",
                "gpu_power_usage_avg",
                "gpu_power_usage_min",
                "gpu_power_usage_max",
                "gpu_power_usage_std",
                "gpu_power_usage_p1",
                "gpu_power_usage_p5",
                "gpu_power_usage_p10",
                "gpu_power_usage_p25",
                "gpu_power_usage_p50",
                "gpu_power_usage_p75",
                "gpu_power_usage_p90",
                "gpu_power_usage_p95",
                "gpu_power_usage_p99",
                "energy_consumption_avg",
                "energy_consumption_min",
                "energy_consumption_max",
                "energy_consumption_std",
                "energy_consumption_p1",
                "energy_consumption_p5",
                "energy_consumption_p10",
                "energy_consumption_p25",
                "energy_consumption_p50",
                "energy_consumption_p75",
                "energy_consumption_p90",
                "energy_consumption_p95",
                "energy_consumption_p99",
            ]
        }

        self.more_is_better_types = {
            record_types[t]
            for t in [
                "request_throughput_avg",
                "request_goodput_avg",
                "request_count_avg",
                "output_token_throughput_avg",
                "output_token_throughput_per_request_min",
                "output_token_throughput_per_request_max",
                "output_token_throughput_per_request_avg",
                "output_token_throughput_per_request_std",
                "output_token_throughput_per_request_p1",
                "output_token_throughput_per_request_p5",
                "output_token_throughput_per_request_p10",
                "output_token_throughput_per_request_p25",
                "output_token_throughput_per_request_p50",
                "output_token_throughput_per_request_p75",
                "output_token_throughput_per_request_p90",
                "output_token_throughput_per_request_p95",
                "output_token_throughput_per_request_p99",
                "output_sequence_length_min",
                "output_sequence_length_max",
                "output_sequence_length_avg",
                "output_sequence_length_std",
                "output_sequence_length_p1",
                "output_sequence_length_p5",
                "output_sequence_length_p10",
                "output_sequence_length_p25",
                "output_sequence_length_p50",
                "output_sequence_length_p75",
                "output_sequence_length_p90",
                "output_sequence_length_p95",
                "output_sequence_length_p99",
                "input_sequence_length_min",
                "input_sequence_length_max",
                "input_sequence_length_avg",
                "input_sequence_length_std",
                "input_sequence_length_p1",
                "input_sequence_length_p5",
                "input_sequence_length_p10",
                "input_sequence_length_p25",
                "input_sequence_length_p50",
                "input_sequence_length_p75",
                "input_sequence_length_p90",
                "input_sequence_length_p95",
                "input_sequence_length_p99",
                "gpu_power_limit_avg",
                "gpu_utilization_min",
                "gpu_utilization_max",
                "gpu_utilization_avg",
                "gpu_utilization_std",
                "gpu_utilization_p1",
                "gpu_utilization_p5",
                "gpu_utilization_p10",
                "gpu_utilization_p25",
                "gpu_utilization_p50",
                "gpu_utilization_p75",
                "gpu_utilization_p90",
                "gpu_utilization_p95",
                "gpu_utilization_p99",
                "total_gpu_memory_avg",
                "gpu_memory_used_min",
                "gpu_memory_used_max",
                "gpu_memory_used_avg",
                "gpu_memory_used_std",
                "gpu_memory_used_p1",
                "gpu_memory_used_p5",
                "gpu_memory_used_p10",
                "gpu_memory_used_p25",
                "gpu_memory_used_p50",
                "gpu_memory_used_p75",
                "gpu_memory_used_p90",
                "gpu_memory_used_p95",
                "gpu_memory_used_p99",
            ]
        }

    def tearDown(self):
        patch.stopall()

    ###########################################################################
    # Completeness Tests
    ###########################################################################
    def test_counts(self):
        """
        Make sure that all 'worse than' and 'better than' tests are tested
        """
        total_count = len(self.all_record_types)
        less_is_better_count = len(self.less_is_better_types)
        more_is_better_count = len(self.more_is_better_types)
        self.assertEqual(total_count, less_is_better_count + more_is_better_count)

    ###########################################################################
    # Basic Operation Tests
    ###########################################################################
    def test_add(self):
        """
        Test __add__ function for
        each record type
        """

        for record_type in self.all_record_types:
            metric1 = record_type(value=5)
            metric2 = record_type(value=9)
            metric3 = metric1 + metric2
            self.assertIsInstance(metric3, record_type)
            self.assertEqual(metric3.value(), 14)

    def test_sub(self):
        """
        Test __sub__ function for
        each record type
        """

        for record_type in self.all_record_types:
            metric1 = record_type(value=10)
            metric2 = record_type(value=3)
            metric3 = metric1 - metric2
            self.assertIsInstance(metric3, record_type)
            if record_type in self.less_is_better_types:
                self.assertEqual(metric3.value(), -7)
            elif record_type in self.more_is_better_types:
                self.assertEqual(metric3.value(), 7)

    def test_mult(self):
        """
        Test __mult__ function for
        each record type
        """

        for record_type in self.all_record_types:
            metric1 = record_type(value=6)
            metric2 = metric1 * 2
            self.assertIsInstance(metric2, record_type)
            self.assertEqual(metric2.value(), 12)

    def test_div(self):
        """
        Test __div__ function for
        each record type
        """

        for record_type in self.all_record_types:
            metric1 = record_type(value=60)
            metric2 = metric1 / 12
            self.assertIsInstance(metric2, record_type)
            self.assertEqual(metric2.value(), 5)

    def test_compare(self):
        """
        Test __lt__, __eq__, __gt__
        functions for each record type
        """

        for record_type in self.all_record_types:
            metric1 = record_type(value=10.6)
            metric2 = record_type(value=3.2)

            # Test __lt__ (True if 1 worse than 2)
            if record_type in self.less_is_better_types:
                self.assertLess(metric1, metric2)
            elif record_type in self.more_is_better_types:
                self.assertLess(metric2, metric1)

            # Test __gt__ (True if 1 better than 2)
            if record_type in self.less_is_better_types:
                self.assertLess(metric1, metric2)
            elif record_type in self.more_is_better_types:
                self.assertGreater(metric1, metric2)

            # Test __eq__
            metric1 = record_type(value=12)
            metric2 = record_type(value=12)
            self.assertEqual(metric1, metric2)

    ###########################################################################
    # Method Tests
    ###########################################################################
    def test_value(self):
        """
        Test the value method
        """
        avg_value = RecordType.get_all_record_types()[
            "request_latency_p99"
        ].value_function()([10, 50, 100, 40])

        total_value = RecordType.get_all_record_types()[
            "request_throughput_avg"
        ].value_function()([10, 50, 100, 40])

        self.assertEqual(avg_value, 50)
        self.assertEqual(total_value, 200)

    def test_calculate_percentage_gain(self):
        """
        Test that percentage gain is calculated correctly
        """
        for record_type in self.all_record_types:
            metric1 = record_type(value=10)
            metric2 = record_type(value=5)

            if record_type in self.less_is_better_types:
                self.assertEqual(metric1.calculate_percentage_gain(metric2), -50)
            else:
                self.assertEqual(metric1.calculate_percentage_gain(metric2), 100)


if __name__ == "__main__":
    unittest.main()
