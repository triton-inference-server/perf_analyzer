# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

        self.less_is_better_types = set()
        self.more_is_better_types = set()

        for record_cls in self.all_record_types:
            metric = record_cls(value=1)
            if metric._positive_is_better():
                self.more_is_better_types.add(record_cls)
            else:
                self.less_is_better_types.add(record_cls)

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
            expected = -7 if record_type in self.less_is_better_types else 7
            self.assertEqual(metric3.value(), expected)

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
                self.assertGreater(metric2, metric1)
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

            expected_gain = -50 if record_type in self.less_is_better_types else 100
            self.assertEqual(metric1.calculate_percentage_gain(metric2), expected_gain)


if __name__ == "__main__":
    unittest.main()
