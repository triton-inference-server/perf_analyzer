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
from math import log2
from unittest.mock import patch

from genai_perf.config.generate.search_parameters import (
    SearchCategory,
    SearchParameters,
    SearchUsage,
)
from genai_perf.config.input.config_command import ConfigCommand, Subcommand
from genai_perf.config.input.config_defaults import AnalyzeDefaults, Range
from genai_perf.exceptions import GenAIPerfException


class TestSearchParameters(unittest.TestCase):
    def setUp(self):
        self.config = ConfigCommand(user_config={})
        self.config.model_names = ["test_model"]

        self.search_parameters = SearchParameters(config=self.config)

        self.search_parameters._add_search_parameter(
            name="concurrency",
            usage=SearchUsage.RUNTIME_PA,
            category=SearchCategory.EXPONENTIAL,
            min_range=log2(AnalyzeDefaults.MIN_CONCURRENCY),
            max_range=log2(AnalyzeDefaults.MAX_CONCURRENCY),
        )

        self.search_parameters._add_search_parameter(
            name="num_dataset_entries",
            usage=SearchUsage.RUNTIME_GAP,
            category=SearchCategory.INTEGER,
            min_range=AnalyzeDefaults.MIN_NUM_DATASET_ENTRIES,
            max_range=AnalyzeDefaults.MAX_NUM_DATASET_ENTRIES,
        )

        self.search_parameters._add_search_parameter(
            name="size",
            usage=SearchUsage.BUILD,
            category=SearchCategory.STR_LIST,
            enumerated_list=["FP8", "FP16", "FP32"],
        )

    def tearDown(self):
        patch.stopall()

    def test_exponential_parameter(self):
        """
        Test exponential parameter, accessing dataclass directly
        """

        parameter = self.search_parameters.get_parameter("concurrency")

        self.assertEqual(SearchUsage.RUNTIME_PA, parameter.usage)
        self.assertEqual(SearchCategory.EXPONENTIAL, parameter.category)
        self.assertEqual(log2(AnalyzeDefaults.MIN_CONCURRENCY), parameter.min_range)
        self.assertEqual(log2(AnalyzeDefaults.MAX_CONCURRENCY), parameter.max_range)

    def test_integer_parameter(self):
        """
        Test integer parameter, using accessor methods
        """

        self.assertEqual(
            SearchUsage.RUNTIME_GAP,
            self.search_parameters.get_type("num_dataset_entries"),
        )
        self.assertEqual(
            SearchCategory.INTEGER,
            self.search_parameters.get_category("num_dataset_entries"),
        )
        self.assertEqual(
            Range(
                min=AnalyzeDefaults.MIN_NUM_DATASET_ENTRIES,
                max=AnalyzeDefaults.MAX_NUM_DATASET_ENTRIES,
            ),
            self.search_parameters.get_range("num_dataset_entries"),
        )

    def test_list_parameter(self):
        """
        Test list parameter, using accessor methods
        """

        self.assertEqual(
            SearchUsage.BUILD,
            self.search_parameters.get_type("size"),
        )
        self.assertEqual(
            SearchCategory.STR_LIST,
            self.search_parameters.get_category("size"),
        )
        self.assertEqual(
            ["FP8", "FP16", "FP32"], self.search_parameters.get_list("size")
        )

    def test_illegal_inputs(self):
        """
        Check that an exception is raised for illegal input combos
        """
        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="concurrency",
                usage=SearchUsage.RUNTIME_PA,
                category=SearchCategory.EXPONENTIAL,
                max_range=10,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="concurrency",
                usage=SearchUsage.RUNTIME_PA,
                category=SearchCategory.EXPONENTIAL,
                min_range=0,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="concurrency",
                usage=SearchUsage.RUNTIME_PA,
                category=SearchCategory.EXPONENTIAL,
                min_range=10,
                max_range=9,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="size",
                usage=SearchUsage.BUILD,
                category=SearchCategory.INT_LIST,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="size",
                usage=SearchUsage.BUILD,
                category=SearchCategory.STR_LIST,
                enumerated_list=["FP8", "FP16", "FP32"],
                min_range=0,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="size",
                usage=SearchUsage.BUILD,
                category=SearchCategory.STR_LIST,
                enumerated_list=["FP8", "FP16", "FP32"],
                max_range=10,
            )

    def test_number_of_configs_range(self):
        """
        Test number of configs for a range (INTEGER/EXPONENTIAL)
        """

        # INTEGER
        # =====================================================================
        num_of_configs = self.search_parameters._number_of_configurations_for_parameter(
            self.search_parameters.get_parameter("num_dataset_entries")
        )
        expected_num_of_configs = (
            AnalyzeDefaults.MAX_NUM_DATASET_ENTRIES
            - AnalyzeDefaults.MIN_NUM_DATASET_ENTRIES
            + 1
        )

        self.assertEqual(expected_num_of_configs, num_of_configs)

        # EXPONENTIAL
        # =====================================================================
        num_of_configs = self.search_parameters._number_of_configurations_for_parameter(
            self.search_parameters.get_parameter("concurrency")
        )
        expected_num_of_configs = (
            log2(AnalyzeDefaults.MAX_CONCURRENCY)
            - log2(AnalyzeDefaults.MIN_CONCURRENCY)
            + 1
        )

        self.assertEqual(expected_num_of_configs, num_of_configs)

    def test_number_of_configs_list(self):
        """
        Test number of configs for a list
        """

        num_of_configs = self.search_parameters._number_of_configurations_for_parameter(
            self.search_parameters.get_parameter("size")
        )
        self.assertEqual(3, num_of_configs)

    def test_total_possible_configurations(self):
        """
        Test number of total possible configurations
        """
        total_num_of_possible_configurations = (
            self.search_parameters.number_of_total_possible_configurations()
        )

        # concurrency (11) * size (3) * num_dataset_entries (901)
        self.assertEqual(11 * 3 * 901, total_num_of_possible_configurations)

    #######################################################################
    # Test Analyze Configs
    #######################################################################
    def test_default_analyze_config(self):
        """
        Test that search parameters are created correctly when calling
        default analyze subcommand
        """
        search_parameters = SearchParameters(config=self.config)

        concurrency = search_parameters.get_parameter("concurrency")
        self.assertEqual(SearchUsage.RUNTIME_PA, concurrency.usage)
        self.assertEqual(SearchCategory.EXPONENTIAL, concurrency.category)
        self.assertEqual(log2(AnalyzeDefaults.MIN_CONCURRENCY), concurrency.min_range)
        self.assertEqual(log2(AnalyzeDefaults.MAX_CONCURRENCY), concurrency.max_range)

    def test_custom_analyze_config(self):
        """
        Test that search parameters are created correctly when calling
        default analyze subcommand
        """
        config = self.config
        config.analyze.sweep_parameters = {
            "num_dataset_entries": [10, 50, 100],
            "request_rate": Range(
                min=AnalyzeDefaults.MIN_REQUEST_RATE,
                max=AnalyzeDefaults.MAX_REQUEST_RATE,
            ),
        }
        search_parameters = SearchParameters(config=config)

        num_dataset_entries = search_parameters.get_parameter("num_dataset_entries")
        self.assertEqual(SearchUsage.RUNTIME_GAP, num_dataset_entries.usage)
        self.assertEqual(SearchCategory.INT_LIST, num_dataset_entries.category)
        self.assertEqual([10, 50, 100], num_dataset_entries.enumerated_list)

        request_rate = search_parameters.get_parameter("request_rate")
        self.assertEqual(SearchUsage.RUNTIME_PA, request_rate.usage)
        self.assertEqual(SearchCategory.EXPONENTIAL, request_rate.category)
        self.assertEqual(log2(AnalyzeDefaults.MIN_REQUEST_RATE), request_rate.min_range)
        self.assertEqual(log2(AnalyzeDefaults.MAX_REQUEST_RATE), request_rate.max_range)


if __name__ == "__main__":
    unittest.main()
