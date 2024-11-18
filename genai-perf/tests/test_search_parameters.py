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
from copy import deepcopy
from math import log2
from unittest.mock import patch

from genai_perf.config.generate.search_parameters import (
    SearchCategory,
    SearchParameters,
    SearchUsage,
)
from genai_perf.config.input.config_command import (
    ConfigCommand,
    Range,
    RunConfigDefaults,
)
from genai_perf.exceptions import GenAIPerfException


class TestSearchParameters(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(ConfigCommand(model_names=["test_model"]))

        self.search_parameters = SearchParameters(config=self.config.optimize)

        self.search_parameters._add_search_parameter(
            name="concurrency",
            usage=SearchUsage.RUNTIME_PA,
            category=SearchCategory.EXPONENTIAL,
            min_range=log2(RunConfigDefaults.MIN_CONCURRENCY),
            max_range=log2(RunConfigDefaults.MAX_CONCURRENCY),
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
        self.assertEqual(log2(RunConfigDefaults.MIN_CONCURRENCY), parameter.min_range)
        self.assertEqual(log2(RunConfigDefaults.MAX_CONCURRENCY), parameter.max_range)

    def test_integer_parameter(self):
        """
        Test integer parameter, using accessor methods
        """

        self.assertEqual(
            SearchUsage.MODEL,
            self.search_parameters.get_type("instance_count"),
        )
        self.assertEqual(
            SearchCategory.INTEGER,
            self.search_parameters.get_category("instance_count"),
        )
        self.assertEqual(
            Range(
                min=RunConfigDefaults.MIN_INSTANCE_COUNT,
                max=RunConfigDefaults.MAX_INSTANCE_COUNT,
            ),
            self.search_parameters.get_range("instance_count"),
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

    def test_search_parameter_creation_optimize_default(self):
        """
        Test that search parameters are correctly created in default optimize case
        """

        config = deepcopy(ConfigCommand(model_names=["test_model"]))
        search_parameters = SearchParameters(self.config.optimize)

        #######################################################################
        # Model Config
        #######################################################################

        # Batch Size
        # =====================================================================
        model_batch_size = search_parameters.get_parameter("model_batch_size")
        self.assertEqual(SearchUsage.MODEL, model_batch_size.usage)
        self.assertEqual(SearchCategory.EXPONENTIAL, model_batch_size.category)
        self.assertEqual(
            log2(RunConfigDefaults.MIN_MODEL_BATCH_SIZE),
            model_batch_size.min_range,
        )
        self.assertEqual(
            log2(RunConfigDefaults.MAX_MODEL_BATCH_SIZE),
            model_batch_size.max_range,
        )

        # Instance Count
        # =====================================================================
        instance_count = search_parameters.get_parameter("instance_count")
        self.assertEqual(SearchUsage.MODEL, instance_count.usage)
        self.assertEqual(SearchCategory.INTEGER, instance_count.category)
        self.assertEqual(RunConfigDefaults.MIN_INSTANCE_COUNT, instance_count.min_range)
        self.assertEqual(RunConfigDefaults.MAX_INSTANCE_COUNT, instance_count.max_range)

        # Max Queue Delay
        max_queue_delay = search_parameters.get_parameter("max_queue_delay")
        self.assertIsNone(max_queue_delay)

        #######################################################################
        # PA Config
        #######################################################################

        # Batch size
        # =====================================================================
        runtime_batch_size = search_parameters.get_parameter("runtime_batch_size")
        self.assertEqual(SearchUsage.RUNTIME_PA, runtime_batch_size.usage)
        self.assertEqual(SearchCategory.INT_LIST, runtime_batch_size.category)
        self.assertEqual(
            [RunConfigDefaults.PA_BATCH_SIZE], runtime_batch_size.enumerated_list
        )

        # Concurrency - this is not set because use_concurrency_formula is True
        # =====================================================================
        concurrency = search_parameters.get_parameter("concurrency")

        self.assertIsNone(concurrency)

        # Request Rate
        # =====================================================================
        request_rate = search_parameters.get_parameter("request_rate")
        self.assertIsNone(request_rate)

    def test_search_parameter_no_concurrency_formula(self):
        """
        Test that search parameters are correctly created when concurrency formula is disabled
        """
        config = deepcopy(ConfigCommand(model_names=["test_model"]))
        config.optimize.perf_analyzer.use_concurrency_formula = False

        search_parameters = SearchParameters(config.optimize)

        concurrency = search_parameters.get_parameter("concurrency")
        self.assertEqual(SearchUsage.RUNTIME_PA, concurrency.usage)
        self.assertEqual(SearchCategory.EXPONENTIAL, concurrency.category)
        self.assertEqual(log2(RunConfigDefaults.MIN_CONCURRENCY), concurrency.min_range)
        self.assertEqual(log2(RunConfigDefaults.MAX_CONCURRENCY), concurrency.max_range)

    def test_search_parameter_request_rate(self):
        """
        Test that request rate is used when specified in config
        """
        config = deepcopy(ConfigCommand(model_names=["test_model"]))
        config.optimize.perf_analyzer.stimulus_type = "request_rate"

        search_parameters = SearchParameters(config.optimize)

        request_rate = search_parameters.get_parameter("request_rate")
        self.assertEqual(SearchUsage.RUNTIME_PA, request_rate.usage)
        self.assertEqual(SearchCategory.EXPONENTIAL, request_rate.category)
        self.assertEqual(
            log2(RunConfigDefaults.MIN_REQUEST_RATE), request_rate.min_range
        )
        self.assertEqual(
            log2(RunConfigDefaults.MAX_REQUEST_RATE), request_rate.max_range
        )

    def test_number_of_configs_range(self):
        """
        Test number of configs for a range (INTEGER/EXPONENTIAL)
        """

        # INTEGER
        # =====================================================================
        num_of_configs = self.search_parameters._number_of_configurations_for_parameter(
            self.search_parameters.get_parameter("instance_count")
        )
        self.assertEqual(5, num_of_configs)

        # EXPONENTIAL
        # =====================================================================
        num_of_configs = self.search_parameters._number_of_configurations_for_parameter(
            self.search_parameters.get_parameter("concurrency")
        )
        self.assertEqual(11, num_of_configs)

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

        # model_batch_size (8) * instance count (5) * concurrency (11) * size (3)
        self.assertEqual(8 * 5 * 11 * 3, total_num_of_possible_configurations)

    #######################################################################
    # Test Analyze Configs
    #######################################################################
    def test_default_analyze_config(self):
        """
        Test that search parameters are created correctly when calling
        default analyze subcommand
        """
        search_parameters = SearchParameters(config=self.config.analyze)

        concurrency = search_parameters.get_parameter("concurrency")
        self.assertEqual(SearchUsage.RUNTIME_PA, concurrency.usage)
        self.assertEqual(SearchCategory.EXPONENTIAL, concurrency.category)
        self.assertEqual(log2(RunConfigDefaults.MIN_CONCURRENCY), concurrency.min_range)
        self.assertEqual(log2(RunConfigDefaults.MAX_CONCURRENCY), concurrency.max_range)

    def test_custom_analyze_config(self):
        """
        Test that search parameters are created correctly when calling
        default analyze subcommand
        """
        config = deepcopy(self.config.analyze)
        config.sweep_parameters = {
            "num_prompts": [10, 50, 100],
            "request_rate": Range(
                min=RunConfigDefaults.MIN_REQUEST_RATE,
                max=RunConfigDefaults.MAX_REQUEST_RATE,
            ),
        }
        search_parameters = SearchParameters(config=config)

        num_prompts = search_parameters.get_parameter("num_prompts")
        self.assertEqual(SearchUsage.RUNTIME_GAP, num_prompts.usage)
        self.assertEqual(SearchCategory.INT_LIST, num_prompts.category)
        self.assertEqual([10, 50, 100], num_prompts.enumerated_list)

        request_rate = search_parameters.get_parameter("request_rate")
        self.assertEqual(SearchUsage.RUNTIME_PA, request_rate.usage)
        self.assertEqual(SearchCategory.EXPONENTIAL, request_rate.category)
        self.assertEqual(
            log2(RunConfigDefaults.MIN_REQUEST_RATE), request_rate.min_range
        )
        self.assertEqual(
            log2(RunConfigDefaults.MAX_REQUEST_RATE), request_rate.max_range
        )


if __name__ == "__main__":
    unittest.main()
