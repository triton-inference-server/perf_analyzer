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

        self.search_parameters = SearchParameters(config=self.config)

        self.search_parameters._add_search_parameter(
            name="concurrency",
            usage=SearchUsage.RUNTIME,
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

        self.assertEqual(SearchUsage.RUNTIME, parameter.usage)
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
                usage=SearchUsage.RUNTIME,
                category=SearchCategory.EXPONENTIAL,
                max_range=10,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="concurrency",
                usage=SearchUsage.RUNTIME,
                category=SearchCategory.EXPONENTIAL,
                min_range=0,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="concurrency",
                usage=SearchUsage.RUNTIME,
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
        search_parameters = SearchParameters(config)

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
        self.assertEqual(SearchUsage.RUNTIME, runtime_batch_size.usage)
        self.assertEqual(SearchCategory.INT_LIST, runtime_batch_size.category)
        self.assertEqual(
            RunConfigDefaults.PA_BATCH_SIZE, runtime_batch_size.enumerated_list
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

        search_parameters = SearchParameters(config)

        concurrency = search_parameters.get_parameter("concurrency")
        self.assertEqual(SearchUsage.RUNTIME, concurrency.usage)
        self.assertEqual(SearchCategory.EXPONENTIAL, concurrency.category)
        self.assertEqual(log2(RunConfigDefaults.MIN_CONCURRENCY), concurrency.min_range)
        self.assertEqual(log2(RunConfigDefaults.MAX_CONCURRENCY), concurrency.max_range)

    def test_search_parameter_request_rate(self):
        """
        Test that request rate is used when specified in config
        """
        config = deepcopy(ConfigCommand(model_names=["test_model"]))
        config.optimize.perf_analyzer.stimulus_type = "request_rate"

        search_parameters = SearchParameters(config)

        request_rate = search_parameters.get_parameter("request_rate")
        self.assertEqual(SearchUsage.RUNTIME, request_rate.usage)
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

    # TODO: OPTIMIZE:
    # This will be enabled once BLS support is added
    #
    # def test_search_parameter_creation_bls_default(self):
    #     """
    #     Test that search parameters are correctly created in default BLS optuna case
    #     """

    #     args = [
    #         "model-analyzer",
    #         "profile",
    #         "--model-repository",
    #         "cli-repository",
    #         "-f",
    #         "path-to-config-file",
    #         "--run-config-search-mode",
    #         "optuna",
    #     ]

    #     yaml_content = """
    #     profile_models: add_sub
    #     bls_composing_models: add,sub
    #     """

    #     config = TestConfig()._evaluate_config(args=args, yaml_content=yaml_content)

    #     analyzer = Analyzer(config, MagicMock(), MagicMock(), MagicMock())

    #     mock_model_config = MockModelConfig()
    #     mock_model_config.start()
    #     analyzer._populate_search_parameters(MagicMock(), MagicMock())
    #     analyzer._populate_composing_search_parameters(MagicMock(), MagicMock())
    #     mock_model_config.stop()

    #     # ADD_SUB
    #     # =====================================================================
    #     # The top level model of a BLS does not search max batch size (always 1)

    #     # max_batch_size
    #     max_batch_size = analyzer._search_parameters["add_sub"].get_parameter(
    #         "max_batch_size"
    #     )
    #     self.assertIsNone(max_batch_size)

    #     # concurrency
    #     concurrency = analyzer._search_parameters["add_sub"].get_parameter(
    #         "concurrency"
    #     )
    #     self.assertEqual(SearchUsage.RUNTIME, concurrency.usage)
    #     self.assertEqual(SearchCategory.EXPONENTIAL, concurrency.category)
    #     self.assertEqual(
    #         log2(default.DEFAULT_RUN_CONFIG_MIN_CONCURRENCY), concurrency.min_range
    #     )
    #     self.assertEqual(
    #         log2(default.DEFAULT_RUN_CONFIG_MAX_CONCURRENCY), concurrency.max_range
    #     )

    #     # instance_group
    #     instance_group = analyzer._search_parameters["add_sub"].get_parameter(
    #         "instance_group"
    #     )
    #     self.assertEqual(SearchUsage.MODEL, instance_group.usage)
    #     self.assertEqual(SearchCategory.INTEGER, instance_group.category)
    #     self.assertEqual(
    #         default.DEFAULT_RUN_CONFIG_MIN_INSTANCE_COUNT, instance_group.min_range
    #     )
    #     self.assertEqual(
    #         default.DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT, instance_group.max_range
    #     )

    #     # ADD/SUB (composing models)
    #     # =====================================================================
    #     # Composing models do not search concurrency and has no max batch size

    #     # max_batch_size
    #     max_batch_size = analyzer._composing_search_parameters["add"].get_parameter(
    #         "max_batch_size"
    #     )
    #     self.assertIsNone(max_batch_size)

    #     # concurrency
    #     concurrency = analyzer._composing_search_parameters["sub"].get_parameter(
    #         "concurrency"
    #     )
    #     self.assertIsNone(concurrency)

    #     # instance_group
    #     instance_group = analyzer._composing_search_parameters["sub"].get_parameter(
    #         "instance_group"
    #     )
    #     self.assertEqual(SearchUsage.MODEL, instance_group.usage)
    #     self.assertEqual(SearchCategory.INTEGER, instance_group.category)
    #     self.assertEqual(
    #         default.DEFAULT_RUN_CONFIG_MIN_INSTANCE_COUNT, instance_group.min_range
    #     )
    #     self.assertEqual(
    #         default.DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT, instance_group.max_range
    #     )


if __name__ == "__main__":
    unittest.main()
