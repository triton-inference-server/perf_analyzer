#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import MagicMock, patch

#import model_analyzer.config.input.config_defaults as default

from genai_perf.config.input.config_command import ConfigCommand, RunConfigDefaults

from genai_perf.config.generate.search_parameters import (
    ParameterCategory,
    ParameterUsage,
    SearchParameters,
)

from genai_perf.exceptions import GenAIPerfException

class TestSearchParameters(unittest.TestCase):
    def setUp(self):
        self.config = ConfigCommand()

        self.search_parameters = SearchParameters(config=self.config)

        self.search_parameters._add_search_parameter(
            name="concurrency",
            usage=ParameterUsage.RUNTIME,
            category=ParameterCategory.EXPONENTIAL,
            min_range=0,
            max_range=10,
        )

        self.search_parameters._add_search_parameter(
            name="instance_group",
            usage=ParameterUsage.MODEL,
            category=ParameterCategory.INTEGER,
            min_range=1,
            max_range=8,
        )

        self.search_parameters._add_search_parameter(
            name="size",
            usage=ParameterUsage.BUILD,
            category=ParameterCategory.STR_LIST,
            enumerated_list=["FP8", "FP16", "FP32"],
        )

    def tearDown(self):
        patch.stopall()

    def test_exponential_parameter(self):
        """
        Test exponential parameter, accessing dataclass directly
        """

        # concurrency
        parameter = self.search_parameters.get_parameter("concurrency")

        self.assertEqual(ParameterUsage.RUNTIME, parameter.usage)
        self.assertEqual(ParameterCategory.EXPONENTIAL, parameter.category)
        self.assertEqual(0, parameter.min_range)
        self.assertEqual(10, parameter.max_range)

    def test_integer_parameter(self):
        """
        Test integer parameter, using accessor methods
        """

        self.assertEqual(
            ParameterUsage.MODEL,
            self.search_parameters.get_type("instance_group"),
        )
        self.assertEqual(
            ParameterCategory.INTEGER,
            self.search_parameters.get_category("instance_group"),
        )
        self.assertEqual((1, 8), self.search_parameters.get_range("instance_group"))

    def test_list_parameter(self):
        """
        Test list parameter, using accessor methods
        """

        self.assertEqual(
            ParameterUsage.BUILD,
            self.search_parameters.get_type("size"),
        )
        self.assertEqual(
            ParameterCategory.STR_LIST,
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
                usage=ParameterUsage.RUNTIME,
                category=ParameterCategory.EXPONENTIAL,
                max_range=10,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="concurrency",
                usage=ParameterUsage.RUNTIME,
                category=ParameterCategory.EXPONENTIAL,
                min_range=0,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="concurrency",
                usage=ParameterUsage.RUNTIME,
                category=ParameterCategory.EXPONENTIAL,
                min_range=10,
                max_range=9,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="size",
                usage=ParameterUsage.BUILD,
                category=ParameterCategory.INT_LIST,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="size",
                usage=ParameterUsage.BUILD,
                category=ParameterCategory.STR_LIST,
                enumerated_list=["FP8", "FP16", "FP32"],
                min_range=0,
            )

        with self.assertRaises(GenAIPerfException):
            self.search_parameters._add_search_parameter(
                name="size",
                usage=ParameterUsage.BUILD,
                category=ParameterCategory.STR_LIST,
                enumerated_list=["FP8", "FP16", "FP32"],
                max_range=10,
            )

    def test_search_parameter_creation_optimize_default(self):
        """
        Test that search parameters are correctly created in default optimize case
        """

        #######################################################################
        # Model Config
        #######################################################################
        
        # Batch Size
        model_batch_size = self.search_parameters.get_parameter(
            "model_batch_size"
        )
        self.assertEqual(ParameterUsage.MODEL, model_batch_size.usage)
        self.assertEqual(ParameterCategory.EXPONENTIAL, model_batch_size.category)
        self.assertEqual(
            log2(RunConfigDefaults.MIN_MODEL_BATCH_SIZE),
            model_batch_size.min_range,
        )
        self.assertEqual(
            log2(RunConfigDefaults.MAX_MODEL_BATCH_SIZE),
            model_batch_size.max_range,
        )

        # Instance Count
        instance_count = self.search_parameters.get_parameter(
            "instance_count"
        )
        self.assertEqual(ParameterUsage.MODEL, instance_count.usage)
        self.assertEqual(ParameterCategory.INTEGER, instance_count.category)
        self.assertEqual(
            RunConfigDefaults.MIN_INSTANCE_COUNT, instance_count.min_range
        )
        self.assertEqual(
            RunConfigDefaults.MAX_INSTANCE_COUNT, instance_count.max_range
        )
        
        # Max Queue Delay
        max_queue_delay = self.search_parameters.get_parameter("max_queue_delay")
        self.assertIsNone(max_queue_delay)
        
        #######################################################################
        # PA Config
        #######################################################################
        
        # Batch size
        runtime_batch_size = self.search_parameters.get_parameter("runtime_batch_size")
        self.assertEqual(ParameterUsage.RUNTIME, runtime_batch_size.usage)
        self.assertEqual(ParameterCategory.INT_LIST, runtime_batch_size.category)
        self.assertEqual(RunConfigDefaults.PA_BATCH_SIZE, runtime_batch_size.enumerated_list)
        
        # Concurrency
        concurrency = self.search_parameters.get_parameter(
            "concurrency"
        ) 
        self.assertEqual(ParameterUsage.RUNTIME, concurrency.usage)
        self.assertEqual(ParameterCategory.EXPONENTIAL, concurrency.category)
        self.assertEqual(
            log2(RunConfigDefaults.MIN_CONCURRENCY), concurrency.min_range
        )
        self.assertEqual(
            log2(RunConfigDefaults.MAX_CONCURRENCY), concurrency.max_range
        )
        
        # Request Rate
        request_rate = self.search_parameters.get_parameter("request_rate")
        self.assertIsNone(request_rate)
  

    # def test_search_parameter_concurrency_formula(self):
    #     """
    #     Test that when concurrency formula is specified it is
    #     not added as a search parameter
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
    #         "--use-concurrency-formula",
    #     ]

    #     yaml_content = """
    #     profile_models: add_sub
    #     """
    #     config = TestConfig()._evaluate_config(args=args, yaml_content=yaml_content)

    #     analyzer = Analyzer(config, MagicMock(), MagicMock(), MagicMock())
    #     analyzer._populate_search_parameters(MagicMock(), MagicMock())

    #     concurrency = analyzer._search_parameters["add_sub"].get_parameter(
    #         "concurrency"
    #     )

    #     self.assertEqual(concurrency, None)

    # def test_search_parameter_request_rate(self):
    #     """
    #     Test that request rate is correctly set in
    #     a non-default optuna case
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
    #     run_config_search_mode: optuna
    #     profile_models:
    #         mult_div:
    #             parameters:
    #                 request_rate: [1, 8, 64, 256]

    #     """
    #     config = TestConfig()._evaluate_config(args, yaml_content)
    #     analyzer = Analyzer(config, MagicMock(), MagicMock(), MagicMock())
    #     mock_model_config = MockModelConfig()
    #     mock_model_config.start()
    #     analyzer._populate_search_parameters(MagicMock(), MagicMock())
    #     mock_model_config.stop()

    #     # request_rate
    #     # ===================================================================

    #     request_rate = analyzer._search_parameters["mult_div"].get_parameter(
    #         "request_rate"
    #     )
    #     self.assertEqual(ParameterUsage.RUNTIME, request_rate.usage)
    #     self.assertEqual(ParameterCategory.INT_LIST, request_rate.category)
    #     self.assertEqual([1, 8, 64, 256], request_rate.enumerated_list)

    # def test_number_of_configs_range(self):
    #     """
    #     Test number of configs for a range (INTEGER/EXPONENTIAL)
    #     """

    #     # INTEGER
    #     # =====================================================================
    #     num_of_configs = self.search_parameters._number_of_configurations_for_parameter(
    #         self.search_parameters.get_parameter("instance_group")
    #     )
    #     self.assertEqual(8, num_of_configs)

    #     # EXPONENTIAL
    #     # =====================================================================
    #     num_of_configs = self.search_parameters._number_of_configurations_for_parameter(
    #         self.search_parameters.get_parameter("concurrency")
    #     )
    #     self.assertEqual(11, num_of_configs)

    # def test_number_of_configs_list(self):
    #     """
    #     Test number of configs for a list
    #     """

    #     num_of_configs = self.search_parameters._number_of_configurations_for_parameter(
    #         self.search_parameters.get_parameter("size")
    #     )
    #     self.assertEqual(3, num_of_configs)

    # def test_total_possible_configurations(self):
    #     """
    #     Test number of total possible configurations
    #     """
    #     total_num_of_possible_configurations = (
    #         self.search_parameters.number_of_total_possible_configurations()
    #     )

    #     # max_batch_size (8) * instance group (8) * concurrency (11) * size (3)
    #     self.assertEqual(8 * 8 * 11 * 3, total_num_of_possible_configurations)

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
    #     self.assertEqual(ParameterUsage.RUNTIME, concurrency.usage)
    #     self.assertEqual(ParameterCategory.EXPONENTIAL, concurrency.category)
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
    #     self.assertEqual(ParameterUsage.MODEL, instance_group.usage)
    #     self.assertEqual(ParameterCategory.INTEGER, instance_group.category)
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
    #     self.assertEqual(ParameterUsage.MODEL, instance_group.usage)
    #     self.assertEqual(ParameterCategory.INTEGER, instance_group.category)
    #     self.assertEqual(
    #         default.DEFAULT_RUN_CONFIG_MIN_INSTANCE_COUNT, instance_group.min_range
    #     )
    #     self.assertEqual(
    #         default.DEFAULT_RUN_CONFIG_MAX_INSTANCE_COUNT, instance_group.max_range
    #     )


if __name__ == "__main__":
    unittest.main()
