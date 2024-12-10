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

from math import log2
from typing import Any, Dict, List, Optional

from genai_perf.config.generate.objective_parameter import ObjectiveCategory
from genai_perf.config.generate.search_parameter import (
    ParameterList,
    SearchCategory,
    SearchParameter,
    SearchUsage,
)
from genai_perf.config.input.config_command import ConfigCommand, Range, Subcommand
from genai_perf.exceptions import GenAIPerfException


class SearchParameters:
    """
    Contains information about all configuration parameters the user wants to search
    """

    # These map to the various fields that can be set for PA and model configs
    # See github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md
    exponential_range_parameters = [
        "model_batch_size",
        "runtime_batch_size",
        "concurrency",
        "request_rate",
        "input_sequence_length",
    ]

    linear_range_parameters = ["instance_count", "num_dataset_entries"]

    model_parameters = [
        "model_batch_size",
        "instance_count",
        "max_queue_delay",
    ]

    runtime_pa_parameters = ["runtime_batch_size", "concurrency", "request_rate"]

    runtime_gap_parameters = ["num_dataset_entries", "input_sequence_length"]

    all_parameters = model_parameters + runtime_pa_parameters + runtime_gap_parameters

    def __init__(
        self,
        config: ConfigCommand,
        subcommand: Subcommand,
        is_bls_model: bool = False,
        is_ensemble_model: bool = False,
        is_composing_model: bool = False,
    ):
        self._subcommand = subcommand
        if subcommand == Subcommand.OPTIMIZE:
            self._config = config.optimize
        elif subcommand == Subcommand.ANALYZE:
            self._config = config.analyze  # type: ignore

        # TODO: OPTIMIZE
        # self._supports_model_batch_size = model.supports_batching()
        self._supports_model_batch_size = True

        self._search_parameters: Dict[str, SearchParameter] = {}
        self._is_ensemble_model = is_ensemble_model
        self._is_bls_model = is_bls_model
        self._is_composing_model = is_composing_model

        self._populate_search_parameters()

    ###########################################################################
    # Accessor Methods
    ###########################################################################
    def get_parameter(self, name: str) -> Optional[SearchParameter]:
        return self._search_parameters.get(name)

    def get_parameter_names(self) -> Optional[List[str]]:
        return list(self._search_parameters.keys())

    def get_type(self, name: str) -> SearchUsage:
        if name in self._search_parameters:
            return self._search_parameters[name].usage
        else:
            return self._determine_parameter_usage(name)

    def get_category(self, name: str) -> SearchCategory:
        if name in self._search_parameters:
            return self._search_parameters[name].category
        else:
            return self._determine_parameter_category(name)

    def get_objective_category(self, name: str) -> ObjectiveCategory:
        # The difference here is that for objectives lists are not possible
        search_category = self.get_category(name)

        if search_category is SearchCategory.EXPONENTIAL:
            return ObjectiveCategory.EXPONENTIAL
        elif (
            search_category is SearchCategory.INTEGER
            or search_category is SearchCategory.INT_LIST
        ):
            return ObjectiveCategory.INTEGER
        else:
            return ObjectiveCategory.STR

    def get_range(self, name: str) -> Range:
        min_range = int(self._search_parameters[name].min_range or 0)
        max_range = int(self._search_parameters[name].max_range or 0)

        return Range(
            min=min_range,
            max=max_range,
        )

    def get_list(self, name: str) -> List[Any]:
        return self._search_parameters[name].get_list()

    ###########################################################################
    # Search Parameters
    ###########################################################################
    def _populate_search_parameters(self) -> None:
        if self._subcommand == Subcommand.OPTIMIZE:
            self._populate_model_config_parameters()
            self._populate_perf_analyzer_parameters()
            self._populate_genai_perf_parameters()
        else:
            self._populate_analyze_parameters()

    ###########################################################################
    # Perf Analyzer Parameters
    ###########################################################################
    def _populate_perf_analyzer_parameters(self) -> None:
        self._populate_perf_analyzer_batch_size()

        if not self._is_composing_model:
            if self._config.is_request_rate_specified():
                self._populate_request_rate()
            else:
                self._populate_concurrency()

    def _populate_perf_analyzer_batch_size(self) -> None:
        if isinstance(self._config.perf_analyzer.batch_size, list):
            self._populate_list_parameter(
                parameter_name="runtime_batch_size",
                parameter_list=list(self._config.perf_analyzer.batch_size),
                parameter_category=SearchCategory.INT_LIST,
            )
        elif isinstance(self._config.perf_analyzer.batch_size, Range):
            self._populate_range_parameter(
                parameter_name="runtime_batch_size",
                parameter_min_value=self._config.perf_analyzer.batch_size.min,
                parameter_max_value=self._config.perf_analyzer.batch_size.max,
            )

    def _populate_concurrency(self) -> None:
        if self._config.perf_analyzer.use_concurrency_formula:
            return
        elif isinstance(self._config.perf_analyzer.concurrency, list):
            self._populate_list_parameter(
                parameter_name="concurrency",
                parameter_list=list(self._config.perf_analyzer.concurrency),
                parameter_category=SearchCategory.INT_LIST,
            )
        elif isinstance(self._config.perf_analyzer.concurrency, Range):
            self._populate_range_parameter(
                parameter_name="concurrency",
                parameter_min_value=self._config.perf_analyzer.concurrency.min,
                parameter_max_value=self._config.perf_analyzer.concurrency.max,
            )

    def _populate_request_rate(self) -> None:
        if isinstance(self._config.perf_analyzer.request_rate, list):
            self._populate_list_parameter(
                parameter_name="request_rate",
                parameter_list=list(self._config.perf_analyzer.request_rate),
                parameter_category=SearchCategory.INT_LIST,
            )
        elif isinstance(self._config.perf_analyzer.request_rate, Range):
            self._populate_range_parameter(
                parameter_name="request_rate",
                parameter_min_value=self._config.perf_analyzer.request_rate.min,
                parameter_max_value=self._config.perf_analyzer.request_rate.max,
            )

    ###########################################################################
    # GenAI Perf Parameters
    ###########################################################################
    def _populate_genai_perf_parameters(self) -> None:
        self._populate_genai_perf_num_dataset_entries()

    def _populate_genai_perf_num_dataset_entries(self) -> None:
        if isinstance(self._config.genai_perf.num_dataset_entries, list):
            self._populate_list_parameter(
                parameter_name="num_dataset_entries",
                parameter_list=list(self._config.genai_perf.num_dataset_entries),
                parameter_category=SearchCategory.INT_LIST,
            )
        elif isinstance(self._config.genai_perf.num_dataset_entries, Range):
            self._populate_range_parameter(
                parameter_name="num_dataset_entries",
                parameter_min_value=self._config.genai_perf.num_dataset_entries.min,
                parameter_max_value=self._config.genai_perf.num_dataset_entries.max,
            )

    ###########################################################################
    # Model Config Parameters
    ###########################################################################
    def _populate_model_config_parameters(self) -> None:
        self._populate_model_batch_size()
        self._populate_instance_count()
        self._populate_max_queue_delay()

    def _populate_model_batch_size(self) -> None:
        if isinstance(self._config.model_config.batch_size, list):
            self._populate_list_parameter(
                parameter_name="model_batch_size",
                parameter_list=list(self._config.model_config.batch_size),
                parameter_category=SearchCategory.INT_LIST,
            )
        elif (
            self._supports_model_batch_size
            and not self._is_bls_model
            and isinstance(self._config.model_config.batch_size, Range)
        ):
            # Need to populate max_batch_size based on range values
            # when no model config parameters are present
            self._populate_range_parameter(
                parameter_name="model_batch_size",
                parameter_min_value=self._config.model_config.batch_size.min,
                parameter_max_value=self._config.model_config.batch_size.max,
            )

    def _populate_instance_count(self) -> None:
        if isinstance(self._config.model_config.instance_count, list):
            self._populate_list_parameter(
                parameter_name="instance_count",
                parameter_list=list(self._config.model_config.instance_count),
                parameter_category=SearchCategory.INT_LIST,
            )
        elif not self._is_ensemble_model and isinstance(
            self._config.model_config.instance_count, Range
        ):
            # Need to populate instance_count based on range values
            # when no model config parameters are present
            self._populate_range_parameter(
                parameter_name="instance_count",
                parameter_min_value=self._config.model_config.instance_count.min,
                parameter_max_value=self._config.model_config.instance_count.max,
            )

    def _populate_max_queue_delay(self) -> None:
        if isinstance(self._config.model_config.max_queue_delay, list):
            self._populate_list_parameter(
                parameter_name="max_queue_delay",
                parameter_list=list(self._config.model_config.max_queue_delay),
                parameter_category=SearchCategory.INT_LIST,
            )
        elif isinstance(self._config.model_config.max_queue_delay, Range):
            self._populate_range_parameter(
                parameter_name="max_queue_delay",
                parameter_min_value=self._config.model_config.max_queue_delay.min,
                parameter_max_value=self._config.model_config.max_queue_delay.max,
            )

    ###########################################################################
    # Analyze Parameters
    ###########################################################################
    def _populate_analyze_parameters(self) -> None:
        for name, value in self._config.sweep_parameters.items():  # type: ignore
            if isinstance(value, list):
                category = (
                    SearchCategory.STR_LIST
                    if isinstance(value[0], str)
                    else SearchCategory.INT_LIST
                )
                self._populate_list_parameter(
                    parameter_name=name,
                    parameter_list=value,
                    parameter_category=category,
                )
            elif isinstance(value, Range):
                self._populate_range_parameter(
                    parameter_name=name,
                    parameter_min_value=value.min,
                    parameter_max_value=value.max,
                )

    ###########################################################################
    # Populate Methods
    ###########################################################################
    def _populate_list_parameter(
        self,
        parameter_name: str,
        parameter_list: ParameterList,
        parameter_category: SearchCategory,
    ) -> None:
        usage = self._determine_parameter_usage(parameter_name)

        self._add_search_parameter(
            name=parameter_name,
            usage=usage,
            category=parameter_category,
            enumerated_list=parameter_list,
        )

    def _populate_range_parameter(
        self,
        parameter_name: str,
        parameter_min_value: int,
        parameter_max_value: int,
    ) -> None:
        usage = self._determine_parameter_usage(parameter_name)
        category = self._determine_parameter_category(parameter_name)

        if category == SearchCategory.EXPONENTIAL:
            min_range = int(log2(parameter_min_value))  # type: ignore
            max_range = int(log2(parameter_max_value))  # type: ignore
        else:
            min_range = parameter_min_value  # type: ignore
            max_range = parameter_max_value  # type: ignore

        self._add_search_parameter(
            name=parameter_name,
            usage=usage,
            category=category,
            min_range=min_range,
            max_range=max_range,
        )

    def _determine_parameter_category(self, name: str) -> SearchCategory:
        if name in SearchParameters.exponential_range_parameters:
            category = SearchCategory.EXPONENTIAL
        elif name in SearchParameters.linear_range_parameters:
            category = SearchCategory.INTEGER
        else:
            raise (GenAIPerfException(f"SearchCategory not found for {name}"))

        return category

    def _determine_parameter_usage(self, name: str) -> SearchUsage:
        if name in SearchParameters.model_parameters:
            usage = SearchUsage.MODEL
        elif name in SearchParameters.runtime_pa_parameters:
            usage = SearchUsage.RUNTIME_PA
        elif name in SearchParameters.runtime_gap_parameters:
            usage = SearchUsage.RUNTIME_GAP
        else:
            raise (GenAIPerfException(f"SearchUsage not found for {name}"))

        return usage

    def _add_search_parameter(
        self,
        name: str,
        usage: SearchUsage,
        category: SearchCategory,
        min_range: Optional[int] = None,
        max_range: Optional[int] = None,
        enumerated_list: List[Any] = [],
    ) -> None:
        self._check_for_illegal_input(category, min_range, max_range, enumerated_list)

        self._search_parameters[name] = SearchParameter(
            usage=usage,
            category=category,
            enumerated_list=enumerated_list,
            min_range=min_range,
            max_range=max_range,
        )

    ###########################################################################
    # Info/Debug Methods
    ###########################################################################
    def number_of_total_possible_configurations(self) -> int:
        total_number_of_configs = 1
        for parameter in self._search_parameters.values():
            total_number_of_configs *= self._number_of_configurations_for_parameter(
                parameter
            )

        return total_number_of_configs

    def print_info(self, name: str) -> str:
        info_string = f"  {name}: "

        parameter = self._search_parameters[name]
        if parameter.category is SearchCategory.INTEGER:
            info_string += f"{parameter.min_range} to {parameter.max_range}"
        elif parameter.category is SearchCategory.EXPONENTIAL:
            info_string += f"{2**parameter.min_range} to {2**parameter.max_range}"  # type: ignore
        elif (
            parameter.category is SearchCategory.INT_LIST
            or parameter.category is SearchCategory.STR_LIST
        ):
            info_string += f"{parameter.enumerated_list}"

        info_string += f" ({self._number_of_configurations_for_parameter(parameter)})"

        return info_string

    def _number_of_configurations_for_parameter(
        self, parameter: SearchParameter
    ) -> int:
        if (
            parameter.category is SearchCategory.INTEGER
            or parameter.category is SearchCategory.EXPONENTIAL
        ):
            number_of_parameter_configs = parameter.max_range - parameter.min_range + 1  # type: ignore
        else:
            number_of_parameter_configs = len(parameter.enumerated_list)  # type: ignore

        return number_of_parameter_configs

    ###########################################################################
    # Error Checking Methods
    ###########################################################################
    def _check_for_illegal_input(
        self,
        category: SearchCategory,
        min_range: Optional[int],
        max_range: Optional[int],
        enumerated_list: List[Any],
    ) -> None:
        if category is SearchCategory.INT_LIST or category is SearchCategory.STR_LIST:
            self._check_for_illegal_list_input(min_range, max_range, enumerated_list)
        else:
            if min_range is None or max_range is None:
                raise GenAIPerfException(
                    f"Both min_range and max_range must be specified"
                )

            if min_range and max_range:
                if min_range > max_range:
                    raise GenAIPerfException(
                        f"min_range cannot be larger than max_range"
                    )

    def _check_for_illegal_list_input(
        self,
        min_range: Optional[int],
        max_range: Optional[int],
        enumerated_list: List[Any],
    ) -> None:
        if not enumerated_list:
            raise GenAIPerfException(
                f"enumerated_list must be specified for a SearchCategory.LIST"
            )
        elif min_range is not None:
            raise GenAIPerfException(
                f"min_range cannot be specified for a SearchCategory.LIST"
            )
        elif max_range is not None:
            raise GenAIPerfException(
                f"max_range cannot be specified for a SearchCategory.LIST"
            )
