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

from copy import copy
from dataclasses import dataclass, field
from typing import List, Optional, TypeAlias, Union

from genai_perf.types import ModelName


def default_field(obj):
    return field(default_factory=lambda: copy(obj))


@dataclass
class Range:
    min: int
    max: int


ConfigRangeOrList: TypeAlias = Optional[Union[Range, List[int]]]


# TODO: OPTIMIZE
# These will be moved to RunConfig once it's created
@dataclass(frozen=True)
class RunConfigDefaults:
    # Optimize: Top-level Defaults
    OBJECTIVE = "throughput"
    CONSTRAINT = None
    SEARCH_SPACE_PERCENTAGE = Range(min=5, max=10)
    NUMBER_OF_TRIALS = Range(min=0, max=0)
    EARLY_EXIT_THRESHOLD = 10

    # Optimize: Model Defaults
    MIN_MODEL_BATCH_SIZE = 1
    MAX_MODEL_BATCH_SIZE = 128
    MIN_INSTANCE_COUNT = 1
    MAX_INSTANCE_COUNT = 5
    MAX_QUEUE_DELAY = None
    DYNAMIC_BATCHING = True
    CPU_ONLY = False

    # Optimize: PA Defaults
    STIMULUS_TYPE = "concurrency"
    PA_BATCH_SIZE = [1]
    MIN_CONCURRENCY = 1
    MAX_CONCURRENCY = 1024
    MIN_REQUEST_RATE = 16
    MAX_REQUEST_RATE = 8192
    USE_CONCURRENCY_FORMULA = True


# TODO: OPTIMIZE
# These are placeholder dataclasses until the real Command Parser is written
@dataclass
class ConfigModelConfig:
    batch_size: ConfigRangeOrList = default_field(
        Range(
            min=RunConfigDefaults.MIN_MODEL_BATCH_SIZE,
            max=RunConfigDefaults.MAX_MODEL_BATCH_SIZE,
        )
    )
    instance_count: ConfigRangeOrList = default_field(
        Range(
            min=RunConfigDefaults.MIN_INSTANCE_COUNT,
            max=RunConfigDefaults.MAX_INSTANCE_COUNT,
        )
    )
    max_queue_delay: Optional[List[int]] = default_field(
        RunConfigDefaults.MAX_QUEUE_DELAY
    )
    dynamic_batching: bool = default_field(RunConfigDefaults.DYNAMIC_BATCHING)
    cpu_only: bool = default_field(RunConfigDefaults.CPU_ONLY)


@dataclass
class ConfigPerfAnalyzer:
    stimulus_type: str = default_field(RunConfigDefaults.STIMULUS_TYPE)
    batch_size: ConfigRangeOrList = default_field(RunConfigDefaults.PA_BATCH_SIZE)
    concurrency: ConfigRangeOrList = default_field(
        Range(
            min=RunConfigDefaults.MIN_CONCURRENCY, max=RunConfigDefaults.MAX_CONCURRENCY
        )
    )
    request_rate: ConfigRangeOrList = default_field(
        Range(
            min=RunConfigDefaults.MIN_REQUEST_RATE,
            max=RunConfigDefaults.MAX_REQUEST_RATE,
        )
    )
    use_concurrency_formula: bool = default_field(
        RunConfigDefaults.USE_CONCURRENCY_FORMULA
    )

    def is_request_rate_specified(self) -> bool:
        rr_specified = self.stimulus_type == "request_rate"

        return rr_specified


@dataclass
class ConfigOptimize:
    objective: str = default_field(RunConfigDefaults.OBJECTIVE)
    constraint: Optional[str] = default_field(RunConfigDefaults.CONSTRAINT)
    search_space_percentage: Range = default_field(
        RunConfigDefaults.SEARCH_SPACE_PERCENTAGE
    )
    number_of_trials: Range = default_field(RunConfigDefaults.NUMBER_OF_TRIALS)
    early_exit_threshold: int = default_field(RunConfigDefaults.EARLY_EXIT_THRESHOLD)

    model_config: ConfigModelConfig = ConfigModelConfig()
    perf_analyzer: ConfigPerfAnalyzer = ConfigPerfAnalyzer()

    def is_request_rate_specified(self) -> bool:
        return self.perf_analyzer.is_request_rate_specified()

    def is_set_by_user(self, field: str) -> bool:
        # FIXME: OPTIMIZE - we have no way of knowing this until a real config class is created
        return False


@dataclass
class ConfigCommand:
    model_names: List[ModelName]
    optimize: ConfigOptimize = ConfigOptimize()

    def get_max(self, config_value: ConfigRangeOrList) -> int:
        if type(config_value) is list:
            return max(config_value)
        elif type(config_value) is Range:
            return config_value.max
        else:
            return 0

    def get_min(self, config_value: ConfigRangeOrList) -> int:
        if type(config_value) is list:
            return min(config_value)
        elif type(config_value) is Range:
            return config_value.min
        else:
            return 0
