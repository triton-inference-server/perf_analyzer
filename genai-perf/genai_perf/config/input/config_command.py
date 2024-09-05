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
from typing import List, Optional, Union


def default_field(obj):
    return field(default_factory=lambda: copy(obj))


# TODO: OPTIMIZE
# These will be moved to RunConfig once it's created
@dataclass
class RunConfigDefaults:
    # Model Defaults
    MIN_MODEL_BATCH_SIZE = 1
    MAX_MODEL_BATCH_SIZE = 128
    MIN_INSTANCE_COUNT = 1
    MAX_INSTANCE_COUNT = 5
    MAX_QUEUE_DELAY = None
    DYNAMIC_BATCHING = True
    CPU_ONLY = False

    # PA Defaults
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
class Range:
    min: int
    max: int


@dataclass
class ConfigModelConfig:
    batch_size: Optional[Union[Range, List[int]]] = default_field(
        Range(
            min=RunConfigDefaults.MIN_MODEL_BATCH_SIZE,
            max=RunConfigDefaults.MAX_MODEL_BATCH_SIZE,
        )
    )
    instance_count: Optional[Union[Range, List[int]]] = default_field(
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
    batch_size: Optional[Union[Range, List[int]]] = default_field(
        RunConfigDefaults.PA_BATCH_SIZE
    )
    concurrency: Optional[Union[Range, List[int]]] = default_field(
        Range(
            min=RunConfigDefaults.MIN_CONCURRENCY, max=RunConfigDefaults.MAX_CONCURRENCY
        )
    )
    request_rate: Optional[Union[Range, List[int]]] = default_field(
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
    model_config: ConfigModelConfig = ConfigModelConfig()
    perf_analyzer: ConfigPerfAnalyzer = ConfigPerfAnalyzer()

    def is_request_rate_specified(self) -> bool:
        return self.perf_analyzer.is_request_rate_specified()


@dataclass
class ConfigCommand:
    optimize: ConfigOptimize = ConfigOptimize()
