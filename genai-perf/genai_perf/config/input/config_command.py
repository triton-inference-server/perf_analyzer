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
from enum import Enum, auto
from typing import Dict, List, Optional, TypeAlias, Union

from genai_perf.types import ModelName


class Subcommand(Enum):
    ANALYZE = auto()
    OPTIMIZE = auto()


def default_field(obj):
    return field(default_factory=lambda: copy(obj))


@dataclass
class Range:
    min: int
    max: int


ConfigRangeOrList: TypeAlias = Optional[Union[Range, List[int]]]
AnalyzeParameter: TypeAlias = Dict[str, ConfigRangeOrList]


# TODO: OPTIMIZE
# These will be moved to RunConfig once it's created
@dataclass(frozen=True)
class RunConfigDefaults:
    # Top-level Defaults
    CHECKPOINT_DIRECTORY = "./"

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
    PA_BATCH_SIZE = 1
    MIN_CONCURRENCY = 1
    MAX_CONCURRENCY = 1024
    MIN_REQUEST_RATE = 16
    MAX_REQUEST_RATE = 8192
    USE_CONCURRENCY_FORMULA = True

    # Analyze Defaults
    SWEEP = {"concurrency": Range(min=MIN_CONCURRENCY, max=MAX_CONCURRENCY)}

    # Perf Analyzer Defaults
    PA_PATH = "perf_analyzer"
    TIMEOUT_THRESHOLD = 600
    MAX_CPU_UTILIZATION = 80.0
    OUTPUT_LOGGING = False
    OUTPUT_PATH = "./logs"
    MAX_AUTO_ADJUSTS = 10
    STABILITY_THRESHOLD = 999

    # GAP Input Defaults
    DATASET = "openorca"
    FILE = None
    NUM_PROMPTS = 100
    SEED = 0

    # GAP Input Synthetic Tokens Defaults
    INPUT_MEAN = -1
    INPUT_STDDEV = 0

    # GAP Output Token Defaults
    OUTPUT_MEAN = -1
    DETERMINISTIC = False
    OUTPUT_STDDEV = 0


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
class ConfigOptimizePerfAnalyzer:
    stimulus_type: str = default_field(RunConfigDefaults.STIMULUS_TYPE)
    batch_size: ConfigRangeOrList = default_field([RunConfigDefaults.PA_BATCH_SIZE])
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
class ConfigOptimizeGenAIPerf:
    num_dataset_entries: ConfigRangeOrList = default_field(
        [RunConfigDefaults.NUM_PROMPTS]
    )


@dataclass
class ConfigOptimize:
    objective: str = default_field(RunConfigDefaults.OBJECTIVE)
    constraint: Optional[str] = default_field(RunConfigDefaults.CONSTRAINT)
    search_space_percentage: Range = default_field(
        RunConfigDefaults.SEARCH_SPACE_PERCENTAGE
    )
    number_of_trials: Range = default_field(RunConfigDefaults.NUMBER_OF_TRIALS)
    early_exit_threshold: int = default_field(RunConfigDefaults.EARLY_EXIT_THRESHOLD)

    model_config: ConfigModelConfig = default_field(ConfigModelConfig())
    perf_analyzer: ConfigOptimizePerfAnalyzer = default_field(
        ConfigOptimizePerfAnalyzer()
    )
    genai_perf: ConfigOptimizeGenAIPerf = default_field(ConfigOptimizeGenAIPerf())

    def is_request_rate_specified(self) -> bool:
        return self.perf_analyzer.is_request_rate_specified()

    def is_set_by_user(self, field: str) -> bool:
        # FIXME: OPTIMIZE - we have no way of knowing this until a real config class is created
        return False


@dataclass
class ConfigAnalyze:
    sweep_parameters: AnalyzeParameter = default_field(RunConfigDefaults.SWEEP)


@dataclass
class ConfigPerfAnalyzer:
    path: str = default_field(RunConfigDefaults.PA_PATH)
    timeout_threshold: int = default_field(RunConfigDefaults.TIMEOUT_THRESHOLD)
    max_cpu_utilization: float = default_field(RunConfigDefaults.MAX_CPU_UTILIZATION)
    output_logging: bool = default_field(RunConfigDefaults.OUTPUT_LOGGING)
    output_path: str = default_field(RunConfigDefaults.OUTPUT_PATH)
    max_auto_adjusts: int = default_field(RunConfigDefaults.MAX_AUTO_ADJUSTS)
    stability_threshold: float = default_field(RunConfigDefaults.STABILITY_THRESHOLD)


@dataclass
class ConfigSyntheticTokens:
    mean: int = default_field(RunConfigDefaults.INPUT_MEAN)
    stddev: int = default_field(RunConfigDefaults.INPUT_STDDEV)


@dataclass
class ConfigInput:
    dataset: str = default_field(RunConfigDefaults.DATASET)
    file: str = default_field(RunConfigDefaults.FILE)
    num_dataset_entries: int = default_field(RunConfigDefaults.NUM_PROMPTS)
    seed: int = default_field(RunConfigDefaults.SEED)
    synthetic_tokens: ConfigSyntheticTokens = default_field(ConfigSyntheticTokens())


@dataclass
class ConfigOutputTokens:
    mean: int = default_field(RunConfigDefaults.OUTPUT_MEAN)
    deterministic: bool = default_field(RunConfigDefaults.DETERMINISTIC)
    stddev: int = default_field(RunConfigDefaults.OUTPUT_STDDEV)


@dataclass
class ConfigCommand:
    model_names: List[ModelName]
    checkpoint_directory: str = default_field(RunConfigDefaults.CHECKPOINT_DIRECTORY)
    optimize: ConfigOptimize = default_field(ConfigOptimize())
    analyze: ConfigAnalyze = default_field(ConfigAnalyze())
    perf_analyzer: ConfigPerfAnalyzer = default_field(ConfigPerfAnalyzer())
    input: ConfigInput = default_field(ConfigInput())
    output_tokens: ConfigOutputTokens = default_field(ConfigOutputTokens())

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
