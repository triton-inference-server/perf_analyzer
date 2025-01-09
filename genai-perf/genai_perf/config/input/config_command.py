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
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

from genai_perf.constants import (
    exponential_range_parameters,
    linear_range_parameters,
    runtime_gap_parameters,
    runtime_pa_parameters,
)
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
    MODEL_NAME = ""
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
    MIN_NUM_DATASET_ENTRIES = 100
    MAX_NUM_DATASET_ENTRIES = 1000
    SEED = 0

    # GAP Input Synthetic Tokens Defaults
    MIN_INPUT_SEQUENCE_LENGTH = 100
    MAX_INPUT_SEQUENCE_LENGTH = 1000
    INPUT_SEQUENCE_LENGTH_MEAN = 550
    INPUT_STDDEV = 0

    # GAP Output Token Defaults
    OUTPUT_MEAN = -1
    DETERMINISTIC = False
    OUTPUT_STDDEV = 0


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
        [RunConfigDefaults.MIN_NUM_DATASET_ENTRIES]
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
    mean: int = default_field(RunConfigDefaults.INPUT_SEQUENCE_LENGTH_MEAN)
    stddev: int = default_field(RunConfigDefaults.INPUT_STDDEV)


@dataclass
class ConfigInput:
    dataset: str = default_field(RunConfigDefaults.DATASET)
    file: str = default_field(RunConfigDefaults.FILE)
    num_dataset_entries: int = default_field(RunConfigDefaults.MIN_NUM_DATASET_ENTRIES)
    seed: int = default_field(RunConfigDefaults.SEED)
    synthetic_tokens: ConfigSyntheticTokens = default_field(ConfigSyntheticTokens())


@dataclass
class ConfigOutputTokens:
    mean: int = default_field(RunConfigDefaults.OUTPUT_MEAN)
    deterministic: bool = default_field(RunConfigDefaults.DETERMINISTIC)
    stddev: int = default_field(RunConfigDefaults.OUTPUT_STDDEV)


@dataclass
class ConfigCommand:
    user_config: Dict[str, Any]
    model_names: List[ModelName] = default_field([RunConfigDefaults.MODEL_NAME])
    checkpoint_directory: str = default_field(RunConfigDefaults.CHECKPOINT_DIRECTORY)
    optimize: ConfigOptimize = default_field(ConfigOptimize())
    analyze: ConfigAnalyze = default_field(ConfigAnalyze())
    perf_analyzer: ConfigPerfAnalyzer = default_field(ConfigPerfAnalyzer())
    input: ConfigInput = default_field(ConfigInput())
    output_tokens: ConfigOutputTokens = default_field(ConfigOutputTokens())

    def __post_init__(self):
        self._parse_yaml(self.user_config)

    ###########################################################################
    # Utility Methods
    ###########################################################################
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

    ###########################################################################
    # Parsing Methods
    ###########################################################################
    def _parse_yaml(self, user_config: Dict[str, Any]) -> None:
        for key, value in user_config.items():
            if key == "model_name":
                self._parse_model_name(value)
            elif key == "checkpoint_directory":
                self._parse_checkpoint_directory(value)
            # elif key == "optimize":
            #     self._parse_optimize(value)
            elif key == "analyze":
                self._parse_analyze(value)
            elif key == "perf_analyzer":
                self._parse_perf_analyzer(value)
            elif key == "input":
                self._parse_input(value)
            elif key == "output_tokens":
                self._parse_output_tokens(value)

    def _parse_model_name(self, model_name: str) -> None:
        if type(model_name) is str:
            self.model_names = [model_name]
        else:
            raise ValueError("User Config: model_name must be a string")

    def _parse_checkpoint_directory(self, checkpoint_directory: str) -> None:
        if type(checkpoint_directory) is str:
            self.checkpoint_directory = checkpoint_directory
        else:
            raise ValueError("User Config: checkpoint_directory must be a string")

    # def _parse_optimize(self, optimize: Dict[str, Any]) -> None:
    #     pass

    def _parse_analyze(self, analyze: Dict[str, Any]) -> None:
        sweep_parameters: Dict[str, Any] = {}
        for sweep_type, range_dict in analyze.items():
            if (
                sweep_type in runtime_pa_parameters
                or sweep_type in runtime_gap_parameters
            ):
                if sweep_type in linear_range_parameters or "step" in range_dict:
                    linear_range_list = self._create_linear_range_list(
                        sweep_type, range_dict
                    )
                    sweep_parameters[sweep_type] = linear_range_list
                elif sweep_type in exponential_range_parameters:
                    start, stop = self._determine_start_and_stop(sweep_type, range_dict)
                    sweep_parameters[sweep_type] = Range(min=start, max=stop)
            else:
                raise ValueError(
                    f"User Config: {sweep_type} is not a valid analyze parameter"
                )
        self.analyze.sweep_parameters = sweep_parameters

    def _parse_perf_analyzer(self, perf_analyzer: Dict[str, Any]) -> None:
        pass

    def _parse_input(self, input: Dict[str, Any]) -> None:
        pass

    def _parse_output_tokens(self, output_tokens: Dict[str, Any]) -> None:
        pass

    def _create_linear_range_list(
        self, sweep_type: str, range_dict: Dict[str, int]
    ) -> List[int]:
        start = self._get_start(sweep_type, range_dict)
        stop = self._get_stop(sweep_type, range_dict)
        step = self._get_step(sweep_type, range_dict)

        return [value for value in range(start, stop + 1, step)]

    def _determine_start_and_stop(
        self, sweep_type: str, range_dict: Dict[str, int]
    ) -> Tuple[int, int]:
        start = self._get_start(sweep_type, range_dict)
        stop = self._get_stop(sweep_type, range_dict)

        return start, stop

    def _get_start(self, sweep_type: str, range_dict: Dict[str, int]) -> int:
        if "start" in range_dict:
            return range_dict["start"]
        else:
            return self._get_default_start(sweep_type)

    def _get_stop(self, sweep_type: str, range_dict: Dict[str, int]) -> int:
        if "stop" in range_dict:
            return range_dict["stop"]
        else:
            return self._get_default_stop(sweep_type)

    def _get_step(self, sweep_type: str, range_dict: Dict[str, int]) -> int:
        if "step" in range_dict:
            return range_dict["step"]
        else:
            return self._get_default_step(sweep_type)

    def _get_default_start(self, sweep_type: str) -> int:
        if sweep_type == "concurrency":
            return RunConfigDefaults.MIN_CONCURRENCY
        elif sweep_type == "runtime_batch_size":
            return RunConfigDefaults.MIN_MODEL_BATCH_SIZE
        elif sweep_type == "request_rate":
            return RunConfigDefaults.MIN_REQUEST_RATE
        elif sweep_type == "num_dataset_entries":
            return RunConfigDefaults.MIN_NUM_DATASET_ENTRIES
        elif sweep_type == "input_sequence_length":
            return RunConfigDefaults.MIN_INPUT_SEQUENCE_LENGTH
        else:
            raise ValueError(f"User Config: {sweep_type} is not a valid sweep type")

    def _get_default_stop(self, sweep_type: str) -> int:
        if sweep_type == "concurrency":
            return RunConfigDefaults.MAX_CONCURRENCY
        elif sweep_type == "runtime_batch_size":
            return RunConfigDefaults.MAX_MODEL_BATCH_SIZE
        elif sweep_type == "request_rate":
            return RunConfigDefaults.MAX_REQUEST_RATE
        elif sweep_type == "num_dataset_entries":
            return RunConfigDefaults.MAX_NUM_DATASET_ENTRIES
        elif sweep_type == "input_sequence_length":
            return RunConfigDefaults.MAX_INPUT_SEQUENCE_LENGTH
        else:
            raise ValueError(f"User Config: {sweep_type} is not a valid sweep type")

    def _get_default_step(self, sweep_type: str) -> int:
        return 1
