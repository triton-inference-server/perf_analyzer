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

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

from genai_perf.config.endpoint_config import _endpoint_type_map
from genai_perf.config.input.config_defaults import (
    AnalyzeDefaults,
    EndPointDefaults,
    ImageDefaults,
    InputDefaults,
    OutputDefaults,
    OutputTokenDefaults,
    PerfAnalyzerDefaults,
    PrefixPromptDefaults,
    Range,
    RequestCountDefaults,
    SyntheticTokenDefaults,
    TokenizerDefaults,
    TopLevelDefaults,
    default_field,
)
from genai_perf.config.input.config_fields import ConfigField, ConfigFields
from genai_perf.constants import (
    all_parameters,
    runtime_gap_parameters,
    runtime_pa_parameters,
)
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.inputs.retrievers.synthetic_image_generator import ImageFormat
from genai_perf.types import CheckpointObject, ModelName


class Subcommand(Enum):
    COMPARE = auto()
    PROFILE = auto()
    ANALYZE = auto()


ConfigRangeOrList: TypeAlias = Optional[Union[Range, List[int]]]
AnalyzeParameter: TypeAlias = Dict[str, ConfigRangeOrList]


###########################################################################
# Base Config
###########################################################################
class BaseConfig:
    def __init__(self, name=None, parent=None):
        self._fields = ConfigFields()

        # This exists just to make looking up values when debugging easier
        self._values = self._fields._values

    def get_field(self, name: str) -> ConfigField:
        return self._fields.get_field(name)

    def __getattr__(self, name: str) -> Any:
        return self._fields.__getattr__(name)

    def __setattr__(self, name: str, value: Any):
        # This prevents recursion failure in __init__
        if name == "_fields" or name == "_values":
            self.__dict__[name] = value
        else:
            self._fields.__setattr__(name, value)

    def __deepcopy__(self, memo):
        new_copy = self.__class__()
        new_copy._fields = deepcopy(self._fields, memo)
        new_copy._values = new_copy._fields._values
        return new_copy

    def __eq__(self, other):
        foo = self._fields == other._fields
        return self._fields == other._fields


###########################################################################
# Analyze Subcommand Config
###########################################################################
class ConfigAnalyze(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.sweep_parameters = ConfigField(
            default=AnalyzeDefaults.SWEEP_PARAMETER, choices=all_parameters
        )


###########################################################################
# EndPoint Config
###########################################################################
class ConfigEndPoint(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.model_selection_strategy = ConfigField(
            default=EndPointDefaults.MODEL_SELECTION_STRATEGY,
            choices=ModelSelectionStrategy,
        )
        self._fields.backend = ConfigField(
            default=EndPointDefaults.BACKEND, choices=OutputFormat
        )
        self._fields.custom = ConfigField(default=EndPointDefaults.CUSTOM)
        self._fields.type = ConfigField(
            default=EndPointDefaults.TYPE,
            choices=list(_endpoint_type_map.keys()),
        )
        self._fields.service_kind = ConfigField(
            default=EndPointDefaults.SERVICE_KIND,
            choices=["triton", "openai", "tensorrtllm_engine"],
        )
        self._fields.streaming = ConfigField(default=EndPointDefaults.STREAMING)
        self._fields.server_metrics_url = ConfigField(
            default=EndPointDefaults.SERVER_METRICS_URL
        )
        self._fields.url = ConfigField(default=EndPointDefaults.URL)


###########################################################################
# PerfAnalzyer (PA) Config
###########################################################################
class ConfigPerfAnalyzer(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.path = ConfigField(default=PerfAnalyzerDefaults.PATH)
        self._fields.stimulus = ConfigField(
            default=PerfAnalyzerDefaults.STIMULUS,
            choices=["concurrency", "request_rate"],
        )
        self._fields.stability_percentage = ConfigField(
            default=PerfAnalyzerDefaults.STABILITY_PERCENTAGE,
            bounds={"min": 1, "max": 999},
        )
        self._fields.measurement_interval = ConfigField(
            default=PerfAnalyzerDefaults.MEASUREMENT_INTERVAL, bounds={"min": 1}
        )
        self._fields.skip_args = ConfigField(default=PerfAnalyzerDefaults.SKIP_ARGS)


###########################################################################
# Input Config
###########################################################################
class ConfigImage(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.batch_size = ConfigField(
            default=ImageDefaults.BATCH_SIZE, bounds={"min": 0}
        )
        self._fields.width_mean = ConfigField(
            default=ImageDefaults.WIDTH_MEAN, bounds={"min": 0}
        )
        self._fields.width_stddev = ConfigField(
            default=ImageDefaults.WIDTH_STDDEV, bounds={"min": 0}
        )
        self._fields.height_mean = ConfigField(
            default=ImageDefaults.HEIGHT_MEAN, bounds={"min": 0}
        )
        self._fields.height_stddev = ConfigField(
            default=ImageDefaults.HEIGHT_STDDEV, bounds={"min": 0}
        )
        self._fields.format = ConfigField(
            default=ImageDefaults.FORMAT, choices=ImageFormat
        )


class ConfigOutputTokens(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.mean = ConfigField(
            default=OutputTokenDefaults.MEAN, bounds={"min": 0}
        )
        self._fields.deterministic = ConfigField(
            default=OutputTokenDefaults.DETERMINISTIC
        )
        self._fields.stddev = ConfigField(
            default=OutputTokenDefaults.STDDEV, bounds={"min": 0}
        )


class ConfigSyntheticTokens(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.mean = ConfigField(
            default=SyntheticTokenDefaults.MEAN, bounds={"min": 0}
        )
        self._fields.stddev = ConfigField(
            default=SyntheticTokenDefaults.STDDEV, bounds={"min": 0}
        )


class ConfigPrefixPrompt(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.num = ConfigField(
            default=PrefixPromptDefaults.NUM, bounds={"min": 0}
        )
        self._fields.length = ConfigField(
            default=PrefixPromptDefaults.LENGTH, bounds={"min": 0}
        )


class ConfigRequestCount(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.warmup = ConfigField(
            default=RequestCountDefaults.WARMUP, bounds={"min": 0}
        )


class ConfigInput(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.batch_size = ConfigField(default=InputDefaults.BATCH_SIZE)
        self._fields.extra = ConfigField(default=InputDefaults.EXTRA)
        self._fields.goodput = ConfigField(default=InputDefaults.GOODPUT)
        self._fields.header = ConfigField(default=InputDefaults.HEADER)
        self._fields.file = ConfigField(default=InputDefaults.FILE)
        self._fields.num_dataset_entries = ConfigField(
            default=InputDefaults.NUM_DATASET_ENTRIES
        )
        self._fields.random_seed = ConfigField(default=InputDefaults.RANDOM_SEED)

        self.image = ConfigImage()
        self.output_tokens = ConfigOutputTokens()
        self.synthetic_tokens = ConfigSyntheticTokens()
        self.prefix_prompt = ConfigPrefixPrompt()
        self.request_count = ConfigRequestCount()


###########################################################################
# Output Config
###########################################################################
class ConfigOutput(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.artifact_directory = ConfigField(
            default=OutputDefaults.ARTIFACT_DIRECTORY
        )
        self._fields.checkpoint_directory = ConfigField(
            default=OutputDefaults.CHECKPOINT_DIRECTORY
        )
        self._fields.profile_export_file = ConfigField(
            default=OutputDefaults.PROFILE_EXPORT_FILE
        )
        self._fields.generate_plots = ConfigField(default=OutputDefaults.GENERATE_PLOTS)


###########################################################################
# Tokenizer Config
###########################################################################
class ConfigTokenizer(BaseConfig):
    def __init__(self):
        super().__init__()
        self._fields.name = ConfigField(default=TokenizerDefaults.NAME)
        self._fields.revision = ConfigField(default=TokenizerDefaults.REVISION)
        self._fields.trust_remote_code = ConfigField(
            default=TokenizerDefaults.TRUST_REMOTE_CODE
        )


###########################################################################
# Top-Level Config
###########################################################################
class ConfigCommand(BaseConfig):
    def __init__(self, user_config: Optional[Dict[str, Any]] = None):
        super().__init__()

        self._fields.model_names = ConfigField(
            default=TopLevelDefaults.MODEL_NAME, required=True
        )

        self._fields.analyze = ConfigAnalyze()
        self._fields.endpoint = ConfigEndPoint()
        self._fields.perf_analyzer = ConfigPerfAnalyzer()
        self._fields.input = ConfigInput()
        self._fields.output = ConfigOutput()
        self._fields.tokenizer = ConfigTokenizer()

        self._parse_yaml(user_config)

    # def __deepcopy__(self, memo):
    #     new_copy = ConfigCommand(user_config={})
    #     new_copy._fields = deepcopy(self._fields, memo)
    #     new_copy._values = new_copy._fields._values

    #     return new_copy

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
    # Top-Level Parsing Methods
    ###########################################################################
    def _parse_yaml(self, user_config: Optional[Dict[str, Any]] = None) -> None:
        if user_config is None:
            return

        for key, value in user_config.items():
            if key == "model_name" or key == "model_names":
                self._parse_model_names(value)
            elif key == "analyze":
                self._parse_analyze(value)
            elif key == "endpoint":
                self._parse_endpoint(value)
            elif key == "perf_analyzer":
                self._parse_perf_analyzer(value)
            elif key == "input":
                self._parse_input(value)
            elif key == "output":
                self._parse_output(value)
            elif key == "tokenizer":
                self._parse_tokenizer(value)
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid top-level parameter"
                )

    ###########################################################################
    # Model Names Parsing Methods
    ###########################################################################
    def _parse_model_names(self, model_names: str) -> None:
        if type(model_names) is str:
            self.model_names = [model_names]
        elif type(model_names) is list:
            self.model_names = [model_names[0] + "_multi"]
        else:
            raise ValueError("User Config: model_names must be a string or list")

    ###########################################################################
    # Analyze Parsing Methods
    ###########################################################################
    def _parse_analyze(self, analyze: Dict[str, Any]) -> None:
        sweep_parameters: Dict[str, Any] = {}
        for sweep_type, range_dict in analyze.items():
            if (
                sweep_type in runtime_pa_parameters
                or sweep_type in runtime_gap_parameters
            ):
                if "step" in range_dict:
                    range_list = self._create_range_list(sweep_type, range_dict)
                    sweep_parameters[sweep_type] = range_list
                else:
                    start, stop = self._determine_start_and_stop(sweep_type, range_dict)
                    sweep_parameters[sweep_type] = Range(min=start, max=stop)
            else:
                raise ValueError(
                    f"User Config: {sweep_type} is not a valid analyze parameter"
                )
        self.analyze.sweep_parameters = sweep_parameters

    ###########################################################################
    # Endpoint Parsing Methods
    ###########################################################################
    def _parse_endpoint(self, endpoint: Dict[str, Any]) -> None:
        for key, value in endpoint.items():
            if key == "model_selection_strategy":
                self.endpoint.model_selection_strategy = ModelSelectionStrategy(
                    value.upper()
                )
            elif key == "backend":
                self.endpoint.backend = OutputFormat(value.upper())
            elif key == "custom":
                self.endpoint.custom = value
            elif key == "type":
                self.endpoint.type = value
            elif key == "service_kind":
                self.endpoint.service_kind = value
            elif key == "streaming":
                self.endpoint.streaming = value
            elif key == "server_metrics_url":
                self.endpoint.server_metrics_url = value
            elif key == "url":
                self.endpoint.url = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid endpoint parameter"
                )

    ###########################################################################
    # Perf Analyzer Parsing Methods
    ###########################################################################
    def _parse_perf_analyzer(self, perf_analyzer: Dict[str, Any]) -> None:
        for key, value in perf_analyzer.items():
            if key == "path":
                self.perf_analyzer.path = value
            elif key == "stimulus":
                self.perf_analyzer.stimulus = value
            elif key == "stability_percentage":
                self.perf_analyzer.stability_percentage = value
            elif key == "measurement_interval":
                self.perf_analyzer.measurement_interval = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid perf_analyzer parameter"
                )

    ###########################################################################
    # Input Parsing Methods
    ###########################################################################
    def _parse_input(self, input: Dict[str, Any]) -> None:
        for key, value in input.items():
            if key == "batch_size":
                self.input.batch_size = value
            elif key == "extra":
                self.input.extra = value
            elif key == "goodput":
                self.input.goodput = value
            elif key == "header":
                self.input.header = value
            elif key == "file":
                self.input.file = value
            elif key == "num_dataset_entries":
                self.input.num_dataset_entries = value
            elif key == "random_seed":
                self.input.random_seed = value
            elif key == "image":
                self._parse_image(value)
            elif key == "output_tokens":
                self._parse_output_tokens(value)
            elif key == "synthetic_tokens":
                self._parse_synthetic_tokens(value)
            elif key == "prefix_prompt":
                self._parse_prefix_prompt(value)
            elif key == "request_count":
                self._parse_request_count(value)
            else:
                raise ValueError(f"User Config: {key} is not a valid input parameter")

    def _parse_image(self, image: Dict[str, Any]) -> None:
        for key, value in image.items():
            if key == "batch_size":
                self.input.image.batch_size = value
            elif key == "width_mean":
                self.input.image.width_mean = value
            elif key == "width_stddev":
                self.input.image.width_stddev = value
            elif key == "height_mean":
                self.input.image.height_mean = value
            elif key == "height_stddev":
                self.input.image.height_stddev = value
            elif key == "format":
                self.input.image.format = ImageFormat(value.upper())
            else:
                raise ValueError(f"User Config: {key} is not a valid image parameter")

    def _parse_output_tokens(self, output_tokens: Dict[str, Any]) -> None:
        for key, value in output_tokens.items():
            if key == "mean":
                self.input.output_tokens.mean = value
            elif key == "deterministic":
                self.input.output_tokens.deterministic = value
            elif key == "stddev":
                self.input.output_tokens.stddev = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid output_tokens parameter"
                )

    def _parse_synthetic_tokens(self, synthetic_tokens: Dict[str, Any]) -> None:
        for key, value in synthetic_tokens.items():
            if key == "mean":
                if type(value) is int:
                    self.input.synthetic_tokens.mean = value
                else:
                    raise ValueError(
                        "User Config: synthetic_tokens mean must be an integer"
                    )
            elif key == "stddev":
                self.input.synthetic_tokens.stddev = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid synthetic_tokens parameter"
                )

    def _parse_prefix_prompt(self, prefix_prompt: Dict[str, Any]) -> None:
        for key, value in prefix_prompt.items():
            if key == "num":
                self.input.prefix_prompt.num = value
            elif key == "length":
                self.input.prefix_prompt.length = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid prefix_prompt parameter"
                )

    def _parse_request_count(self, request_count: Dict[str, Any]) -> None:
        for key, value in request_count.items():
            if key == "warmup":
                self.input.request_count.warmup = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid request_count parameter"
                )

    ###########################################################################
    # Output Parsing Methods
    ###########################################################################
    def _parse_output(self, output: Dict[str, Any]) -> None:
        for key, value in output.items():
            if key == "artifact_directory":
                self.output.artifact_directory = value
            elif key == "checkpoint_directory":
                self.output.checkpoint_directory = value
            elif key == "profile_export_file":
                self.output.profile_export_file = value
            elif key == "generate_plots":
                self.output.generate_plots = value
            else:
                raise ValueError(f"User Config: {key} is not a valid output parameter")

    ###########################################################################
    # Tokenizer Parsing Methods
    ###########################################################################
    def _parse_tokenizer(self, tokenizer: Dict[str, Any]) -> None:
        for key, value in tokenizer.items():
            if key == "name":
                self.tokenizer.name = value
            elif key == "revision":
                self.tokenizer.revision = value
            elif key == "trust_remote_code":
                self.tokenizer.trust_remote_code = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid tokenizer parameter"
                )

    ###########################################################################
    # Utility Parsing Methods
    ###########################################################################
    def _create_range_list(
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
            return AnalyzeDefaults.MIN_CONCURRENCY
        elif sweep_type == "runtime_batch_size":
            return AnalyzeDefaults.MIN_MODEL_BATCH_SIZE
        elif sweep_type == "request_rate":
            return AnalyzeDefaults.MIN_REQUEST_RATE
        elif sweep_type == "num_dataset_entries":
            return AnalyzeDefaults.MIN_NUM_DATASET_ENTRIES
        elif sweep_type == "input_sequence_length":
            return AnalyzeDefaults.MIN_INPUT_SEQUENCE_LENGTH
        else:
            raise ValueError(f"User Config: {sweep_type} is not a valid sweep type")

    def _get_default_stop(self, sweep_type: str) -> int:
        if sweep_type == "concurrency":
            return AnalyzeDefaults.MAX_CONCURRENCY
        elif sweep_type == "runtime_batch_size":
            return AnalyzeDefaults.MAX_MODEL_BATCH_SIZE
        elif sweep_type == "request_rate":
            return AnalyzeDefaults.MAX_REQUEST_RATE
        elif sweep_type == "num_dataset_entries":
            return AnalyzeDefaults.MAX_NUM_DATASET_ENTRIES
        elif sweep_type == "input_sequence_length":
            return AnalyzeDefaults.MAX_INPUT_SEQUENCE_LENGTH
        else:
            raise ValueError(f"User Config: {sweep_type} is not a valid sweep type")

    def _get_default_step(self, sweep_type: str) -> int:
        return AnalyzeDefaults.STEP
