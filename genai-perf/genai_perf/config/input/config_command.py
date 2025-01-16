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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

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
from genai_perf.constants import runtime_gap_parameters, runtime_pa_parameters
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.types import CheckpointObject, ModelName


class Subcommand(Enum):
    COMPARE = auto()
    PROFILE = auto()
    ANALYZE = auto()


ConfigRangeOrList: TypeAlias = Optional[Union[Range, List[int]]]
AnalyzeParameter: TypeAlias = Dict[str, ConfigRangeOrList]


###########################################################################
# Analyze Subcommand Config
###########################################################################
@dataclass
class ConfigAnalyze:
    sweep_parameters: AnalyzeParameter = default_field(AnalyzeDefaults.SWEEP_PARAMETER)

    @classmethod
    def _create_class_from_checkpoint(
        cls, config_analyze_dict: CheckpointObject
    ) -> "ConfigAnalyze":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of a ConfigAnalyze
        """
        config_analyze = ConfigAnalyze(**config_analyze_dict)

        return config_analyze


###########################################################################
# EndPoint Config
###########################################################################
@dataclass
class ConfigEndPoint:
    model_selection_strategy: ModelSelectionStrategy = default_field(
        EndPointDefaults.MODEL_SELECTION_STRATEGY
    )
    backend: OutputFormat = default_field(EndPointDefaults.BACKEND)
    custom: str = default_field(EndPointDefaults.CUSTOM)
    type: str = default_field(EndPointDefaults.TYPE)
    service_kind: str = default_field(EndPointDefaults.SERVICE_KIND)
    streaming: bool = default_field(EndPointDefaults.STREAMING)
    server_metrics_url: str = default_field(EndPointDefaults.SERVER_METRICS_URL)
    url: str = default_field(EndPointDefaults.URL)

    @classmethod
    def _create_class_from_checkpoint(
        cls, config_endpoint_dict: CheckpointObject
    ) -> "ConfigEndPoint":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of ConfigEndPoint
        """
        config_endpoint = ConfigEndPoint(**config_endpoint_dict)

        return config_endpoint


###########################################################################
# PerfAnalzyer (PA) Config
###########################################################################
@dataclass
class ConfigPerfAnalyzer:
    path: str = default_field(PerfAnalyzerDefaults.PATH)
    stimulus: Dict[str, Any] = default_field(PerfAnalyzerDefaults.STIMULUS)
    stability_percentage: float = default_field(
        PerfAnalyzerDefaults.STABILITY_PERCENTAGE
    )
    measurement_interval: int = default_field(PerfAnalyzerDefaults.MEASUREMENT_INTERVAL)

    @classmethod
    def _create_class_from_checkpoint(
        cls, config_perf_analyzer_dict: CheckpointObject
    ) -> "ConfigPerfAnalyzer":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of ConfigPerfAnalyzer
        """
        config_perf_analyzer = ConfigPerfAnalyzer(**config_perf_analyzer_dict)

        return config_perf_analyzer


###########################################################################
# Input Config
###########################################################################
@dataclass
class ConfigImage:
    batch_size: int = default_field(ImageDefaults.BATCH_SIZE)
    width_mean: int = default_field(ImageDefaults.WIDTH_MEAN)
    width_stddev: int = default_field(ImageDefaults.WIDTH_STDDEV)
    height_mean: int = default_field(ImageDefaults.HEIGHT_MEAN)
    height_stddev: int = default_field(ImageDefaults.HEIGHT_STDDEV)
    format: Optional[str] = default_field(ImageDefaults.FORMAT)

    @classmethod
    def _create_class_from_checkpoint(
        cls, config_image_dict: CheckpointObject
    ) -> "ConfigImage":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of ConfigImage
        """
        config_image = ConfigImage(**config_image_dict)

        return config_image


@dataclass
class ConfigOutputTokens:
    mean: int = default_field(OutputTokenDefaults.MEAN)
    deterministic: bool = default_field(OutputTokenDefaults.DETERMINISTIC)
    stddev: int = default_field(OutputTokenDefaults.STDDEV)

    @classmethod
    def _create_class_from_checkpoint(
        cls, config_output_tokens_dict: CheckpointObject
    ) -> "ConfigOutputTokens":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of ConfigOutputTokens
        """
        config_output_tokens = ConfigOutputTokens(**config_output_tokens_dict)

        return config_output_tokens


@dataclass
class ConfigSyntheticTokens:
    mean: int = default_field(SyntheticTokenDefaults.MEAN)
    stddev: int = default_field(SyntheticTokenDefaults.STDDEV)

    @classmethod
    def _create_class_from_checkpoint(
        cls, config_synthetic_tokens_dict: CheckpointObject
    ) -> "ConfigSyntheticTokens":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of ConfigSyntheticTokens
        """
        config_synthetic_tokens = ConfigSyntheticTokens(**config_synthetic_tokens_dict)

        return config_synthetic_tokens


@dataclass
class ConfigPrefixPrompt:
    num: int = default_field(PrefixPromptDefaults.NUM)
    length: int = default_field(PrefixPromptDefaults.LENGTH)

    @classmethod
    def _create_class_from_checkpoint(
        cls, config_prefix_prompt_dict: CheckpointObject
    ) -> "ConfigPrefixPrompt":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of ConfigPrefixPrompt
        """
        config_prefix_prompt = ConfigPrefixPrompt(**config_prefix_prompt_dict)

        return config_prefix_prompt


@dataclass
class ConfigRequestCount:
    num: int = default_field(RequestCountDefaults.NUM)
    warmup: int = default_field(RequestCountDefaults.WARMUP)

    @classmethod
    def _create_class_from_checkpoint(
        cls, config_request_count_dict: CheckpointObject
    ) -> "ConfigRequestCount":
        """
        Takes the checkpoint's representation of the class and creates
        (and populates) a new instance of ConfigRequestCount
        """
        config_request_count = ConfigRequestCount(**config_request_count_dict)

        return config_request_count


@dataclass
class ConfigInput:
    batch_size: int = default_field(InputDefaults.BATCH_SIZE)
    extra: str = default_field(InputDefaults.EXTRA)
    goodput: Dict[str, Any] = default_field(InputDefaults.GOODPUT)
    header: Dict[str, Any] = default_field(InputDefaults.HEADER)
    file: Dict[str, Any] = default_field(InputDefaults.FILE)
    num_dataset_entries: int = default_field(InputDefaults.NUM_DATASET_ENTRIES)
    random_seed: int = default_field(InputDefaults.RANDOM_SEED)

    image: ConfigImage = default_field(ConfigImage())
    output_tokens: ConfigOutputTokens = default_field(ConfigOutputTokens())
    synthetic_tokens: ConfigSyntheticTokens = default_field(ConfigSyntheticTokens())
    prefix_prompt: ConfigPrefixPrompt = default_field(ConfigPrefixPrompt())
    request_count: ConfigRequestCount = default_field(ConfigRequestCount())

    @classmethod
    def create_class_from_checkpoint(
        cls, config_input_dict: CheckpointObject
    ) -> "ConfigInput":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of ConfigInput
        """
        config_input = ConfigInput(**config_input_dict)
        config_input.image = ConfigImage._create_class_from_checkpoint(
            config_input_dict["image"]
        )
        config_input.output_tokens = ConfigOutputTokens._create_class_from_checkpoint(
            config_input_dict["output_tokens"]
        )
        config_input.synthetic_tokens = (
            ConfigSyntheticTokens._create_class_from_checkpoint(
                config_input_dict["synthetic_tokens"]
            )
        )
        config_input.prefix_prompt = ConfigPrefixPrompt._create_class_from_checkpoint(
            config_input_dict["prefix_prompt"]
        )
        config_input.request_count = ConfigRequestCount._create_class_from_checkpoint(
            config_input_dict["request_count"]
        )

        return config_input


###########################################################################
# Output Config
###########################################################################
@dataclass
class ConfigOutput:
    artifact_directory: str = default_field(OutputDefaults.ARTIFACT_DIRECTORY)
    checkpoint_directory: str = default_field(OutputDefaults.CHECKPOINT_DIRECTORY)
    profile_export_file: str = default_field(OutputDefaults.PROFILE_EXPORT_FILE)
    generate_plots: bool = default_field(OutputDefaults.GENERATE_PLOTS)

    @classmethod
    def _create_class_from_checkpoint(
        cls, config_output_dict: CheckpointObject
    ) -> "ConfigOutput":
        """
        Takes the checkpoint's representation of the class and creates
        (and populates) a new instance of ConfigOutput
        """
        config_output = ConfigOutput(**config_output_dict)

        return config_output


###########################################################################
# Tokenizer Config
###########################################################################
@dataclass
class ConfigTokenizer:
    name: str = default_field(TokenizerDefaults.NAME)
    revision: str = default_field(TokenizerDefaults.REVISION)
    trust_remote_code: bool = default_field(TokenizerDefaults.TRUST_REMOTE_CODE)

    @classmethod
    def _create_class_from_checkpoint(
        cls, config_tokenizer_dict: CheckpointObject
    ) -> "ConfigTokenizer":
        """
        Takes the checkpoint's representation of the class and creates
        (and populates) a new instance of ConfigTokenizer
        """
        config_tokenizer = ConfigTokenizer(**config_tokenizer_dict)

        return config_tokenizer


###########################################################################
# Top-Level Config
###########################################################################
@dataclass
class ConfigCommand:
    user_config: Dict[str, Any]
    model_names: List[ModelName] = default_field([TopLevelDefaults.MODEL_NAME])

    analyze: ConfigAnalyze = default_field(ConfigAnalyze())
    endpoint: ConfigEndPoint = default_field(ConfigEndPoint())
    perf_analyzer: ConfigPerfAnalyzer = default_field(ConfigPerfAnalyzer())
    input: ConfigInput = default_field(ConfigInput())
    output: ConfigOutput = default_field(ConfigOutput())
    tokenizer: ConfigTokenizer = default_field(ConfigTokenizer())

    def __post_init__(self):
        self._parse_yaml(self.user_config)

    @classmethod
    def create_class_from_checkpoint(
        cls, config_command_dict: CheckpointObject
    ) -> "ConfigCommand":
        """
        Takes the checkpoint's representation of the class and creates
        (and populates) a new instance of ConfigCommand
        """
        config_command = ConfigCommand(**config_command_dict)
        config_command.analyze = ConfigAnalyze._create_class_from_checkpoint(
            config_command_dict["analyze"]
        )
        config_command.endpoint = ConfigEndPoint._create_class_from_checkpoint(
            config_command_dict["endpoint"]
        )
        config_command.perf_analyzer = ConfigPerfAnalyzer._create_class_from_checkpoint(
            config_command_dict["perf_analyzer"]
        )
        config_command.input = ConfigInput.create_class_from_checkpoint(
            config_command_dict["input"]
        )
        config_command.output = ConfigOutput._create_class_from_checkpoint(
            config_command_dict["output"]
        )
        config_command.tokenizer = ConfigTokenizer._create_class_from_checkpoint(
            config_command_dict["tokenizer"]
        )

        return config_command

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
    def _parse_yaml(self, user_config: Dict[str, Any]) -> None:
        for key, value in user_config.items():
            if key == "model_name":
                self._parse_model_name(value)
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
    # Model Name Parsing Methods
    ###########################################################################
    def _parse_model_name(self, model_name: str) -> None:
        if type(model_name) is str:
            self.model_names = [model_name]
        else:
            raise ValueError("User Config: model_name must be a string")

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
                self.input.image.format = value
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
            if key == "num":
                self.input.request_count.num = value
            elif key == "warmup":
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
