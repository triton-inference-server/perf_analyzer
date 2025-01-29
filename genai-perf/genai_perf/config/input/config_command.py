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

from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_analyze import ConfigAnalyze
from genai_perf.config.input.config_defaults import (
    AnalyzeDefaults,
    Range,
    TopLevelDefaults,
)
from genai_perf.config.input.config_endpoint import ConfigEndPoint
from genai_perf.config.input.config_field import ConfigField
from genai_perf.config.input.config_input import ConfigInput
from genai_perf.config.input.config_output import ConfigOutput
from genai_perf.config.input.config_perf_analyzer import ConfigPerfAnalyzer
from genai_perf.config.input.config_tokenizer import ConfigTokenizer
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.types import CheckpointObject, ModelName


class Subcommand(Enum):
    COMPARE = auto()
    PROFILE = auto()
    ANALYZE = auto()


ConfigRangeOrList: TypeAlias = Optional[Union[Range, List[int]]]


class ConfigCommand(BaseConfig):
    """
    Describes the top-level configuration options for GAP
    """

    def __init__(self, user_config: Optional[Dict[str, Any]] = None):
        super().__init__()

        self.model_names: Any = ConfigField(
            default=TopLevelDefaults.MODEL_NAME, required=True
        )

        self.analyze = ConfigAnalyze()
        self.endpoint = ConfigEndPoint()
        self.perf_analyzer = ConfigPerfAnalyzer()
        self.input = ConfigInput()
        self.output = ConfigOutput()
        self.tokenizer = ConfigTokenizer()

        self._parse_yaml(user_config)

    ###########################################################################
    # Top-Level Parsing Methods
    ###########################################################################
    def _parse_yaml(self, user_config: Optional[Dict[str, Any]] = None) -> None:
        if not user_config:
            return

        for key, value in user_config.items():
            if key == "model_name" or key == "model_names":
                self._parse_model_names(value)
            elif key == "analyze":
                self.analyze.parse(value)
            elif key == "endpoint":
                self.endpoint.parse(value)
            elif key == "perf_analyzer":
                self.perf_analyzer.parse(value)
            elif key == "input":
                self.input.parse(value)
            elif key == "output":
                self.output.parse(value)
            elif key == "tokenizer":
                self.tokenizer.parse(value)
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid top-level parameter"
                )

            self._infer_settings()
            self._check_for_illegal_combinations()

    def _parse_model_names(self, model_names: str) -> None:
        if type(model_names) is str:
            self.model_names = [model_names]
        elif type(model_names) is list:
            self.model_names = [model_names[0] + "_multi"]
        else:
            raise ValueError("User Config: model_names must be a string or list")

    ###########################################################################
    # Infer Methods
    ###########################################################################
    def _infer_settings(self) -> None:
        self.endpoint.infer_settings(model_name=self.model_names[0])

        self.input.infer_prompt_source()
        self.input.infer_synthetic_input_files()

    ###########################################################################
    # Illegal Combination Methods
    ###########################################################################
    def _check_for_illegal_combinations(self) -> None:
        self._check_output_tokens_and_service_kind()
        self._check_output_format_and_generate_plots()

    def _check_output_tokens_and_service_kind(self) -> None:
        if self.endpoint.service_kind not in ["triton", "tensorrtllm_engine"]:
            if self.input.output_tokens.get_field("deterministic").is_set_by_user:
                raise ValueError(
                    "User Config: input.output_tokens.deterministic is only supported with Triton or TensorRT-LLM Engine service kinds"
                )

    def _check_output_format_and_generate_plots(self) -> None:
        if self.endpoint.output_format in [
            OutputFormat.IMAGE_RETRIEVAL,
            OutputFormat.NVCLIP,
            OutputFormat.OPENAI_EMBEDDINGS,
            OutputFormat.RANKINGS,
        ]:
            if self.output.generate_plots:
                raise ValueError(
                    "User Config: generate_plots is not supported with the {self.endpoint.output_format} output format"
                )

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
