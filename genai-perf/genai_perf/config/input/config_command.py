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

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import genai_perf.logging as logging
from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_analyze import ConfigAnalyze
from genai_perf.config.input.config_defaults import TopLevelDefaults
from genai_perf.config.input.config_endpoint import ConfigEndPoint
from genai_perf.config.input.config_field import ConfigField
from genai_perf.config.input.config_input import ConfigInput
from genai_perf.config.input.config_output import ConfigOutput
from genai_perf.config.input.config_perf_analyzer import ConfigPerfAnalyzer
from genai_perf.config.input.config_process import ConfigProcess
from genai_perf.config.input.config_tokenizer import ConfigTokenizer
from genai_perf.inputs.input_constants import (
    OutputFormat,
    PerfAnalyzerMeasurementMode,
    PromptSource,
    Subcommand,
)
from genai_perf.utils import split_and_strip_whitespace

logger = logging.getLogger(__name__)


class ConfigCommand(BaseConfig):
    """
    Describes the top-level configuration options for GAP
    """

    def __init__(
        self,
        user_config: Optional[Dict[str, Any]] = None,
        skip_inferencing_and_checking: bool = False,
        enable_debug_logging: bool = True,
    ):
        super().__init__()

        self.model_names: Any = ConfigField(
            default=TopLevelDefaults.MODEL_NAMES,
            required=True,
            add_to_template=True,
            verbose_template_comment="The name of the model(s) to benchmark.",
        )

        self.subcommand: Any = ConfigField(
            default=TopLevelDefaults.SUBCOMMAND,
            choices=Subcommand,
            add_to_template=False,
        )

        self.verbose: Any = ConfigField(
            default=TopLevelDefaults.VERBOSE, add_to_template=False
        )

        self.template_filename: Any = ConfigField(
            default=Path(TopLevelDefaults.TEMPLATE_FILENAME),
            add_to_template=False,
        )

        self.analyze = ConfigAnalyze()
        self.endpoint = ConfigEndPoint()
        self.perf_analyzer = ConfigPerfAnalyzer()
        self.input = ConfigInput()
        self.output = ConfigOutput()
        self.tokenizer = ConfigTokenizer(enable_debug_logging)
        self.process = ConfigProcess()

        self._parse_yaml(user_config, skip_inferencing_and_checking)

    ###########################################################################
    # Top-Level Parsing Methods
    ###########################################################################
    def infer_and_check_options(self) -> None:
        """
        Infers and checks the configuration options.
        """
        self._infer_settings()
        self._check_required_fields_are_set()
        self._check_for_illegal_combinations()
        self._check_fixed_schedule_requirements()
        self._check_profile_export_file()
        if self.subcommand == Subcommand.PROCESS:
            self._check_input_path_is_valid()

    def _parse_yaml(
        self,
        user_config: Optional[Dict[str, Any]] = None,
        skip_inferencing_and_checking: bool = False,
    ) -> None:
        if user_config:
            for key, value in user_config.items():
                if key == "model_name" or key == "model_names":
                    self._parse_model_names(value)
                elif key == "analyze":
                    self.analyze.parse(value)
                elif key == "process":
                    self.process.parse(value)
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

        if not skip_inferencing_and_checking:
            self.infer_and_check_options()

    def _parse_model_names(self, model_names: Any) -> None:
        if type(model_names) is str:
            self.model_names = split_and_strip_whitespace(model_names)
        elif type(model_names) is list:
            self.model_names = model_names
        else:
            raise ValueError("User Config: model_names must be a string or list")

    def _check_required_fields_are_set(self) -> None:
        if self.subcommand == Subcommand.PROCESS:
            # Skip checking model_names for process-export-files subcommand
            self.get_field("model_names").required = False
        super().check_required_fields_are_set()

    ###########################################################################
    # Infer Methods
    ###########################################################################
    def _infer_settings(self) -> None:
        # covers the template creation case
        model_name = self.model_names[0] if self.model_names else ""
        self._infer_subcommand()

        self.endpoint.infer_settings(model_name)
        self.input.infer_settings()
        self.perf_analyzer.infer_settings()
        self.tokenizer.infer_settings(model_name)

    def _infer_subcommand(self) -> None:
        if self.subcommand != Subcommand.CONFIG:
            return

        if self.analyze.any_field_set_by_user():
            self.subcommand = Subcommand.ANALYZE
        elif self.process.any_field_set_by_user():
            self.subcommand = Subcommand.PROCESS
        else:
            self.subcommand = Subcommand.PROFILE

    ###########################################################################
    # Illegal Combination Methods
    ###########################################################################
    def _check_for_illegal_combinations(self) -> None:
        self._check_output_tokens_and_service_kind()
        self._check_output_format_and_generate_plots()

        self.endpoint.check_for_illegal_combinations()
        self.input.check_for_illegal_combinations()

    def _check_fixed_schedule_requirements(self) -> None:
        if self.perf_analyzer.stimulus == "fixed_schedule":
            try:
                with open(self.input.file, "r") as file:
                    for line in file:
                        self._check_for_timestamp_on_line(line)
            except Exception as e:
                raise ValueError(f"Error validating 'timestamp' field: {e}")

    def _check_for_timestamp_on_line(self, line: str) -> None:
        if not line.strip():
            return
        data = json.loads(line)
        if "timestamp" not in data:
            raise ValueError(
                "User Config: Stimulus 'fixed_schedule' requires a 'timestamp' field in every entry in the input file."
            )
        if not isinstance(data["timestamp"], int):
            raise ValueError(
                "User Config: Stimulus 'fixed_schedule' requires the 'timestamp' field to be an integer."
            )

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
                    f"User Config: generate_plots is not supported with the {self.endpoint.output_format} output format"
                )

    ###########################################################################
    # Set Path Methods
    ###########################################################################
    def _set_artifact_directory(self) -> None:
        if not self.output.get_field("artifact_directory").is_set_by_user:
            name = self._preprocess_model_name(self.model_names[0])
            name += self._process_service_kind()
            name += self._process_stimulus()

            self.output.artifact_directory = self.output.artifact_directory / Path(
                "-".join(name)
            )

    def _check_profile_export_file(self) -> None:
        if self.output.get_field("profile_export_file").is_set_by_user:
            if Path(self.output.profile_export_file).parent != Path(""):
                raise ValueError(
                    "Please use artifact_directory option to define intermediary paths to "
                    "the profile_export_file."
                )

    def _check_input_path_is_valid(self) -> None:
        input_path = Path(self.process.input_path)
        if not input_path.is_dir():
            raise ValueError(
                f"User Config: input path {Path(self.process.input_path)} is not a valid directory."
            )
        if Path(self.process.input_path) == self.output.artifact_directory:
            logger.warning(
                "Input directory and artifact directory are the same. "
                "This could lead to potential issues such as file overwriting "
                "or unintended aggregation of results from previous runs."
            )
        self.process.input_path = input_path

    def _preprocess_model_name(self, model_name: str) -> List[str]:
        # Preprocess Huggingface model names that include '/' in their model name.
        if (model_name is not None) and ("/" in model_name):
            filtered_name = "_".join(model_name.split("/"))
            logger.info(
                f"Model name '{model_name}' cannot be used to create artifact "
                f"directory. Instead, '{filtered_name}' will be used."
            )
            return [f"{filtered_name}"]
        else:
            return [f"{model_name}"]

    def _process_service_kind(self) -> List[str]:
        if self.endpoint.service_kind == "openai":
            return [f"{self.endpoint.service_kind}-{self.endpoint.type}"]
        elif self.endpoint.service_kind == "triton":
            return [
                f"{self.endpoint.service_kind}-{self.endpoint.backend.to_lowercase()}"
            ]
        elif self.endpoint.service_kind == "tensorrtllm_engine":
            return [f"{self.endpoint.service_kind}"]
        else:
            raise ValueError(f"Unknown service kind '{self.endpoint.service_kind}'.")

    def _process_stimulus(self) -> List[str]:
        if "concurrency" in self.perf_analyzer.stimulus:
            concurrency = self.perf_analyzer.stimulus["concurrency"]
            return [f"concurrency{concurrency}"]
        elif "request_rate" in self.perf_analyzer.stimulus:
            request_rate = self.perf_analyzer.stimulus["request_rate"]
            return [f"request_rate{request_rate}"]
        else:
            return []

    ###########################################################################
    # Template Creation Methods
    ###########################################################################
    def make_template(self) -> str:
        return self.create_template(header="", level=0, verbose=self.verbose)
