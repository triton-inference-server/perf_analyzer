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
from pathlib import Path
from typing import Any, List, Optional

from genai_perf.config.generate.search_parameter import SearchUsage
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.input.config_defaults import (
    AnalyzeDefaults,
    PerfAnalyzerMeasurementDefaults,
)
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import (
    DEFAULT_INPUT_DATA_JSON,
    OutputFormat,
    PerfAnalyzerMeasurementMode,
    PromptSource,
)
from genai_perf.logging import logging
from genai_perf.types import CheckpointObject, ModelObjectiveParameters, Parameters

logger = logging.getLogger(__name__)


class InferenceType(Enum):
    NONE = auto()
    CONCURRENCY = auto()
    REQUEST_RATE = auto()


@dataclass
class PerfAnalyzerConfig:
    """
    Contains all the methods necessary for handling calls
    to PerfAnalyzer
    """

    def __init__(
        self,
        config: ConfigCommand,
        model_objective_parameters: Optional[ModelObjectiveParameters] = None,
        extra_args: Optional[List[str]] = None,
    ):
        self._model_name = config.model_names[0]
        self._artifact_directory = self._set_artifact_directory(
            config, model_objective_parameters
        )
        self._profile_export_file = self._set_profile_export_file(
            config, model_objective_parameters
        )

        self._set_options_based_on_objective(model_objective_parameters)

        self._cli_args = self._set_cli_args_based_on_config(config, extra_args)
        self._cli_args += self._get_artifact_paths()

    ###########################################################################
    # Set Options Methods
    ###########################################################################
    def _set_cli_args_based_on_config(
        self, config: ConfigCommand, extra_args: Optional[List[str]] = None
    ) -> List[str]:
        cli_args = []

        cli_args += self._add_required_args(config)
        cli_args += self._add_verbose_args(config)
        cli_args += self._add_perf_analyzer_args(config)
        cli_args += self._add_protocol_args(config)
        cli_args += self._add_url_args(config)
        cli_args += self._add_dynamic_grpc_args(config)
        cli_args += self._add_inference_load_args(config)
        cli_args += self._add_endpoint_args(config)
        cli_args += self._add_header_args(config)
        cli_args += self._add_extra_args(extra_args)

        return cli_args

    def _set_options_based_on_objective(
        self, model_objective_parameters: Optional[ModelObjectiveParameters]
    ) -> None:
        self._parameters: Parameters = {}

        if not model_objective_parameters:
            return

        for objective in model_objective_parameters.values():
            for name, parameter in objective.items():
                if parameter.usage == SearchUsage.RUNTIME_PA:
                    self._parameters[name] = parameter.get_value_based_on_category()

    def _set_artifact_directory(
        self,
        config: ConfigCommand,
        model_objective_parameters: Optional[ModelObjectiveParameters],
    ) -> Path:
        artifact_name = [self._get_artifact_model_name(config)]
        artifact_name += self._get_artifact_service_kind(config)

        stimulus = self._get_artifact_stimulus_type(config, model_objective_parameters)
        if stimulus:
            artifact_name += stimulus

        artifact_directory = config.output.artifact_directory / Path(
            "-".join(artifact_name)
        )

        return artifact_directory

    def _set_profile_export_file(
        self,
        config: ConfigCommand,
        model_objective_parameters: Optional[ModelObjectiveParameters],
    ) -> Path:
        profile_export_file = (
            self._artifact_directory / config.output.profile_export_file
        )

        return profile_export_file

    ###########################################################################
    # Misc. Private Methods
    ###########################################################################
    def _get_artifact_paths(self) -> List[str]:
        artifact_paths = [
            f"--input-data",
            f"{self._artifact_directory / DEFAULT_INPUT_DATA_JSON}",
            f"--profile-export-file",
            f"{self._profile_export_file}",
        ]

        return artifact_paths

    def _get_artifact_model_name(self, config: ConfigCommand) -> str:
        if len(config.model_names) > 1:
            model_name: str = config.model_names[0] + "_multi"
        else:
            model_name = config.model_names[0]

        # Preprocess Huggingface model names that include '/' in their model name.
        if (model_name is not None) and ("/" in model_name):
            filtered_name = "_".join(model_name.split("/"))
            logger.info(
                f"Model name '{model_name}' cannot be used to create artifact "
                f"directory. Instead, '{filtered_name}' will be used."
            )
            return filtered_name
        else:
            return model_name

    def _get_artifact_service_kind(self, config: ConfigCommand) -> List[str]:
        if config.endpoint.service_kind in ["openai", "dynamic_grpc"]:
            service_kind = [f"{config.endpoint.service_kind}-{config.endpoint.type}"]
        elif config.endpoint.service_kind == "triton":
            service_kind = [
                f"{config.endpoint.service_kind}-{config.endpoint.backend.value.lower()}"
            ]
        elif config.endpoint.service_kind == "tensorrtllm_engine":
            service_kind = [f"{config.endpoint.service_kind}"]
        else:
            raise ValueError(f"Unknown service kind '{config.endpoint.service_kind}'.")

        return service_kind

    def _get_artifact_stimulus_type(
        self,
        config: ConfigCommand,
        model_objective_parameters: Optional[ModelObjectiveParameters],
    ) -> Optional[List[str]]:
        if model_objective_parameters:
            stimulus = self._get_artifact_stimulus_based_on_objective(
                model_objective_parameters
            )
        else:
            stimulus = self._get_artifact_stimulus_based_on_config(config)

        return stimulus

    def _get_artifact_stimulus_based_on_config(
        self, config: ConfigCommand
    ) -> Optional[List[str]]:
        if (
            config.input.prompt_source == PromptSource.PAYLOAD
            and not "session_concurrency" in config.perf_analyzer.stimulus
            and not "fixed_schedule" in config.perf_analyzer.stimulus
        ):
            stimulus = None
        elif "concurrency" in config.perf_analyzer.stimulus:
            concurrency = config.perf_analyzer.stimulus["concurrency"]
            stimulus = [f"concurrency{concurrency}"]
        elif "request_rate" in config.perf_analyzer.stimulus:
            request_rate = config.perf_analyzer.stimulus["request_rate"]
            stimulus = [f"request_rate{request_rate}"]
        elif "session_concurrency" in config.perf_analyzer.stimulus:
            session_concurrency = config.perf_analyzer.stimulus["session_concurrency"]
            stimulus = [f"session_concurrency{session_concurrency}"]
        elif "fixed_schedule" in config.perf_analyzer.stimulus:
            stimulus = [f"fixed_schedule"]
        else:
            raise GenAIPerfException(f"Stimulus type not found in config")

        return stimulus

    def _get_artifact_stimulus_based_on_objective(
        self, model_objective_parameters: ModelObjectiveParameters
    ) -> Optional[List[str]]:
        stimulus = []
        for objective in model_objective_parameters.values():
            for name, parameter in objective.items():
                # Need to ensure that every artifact directory is unique
                # so we also include GAP parameters in the name
                if (
                    parameter.usage == SearchUsage.RUNTIME_PA
                    or parameter.usage == SearchUsage.RUNTIME_GAP
                ):
                    stimulus.append(f"{name}{parameter.get_value_based_on_category()}")

        return stimulus

    def _add_verbose_args(self, config: ConfigCommand) -> List[str]:
        verbose_args = []
        if config.perf_analyzer.verbose:
            verbose_args += ["-v"]

        return verbose_args

    def _add_required_args(self, config: ConfigCommand) -> List[str]:
        required_args = [f"{config.perf_analyzer.path}"]

        if config.endpoint.service_kind != "dynamic_grpc":
            required_args += [f"-m", f"{config.model_names[0]}", f"--async"]

        return required_args

    def _add_perf_analyzer_args(self, config: ConfigCommand) -> List[str]:
        perf_analyzer_args = []

        if config.perf_analyzer.warmup_request_count > 0:
            perf_analyzer_args += [
                "--warmup-request-count",
                f"{config.perf_analyzer.warmup_request_count}",
            ]

        if config.input.prompt_source != PromptSource.PAYLOAD:
            perf_analyzer_args += [
                f"--stability-percentage",
                f"{config.perf_analyzer.stability_percentage}",
            ]

            mode = config.perf_analyzer.measurement.mode
            if mode == PerfAnalyzerMeasurementMode.REQUEST_COUNT:
                perf_analyzer_args += [
                    "--request-count",
                    f"{self._calculate_request_count(config)}",
                ]
            elif mode == PerfAnalyzerMeasurementMode.INTERVAL:
                perf_analyzer_args += [
                    f"--measurement-interval",
                    f"{config.perf_analyzer.measurement.num}",
                ]

        return perf_analyzer_args

    def _add_url_args(self, config: ConfigCommand) -> List[str]:
        url_args = []

        if (
            config.endpoint.get_field("url").is_set_by_user
            or config.endpoint.service_kind == "triton"
            or config.endpoint.service_kind == "dynamic_grpc"
        ):
            url_args += ["-u", config.endpoint.url]

        return url_args

    def _add_protocol_args(self, config: ConfigCommand) -> List[str]:
        protocol_args = []

        if config.endpoint.service_kind == "triton":
            protocol_args += ["-i", "grpc", "--streaming"]

            if config.endpoint.backend == OutputFormat.TENSORRTLLM:
                protocol_args += ["--shape", "max_tokens:1", "--shape", "text_input:1"]
        elif config.endpoint.service_kind == "openai":
            protocol_args += ["-i", "http"]

        return protocol_args

    def _add_dynamic_grpc_args(self, config: ConfigCommand) -> List[str]:
        dynamic_grpc_args = []

        if config.endpoint.service_kind == "dynamic_grpc":
            dynamic_grpc_args += ["--grpc-method", f"{config.endpoint.grpc_method}"]

        return dynamic_grpc_args

    def _add_inference_load_args(self, config: ConfigCommand) -> List[str]:
        inference_load_args = []

        if not self._parameters:
            if config.perf_analyzer.get_field("stimulus").is_set_by_user:
                if "concurrency" in config.perf_analyzer.stimulus:
                    inference_load_args += [
                        "--concurrency-range",
                        f'{config.perf_analyzer.stimulus["concurrency"]}',
                    ]
                elif "request_rate" in config.perf_analyzer.stimulus:
                    inference_load_args += [
                        "--request-rate-range",
                        f'{config.perf_analyzer.stimulus["request_rate"]}',
                    ]
                elif "session_concurrency" in config.perf_analyzer.stimulus:
                    inference_load_args += [
                        "--session-concurrency",
                        f"{config.perf_analyzer.stimulus['session_concurrency']}",
                    ]
                elif "fixed_schedule" in config.perf_analyzer.stimulus:
                    inference_load_args += [
                        "--fixed-schedule",
                    ]
        else:
            for parameter, value in self._parameters.items():
                if parameter == "concurrency":
                    inference_load_args += ["--concurrency-range", f"{value}"]
                elif parameter == "request_rate":
                    inference_load_args += ["--request-rate-range", f"{value}"]
                elif parameter == "runtime_batch_size":
                    inference_load_args += ["-b", f"{value}"]

        return inference_load_args

    def _add_endpoint_args(self, config: ConfigCommand) -> List[str]:
        endpoint_args = []
        if config.endpoint.service_kind == "tensorrtllm_engine":
            endpoint_args += ["--service-kind", "triton_c_api", "--streaming"]
        else:
            endpoint_args += ["--service-kind", f"{config.endpoint.service_kind}"]

        if config.endpoint.custom:
            endpoint_args += ["--endpoint", f"{config.endpoint.custom}"]

        return endpoint_args

    def _add_header_args(self, config: ConfigCommand) -> List[str]:
        header_args = []

        for h in config.input.header:
            header_args += ["-H", h]

        return header_args

    def _add_extra_args(self, extra_args: Optional[List[str]]) -> List[str]:
        if not extra_args:
            return []

        args = []
        for extra_arg in extra_args:
            args += [f"{extra_arg}"]

        return args

    def _calculate_request_count(self, config: ConfigCommand) -> int:
        """
        Calculate the request count for performance analysis based on the configuration.

        This method determines the number of requests to be used during performance
        analysis. If the user explicitly sets the `num` field in the measurement
        configuration, that value is returned. Otherwise, the request count is
        calculated as the maximum of the configured request count and a value
        derived from the concurrency level multiplied by a predefined multiplier.
        """
        REQUEST_COUNT_CONCURRENCY_MULTIPLIER = 2

        if config.perf_analyzer.measurement.get_field("num").is_set_by_user:
            return config.perf_analyzer.measurement.num
        else:
            concurrency = self._get_concurrency(config)

            request_count = max(
                PerfAnalyzerMeasurementDefaults.REQUEST_COUNT,
                REQUEST_COUNT_CONCURRENCY_MULTIPLIER * concurrency,
            )

            return request_count

    def _get_concurrency(self, config: ConfigCommand) -> int:
        concurrency = 0
        if not self._parameters:
            if "concurrency" in config.perf_analyzer.stimulus:
                concurrency = config.perf_analyzer.stimulus["concurrency"]
        else:
            for parameter, value in self._parameters.items():
                if parameter == "concurrency":
                    concurrency = value

        return concurrency

    ###########################################################################
    # Get Accessor Methods
    ###########################################################################
    def get_parameters(self) -> Parameters:
        """
        Returns a dictionary of parameters and their values
        """
        return self._parameters

    def get_inference_type(self) -> InferenceType:
        """
        Returns the type of inferencing: concurrency or request-rate
        """
        cmd = self.create_command()
        if "--concurrency-range" in cmd:
            return InferenceType.CONCURRENCY
        elif "--request-rate-range" in cmd:
            return InferenceType.REQUEST_RATE
        else:
            return InferenceType.NONE

    def get_inference_value(self) -> int:
        """
        Returns the value that we are inferencing
        """
        infer_type = self.get_inference_type()

        if infer_type == InferenceType.NONE:
            infer_value = AnalyzeDefaults.MIN_CONCURRENCY
        else:
            infer_cmd_option = (
                "--concurrency-range"
                if infer_type == InferenceType.CONCURRENCY
                else "--request-rate-range"
            )

            cmd = self.create_command()
            infer_value_index = cmd.index(infer_cmd_option) + 1
            infer_value = int(cmd[infer_value_index])

        return infer_value

    def get_artifact_directory(self) -> Path:
        return self._artifact_directory

    def get_profile_export_file(self) -> Path:
        return self._profile_export_file

    ###########################################################################
    # CLI String Creation Methods
    ###########################################################################
    def create_command(self) -> List[str]:
        """
        Returns the PA command a list of strings
        """
        cli_args = self._create_pa_cli_cmd_args()

        return cli_args

    def _create_pa_cli_cmd_args(self) -> List[str]:
        cli_args = self._cli_args
        for name, value in self._parameters.items():
            cli_name = self._convert_objective_to_cli_option(name)
            if cli_name in cli_args:
                cli_args[cli_args.index(cli_name) + 1] = str(value)
            else:
                cli_args.append(cli_name)
                cli_args.append(str(value))

        return cli_args

    def _convert_objective_to_cli_option(self, objective_name: str) -> str:
        obj_to_cli_dict = {
            "runtime_batch_size": "-b",
            "concurrency": "--concurrency-range",
            "request_rate": "--request-rate-range",
            "session_concurrency": "--session-concurrency",
        }

        try:
            return obj_to_cli_dict[objective_name]
        except:
            raise GenAIPerfException(f"{objective_name} not found")

    ###########################################################################
    # Representation Methods
    ###########################################################################
    def representation(self) -> str:
        """
        A string representation of the PA command that removes values which
        can vary between runs, but should be ignored when determining
        if a previous (checkpointed) run can be used
        """
        options_with_arg_to_remove = [
            "--url",
            "--metrics-url",
            "--latency-report-file",
            "--measurement-request-count",
            "--input-data",
            "--profile-export-file",
            "-i",
            "-u",
        ]
        options_only_to_remove = ["--verbose", "--extra-verbose", "--verbose-csv"]

        command = deepcopy(self.create_command())

        # Remove the PA call path which is always the first item
        command.pop(0)

        for option_with_arg in options_with_arg_to_remove:
            if option_with_arg in command:
                index = command.index(option_with_arg)
                del command[index : index + 2]

        for option_only in options_only_to_remove:
            if option_only in command:
                index = command.index(option_only)
                del command[index]

        representation = " ".join(command)
        return representation

    def _remove_option_from_cli_string(
        self, option_to_remove: str, cli_string: str, with_arg: bool
    ) -> str:
        if option_to_remove not in cli_string:
            return cli_string

        cli_str_tokens = cli_string.split(" ")

        removal_index = cli_str_tokens.index(option_to_remove)
        cli_str_tokens.pop(removal_index)

        if with_arg:
            cli_str_tokens.pop(removal_index)

        return " ".join(cli_str_tokens)

    ###########################################################################
    # Checkpoint Methods
    ###########################################################################
    def create_checkpoint_object(self) -> CheckpointObject:
        """
        Converts the class data into a dictionary that can be written to
        the checkpoint file
        """
        pa_config_dict = deepcopy(self.__dict__)

        del pa_config_dict["_artifact_directory"]
        del pa_config_dict["_profile_export_file"]

        return pa_config_dict

    @classmethod
    def create_class_from_checkpoint(
        cls, perf_analyzer_config_dict: CheckpointObject
    ) -> "PerfAnalyzerConfig":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of a PerfAnalyzerConfig
        """
        perf_analyzer_config = PerfAnalyzerConfig(
            config=ConfigCommand(
                user_config={"model_name": perf_analyzer_config_dict["_model_name"]},
                enable_debug_logging=False,
            ),
            model_objective_parameters={},
        )

        perf_analyzer_config._parameters = perf_analyzer_config_dict["_parameters"]
        perf_analyzer_config._cli_args = perf_analyzer_config_dict["_cli_args"]

        return perf_analyzer_config
