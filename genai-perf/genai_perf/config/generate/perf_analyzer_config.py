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

from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, List, Optional

from genai_perf.config.generate.search_parameter import SearchUsage
from genai_perf.config.input.config_command import (
    ConfigCommand,
    ConfigPerfAnalyzer,
    RunConfigDefaults,
)
from genai_perf.constants import DEFAULT_ARTIFACT_DIR
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import DEFAULT_INPUT_DATA_JSON
from genai_perf.logging import logging
from genai_perf.types import (
    CheckpointObject,
    ModelName,
    ModelObjectiveParameters,
    Parameters,
)
from genai_perf.utils import convert_option_name
from genai_perf.wrapper import Profiler

# This is the list of GAP CLI args that are not used when creating
# the PA command line
perf_analyzer_ignore_args = [
    "artifact_dir",
    "backend",
    "batch_size_image",
    "batch_size_text",
    "concurrency",
    "endpoint_type",
    "extra_inputs",
    "formatted_model_name",
    "func",
    "generate_plots",
    "goodput",
    "image_format",
    "image_height_mean",
    "image_height_stddev",
    "image_width_mean",
    "image_width_stddev",
    "input_dataset",
    "input_file",
    "input_format",
    "model",
    "model_selection_strategy",
    "num_dataset_entries",
    "num_prefix_prompts",
    "prefix_prompt_length",
    "request_count",
    "warmup_request_count",
    "output_format",
    "output_tokens_mean",
    "output_tokens_mean_deterministic",
    "output_tokens_stddev",
    "profile_export_file",
    "prompt_source",
    "random_seed",
    "request_rate",
    "server_metrics_url",
    # The 'streaming' passed in to this script is to determine if the
    # LLM response should be streaming. That is different than the
    # 'streaming' that PA takes, which means something else (and is
    # required for decoupled models into triton).
    "streaming",
    "subcommand",
    "sweep_list",
    "sweep_type",
    "sweep_range",
    "sweep_min",
    "sweep_max",
    "sweep_step",
    "synthetic_input_files",
    "synthetic_input_tokens_mean",
    "synthetic_input_tokens_stddev",
    "tokenizer",
    "tokenizer_trust_remote_code",
    "tokenizer_revision",
]


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
        model_objective_parameters: ModelObjectiveParameters,
        model_name: ModelName,
        args: Namespace = Namespace(),
        extra_args: Optional[List[str]] = None,
    ):
        self._model_name = model_name
        self._args = deepcopy(args)
        self._set_options_based_on_cli(args, extra_args)
        self._set_options_based_on_config(config)
        self._set_options_based_on_objective(model_objective_parameters)
        self._set_artifact_paths(model_objective_parameters)

    ###########################################################################
    # Set Options Methods
    ###########################################################################
    def _set_options_based_on_cli(
        self, args: Namespace, extra_args: Optional[List[str]] = None
    ) -> None:
        self._cli_args = []

        # When restoring from a checkpoint there won't be any args
        if not hasattr(self._args, "subcommand"):
            return

        self._cli_args += self._add_required_args(args)
        self._cli_args += Profiler.add_protocol_args(args)
        self._cli_args += Profiler.add_inference_load_args(args)
        self._cli_args += self._add_misc_args(args)
        self._cli_args += self._add_extra_args(extra_args)

    def _set_options_based_on_config(self, config: ConfigCommand) -> None:
        self._config: ConfigPerfAnalyzer = config.perf_analyzer

    def _set_options_based_on_objective(
        self, model_objective_parameters: ModelObjectiveParameters
    ) -> None:
        self._parameters: Parameters = {}
        for objective in model_objective_parameters.values():
            for name, parameter in objective.items():
                if parameter.usage == SearchUsage.RUNTIME_PA:
                    self._parameters[name] = parameter.get_value_based_on_category()

    def _set_artifact_paths(
        self, model_objective_parameters: ModelObjectiveParameters
    ) -> None:
        # When restoring from a checkpoint there won't be any args
        if not hasattr(self._args, "subcommand"):
            return

        if self._args.artifact_dir == Path(DEFAULT_ARTIFACT_DIR):
            artifact_name = self._get_artifact_model_name()
            artifact_name += self._get_artifact_service_kind()
            artifact_name += self._get_artifact_stimulus_type(
                model_objective_parameters
            )

            self._args.artifact_dir = self._args.artifact_dir / Path(
                "-".join(artifact_name)
            )

        if self._args.profile_export_file.parent != Path(""):
            raise ValueError(
                "Please use --artifact-dir option to define intermediary paths to "
                "the profile export file."
            )

        self._args.profile_export_file = (
            self._args.artifact_dir / self._args.profile_export_file
        )

        self._cli_args += [
            f"--input-data",
            f"{self._args.artifact_dir / DEFAULT_INPUT_DATA_JSON}",
            f"--profile-export-file",
            f"{self._args.profile_export_file}",
        ]

    def _get_artifact_model_name(self) -> List[str]:
        # Preprocess Huggingface model names that include '/' in their model name.
        if (self._args.formatted_model_name is not None) and (
            "/" in self._args.formatted_model_name
        ):
            filtered_name = "_".join(self._args.formatted_model_name.split("/"))
            logger = logging.getLogger(__name__)
            logger.info(
                f"Model name '{self._args.formatted_model_name}' cannot be used to create artifact "
                f"directory. Instead, '{filtered_name}' will be used."
            )
            model_name = [f"{filtered_name}"]
        else:
            model_name = [f"{self._args.formatted_model_name}"]

        return model_name

    def _get_artifact_service_kind(self) -> List[str]:
        if self._args.service_kind == "openai":
            service_kind = [f"{self._args.service_kind}-{self._args.endpoint_type}"]
        elif self._args.service_kind == "triton":
            service_kind = [
                f"{self._args.service_kind}-{self._args.backend.to_lowercase()}"
            ]
        elif self._args.service_kind == "tensorrtllm_engine":
            service_kind = [f"{self._args.service_kind}"]
        else:
            raise ValueError(f"Unknown service kind '{self._args.service_kind}'.")

        return service_kind

    def _get_artifact_stimulus_type(
        self, model_objective_parameters: ModelObjectiveParameters
    ) -> List[str]:
        parameters = model_objective_parameters[self._model_name]

        if "concurrency" in parameters:
            concurrency = str(parameters["concurrency"].get_value_based_on_category())
            stimulus = [f"concurrency{concurrency}"]
        elif "request_rate" in parameters:
            request_rate = str(parameters["request_rate"].get_value_based_on_category())
            stimulus = [f"request_rate{request_rate}"]
        elif "input_sequence_length" in parameters:
            input_sequence_length = str(
                parameters["input_sequence_length"].get_value_based_on_category()
            )
            stimulus = [f"input_sequence_length{input_sequence_length}"]
        elif "num_dataset_entries" in parameters:
            input_sequence_length = str(
                parameters["num_dataset_entries"].get_value_based_on_category()
            )
            stimulus = [f"num_dataset_entries{input_sequence_length}"]
        elif "runtime_batch_size" in parameters:
            runtime_batch_size = str(
                parameters["runtime_batch_size"].get_value_based_on_category()
            )
            stimulus = [f"batch_size{runtime_batch_size}"]

        return stimulus

    def _add_required_args(self, args: Namespace) -> List[str]:
        required_args = [
            f"-m",
            f"{args.formatted_model_name}",
            f"--async",
        ]

        return required_args

    def _add_misc_args(self, args: Namespace) -> List[str]:
        misc_args = []

        for arg, value in vars(args).items():
            if arg in perf_analyzer_ignore_args:
                pass
            elif self._arg_is_tensorrtllm_engine(arg, value):
                misc_args += self._add_tensorrtllm_engine_args()
            elif value is None or value is False:
                pass
            elif value is True:
                misc_args += self._add_boolean_arg(arg)
            else:
                misc_args += self._add_non_boolean_arg(arg, value)

        return misc_args

    def _add_boolean_arg(self, arg: str) -> List[str]:
        if len(arg) == 1:
            return [f"-{arg}"]
        else:
            return [f"--{arg}"]

    def _add_non_boolean_arg(self, arg: str, value: Any) -> List[str]:
        if len(arg) == 1:
            return [f"-{arg}", f"{value}"]
        else:
            converted_arg = convert_option_name(arg)
            return [f"--{converted_arg}", f"{value}"]

    def _add_tensorrtllm_engine_args(self) -> List[str]:
        # GAP needs to call PA using triton_c_api service kind when running
        # against tensorrtllm engine.
        return ["--service-kind", "triton_c_api", "--streaming"]

    def _arg_is_tensorrtllm_engine(self, arg: str, value: str) -> bool:
        return arg == "service_kind" and value == "tensorrtllm_engine"

    def _add_extra_args(self, extra_args: Optional[List[str]]) -> List[str]:
        if not extra_args:
            return []

        args = []
        for extra_arg in extra_args:
            args += [f"{extra_arg}"]

        return args

    ###########################################################################
    # Get Accessor Methods
    ###########################################################################
    def get_parameters(self) -> Parameters:
        """
        Returns a dictionary of parameters and their values
        """
        return self._parameters

    def get_obj_args(self) -> Namespace:
        """
        Returns args that can be used by the existing CLI based methods in GAP
        These will include any objectives that are set via parameters
        """
        obj_args = deepcopy(self._args)
        if "concurrency" in self._parameters:
            obj_args.concurrency = self._parameters["concurrency"]
        if "request_rate" in self._parameters:
            obj_args.request_rate = self._parameters["request_rate"]
        if "runtime_batch_size" in self._parameters:
            obj_args.batch_size = self._parameters["runtime_batch_size"]

        return obj_args

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
            infer_value = RunConfigDefaults.MIN_CONCURRENCY
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

    ###########################################################################
    # CLI String Creation Methods
    ###########################################################################
    def create_command(self) -> List[str]:
        """
        Returns the PA command a list of strings
        """

        cli_args = [self._config.path]
        # FIXME: For now these will come from the CLI until support for a config file is added
        # cli_args = self._create_required_args()
        cli_args += self._cli_args
        cli_args += self._create_parameter_args()

        return cli_args

    def create_cli_string(self) -> str:
        """
        Returns the PA command as a string
        """
        cli_args = self.create_command()

        cli_string = " ".join(cli_args)
        return cli_string

    def _create_required_args(self) -> List[str]:
        # These come from the config and are always used
        required_args = [
            self._config.path,
            "-m",
            self._model_name,
            "--stability-percentage",
            str(self._config.stability_threshold),
        ]

        return required_args

    def _create_parameter_args(self) -> List[str]:
        parameter_args = []
        for name, value in self._parameters.items():
            parameter_args.append(self._convert_objective_to_cli_option(name))
            parameter_args.append(str(value))

        return parameter_args

    def _convert_objective_to_cli_option(self, objective_name: str) -> str:
        obj_to_cli_dict = {
            "runtime_batch_size": "-b",
            "concurrency": "--concurrency-range",
            "request_rate": "--request-rate-range",
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

        command = self.create_command()

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

        # Values set on the CLI are not kept (they can vary from run to run)
        del pa_config_dict["_args"]

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
            model_name=perf_analyzer_config_dict["_model_name"],
            config=ConfigCommand([""]),
            model_objective_parameters={},
        )
        perf_analyzer_config._config = ConfigPerfAnalyzer(
            **perf_analyzer_config_dict["_config"]
        )
        perf_analyzer_config._parameters = perf_analyzer_config_dict["_parameters"]
        perf_analyzer_config._cli_args = perf_analyzer_config_dict["_cli_args"]

        return perf_analyzer_config
