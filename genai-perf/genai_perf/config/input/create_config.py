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

import argparse
from typing import Dict, List, Optional

import genai_perf.logging as logging
import genai_perf.utils as utils
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.input.config_defaults import OutputTokenDefaults
from genai_perf.inputs.input_constants import (
    AudioFormat,
    ImageFormat,
    ModelSelectionStrategy,
    OutputFormat,
    PerfAnalyzerMeasurementMode,
    PromptSource,
    Subcommand,
)
from genai_perf.subcommand.common import get_extra_inputs_as_dict

logger = logging.getLogger(__name__)


class CreateConfig:
    """
    Creates a Config object from args
    """

    @staticmethod
    def create(args: argparse.Namespace, extra_args: List[str] = []) -> ConfigCommand:

        if args.subcommand == Subcommand.TEMPLATE.value:
            config = CreateConfig._create_template_config(args)
        else:
            if args.subcommand == Subcommand.CONFIG.value:
                user_config = utils.load_yaml(args.file)
            else:
                user_config = {}

            config = ConfigCommand(user_config, skip_inferencing_and_checking=True)
            config = CreateConfig._add_cli_options_to_config(config, args)
            config.infer_and_check_options()
            CreateConfig._print_warnings(config)

            if config.subcommand != Subcommand.PROCESS:
                logger.info(f"Profiling these models: {', '.join(config.model_names)}")

        return config

    @staticmethod
    def _create_template_config(args: argparse.Namespace) -> ConfigCommand:
        config = ConfigCommand(skip_inferencing_and_checking=True)

        config.verbose = args.verbose
        config.subcommand = Subcommand(args.subcommand)

        if args.file:
            config.template_filename = args.file

        return config

    @staticmethod
    def _print_warnings(config: ConfigCommand) -> None:
        if config.tokenizer.trust_remote_code:
            logger.warning(
                "--tokenizer-trust-remote-code is enabled. "
                "Custom tokenizer code can be executed. "
                "This should only be used with repositories you trust."
            )
        if (
            config.input.prompt_source == PromptSource.PAYLOAD
            and config.input.output_tokens.mean != OutputTokenDefaults.MEAN
        ):
            logger.warning(
                "--output-tokens-mean is incompatible with output_length"
                " in the payload input file. output-tokens-mean"
                " will be ignored in favour of per payload settings."
            )

    @staticmethod
    def _add_cli_options_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:

        CreateConfig._check_that_override_is_set(args)
        CreateConfig._add_top_level_args_to_config(config, args)
        if config.subcommand == Subcommand.PROCESS:
            CreateConfig._add_process_args_to_config(config, args)
        else:
            CreateConfig._add_profile_analyze_args_to_config(config, args)

        return config

    @staticmethod
    def _check_that_override_is_set(args: argparse.Namespace) -> None:
        """
        Check that the --override-config flag is set if the user is trying to
        override a config value via the CLI.
        """
        if not args.subcommand == Subcommand.CONFIG.value:
            return

        args_exempt_from_override = {
            "func",  # this is the function to call that comes from vars(args)
            "subcommand",
            "override_config",
            "file",
            "verbose",
        }

        for key, value in vars(args).items():
            if key in args_exempt_from_override:
                continue
            if value and not args.override_config:
                raise ValueError(
                    "In order to use the CLI to override the config, the --override-config flag must be set."
                )

    @staticmethod
    def _add_profile_analyze_args_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:
        CreateConfig._add_analyze_args_to_config(config, args)
        CreateConfig._add_endpoint_args_to_config(config, args)
        CreateConfig._add_perf_analyzer_args_to_config(config, args)
        CreateConfig._add_input_args_to_config(config, args)
        CreateConfig._add_output_args_to_config(config, args)
        CreateConfig._add_tokenizer_args_to_config(config, args)

        return config

    @staticmethod
    def _add_process_args_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:
        CreateConfig._add_process_export_files_args_to_config(config, args)
        CreateConfig._add_output_args_to_config(config, args)

        return config

    @staticmethod
    def _add_top_level_args_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:
        if hasattr(args, "model") and args.model:
            config.model_names = args.model
        if args.subcommand:
            config.subcommand = Subcommand(args.subcommand)
        if args.verbose:
            config.verbose = args.verbose

        return config

    @staticmethod
    def _add_analyze_args_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:
        if args.subcommand and args.subcommand == "analyze":
            config.analyze.sweep_parameters = {}
            if args.sweep_list:
                sweep_parameters = {args.sweep_type: args.sweep_list}
            else:
                sweep_parameters = {
                    args.sweep_type: {"start": args.sweep_min, "stop": args.sweep_max}
                }
                if args.sweep_step:
                    sweep_parameters[args.sweep_type]["step"] = args.sweep_step

            config.analyze.parse(sweep_parameters)

        return config

    @staticmethod
    def _add_endpoint_args_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:
        if args.model_selection_strategy:
            config.endpoint.model_selection_strategy = ModelSelectionStrategy(
                args.model_selection_strategy.upper()
            )
        if args.backend:
            config.endpoint.backend = OutputFormat(args.backend.upper())
        if args.endpoint:
            config.endpoint.custom = args.endpoint
        if args.endpoint_type:
            config.endpoint.type = args.endpoint_type
        if args.streaming:
            config.endpoint.streaming = args.streaming
        if args.server_metrics_url:
            config.endpoint.server_metrics_urls = args.server_metrics_url
        if args.u:
            config.endpoint.url = args.u
        if args.grpc_method:
            config.endpoint.grpc_method = args.grpc_method

        return config

    @staticmethod
    def _add_perf_analyzer_args_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:

        # config.perf_analyzer.path - There is no equivalent setting in the CLI
        stimulus = CreateConfig._convert_args_to_stimulus(args)
        if stimulus:
            config.perf_analyzer.stimulus = stimulus

        if args.stability_percentage:
            config.perf_analyzer.stability_percentage = args.stability_percentage
        if args.warmup_request_count:
            config.perf_analyzer.warmup_request_count = args.warmup_request_count

        if args.measurement_interval:
            config.perf_analyzer.measurement.mode = PerfAnalyzerMeasurementMode.INTERVAL
            config.perf_analyzer.measurement.num = args.measurement_interval
        elif args.request_count:
            config.perf_analyzer.measurement.mode = (
                PerfAnalyzerMeasurementMode.REQUEST_COUNT
            )
            config.perf_analyzer.measurement.num = args.request_count

        return config

    @staticmethod
    def _add_process_export_files_args_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:
        if hasattr(args, "input_path") and args.input_path:
            config.process.input_path = args.input_path[0]
        return config

    @staticmethod
    def _add_input_args_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:
        # Input - Top Level
        if args.batch_size_text:
            config.input.batch_size = args.batch_size_text

        extra_inputs = get_extra_inputs_as_dict(args)

        if extra_inputs:
            config.input.extra = extra_inputs

        if args.goodput:
            config.input.goodput = args.goodput
        if args.header:
            config.input.header = args.header
        if args.input_file:
            config.input.file = args.input_file
        if args.num_dataset_entries:
            config.input.num_dataset_entries = args.num_dataset_entries
        if args.random_seed:
            config.input.random_seed = args.random_seed

        # Input - Audio
        if args.batch_size_audio:
            config.input.audio.batch_size = args.batch_size_audio
        if args.audio_length_mean:
            config.input.audio.length.mean = args.audio_length_mean
        if args.audio_length_stddev:
            config.input.audio.length.stddev = args.audio_length_stddev
        if args.audio_format:
            config.input.audio.format = AudioFormat(args.audio_format.upper())
        if args.audio_depths:
            config.input.audio.depths = args.audio_depths
        if args.audio_sample_rates:
            config.input.audio.sample_rates = args.audio_sample_rates
        if args.audio_num_channels:
            config.input.audio.num_channels = args.audio_num_channels

        # Input - Image
        if args.batch_size_image:
            config.input.image.batch_size = args.batch_size_image
        if args.image_width_mean:
            config.input.image.width.mean = args.image_width_mean
        if args.image_width_stddev:
            config.input.image.width.stddev = args.image_width_stddev
        if args.image_height_mean:
            config.input.image.height.mean = args.image_height_mean
        if args.image_height_stddev:
            config.input.image.height.stddev = args.image_height_stddev
        if args.image_format:
            config.input.image.format = ImageFormat(args.image_format.upper())

        # Input - Output Tokens
        if args.output_tokens_mean:
            config.input.output_tokens.mean = args.output_tokens_mean
        if args.output_tokens_mean_deterministic:
            config.input.output_tokens.deterministic = (
                args.output_tokens_mean_deterministic
            )
        if args.output_tokens_stddev:
            config.input.output_tokens.stddev = args.output_tokens_stddev

        # Input - Synthetic Tokens
        if args.synthetic_input_tokens_mean:
            config.input.synthetic_tokens.mean = args.synthetic_input_tokens_mean
        if args.synthetic_input_tokens_stddev:
            config.input.synthetic_tokens.stddev = args.synthetic_input_tokens_stddev

        # Input - Prefix Prompt
        if args.num_prefix_prompts:
            config.input.prefix_prompt.num = args.num_prefix_prompts
        if args.prefix_prompt_length:
            config.input.prefix_prompt.length = args.prefix_prompt_length

        # Input - Sessions
        if args.num_sessions:
            config.input.sessions.num = args.num_sessions
        if args.session_turn_delay_mean:
            config.input.sessions.turn_delay.mean = args.session_turn_delay_mean
        if args.session_delay_ratio:
            config.input.sessions.turn_delay.ratio = args.session_delay_ratio
        if args.session_turn_delay_stddev:
            config.input.sessions.turn_delay.stddev = args.session_turn_delay_stddev
        if args.session_turns_mean:
            config.input.sessions.turns.mean = args.session_turns_mean
        if args.session_turns_stddev:
            config.input.sessions.turns.stddev = args.session_turns_stddev

        return config

    @staticmethod
    def _add_output_args_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:
        if args.artifact_dir:
            config.output.artifact_directory = args.artifact_dir
        if args.checkpoint_dir:
            config.output.checkpoint_directory = args.checkpoint_dir
        if args.profile_export_file:
            config.output.profile_export_file = args.profile_export_file
        if args.generate_plots:
            config.output.generate_plots = args.generate_plots
        if args.enable_checkpointing:
            config.output.enable_checkpointing = args.enable_checkpointing

        return config

    @staticmethod
    def _add_tokenizer_args_to_config(
        config: ConfigCommand, args: argparse.Namespace
    ) -> ConfigCommand:
        if args.tokenizer:
            config.tokenizer.name = args.tokenizer
        if args.tokenizer_revision:
            config.tokenizer.revision = args.tokenizer_revision
        if args.tokenizer_trust_remote_code:
            config.tokenizer.trust_remote_code = args.tokenizer_trust_remote_code

        return config

    @staticmethod
    def _convert_args_to_stimulus(
        args: argparse.Namespace,
    ) -> Optional[Dict[str, int | None]]:
        if args.session_concurrency:
            return {"session_concurrency": args.session_concurrency}
        elif args.concurrency:
            return {"concurrency": args.concurrency}
        elif args.request_rate:
            return {"request_rate": args.request_rate}
        elif args.fixed_schedule:
            return {"fixed_schedule": None}
        else:
            return None
