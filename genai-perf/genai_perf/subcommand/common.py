# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import subprocess  # nosec
from argparse import Namespace
from typing import List, Optional

import genai_perf.logging as logging
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.constants import DEFAULT_TRITON_METRICS_URL
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import DEFAULT_STARTING_INDEX
from genai_perf.inputs.inputs import Inputs
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.profile_data_parser import (
    ImageRetrievalProfileDataParser,
    LLMProfileDataParser,
    ProfileDataParser,
)
from genai_perf.telemetry_data.triton_telemetry_data_collector import (
    TelemetryDataCollector,
    TritonTelemetryDataCollector,
)
from genai_perf.tokenizer import Tokenizer, get_tokenizer
from genai_perf.utils import load_json_str
from genai_perf.wrapper import Profiler

logger = logging.getLogger(__name__)


"""
Contains methods that are used by multiple subcommands
"""


def generate_inputs(config_options: InputsConfig) -> None:
    inputs = Inputs(config_options)
    inputs.create_inputs()


def calculate_metrics(args: Namespace, tokenizer: Tokenizer) -> ProfileDataParser:
    if args.endpoint_type in ["embeddings", "nvclip", "rankings"]:
        return ProfileDataParser(
            args.profile_export_file,
            goodput_constraints=args.goodput,
        )
    elif args.endpoint_type == "image_retrieval":
        return ImageRetrievalProfileDataParser(
            args.profile_export_file,
            goodput_constraints=args.goodput,
        )
    else:
        return LLMProfileDataParser(
            filename=args.profile_export_file,
            tokenizer=tokenizer,
            goodput_constraints=args.goodput,
        )


def get_extra_inputs_as_dict(args: Namespace) -> dict:
    request_inputs = {}
    if args.extra_inputs:
        for input_str in args.extra_inputs:
            if input_str.startswith("{") and input_str.endswith("}"):
                request_inputs.update(load_json_str(input_str))
            else:
                semicolon_count = input_str.count(":")
                if semicolon_count != 1:
                    raise ValueError(
                        f"Invalid input format for --extra-inputs: {input_str}\n"
                        "Expected input format: 'input_name:value'"
                    )
                input_name, value = input_str.split(":", 1)

                if not input_name or not value:
                    raise ValueError(
                        f"Input name or value is empty in --extra-inputs: {input_str}\n"
                        "Expected input format: 'input_name:value'"
                    )

                is_bool = value.lower() in ["true", "false"]
                is_int = value.isdigit()
                is_float = value.count(".") == 1 and (
                    value[0] == "." or value.replace(".", "").isdigit()
                )

                if is_bool:
                    value = value.lower() == "true"
                elif is_int:
                    value = int(value)
                elif is_float:
                    value = float(value)

                if input_name in request_inputs:
                    raise ValueError(
                        f"Input name already exists in request_inputs dictionary: {input_name}"
                    )
                request_inputs[input_name] = value

    return request_inputs


def create_telemetry_data_collector(
    args: Namespace,
) -> Optional[TelemetryDataCollector]:
    telemetry_data_collector = None
    if args.service_kind == "triton":
        server_metrics_url = args.server_metrics_url or DEFAULT_TRITON_METRICS_URL
        telemetry_data_collector = TritonTelemetryDataCollector(server_metrics_url)

    if telemetry_data_collector and not telemetry_data_collector.is_url_reachable():
        logger.warning(
            f"The metrics URL ({telemetry_data_collector.metrics_url}) is unreachable. "
            "GenAI-Perf cannot collect telemetry data."
        )
        telemetry_data_collector = None
    return telemetry_data_collector


def create_artifacts_dirs(args: Namespace) -> None:
    plot_dir = args.artifact_dir / "plots"
    os.makedirs(args.artifact_dir, exist_ok=True)
    if hasattr(args, "generate_plots") and args.generate_plots:
        os.makedirs(plot_dir, exist_ok=True)


def create_config_options(args: Namespace) -> InputsConfig:
    try:
        extra_input_dict = get_extra_inputs_as_dict(args)
    except ValueError as e:
        raise GenAIPerfException(e)

    return InputsConfig(
        input_type=args.prompt_source,
        output_format=args.output_format,
        model_name=args.model,
        model_selection_strategy=args.model_selection_strategy,
        input_filename=args.input_file,
        synthetic_input_filenames=args.synthetic_input_files,
        starting_index=DEFAULT_STARTING_INDEX,
        length=args.num_prompts,
        prompt_tokens_mean=args.synthetic_input_tokens_mean,
        prompt_tokens_stddev=args.synthetic_input_tokens_stddev,
        output_tokens_mean=args.output_tokens_mean,
        output_tokens_stddev=args.output_tokens_stddev,
        output_tokens_deterministic=args.output_tokens_mean_deterministic,
        image_width_mean=args.image_width_mean,
        image_width_stddev=args.image_width_stddev,
        image_height_mean=args.image_height_mean,
        image_height_stddev=args.image_height_stddev,
        image_format=args.image_format,
        random_seed=args.random_seed,
        num_prompts=args.num_prompts,
        add_stream=args.streaming,
        tokenizer=get_tokenizer(
            args.tokenizer, args.tokenizer_trust_remote_code, args.tokenizer_revision
        ),
        extra_inputs=extra_input_dict,
        batch_size_image=args.batch_size_image,
        batch_size_text=args.batch_size_text,
        output_dir=args.artifact_dir,
    )


def run_perf_analyzer(
    args: Namespace,
    extra_args: Optional[List[str]] = None,
    perf_analyzer_config: Optional[PerfAnalyzerConfig] = None,
    telemetry_data_collector: Optional[TelemetryDataCollector] = None,
) -> None:
    try:
        if telemetry_data_collector is not None:
            telemetry_data_collector.start()

        if perf_analyzer_config is not None:
            cmd = perf_analyzer_config.create_command()
            logger.info(
                f"Running Perf Analyzer : '{perf_analyzer_config.create_cli_string()}'"
            )
        else:
            cmd = Profiler.build_cmd(args, extra_args)
            logger.info(f"Running Perf Analyzer : '{' '.join(cmd)}'")

        if args and args.verbose:
            subprocess.run(cmd, check=True, stdout=None)  # nosec
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)  # nosec
    finally:
        if telemetry_data_collector is not None:
            telemetry_data_collector.stop()