#!/usr/bin/env python3
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
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Optional

import genai_perf.logging as logging
from genai_perf import parser
from genai_perf.constants import DEFAULT_TRITON_METRICS_URL
from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.inputs.input_constants import DEFAULT_STARTING_INDEX
from genai_perf.inputs.inputs import Inputs
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.plots.plot_config_parser import PlotConfigParser
from genai_perf.plots.plot_manager import PlotManager
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


def create_artifacts_dirs(args: Namespace) -> None:
    plot_dir = args.artifact_dir / "plots"
    os.makedirs(args.artifact_dir, exist_ok=True)
    if hasattr(args, "generate_plots") and args.generate_plots:
        os.makedirs(plot_dir, exist_ok=True)


def create_config_options(args: Namespace) -> InputsConfig:

    try:
        extra_input_dict = parser.get_extra_inputs_as_dict(args)
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


def create_telemetry_data_collector(
    args: Namespace,
) -> Optional[TelemetryDataCollector]:
    logger = logging.getLogger(__name__)
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


def report_output(
    data_parser: ProfileDataParser,
    telemetry_data_collector: Optional[TelemetryDataCollector],
    args: Namespace,
) -> None:
    if args.concurrency:
        infer_mode = "concurrency"
        load_level = f"{args.concurrency}"
    elif args.request_rate:
        infer_mode = "request_rate"
        load_level = f"{args.request_rate}"
    else:
        raise GenAIPerfException("No valid infer mode specified")

    stats = data_parser.get_statistics(infer_mode, load_level)
    telemetry_stats = (
        telemetry_data_collector.get_statistics() if telemetry_data_collector else None
    )
    reporter = OutputReporter(stats, telemetry_stats, args)

    reporter.report_output()
    if args.generate_plots:
        create_plots(args)


def create_plots(args: Namespace) -> None:
    # TMA-1911: support plots CLI option
    plot_dir = args.artifact_dir / "plots"
    PlotConfigParser.create_init_yaml_config(
        filenames=[args.profile_export_file],  # single run
        output_dir=plot_dir,
    )
    config_parser = PlotConfigParser(plot_dir / "config.yaml")
    plot_configs = config_parser.generate_configs(args.tokenizer)
    plot_manager = PlotManager(plot_configs)
    plot_manager.generate_plots()


# Separate function that can raise exceptions used for testing
# to assert correct errors and messages.
def run():
    # TMA-1900: refactor CLI handler
    logging.init_logging()
    args, extra_args = parser.parse_args()
    config_options = create_config_options(args)
    if args.subcommand == "compare":
        args.func(args)
    else:
        create_artifacts_dirs(args)
        tokenizer = get_tokenizer(
            args.tokenizer,
            args.tokenizer_trust_remote_code,
            args.tokenizer_revision,
        )
        generate_inputs(config_options)
        telemetry_data_collector = create_telemetry_data_collector(args)
        args.func(args, extra_args, telemetry_data_collector)
        data_parser = calculate_metrics(args, tokenizer)
        report_output(data_parser, telemetry_data_collector, args)


def main():
    # Interactive use will catch exceptions and log formatted errors rather than
    # tracebacks.
    try:
        run()
    except Exception as e:
        traceback.print_exc()
        logger = logging.getLogger(__name__)
        logger.error(e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
