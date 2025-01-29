# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from argparse import Namespace
from typing import List, Optional

from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.metrics.telemetry_statistics import TelemetryStatistics
from genai_perf.plots.plot_config_parser import PlotConfigParser
from genai_perf.plots.plot_manager import PlotManager
from genai_perf.profile_data_parser import ProfileDataParser
from genai_perf.subcommand.common import (
    calculate_metrics,
    create_artifacts_dirs,
    create_config_options,
    create_telemetry_data_collectors,
    generate_inputs,
    merge_telemetry_metrics,
    run_perf_analyzer,
)
from genai_perf.telemetry_data.triton_telemetry_data_collector import (
    TelemetryDataCollector,
)
from genai_perf.tokenizer import get_tokenizer


def profile_handler(args: Namespace, extra_args: Optional[List[str]]) -> None:
    """
    Handles `profile` subcommand workflow
    """
    config_options = create_config_options(args)
    create_artifacts_dirs(args)
    tokenizer = get_tokenizer(
        args.tokenizer,
        args.tokenizer_trust_remote_code,
        args.tokenizer_revision,
    )
    generate_inputs(config_options)
    telemetry_data_collectors = create_telemetry_data_collectors(args)
    run_perf_analyzer(
        args=args,
        extra_args=extra_args,
        telemetry_data_collectors=telemetry_data_collectors,
    )
    data_parser = calculate_metrics(args, tokenizer)
    _report_output(data_parser, telemetry_data_collectors, args)


def _report_output(
    data_parser: ProfileDataParser,
    telemetry_data_collectors: List[TelemetryDataCollector],
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
    telemetry_metrics_list = [
        collector.get_metrics() for collector in telemetry_data_collectors
    ]

    merged_telemetry_metrics = merge_telemetry_metrics(telemetry_metrics_list)

    reporter = OutputReporter(
        stats, TelemetryStatistics(merged_telemetry_metrics), args
    )

    reporter.report_output()
    if args.generate_plots:
        _create_plots(args)


def _create_plots(args: Namespace) -> None:
    # TMA-1911: support plots CLI option
    plot_dir = args.artifact_dir / "plots"
    PlotConfigParser.create_init_yaml_config(
        filenames=[args.profile_export_file],  # single run
        output_dir=plot_dir,
    )
    config_parser = PlotConfigParser(plot_dir / "config.yaml")
    plot_configs = config_parser.generate_configs(
        args.tokenizer, args.tokenizer_trust_remote_code, args.tokenizer_revision
    )
    plot_manager = PlotManager(plot_configs)
    plot_manager.generate_plots()
