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

from typing import List, Optional

from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.inputs import input_constants as ic
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.metrics.telemetry_statistics import TelemetryStatistics
from genai_perf.plots.plot_config_parser import PlotConfigParser
from genai_perf.plots.plot_manager import PlotManager
from genai_perf.profile_data_parser import ProfileDataParser
from genai_perf.subcommand.common import (
    calculate_metrics,
    create_artifact_directory,
    create_plot_directory,
    create_telemetry_data_collectors,
    generate_inputs,
    merge_telemetry_metrics,
    run_perf_analyzer,
)
from genai_perf.telemetry_data.triton_telemetry_data_collector import (
    TelemetryDataCollector,
)
from genai_perf.tokenizer import get_tokenizer


def profile_handler(config: ConfigCommand, extra_args: Optional[List[str]]) -> None:
    """
    Handles `profile` subcommand workflow
    """
    perf_analyzer_config = PerfAnalyzerConfig(config=config, extra_args=extra_args)

    create_artifact_directory(perf_analyzer_config.get_artifact_directory())
    create_plot_directory(config, perf_analyzer_config.get_artifact_directory())

    tokenizer = get_tokenizer(config)
    inputs_config = InputsConfig(
        config=config,
        tokenizer=tokenizer,
        output_directory=perf_analyzer_config.get_artifact_directory(),
    )
    generate_inputs(inputs_config)

    telemetry_data_collectors = create_telemetry_data_collectors(config)

    run_perf_analyzer(
        config=config,
        perf_analyzer_config=perf_analyzer_config,
        telemetry_data_collectors=telemetry_data_collectors,
    )
    data_parser = calculate_metrics(config, perf_analyzer_config, tokenizer)
    _report_output(
        data_parser=data_parser,
        telemetry_data_collectors=telemetry_data_collectors,
        config=config,
        perf_analyzer_config=perf_analyzer_config,
    )


def _report_output(
    data_parser: ProfileDataParser,
    telemetry_data_collectors: List[Optional[TelemetryDataCollector]],
    config: ConfigCommand,
    perf_analyzer_config: PerfAnalyzerConfig,
) -> None:
    if "session_concurrency" in config.perf_analyzer.stimulus:
        # [TPA-985] Profile export file should have a session concurrency mode
        infer_mode = "request_rate"
        load_level = "0.0"
    # When using fixed schedule mode, infer mode is not set.
    # Setting to default values to avoid an error.
    elif config.input.prompt_source == ic.PromptSource.PAYLOAD:
        infer_mode = "request_rate"
        load_level = "0.0"
    elif "concurrency" in config.perf_analyzer.stimulus:
        infer_mode = "concurrency"
        load_level = f'{config.perf_analyzer.stimulus["concurrency"]}'
    elif "request_rate" in config.perf_analyzer.stimulus:
        infer_mode = "request_rate"
        load_level = f'{config.perf_analyzer.stimulus["request_rate"]}'
    else:
        raise GenAIPerfException("No valid infer mode specified")

    stats = data_parser.get_statistics(infer_mode, load_level)
    session_stats = data_parser.get_session_statistics()

    if telemetry_data_collectors:
        telemetry_metrics = [c.get_metrics() for c in telemetry_data_collectors]  # type: ignore
    else:
        telemetry_metrics = []

    merged_telemetry_metrics = merge_telemetry_metrics(telemetry_metrics)
    telemetry_stats = TelemetryStatistics(merged_telemetry_metrics)

    reporter = OutputReporter(
        stats=stats,
        telemetry_stats=telemetry_stats,
        config=config,
        perf_analyzer_config=perf_analyzer_config,
        session_stats=session_stats,
    )
    reporter.report_output()

    if config.output.generate_plots:
        _create_plots(config)


def _create_plots(config: ConfigCommand) -> None:
    # TMA-1911: support plots CLI option
    plot_dir = config.output.artifact_directory / "plots"
    PlotConfigParser.create_init_yaml_config(
        filenames=[config.output.profile_export_file],  # single run
        output_dir=plot_dir,
    )
    config_parser = PlotConfigParser(plot_dir / "config.yaml")
    plot_configs = config_parser.generate_configs(config)
    plot_manager = PlotManager(plot_configs)
    plot_manager.generate_plots()
