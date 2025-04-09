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

import json
import shutil
from pathlib import Path
from typing import Any, DefaultDict, Dict

from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.inputs import input_constants as ic
from genai_perf.metrics.telemetry_statistics import TelemetryStatistics
from genai_perf.plots.plot_config_parser import PlotConfigParser
from genai_perf.plots.plot_manager import PlotManager
from genai_perf.profile_data_parser import ProfileDataParser
from genai_perf.subcommand.common import (
    calculate_metrics,
    create_artifact_directory,
    create_plot_directory,
    merge_telemetry_metrics,
)
from genai_perf.tokenizer import get_tokenizer


def process_export_files_handler(config: ConfigCommand) -> None:
    """
    Handles `process-export-files` subcommand workflow
    """
    profile_data = _get_profile_export_json_data(config.process.input_path)
    _update_config_from_profile_data(config, profile_data)

    perf_analyzer_config = PerfAnalyzerConfig(config=config)
    create_artifact_directory(perf_analyzer_config.get_artifact_directory())
    create_plot_directory(config, perf_analyzer_config.get_artifact_directory())
    tokenizer = get_tokenizer(config)
    _copy_profile_export_to_artifact_dir(config, perf_analyzer_config)

    if "telemetry_stats" in profile_data:
        telemetry_stats_dict = profile_data["telemetry_stats"]
    else:
        telemetry_stats_dict = {}
    data_parser = calculate_metrics(config, perf_analyzer_config, tokenizer)
    _report_output(
        data_parser=data_parser,
        telemetry_stats_dict=telemetry_stats_dict,
        config=config,
        perf_analyzer_config=perf_analyzer_config,
    )


def _get_profile_export_json_data(input_directory: Path) -> Dict[str, Any]:
    """
    Loads the profile_export_genai_perf.json file from the provided input path.
    """
    profile_export_json_file = input_directory / "profile_export_genai_perf.json"
    try:
        with open(profile_export_json_file, "r") as f:
            profile_data = json.load(f)
    except FileNotFoundError:
        raise GenAIPerfException(f"File {profile_export_json_file} not found.")
    except json.JSONDecodeError:
        raise GenAIPerfException(f"Invalid JSON format in {profile_export_json_file}.")

    return profile_data


def _update_config_from_profile_data(
    config: ConfigCommand, profile_data: Dict[str, Any]
) -> None:
    """
    Handles uninitialized config fields by loading them from profile_data.
    """
    _set_model_names(config, profile_data)
    _set_input_fields(config, profile_data)
    _set_perf_analyzer_fields(config, profile_data)
    _set_endpoint_fields(config, profile_data)
    _set_tokenizer_fields(config, profile_data)


def _set_model_names(config: ConfigCommand, profile_data: Dict[str, Any]) -> None:
    config.model_names = profile_data["input_config"]["model_names"]


def _set_input_fields(config: ConfigCommand, profile_data: Dict[str, Any]) -> None:
    input_data = profile_data["input_config"]["input"]
    keys_to_exclude = ["prompt_source", "synthetic_files", "payload_file"]
    input_data = {
        k: v
        for k, v in profile_data["input_config"]["input"].items()
        if k not in keys_to_exclude
    }
    config.input.parse(input_data)
    config.input.infer_settings()


def _set_perf_analyzer_fields(
    config: ConfigCommand, profile_data: Dict[str, Any]
) -> None:
    config.perf_analyzer.parse(profile_data["input_config"]["perf_analyzer"])


def _set_endpoint_fields(config: ConfigCommand, profile_data: Dict[str, Any]) -> None:
    endpoint_data = profile_data["input_config"]["endpoint"]
    endpoint_data.pop("output_format", None)
    config.endpoint.parse(endpoint_data)
    config.endpoint.infer_settings(config.model_names[0])


def _set_tokenizer_fields(config: ConfigCommand, profile_data: Dict[str, Any]) -> None:
    config.tokenizer.parse(profile_data["input_config"]["tokenizer"])


def _copy_profile_export_to_artifact_dir(
    config: ConfigCommand, perf_analyzer_config: PerfAnalyzerConfig
) -> None:
    source_file = Path(config.process.input_path) / "profile_export.json"
    destination_file = (
        perf_analyzer_config.get_artifact_directory() / "profile_export.json"
    )
    shutil.copy(source_file, destination_file)


def _report_output(
    data_parser: ProfileDataParser,
    telemetry_stats_dict: DefaultDict[str, Any],
    config: ConfigCommand,
    perf_analyzer_config: PerfAnalyzerConfig,
) -> None:
    if "session_concurrency" in config.perf_analyzer.stimulus:
        # [TPA-985] Profile export file should have a session concurrency mode
        infer_mode = "request_rate"
        load_level = "0.0"
    # When using fixed schedule mode, infer mode is not set.
    # Setting to default values to avoid an error.
    elif (
        hasattr(config.input, "prompt_source")
        and config.input.prompt_source == ic.PromptSource.PAYLOAD
    ):
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

    telemetry_metrics = []  # type: ignore
    merged_telemetry_metrics = merge_telemetry_metrics(telemetry_metrics)

    telemetry_stats = TelemetryStatistics(merged_telemetry_metrics)
    telemetry_stats.set_stats_dict(telemetry_stats_dict)

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
