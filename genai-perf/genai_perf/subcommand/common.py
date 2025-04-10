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

import os
import subprocess  # nosec
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import genai_perf.logging as logging
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.constants import DEFAULT_TRITON_METRICS_URL
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.inputs import Inputs
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.metrics.telemetry_metrics import TelemetryMetrics
from genai_perf.profile_data_parser import (
    ImageRetrievalProfileDataParser,
    LLMProfileDataParser,
    ProfileDataParser,
)
from genai_perf.telemetry_data.triton_telemetry_data_collector import (
    TelemetryDataCollector,
    TritonTelemetryDataCollector,
)
from genai_perf.tokenizer import Tokenizer
from genai_perf.utils import load_json_str, remove_file

logger = logging.getLogger(__name__)


"""
Contains methods that are used by multiple subcommands
"""


def generate_inputs(inputs_config: InputsConfig) -> None:
    inputs = Inputs(inputs_config)
    inputs.create_inputs()


def calculate_metrics(
    config: ConfigCommand,
    perf_analyzer_config: PerfAnalyzerConfig,
    tokenizer: Tokenizer,
) -> ProfileDataParser:
    if config.endpoint.output_format == OutputFormat.TEMPLATE:
        return ProfileDataParser(
            perf_analyzer_config.get_profile_export_file(),
            goodput_constraints=config.input.goodput,
        )
    elif config.endpoint.type in ["embeddings", "nvclip", "rankings", "dynamic_grpc"]:
        return ProfileDataParser(
            perf_analyzer_config.get_profile_export_file(),
            goodput_constraints=config.input.goodput,
        )
    elif config.endpoint.type == "image_retrieval":
        return ImageRetrievalProfileDataParser(
            perf_analyzer_config.get_profile_export_file(),
            goodput_constraints=config.input.goodput,
        )
    else:
        return LLMProfileDataParser(
            filename=perf_analyzer_config.get_profile_export_file(),
            tokenizer=tokenizer,
            goodput_constraints=config.input.goodput,
        )


def get_extra_inputs_as_dict(args: Namespace) -> Dict[str, Any]:
    request_inputs: Dict[str, Any] = {}
    if args.extra_inputs:
        for input_str in args.extra_inputs:
            semicolon_count = input_str.count(":")
            if input_str.startswith("{") and input_str.endswith("}"):
                request_inputs.update(load_json_str(input_str))
            elif semicolon_count == 0:  # extra input as a flag
                request_inputs[input_str] = None
            elif semicolon_count == 1:
                input_name, value = input_str.split(":", 1)

                if not input_name or not value:
                    raise ValueError(
                        f"Input name or value is empty in --extra-inputs: "
                        f"{input_str}\nExpected input format: 'input_name' or "
                        "'input_name:value'"
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
                        f"Input name already exists in request_inputs "
                        f"dictionary: {input_name}"
                    )
                request_inputs[input_name] = value
            else:
                raise ValueError(
                    f"Invalid input format for --extra-inputs: {input_str}\n"
                    "Expected input format: 'input_name' or 'input_name:value'"
                )

    return request_inputs


def create_telemetry_data_collectors(
    config: ConfigCommand,
) -> List[Optional[TelemetryDataCollector]]:
    telemetry_collectors: List[Optional[TelemetryDataCollector]] = []

    if not config.endpoint.service_kind == "triton":
        return telemetry_collectors

    if not config.endpoint.server_metrics_urls:
        config.endpoint.server_metrics_urls = [DEFAULT_TRITON_METRICS_URL]

    for url in config.endpoint.server_metrics_urls:
        collector = TritonTelemetryDataCollector(url.strip())
        if collector.is_url_reachable():
            telemetry_collectors.append(collector)
        else:
            logger.warning(f"Skipping unreachable metrics URL: {url}")

    return telemetry_collectors


def create_artifact_directory(artifact_directory: Path) -> None:
    os.makedirs(artifact_directory, exist_ok=True)


def create_plot_directory(config: ConfigCommand, artifact_directory: Path) -> None:
    if config.output.generate_plots:
        plot_dir = artifact_directory / "plots"
        os.makedirs(plot_dir, exist_ok=True)


def run_perf_analyzer(
    config: ConfigCommand,
    perf_analyzer_config: PerfAnalyzerConfig,
    telemetry_data_collectors: List[Optional[TelemetryDataCollector]] = [],
) -> None:
    try:
        for collector in telemetry_data_collectors:
            if collector:
                collector.start()

        remove_file(perf_analyzer_config.get_profile_export_file())
        cmd = perf_analyzer_config.create_command()
        logger.info(f"Running Perf Analyzer : '{' '.join(cmd)}'")

        if config.verbose or config.perf_analyzer.verbose:
            subprocess.run(cmd, check=True, stdout=None)  # nosec
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)  # nosec
    finally:
        for collector in telemetry_data_collectors:
            if collector:
                collector.stop()


def merge_telemetry_metrics(metrics_list: List[TelemetryMetrics]) -> TelemetryMetrics:
    """
    Merges multiple TelemetryMetrics objects into a single one.

    Args:
        metrics_list (List[TelemetryMetrics]): A list of TelemetryMetrics instances.

    Returns:
        TelemetryMetrics: A new TelemetryMetrics instance with merged raw data.
    """

    merged_metrics = TelemetryMetrics()

    for metrics in metrics_list:
        for metric in TelemetryMetrics.TELEMETRY_METRICS:
            metric_key = metric.name
            metric_dict = getattr(merged_metrics, metric_key)
            source_dict = getattr(metrics, metric_key)

            for gpu_id, values in source_dict.items():
                metric_dict[gpu_id].extend(values)
    return merged_metrics
