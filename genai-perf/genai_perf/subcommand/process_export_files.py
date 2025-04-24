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
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, DefaultDict, Dict, List, Optional

from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.metrics import (
    LLMMetrics,
    Statistics,
    TelemetryMetrics,
    TelemetryStatistics,
)
from genai_perf.subcommand.subcommand import Subcommand
from genai_perf.types import ModelObjectiveParameters


def process_export_files_handler(
    config: ConfigCommand, extra_args: Optional[List[str]] = None
) -> None:
    """
    Handles `process-export-files` subcommand workflow
    """
    process = ProcessExportFiles(config, extra_args)
    process.process_export_files()


class ProcessExportFiles(Subcommand):
    """
    Contains all the methods needed to run the process-export-files subcommand
    """

    def __init__(self, config: ConfigCommand, extra_args: Optional[List[str]]) -> None:
        super().__init__(config, extra_args)

    def process_export_files(self) -> None:

        input_profile_data_list = self._parse_json_profile_data_files(
            self._config.process.input_path
        )

        # TODO: TPA-1086 - Handle when user provides mismatched profile files
        self._input_profile_data = input_profile_data_list[0]

        self._update_config_from_profile_data()

        objectives = self._create_objectives_based_on_stimulus()
        perf_analyzer_config = self._create_perf_analyzer_config(objectives)

        self._create_tokenizer()
        self._create_artifact_directory(perf_analyzer_config)

        self._copy_profile_export_to_artifact_dir(perf_analyzer_config)

        self._set_data_parser(perf_analyzer_config)
        self._add_output_to_artifact_directory(perf_analyzer_config, objectives)

    def _get_all_stats(self, input_profile_data_list: List[Dict[str, Any]]) -> None:
        """
        Return aggregated stats from all input files
        """
        self._perf_stats = self._get_aggregated_perf_stats(input_profile_data_list)
        self._telemetry_stats = self._get_aggregated_telemetry_stats(
            input_profile_data_list
        )

    def _get_aggregated_perf_stats(
        self, input_profile_data_list: List[Dict[str, Any]]
    ) -> Statistics:
        """
        Aggregates the perf_stats from all input files and returns a Statistics object.
        """
        sum_metrics = {
            "request_throughput",
            "output_token_throughput",
            "request_count",
        }
        perf_stats_dicts = []

        if not input_profile_data_list:
            raise GenAIPerfException("No profile data found to aggregate.")

        for profile_data in input_profile_data_list:
            perf_stats_dict = {
                key: value
                for key, value in profile_data.items()
                if key not in {"input_config", "telemetry_stats", "sessions"}
            }

            perf_stats_dicts.append(perf_stats_dict)

        aggregated_perf_stats_dict: Dict[str, Any] = {}
        for metric_name in perf_stats_dicts[0]:
            values = [
                stat_dict[metric_name]["avg"]
                for stat_dict in perf_stats_dicts
                if metric_name in stat_dict and "avg" in stat_dict[metric_name]
            ]
            unit = perf_stats_dicts[0][metric_name].get("unit", "")

            aggregated_value = (
                sum(values) if metric_name in sum_metrics else mean(values)
            )
            aggregated_perf_stats_dict[metric_name] = {
                "unit": unit,
                "avg": float(aggregated_value),
            }

        perf_stats = Statistics(LLMMetrics([]))
        perf_stats.set_stats_dict(aggregated_perf_stats_dict)

        return perf_stats

    def _get_aggregated_telemetry_stats(
        self, input_profile_data_list: List[Dict[str, Any]]
    ) -> TelemetryStatistics:
        """
        Aggregates the telemetry_stats from all input files and returns a TelemetryStatistics object.
        """

        telemetry_dicts = [
            stats_dict.get("telemetry_stats", {})
            for stats_dict in input_profile_data_list
        ]
        telemetry_stats = TelemetryStatistics(TelemetryMetrics(None))
        if not telemetry_dicts:
            return telemetry_stats

        aggregated_telemetry_stats_dict: DefaultDict[str, Any] = defaultdict(dict)
        for metric_name in telemetry_dicts[0]:
            aggregated_telemetry_stats_dict[metric_name] = {}
            unit = telemetry_dicts[0][metric_name].get("unit", "")
            aggregated_telemetry_stats_dict[metric_name]["unit"] = unit

            gpu_ids = set()
            for telemetry_dict in telemetry_dicts:
                gpu_ids.update(telemetry_dict.get(metric_name, {}).keys())
            gpu_ids.discard("unit")

            for gpu_id in gpu_ids:
                values = [
                    telemetry_dict[metric_name][gpu_id]["avg"]
                    for telemetry_dict in telemetry_dicts
                    if metric_name in telemetry_dict
                    and gpu_id in telemetry_dict[metric_name]
                    and "avg" in telemetry_dict[metric_name][gpu_id]
                ]
                if values:
                    aggregated_telemetry_stats_dict[metric_name][gpu_id] = {
                        "avg": mean(values)
                    }

        telemetry_stats.set_stats_dict(aggregated_telemetry_stats_dict)

        return telemetry_stats

    def _parse_json_profile_data_files(
        self, input_directory: Path
    ) -> List[Dict[str, Any]]:
        """
        Loads all *_genai_perf.json files from the provided input directory.
        """
        json_files = list(input_directory.glob("**/*_genai_perf.json"))
        if not json_files:
            raise GenAIPerfException(
                f"No *_genai_perf.json files found in '{input_directory}'"
            )

        input_profile_data_list = []
        for file_path in json_files:
            try:
                with open(file_path, "r") as f:
                    profile_data = json.load(f)
                    input_profile_data_list.append(profile_data)
            except Exception as e:
                raise GenAIPerfException(f"Failed to read '{file_path}': {e}")

        return input_profile_data_list

    def _update_config_from_profile_data(self) -> None:
        """
        Handles uninitialized config fields by loading them from profile_data.
        """
        self._set_model_names()
        self._set_input_fields()
        self._set_perf_analyzer_fields()
        self._set_endpoint_fields()
        self._set_tokenizer_fields()

    def _set_model_names(self) -> None:
        self._config.model_names = self._input_profile_data["input_config"][
            "model_names"
        ]

    def _set_input_fields(self) -> None:
        keys_to_exclude = ["prompt_source", "synthetic_files", "payload_file"]
        input_data = {
            k: v
            for k, v in self._input_profile_data["input_config"]["input"].items()
            if k not in keys_to_exclude
        }

        self._config.input.parse(input_data)
        self._config.input.infer_settings()

    def _set_perf_analyzer_fields(self) -> None:
        self._config.perf_analyzer.parse(
            self._input_profile_data["input_config"]["perf_analyzer"]
        )

    def _set_endpoint_fields(self) -> None:
        keys_to_exclude = ["service_kind", "output_format"]
        endpoint_data = {
            k: v
            for k, v in self._input_profile_data["input_config"]["endpoint"].items()
            if k not in keys_to_exclude
        }
        self._config.endpoint.parse(endpoint_data)
        self._config.endpoint.infer_settings(self._config.model_names[0])

    def _set_tokenizer_fields(self) -> None:
        keys_to_exclude = ["_enable_debug_logging"]
        tokenizer_data = {
            k: v
            for k, v in self._input_profile_data["input_config"]["tokenizer"].items()
            if k not in keys_to_exclude
        }
        self._config.tokenizer.parse(tokenizer_data)
        self._config.tokenizer.infer_settings(self._config.model_names[0])

    def _copy_profile_export_to_artifact_dir(
        self, perf_analyzer_config: PerfAnalyzerConfig
    ) -> None:
        source_file = Path(self._config.process.input_path) / "profile_export.json"
        destination_file = (
            perf_analyzer_config.get_artifact_directory() / "profile_export.json"
        )
        shutil.copy(source_file, destination_file)

    def _add_output_to_artifact_directory(
        self,
        perf_analyzer_config: PerfAnalyzerConfig,
        objectives: ModelObjectiveParameters,
    ) -> None:

        session_stats = self._get_session_stats()

        OutputReporter(
            self._perf_stats,
            self._telemetry_stats,
            self._config,
            perf_analyzer_config,
            session_stats,
        ).report_output()
