# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from typing import Any, Dict, List, Optional

from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.metrics.telemetry_statistics import TelemetryStatistics
from genai_perf.metrics.telemetry_stats_aggregator import TelemetryStatsAggregator
from genai_perf.profile_data_parser.merged_profile_parser import MergedProfileParser
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
    This class encapsulates the logic for handling the `process-export-files` subcommand.

    The class includes methods for parsing input directories, handling
    profile files, updating configurations, and generating output artifacts.
    """

    def __init__(self, config: ConfigCommand, extra_args: Optional[List[str]]) -> None:
        super().__init__(config, extra_args)
        self._pa_profile_data: Dict[str, Any] = {
            "experiments": [],
            "version": "",
            "service_kind": "",
            "endpoint": "",
        }
        self._throughput_metrics_dict: Dict[str, List[float]] = {
            "request_throughput": [],
            "output_token_throughput": [],
        }
        self._telemetry_dicts: List[Dict[str, Any]] = []
        self._input_config: Dict[str, Any] = {}
        self._next_start_timestamp = 0

    def process_export_files(self) -> None:
        self._parse_input_directory(self._config.process.input_path)

        # TPA-1086: Handle when user provides mismatched profile files
        self._update_config_from_profile_data()

        objectives = self._create_objectives_based_on_stimulus()
        perf_analyzer_config = self._create_perf_analyzer_config(objectives)

        self._create_tokenizer()
        self._create_artifact_directory(perf_analyzer_config)

        self._create_merged_profile_export_file(perf_analyzer_config)
        self._set_data_parser(perf_analyzer_config)
        self._set_telemetry_aggregator()

        self._add_output_to_artifact_directory(perf_analyzer_config, objectives)

    def _parse_input_directory(self, input_directory: Path) -> None:
        """
        Parse the input directory to find all valid subdirectories containing profile export files.
        Calls respective functions for parsing PA and GAP profile export files.
        """
        for subdir in input_directory.iterdir():
            if not subdir.is_dir():
                continue

            json_files = list(subdir.glob("*.json"))
            pa_profile_file = next(
                (f for f in json_files if "_genai_perf" not in f.name), None
            )
            gap_profile_file = next(
                (f for f in json_files if "_genai_perf.json" in f.name), None
            )

            if pa_profile_file and gap_profile_file:
                self._process_pa_profile_file(pa_profile_file)
                self._process_gap_profile_file(gap_profile_file)

    def _process_pa_profile_file(self, pa_profile_file: Path) -> None:
        """
        Process a PA profile export file and update the pa_profile_data dictionary.
        """
        try:
            with open(pa_profile_file, "r") as f:

                pa_profile_data = json.load(f)
                if "experiments" in pa_profile_data and pa_profile_data["experiments"]:
                    experiment = pa_profile_data["experiments"][0]
                else:
                    raise GenAIPerfException(
                        f"Invalid profile data in file '{pa_profile_file}': 'experiments' key is missing or empty"
                    )

                if not self._pa_profile_data["experiments"]:
                    self._pa_profile_data.update(
                        {
                            "version": pa_profile_data.get("version", ""),
                            "service_kind": pa_profile_data.get("service_kind", ""),
                            "endpoint": pa_profile_data.get("endpoint", ""),
                            "experiments": [experiment],
                        }
                    )
                else:
                    self._pa_profile_data["experiments"][0]["requests"].extend(
                        experiment["requests"]
                    )

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise GenAIPerfException(f"Error reading file '{pa_profile_file}': {e}")

        except Exception as e:
            raise GenAIPerfException(f"Unexpected error: {e}")

    def _process_gap_profile_file(self, gap_profile_file: Path) -> None:
        """
        Process a GAP JSON profile export file and updates telemetry and input config data.
        """
        try:
            with open(gap_profile_file, "r") as f:
                gap_profile_data = json.load(f)

                for metric in self._throughput_metrics_dict:
                    if metric in gap_profile_data:
                        self._throughput_metrics_dict[metric].append(
                            gap_profile_data[metric]["avg"]
                        )
                telemetry_stats = gap_profile_data.get("telemetry_stats", {})
                if telemetry_stats:
                    self._telemetry_dicts.append(telemetry_stats)
                if not self._input_config:
                    self._input_config = gap_profile_data.get("input_config", {})

        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise GenAIPerfException(f"Error reading file '{gap_profile_file}': {e}")

        except Exception as e:
            raise GenAIPerfException(f"Unexpected error: {e}")

    def _update_config_from_profile_data(self) -> None:
        """
        Handles uninitialized config fields by loading them from profile data.
        """
        self._set_model_names()
        self._set_input_fields()
        self._set_perf_analyzer_fields()
        self._set_endpoint_fields()
        self._set_tokenizer_fields()

    def _set_model_names(self) -> None:
        self._config.model_names = self._input_config.get("model_names", [])

    def _set_input_fields(self) -> None:
        keys_to_exclude = ["prompt_source", "synthetic_files", "payload_file"]
        input_data = {
            k: v
            for k, v in self._input_config["input"].items()
            if k not in keys_to_exclude
        }

        self._config.input.parse(input_data)
        self._config.input.infer_settings()

    def _set_perf_analyzer_fields(self) -> None:
        self._config.perf_analyzer.parse(self._input_config["perf_analyzer"])

    def _set_endpoint_fields(self) -> None:
        keys_to_exclude = ["service_kind", "output_format"]
        endpoint_data = {
            k: v
            for k, v in self._input_config["endpoint"].items()
            if k not in keys_to_exclude
        }
        self._config.endpoint.parse(endpoint_data)
        self._config.endpoint.infer_settings(self._config.model_names[0])

    def _set_tokenizer_fields(self) -> None:
        keys_to_exclude = ["_enable_debug_logging"]
        tokenizer_data = {
            k: v
            for k, v in self._input_config["tokenizer"].items()
            if k not in keys_to_exclude
        }
        self._config.tokenizer.parse(tokenizer_data)
        self._config.tokenizer.infer_settings(self._config.model_names[0])

    def _create_merged_profile_export_file(
        self, perf_analyzer_config: PerfAnalyzerConfig
    ) -> None:
        """
        Creates a merged PA profile export file in artifact directory.
        """
        profile_export_file = perf_analyzer_config.get_profile_export_file()
        try:
            with open(profile_export_file, "w") as f:
                json.dump(self._pa_profile_data, f)
        except (FileNotFoundError, PermissionError, IOError) as e:
            raise GenAIPerfException(f"Error writing file '{profile_export_file}': {e}")

    def _calculate_metrics(
        self, perf_analyzer_config: PerfAnalyzerConfig
    ) -> MergedProfileParser:
        return MergedProfileParser(
            filename=perf_analyzer_config.get_profile_export_file(),
            tokenizer=self._tokenizer,  # type: ignore
            throughput_metrics_dict=self._throughput_metrics_dict,
            goodput_constraints=self._config.input.goodput,
        )

    def _set_telemetry_aggregator(self) -> None:
        self._telemetry_aggregator = TelemetryStatsAggregator(self._telemetry_dicts)

    def _create_telemetry_stats(self) -> TelemetryStatistics:
        return self._telemetry_aggregator.get_telemetry_stats()

    def _add_output_to_artifact_directory(
        self,
        perf_analyzer_config: PerfAnalyzerConfig,
        objectives: ModelObjectiveParameters,
    ) -> None:
        perf_stats = self._create_perf_stats(perf_analyzer_config, objectives)
        telemetry_stats = self._create_telemetry_stats()
        session_stats = self._create_session_stats(perf_analyzer_config, objectives)
        OutputReporter(
            perf_stats,
            telemetry_stats,
            self._config,
            perf_analyzer_config,
            session_stats,
        ).report_output()
