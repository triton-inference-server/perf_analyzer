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
from typing import Any, Dict, List, Optional

from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.plots.plot_config_parser import PlotConfigParser
from genai_perf.plots.plot_manager import PlotManager
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
    if config.output.generate_plots:
        process.create_plots()


class ProcessExportFiles(Subcommand):
    """
    Contains all the methods needed to run the process-export-files subcommand
    """

    def __init__(self, config: ConfigCommand, extra_args: Optional[List[str]]) -> None:
        super().__init__(config, extra_args)

    def process_export_files(self) -> None:

        self._input_profile_data = self._get_profile_export_json_data(
            self._config.process.input_path
        )
        self._update_config_from_profile_data()

        objectives = self._create_objectives_based_on_stimulus()
        perf_analyzer_config = self._create_perf_analyzer_config(objectives)

        self._create_tokenizer()
        self._create_artifact_directory(perf_analyzer_config)
        self._create_plot_directory(perf_analyzer_config)

        self._copy_profile_export_to_artifact_dir(perf_analyzer_config)

        self._set_data_parser(perf_analyzer_config)
        self._add_output_to_artifact_directory(perf_analyzer_config, objectives)

    def _get_profile_export_json_data(self, input_directory: Path) -> Dict[str, Any]:
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
            raise GenAIPerfException(
                f"Invalid JSON format in {profile_export_json_file}."
            )

        return profile_data

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
        self._config.tokenizer.parse(
            self._input_profile_data["input_config"]["tokenizer"]
        )

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
        perf_stats = self._create_perf_stats(perf_analyzer_config, objectives)
        telemetry_stats = self._create_merged_telemetry_stats()
        session_stats = self._create_session_stats(perf_analyzer_config, objectives)

        telemetry_stats_dict = self._input_profile_data["telemetry_stats"]
        telemetry_stats.set_stats_dict(telemetry_stats_dict)

        OutputReporter(
            perf_stats,
            telemetry_stats,
            self._config,
            perf_analyzer_config,
            session_stats,
        ).report_output()
