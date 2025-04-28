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

from typing import Dict

from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.export_data.data_exporter_factory import DataExporterFactory
from genai_perf.export_data.exporter_config import ExporterConfig
from genai_perf.inputs import input_constants as ic
from genai_perf.metrics import Metrics, Statistics, TelemetryStatistics


class OutputReporter:
    """
    A class to orchestrate output generation.
    """

    def __init__(
        self,
        stats: Statistics,
        telemetry_stats: TelemetryStatistics,
        config: ConfigCommand,
        perf_analyzer_config: PerfAnalyzerConfig,
        session_stats: Dict[str, Statistics],
    ):
        self.config = config
        self.perf_analyzer_config = perf_analyzer_config
        self.stats = stats
        self.telemetry_stats = telemetry_stats
        self.session_stats = session_stats

        # scale the data to be in milliseconds
        self.stats.scale_data()
        # For the process-export-files subcommand, telemetry_stats
        # are loaded from a previously generated profile_export_genai_perf.json file.
        # As the data is already preprocessed, scaling is not required.
        if config.subcommand != ic.Subcommand.PROCESS:
            self.telemetry_stats.scale_data()
        for stat in self.session_stats.values():
            stat.scale_data()

    def report_output(self) -> None:
        factory = DataExporterFactory()
        exporter_config = self._create_exporter_config()
        data_exporters = factory.create_data_exporters(exporter_config)

        for exporter in data_exporters:
            exporter.export()

    def _create_exporter_config(self) -> ExporterConfig:
        assert isinstance(self.stats.metrics, Metrics)
        telemetry_stats = self.telemetry_stats.stats_dict
        session_stats = {k: v.stats_dict for k, v in self.session_stats.items()}
        config = ExporterConfig(
            stats=self.stats.stats_dict,
            metrics=self.stats.metrics,
            config=self.config,
            perf_analyzer_config=self.perf_analyzer_config,
            extra_inputs=self.config.input.extra,
            telemetry_stats=telemetry_stats,
            session_stats=session_stats,
        )

        return config
