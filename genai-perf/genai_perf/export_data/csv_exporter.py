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


import csv

import genai_perf.logging as logging

from . import exporter_utils
from . import telemetry_data_exporter_util as telem_utils
from .exporter_config import ExporterConfig

logger = logging.getLogger(__name__)


class CsvExporter:
    """
    A class to export the statistics and arg values in a csv format.
    """

    REQUEST_METRICS_HEADER = [
        "Metric",
        "avg",
        "min",
        "max",
        "p99",
        "p95",
        "p90",
        "p75",
        "p50",
        "p25",
        "p10",
        "p5",
        "p1",
    ]

    SYSTEM_METRICS_HEADER = [
        "Metric",
        "Value",
    ]

    def __init__(self, config: ExporterConfig):
        self._stats = config.stats
        self._telemetry_stats = config.telemetry_stats
        self._metrics = config.metrics
        self._output_dir = config.perf_analyzer_config.get_artifact_directory()
        self._profile_export_file = config.config.output.profile_export_file
        self._config = config.config

    def export(self) -> None:
        filename = self._output_dir / f"{self._profile_export_file.stem}_genai_perf.csv"
        logger.info(f"Generating {filename}")

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            self._write_request_metrics(writer)
            writer.writerow([])
            self._write_system_metrics(writer)
            telem_utils.export_telemetry_stats_csv(self._telemetry_stats, writer)

    def _write_request_metrics(self, csv_writer) -> None:
        csv_writer.writerow(self.REQUEST_METRICS_HEADER)
        for metric in self._metrics.request_metrics:
            if self._should_skip(metric.name):
                continue

            metric_str = exporter_utils.format_metric_name(metric.name, metric.unit)
            row_values = [metric_str]
            for stat in self.REQUEST_METRICS_HEADER[1:]:
                row_values.append(
                    exporter_utils.fetch_stat(self._stats, metric.name, stat)
                )

            csv_writer.writerow(row_values)

    def _write_system_metrics(self, csv_writer) -> None:
        csv_writer.writerow(self.SYSTEM_METRICS_HEADER)
        for metric in self._metrics.system_metrics:
            metric_str = exporter_utils.format_metric_name(metric.name, metric.unit)
            if metric.name == "request_goodput" and not self._config.input.goodput:
                continue
            value = exporter_utils.fetch_stat(self._stats, metric.name, "avg")
            csv_writer.writerow([metric_str, exporter_utils.format_stat_value(value)])

    def _should_skip(self, metric_name: str) -> bool:
        if self._config.endpoint.type == "embeddings":
            return False  # skip nothing

        # Skip following streaming metrics when non-streaming mode
        streaming_metrics = [
            "inter_token_latency",
            "time_to_first_token",
            "time_to_second_token",
            "output_token_throughput_per_user",
        ]
        if not self._config.endpoint.streaming and metric_name in streaming_metrics:
            return True
        return False
