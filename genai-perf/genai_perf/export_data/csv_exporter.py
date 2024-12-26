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


import csv

import genai_perf.logging as logging
from genai_perf.export_data import telemetry_data_exporter_util as telem_utils
from genai_perf.export_data.exporter_config import ExporterConfig

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
    ]

    SYSTEM_METRICS_HEADER = [
        "Metric",
        "Value",
    ]

    def __init__(self, config: ExporterConfig):
        self._stats = config.stats
        self._telemetry_stats = config.telemetry_stats
        self._metrics = config.metrics
        self._output_dir = config.artifact_dir
        self._args = config.args

    def export(self) -> None:
        filename = (
            self._output_dir / f"{self._args.profile_export_file.stem}_genai_perf.csv"
        )
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

            metric_str = self.format_metric_name(metric.name, metric.unit)
            row_values = [metric_str]
            for stat in self.REQUEST_METRICS_HEADER[1:]:
                row_values.append(self.fetch_stat(metric.name, stat))

            csv_writer.writerow(row_values)

    def _write_system_metrics(self, csv_writer) -> None:
        csv_writer.writerow(self.SYSTEM_METRICS_HEADER)
        for metric in self._metrics.system_metrics:
            metric_str = self.format_metric_name(metric.name, metric.unit)
            if metric.name == "request_goodput" and not self._args.goodput:
                continue
            value = self.fetch_stat(metric.name, "avg")
            row = [metric_str, self.format_stat_value(value)]
            print(row)
            csv_writer.writerow([metric_str, self.format_stat_value(value)])

    def _should_skip(self, metric_name: str) -> bool:
        if self._args.endpoint_type == "embeddings":
            return False  # skip nothing

        # TODO (TMA-1712): need to decide if we need this metric. Remove
        # from statistics display for now.
        # TODO (TMA-1678): output_token_throughput_per_request is treated
        # separately since the current code treats all throughput metrics to
        # be displayed outside of the statistics table.
        if metric_name == "output_token_throughput_per_request":
            return True

        # When non-streaming, skip ITL and TTFT
        streaming_metrics = [
            "inter_token_latency",
            "time_to_first_token",
            "time_to_second_token",
        ]
        if not self._args.streaming and metric_name in streaming_metrics:
            return True
        return False

    def format_metric_name(self, name, unit):
        """Helper to format metric name with its unit."""
        metric_str = name.replace("_", " ").title()
        return f"{metric_str} ({unit})" if unit else metric_str

    def format_stat_value(self, value):
        """Helper to format a statistic value for printing."""
        return f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)

    def fetch_stat(self, metric_name: str, stat: str):
        """
        Fetches a statistic value for a metric.
        Logs errors and returns 'N/A' if the value is missing
        """
        if metric_name not in self._stats:
            logger.error(
                f"Metric '{metric_name}' is missing in the provided statistics."
            )
            return "N/A"

        metric_stats = self._stats[metric_name]
        if not isinstance(metric_stats, dict):
            logger.error(
                f"Expected statistics for metric '{metric_name}' to be a dictionary. "
                f"Got: {type(metric_stats).__name__}."
            )
            return "N/A"

        if stat not in metric_stats:
            logger.error(
                f"Statistic '{stat}' for metric '{metric_name}' is missing. "
                f"Available stats: {list(metric_stats.keys())}."
            )
            return "N/A"

        return self.format_stat_value(metric_stats[stat])
