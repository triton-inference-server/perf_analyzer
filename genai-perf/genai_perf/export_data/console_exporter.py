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

from rich.console import Console
from rich.table import Table

from . import exporter_utils
from . import telemetry_data_exporter_util as telem_utils
from .exporter_config import ExporterConfig


class ConsoleExporter:
    """
    A class to export the statistics and arg values to the console.
    """

    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p75"]

    def __init__(self, config: ExporterConfig):
        self._stats = config.stats
        self._telemetry_stats = config.telemetry_stats
        self._metrics = config.metrics
        self._config = config.config

        # Set the maximum width of the 'Statistic' column.
        # Any metric name+unit longer than this width will be wrapped.
        self._max_width = 36

    def _get_title(self):
        title = "NVIDIA GenAI-Perf | "
        if self._config.endpoint.type == "embeddings":
            title += "Embeddings Metrics"
        elif self._config.endpoint.type == "rankings":
            title += "Rankings Metrics"
        elif self._config.endpoint.type == "image_retrieval":
            title += "Image Retrieval Metrics"
        elif self._config.endpoint.type == "multimodal":
            title += "Multi-Modal Metrics"
        else:
            title += "LLM Metrics"
        return title

    def export(self, **kwargs) -> None:
        table = Table(title=self._get_title())

        table.add_column("Statistic", justify="right", style="cyan")
        for stat in self.STAT_COLUMN_KEYS:
            table.add_column(stat, justify="right", style="green")

        # Request metrics table
        self._construct_table(table)

        console = Console(**kwargs)
        console.print(table)
        if self._config.verbose:
            telem_utils.export_telemetry_stats_console(
                self._telemetry_stats, self.STAT_COLUMN_KEYS, console
            )

    def _construct_table(self, table: Table) -> None:

        for metric in self._metrics.request_metrics:
            if self._should_skip(metric.name):
                continue

            metric_str = exporter_utils.format_metric_name(
                metric.name, metric.unit, self._max_width
            )
            row_values = [metric_str]

            for stat in self.STAT_COLUMN_KEYS:
                row_values.append(
                    exporter_utils.fetch_stat(self._stats, metric.name, stat)
                )

            table.add_row(*row_values)

        for metric in self._metrics.system_metrics:
            metric_str = exporter_utils.format_metric_name(
                metric.name, metric.unit, self._max_width
            )
            if metric.name == "request_goodput" and not self._config.input.goodput:
                continue

            row_values = [metric_str]
            for stat in self.STAT_COLUMN_KEYS:
                if stat == "avg":
                    row_values.append(
                        exporter_utils.fetch_stat(self._stats, metric.name, "avg")
                    )
                else:
                    row_values.append("N/A")

            table.add_row(*row_values)

    # (TMA-1976) Refactor this method as the csv exporter shares identical method.
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
