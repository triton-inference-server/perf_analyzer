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

    TELEMETRY_AGGREGATED_METRICS_HEADER = [
        "Metric",
        "GPU",
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

    TELEMETRY_CONSTANT_METRICS_HEADER = [
        "Metric",
        "GPU",
        "Value",
    ]

    def __init__(self, config: ExporterConfig):
        self._stats = config.stats
        self._metrics = config.metrics
        self._output_dir = config.artifact_dir
        self._args = config.args
        self._is_telemetry_data = config.is_telemetry_data

    def export(self) -> None:
        if self._is_telemetry_data:
            self._export_telemetry_metrics()
        else:
            self._export_llm_metrics()

    def _export_llm_metrics(self) -> None:
        filename = (
            self._output_dir
            / f"{self._args.profile_export_file.stem}_llm_genai_perf.csv"
        )
        logger.info(f"Generating {filename}")

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            self._write_request_metrics(writer)
            writer.writerow([])
            self._write_system_metrics(writer)

    def _export_telemetry_metrics(self) -> None:
        telemetry_filename = (
            self._output_dir / f"{self._args.profile_export_file.stem}_telemetry_genai_perf.csv"
        )
        logger.info(f"Generating {telemetry_filename}")

        with open(telemetry_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            self._write_telemetry_metrics(writer)
            writer.writerow([])
            self._write_constant_metrics(writer)

    def _write_request_metrics(self, csv_writer) -> None:
        csv_writer.writerow(self.REQUEST_METRICS_HEADER)
        for metric in self._metrics.request_metrics:
            if self._should_skip(metric.name):
                continue

            metric_str = metric.name.replace("_", " ").title()
            metric_str += f" ({metric.unit})" if metric.unit != "tokens" else ""
            row_values = [metric_str]
            for stat in self.REQUEST_METRICS_HEADER[1:]:
                value = self._stats[metric.name][stat]
                row_values.append(f"{value:,.2f}")

            csv_writer.writerow(row_values)

    def _write_system_metrics(self, csv_writer) -> None:
        csv_writer.writerow(self.SYSTEM_METRICS_HEADER)
        for metric in self._metrics.system_metrics:
            metric_str = metric.name.replace("_", " ").title()
            metric_str += f" ({metric.unit})"
            if metric.name == "request_goodput":
                if not self._args.goodput:
                    continue
            value = self._stats[metric.name]["avg"]
            csv_writer.writerow([metric_str, f"{value:.2f}"])

    def _write_telemetry_metrics(self, csv_writer) -> None:
        csv_writer.writerow(self.TELEMETRY_AGGREGATED_METRICS_HEADER)

        for metric_name, metric_data in self._stats.items():
            # Skip constant metrics
            if metric_name in ["gpu_power_limit", "total_gpu_memory"]:
                continue

            metric_str = metric_name.replace("_", " ").title()
            metric_str += f" ({metric_data['unit']})"

            # Iterate through each GPU
            for gpu in metric_data["avg"].keys():
                row_values = [metric_str, gpu]

                for stat in self.TELEMETRY_AGGREGATED_METRICS_HEADER[2:]:
                    value = metric_data[stat][gpu]
                    row_values.append(f"{value:,.2f}")

                csv_writer.writerow(row_values)

    def _write_constant_metrics(self, csv_writer) -> None:
        csv_writer.writerow(self.TELEMETRY_CONSTANT_METRICS_HEADER)

        for metric_name, metric_data in self._stats.items():
            # Skip non-constant metrics
            if metric_name not in ["gpu_power_limit", "total_gpu_memory"]:
                continue

            metric_str = metric_name.replace("_", " ").title()
            metric_str += f" ({metric_data['unit']})"
            for gpu in metric_data.keys():
                if gpu == "unit":
                    continue
                value = metric_data[gpu]
                csv_writer.writerow([metric_str, gpu, f"{value:.2f}"])

    def _format_value(self, value) -> str:
        if isinstance(value, (int, float)):
            return f"{value:,.2f}"  # Format with commas and two decimal places
        return value

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
        ]
        if not self._args.streaming and metric_name in streaming_metrics:
            return True
        return False
