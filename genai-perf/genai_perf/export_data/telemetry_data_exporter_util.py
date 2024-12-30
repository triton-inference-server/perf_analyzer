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

from typing import Dict, List, Optional

from genai_perf.constants import ABBREVIATIONS
from rich.console import Console
from rich.table import Table

TELEMETRY_DYNAMIC_METRICS_HEADER = [
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

TELEMETRY_CONSTANT_METRICS = ["gpu_power_limit", "total_gpu_memory"]

TELEMETRY_GROUPS = {
    "Power": ["gpu_power_usage", "gpu_power_limit", "energy_consumption"],
    "Memory": ["gpu_memory_used", "total_gpu_memory"],
    "Utilization": ["gpu_utilization"],
}


def merge_telemetry_stats_json(
    telemetry_stats: Optional[Dict], stats_and_args: Dict
) -> None:
    if telemetry_stats is not None:
        stats_and_args.update({"telemetry_stats": telemetry_stats})


def export_telemetry_stats_csv(telemetry_stats: Optional[Dict], csv_writer) -> None:
    if telemetry_stats:
        _write_dynamic_telemetry_stats(telemetry_stats, csv_writer)
        _write_constant_telemetry_stats(telemetry_stats, csv_writer)


def export_telemetry_stats_console(
    telemetry_stats: Optional[Dict], stat_column_keys: List[str], console: Console
) -> None:
    if telemetry_stats:
        _construct_telemetry_stats_table(telemetry_stats, stat_column_keys, console)


def _construct_telemetry_stats_table(
    telemetry_stats, stat_column_keys: List[str], console: Console
) -> None:
    for group_name, metrics in TELEMETRY_GROUPS.items():
        table = Table(title=f"NVIDIA GenAI-Perf | {group_name} Metrics")

        for metric_name in metrics:
            metric_data = telemetry_stats.get(metric_name, {})

            unit = metric_data.get("unit", "N/A")
            metric_name_display = _format_metric_name(metric_name.replace("_", " "))
            table_title = f"{metric_name_display}{f' ({unit})' if unit else ''}"
            sub_table = Table(title=table_title)

            sub_table.add_column(
                "GPU Index", justify="right", style="cyan", no_wrap=True
            )
            for stat in stat_column_keys:
                sub_table.add_column(stat, justify="right", style="green")

            _construct_telemetry_stats_subtable(
                sub_table, metric_data, metric_name, stat_column_keys
            )
            table.add_row(sub_table)

        console.print(table)


def _construct_telemetry_stats_subtable(
    table: Table,
    metric_data: Dict[str, Dict[str, float]],
    metric_name: str,
    stat_column_keys: List[str],
) -> None:
    gpu_indices = sorted(key for key in metric_data if key != "unit")

    for gpu_index in gpu_indices:
        row = [f"{gpu_index}"]
        for stat in stat_column_keys:
            value = metric_data.get(gpu_index, {}).get(stat, "N/A")
            if isinstance(value, (float)):
                if metric_name == "gpu_utilization":
                    value = str(int(round(value)))
                else:
                    value = f"{value:,.2f}"
            row.append(value)
        table.add_row(*row)


def _write_dynamic_telemetry_stats(telemetry_stats: Dict, csv_writer) -> None:
    csv_writer.writerow([])
    csv_writer.writerow(TELEMETRY_DYNAMIC_METRICS_HEADER)

    for metric_name, metric_data in telemetry_stats.items():
        if metric_name in TELEMETRY_CONSTANT_METRICS:
            continue
        metric_str = _format_metric_name(metric_name.replace("_", " "))
        metric_str += f" ({metric_data['unit']})"

        for key, gpu_data in metric_data.items():
            if key == "unit":
                continue

            row_values = [metric_str, key]

            for stat in TELEMETRY_DYNAMIC_METRICS_HEADER[2:]:
                value = gpu_data.get(stat, 0.0)
                row_values.append(f"{value:,.2f}")

            csv_writer.writerow(row_values)


def _write_constant_telemetry_stats(telemetry_stats: Dict, csv_writer) -> None:
    csv_writer.writerow([])
    csv_writer.writerow(TELEMETRY_CONSTANT_METRICS_HEADER)

    for metric_name, metric_data in telemetry_stats.items():
        if metric_name not in TELEMETRY_CONSTANT_METRICS:
            continue

        metric_str = _format_metric_name(metric_name.replace("_", " "))
        metric_str += f" ({metric_data['unit']})"
        for gpu in metric_data.keys():
            if gpu == "unit":
                continue
            value = metric_data[gpu]["avg"]
            csv_writer.writerow([metric_str, gpu, f"{value:.2f}"])


def _format_metric_name(metric_name: str) -> str:

    words = metric_name.split()
    capitalized_words = [
        word.upper() if word.lower() in ABBREVIATIONS else word.capitalize()
        for word in words
    ]
    return " ".join(capitalized_words)
