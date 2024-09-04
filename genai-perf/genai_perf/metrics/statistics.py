#!/usr/bin/env python3

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

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from genai_perf.metrics.metrics import Metrics
from genai_perf.metrics.telemetry_metrics import TelemetryMetrics


class Statistics:
    """A class that aggregates various statistics from given metrics class.

    The Statistics class goes through each metric in the metrics class and
    calculates several statistics such as:
      - average (arithmetic mean)
      - percentiles (p25, p50, p75, p90, p95, p99)
      - minimum & maximum
      - standard deviation
    The class will store each calculated statistics as part of its attribute.

    Example:

      >>> metrics = LLMMetrics(request_throughputs=[2, 4])
      >>> stats = Statistics(metrics)
      >>> print(stats.avg_request_throughput)  # output: 3
    """

    def __init__(self, metrics: Union[Metrics, TelemetryMetrics]):
        # iterate through Metrics to calculate statistics and set attributes
        self._metrics = metrics
        self._stats_dict: Dict = defaultdict(dict)
        for attr, data in metrics.data.items():
            if self._should_skip(data, attr):
                continue

            if hasattr(metrics, "get_base_name"):
                attr = metrics.get_base_name(attr)
                self._add_units(attr)
                self._stats_dict[attr]["avg"] = self._calculate_mean(data)
                if not self._is_system_metric(attr):
                    percentile_results = self._calculate_percentiles(data)
                    for (
                        percentile_label,
                        percentile_value,
                    ) in percentile_results.items():
                        self._stats_dict[attr][percentile_label] = percentile_value
                    min, max = self._calculate_minmax(data)
                    self._stats_dict[attr]["min"] = min
                    self._stats_dict[attr]["max"] = max
                    self._stats_dict[attr]["std"] = self._calculate_std(data)

    def _should_skip(self, data: List[Union[int, float]], attr: str) -> bool:
        """Checks if some metrics should be skipped."""
        # No data points
        if len(data) == 0:
            return True
        # Skip ITL when non-streaming (all zero)
        elif attr == "inter_token_latencies" and sum(data) == 0:
            return True
        return False

    def _calculate_mean(self, data: List[Union[int, float]]) -> float:
        avg = np.mean(data)
        return float(avg)

    def _calculate_percentiles(self, data: List[Union[int, float]]) -> Dict[str, float]:
        percentiles = [25, 50, 75, 90, 95, 99]
        percentile_labels = [f"p{p}" for p in percentiles]
        percentile_values = [float(np.percentile(data, p)) for p in percentiles]
        return dict(zip(percentile_labels, percentile_values))

    def _calculate_minmax(self, data: List[Union[int, float]]) -> Tuple[float, float]:
        min, max = np.min(data), np.max(data)
        return (float(min), float(max))

    def _calculate_std(self, data: List[Union[int, float]]) -> float:
        std = np.std(data)
        return float(std)

    def scale_data(self, factor: float = 1 / 1e6) -> None:
        for k1, v1 in self.stats_dict.items():
            if self._is_time_metric(k1):
                for k2, v2 in v1.items():
                    if k2 != "unit":
                        self.stats_dict[k1][k2] = self._scale(v2, factor)

    def _scale(self, metric: float, factor: float) -> float:
        """
        Scale metrics by the factor
        """
        return metric * factor

    def _add_units(self, key) -> None:
        if self._is_time_metric(key):
            self._stats_dict[key]["unit"] = "ms"
        elif key == "request_throughput" or "request_goodput":
            self._stats_dict[key]["unit"] = "requests/sec"
        elif key == "image_throughput":
            self._stats_dict[key]["unit"] = "pages/sec"
        elif key.startswith("output_token_throughput"):
            self._stats_dict[key]["unit"] = "tokens/sec"
        elif "sequence_length" in key:
            self._stats_dict[key]["unit"] = "tokens"
        else:
            self._stats_dict[key]["unit"] = ""

    def __repr__(self) -> str:
        attr_strs = []
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                attr_strs.append(f"{k}={v}")
        return f"Statistics({','.join(attr_strs)})"

    @property
    def data(self) -> dict:
        """Return all the aggregated statistics."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def metrics(self) -> Union[Metrics, TelemetryMetrics]:
        """Return the underlying metrics used to calculate the statistics."""
        return self._metrics

    @property
    def stats_dict(self) -> Dict:
        return self._stats_dict

    def _is_system_metric(self, attr: str) -> bool:
        if isinstance(self._metrics, Metrics):
            return attr in [m.name for m in self._metrics.system_metrics]
        return False

    def _is_time_metric(self, field: str) -> bool:
        # TPA-188: Remove the hardcoded time metrics list
        time_metrics = [
            "inter_token_latency",
            "time_to_first_token",
            "request_latency",
            "image_latency",
        ]
        return field in time_metrics

    def export_parquet(self, artifact_dir: Path, filename: str) -> None:
        max_length = -1
        col_index = 0
        filler_list = []
        df = pd.DataFrame()

        # Data frames require all columns of the same length
        # find the max length column
        for key, value in self._metrics.data.items():
            max_length = max(max_length, len(value))

        # Insert None for shorter columns to match longest column
        for key, value in self._metrics.data.items():
            if len(value) < max_length:
                diff = max_length - len(value)
                filler_list = [None] * diff
            df.insert(col_index, key, value + filler_list)
            diff = 0
            filler_list = []
            col_index = col_index + 1

        filepath = artifact_dir / f"{filename}.gzip"
        df.to_parquet(filepath, compression="gzip")
