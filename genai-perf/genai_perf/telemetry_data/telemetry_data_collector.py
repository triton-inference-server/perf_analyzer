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

import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List

import requests
from genai_perf.metrics.telemetry_metrics import TelemetryMetrics


class TelemetryDataCollector(ABC):
    def __init__(
        self, server_metrics_url: str, collection_interval: float = 1.0  # in seconds
    ) -> None:
        self._server_metrics_url = server_metrics_url
        self._collection_interval = collection_interval
        self._metrics = TelemetryMetrics()
        self._stop_event = threading.Event()
        self._thread = None

    def _fetch_metrics(self) -> None:
        """Fetch the metrics from the metrics endpoint"""
        response = requests.get(self._server_metrics_url)
        response.raise_for_status()
        return response.text

    @abstractmethod
    def _parse_metrics(self) -> None:
        """Parse metrics data. This method should be implemented by subclasses."""
        pass

    def _update_metrics(self, parsed_data) -> None:
        for metric_name, metric_values in parsed_data.items():
            if len(metric_values) > len(getattr(self.metrics, metric_name, [])):
                current_values = getattr(self.metrics, metric_name, [])
                current_values.append(metric_values)
                setattr(self.metrics, metric_name, current_values)
        print(self.metrics)

    def _collect_metrics(self) -> None:
        while not self._stop_event.is_set():
            metrics_data = self._fetch_metrics()
            parsed_data = self._parse_metrics(metrics_data)
            self._update_metrics(parsed_data)

            self.metrics.gpu_power_usage.append(parsed_data["gpu_power_usage"])
            self.metrics.gpu_power_limit.append(parsed_data["gpu_power_limit"])
            self.metrics.energy_consumption.append(parsed_data["energy_consumption"])
            self.metrics.gpu_utilization.append(parsed_data["gpu_utilization"])
            self.metrics.total_gpu_memory.append(parsed_data["total_gpu_memory"])
            self.metrics.gpu_memory_used.append(parsed_data["gpu_memory_used"])

            time.sleep(self._collection_interval)

    def start(self) -> None:
        """Start the telemetry data collection thread."""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._collect_metrics)
            self._thread.start()

    def stop(self) -> None:
        """Stop the telemetry data collection thread."""
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()

    @property
    def metrics(self) -> TelemetryMetrics:
        """Return the collected metrics."""
        return self._metrics
